# Working in this repo

**Reset on 2026-08-02.** All prior results and the old `SESSION_NOTES.md` are archived
under `stash/archive-2026-08-02/` (local only — `stash/` is gitignored, so from a fresh
clone they are simply absent). They were retired rather than deleted because two defects
invalidated the query numbers in them:

- `/_search` silently ignored `_source_includes`, so every `topk-es` hit carried its
  768-float embedding. ES query numbers there are inflated. Fixed 2026-08-02.
- A stale `ES_URL` in the tmux server environment pointed the `topk-es` arm at an
  unrelated Elastic Cloud project for part of one run, producing parquet files with
  `oks = 0` that still looked well-formed.

Both were invisible until someone read the numbers, which is why measurement now starts
with `stash/preflight.py`: it asserts endpoint identity, collection counts and
cross-client result parity **before** any timing, writes a `manifest.json` recording the
conditions next to the results, and exits non-zero so a sweep refuses to start on a
broken setup. Run it first; treat a run without a manifest as unattributable.

## Where things run

The repo is checked out twice. **Edit on the laptop, execute on the VM.**

- Laptop `~/Code/bench` — where Claude Code runs and where code is edited. No benchmark
  ever runs here: the datasets, staging creds, `stash/` and every `results*/` dir live
  only on the VM.
- VM `ssh bench` (`ubuntu@bench:~/Code/bench`) — where every run happens.

Sync is git on the working branch, not rsync of source:

```sh
git push                                        # laptop
ssh bench 'cd ~/Code/bench && git pull --ff-only'
```

After a squash + force-push the VM's `pull --ff-only` fails on diverged history. The VM
holds no commits of its own worth keeping — its working tree is where runs happen, not
where work is authored — so re-point it instead:

```sh
ssh bench 'cd ~/Code/bench && git fetch origin && git reset --hard origin/<branch>'
```

Check `git status` on the VM first: `reset --hard` discards any edit made there.

Data stays on the VM and is pulled per-run, on demand:

```sh
mkdir -p results && rsync -avz bench:~/Code/bench/results/<file> results/
```

`mkdir -p` first: the laptop ships openrsync (advertises "2.6.9 compatible"), which has
neither `--mkpath` nor `--info=` — it prints its usage instead of an error, so a missing
destination dir looks like a silent no-op.

`results/`, `results-idle/`, `results-interleaved/`, `results-ingest/`, `stash/` and
`target/` are gitignored — they exist on the VM only, and a local checkout will be
missing them. Don't recreate them locally; fetch the one run being looked at.

## The notebook

`notebooks/bench.ipynb` reads the `results*/` dirs, so its kernel must run on the VM.
Two ways to see it on the laptop:

**Live (edit and re-run cells)** — Jupyter on the VM, reached through an ssh tunnel:

```sh
ssh bench 'cd ~/Code/bench && tmux new -d -s jupyter \
  ".venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8888 > /tmp/jupyter.log 2>&1"'
ssh bench '~/Code/bench/.venv/bin/jupyter server list'   # grab the token URL
ssh -N -L 8888:localhost:8888 bench                      # leave running, then open the URL
```

`--ip=127.0.0.1` is not optional: the VM has a public IP, and binding `0.0.0.0` would put
a token-authed kernel with shell access on the open internet. The tunnel is the only path in.

**Static (just look at the figures)** — execute headless on the VM, pull the result:

```sh
ssh bench 'cd ~/Code/bench && .venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/bench.ipynb --output-dir /tmp --output bench-executed.ipynb \
  --ExecutePreprocessor.timeout=600'
rsync -avz bench:/tmp/bench-executed.ipynb notebooks/
```

~15 s, 15 figures. The plotly output is embedded in the `.ipynb`, so the pulled copy
renders on the laptop with none of the parquet present. `*-executed.ipynb` is gitignored.

Launch anything long detached, so a dropped ssh can't kill it:

```sh
ssh bench 'cd ~/Code/bench && tmux new -d -s bench "uv run local.py ... 2>&1 | tee /tmp/bench.log"'
ssh bench 'tail -50 /tmp/bench.log'             # poll
```

## Hard constraints

- **Staging only.** `TOPK_HOST=topk.dev`, `TOPK_REGION=sunflower`. Never production,
  never a production ingest — not even to settle a benchmark question. Stated directly by
  the user on 2026-07-30.
- **Never publish results off-machine.** Terminal output only. No Artifacts, no uploads.

## Things that have already gone wrong here — don't repeat them

- **Measure before queueing a long run.** A provider "optimisation" was landed and a
  multi-hour rerun started 48 seconds later, with no before/after check. It cost ~1.75 h
  and 15 unusable files. A 60-second A/B would have caught it. See §4c.5.
- **A size-independent latency floor usually means a fixed per-request cost — and it can
  be on the client side.** That signature produced one true finding (§4a.1) and one false
  one (§4c.5) in the same week. Rule out the client before concluding "server".
- **Never measure one provider per day.** Staging drifts ~20% between days while
  within-session spread is ~2%, so cross-day provider comparisons are mostly noise. Use
  `stash/rerun_interleaved.sh`, which interleaves providers. See §4b-ter.
- **Always count-guard collections before trusting query numbers.** Ingest has silently
  dropped docs (all batches ACKed, 0 errors, 52k/100k landed). See §4c.3.
- **Kill and wait by explicit PID or `pgrep -x`.** A substring `pkill`/`pgrep` has twice
  matched the command issuing it and deadlocked or self-killed.

## Layout

- `local.py` — Modal-free runner: `--provider {topk,topk-sql,topk-es}`, `--size`, `--runs`
- `stash/` — gitignored scratch: comparison scripts, December baseline, rerun script
- `stash/preflight.py` — Phase 0: identity + parity gates, writes `manifest.json`
- results live in a per-sweep directory named by its `sweep_id`; the manifest beside
  them records endpoints, collection counts, params and client versions
- Expired AWS creds in `.env` are expected and harmless; datasets are cached in
  `/tmp/topk-bench/`. Don't go refreshing them. See §7.
