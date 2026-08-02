# TopK Bench — session handoff notes

Written 2026-07-30. Lives at the **repo root** and is **committed**, so any agent or
person landing in this directory finds it and it survives a machine loss. `CLAUDE.md`
sits next to it and points here.

Read this first when resuming. The helper scripts it refers to are in `stash/`, which
*is* gitignored — so the scripts are local-only even though these notes are not. If you
are reading this from a fresh clone, `stash/` will be missing; §5 lists what was in it.

---

## 1. What this work is

Two goals, in order:

1. **Re-run the December 2025 published benchmark against TopK only**, to see how TopK
   evolved since. Blog: https://www.topk.io/blog/20251201-topk-bench (actual run date
   2025-11-27).
2. **Benchmark the two TopK shims — `topk-sql` and `topk-es` — against the native
   proto/gRPC base**, to measure translation overhead. Both shims hit the *same* backend
   and *same* collections as native, so differences are pure protocol cost.

Everything runs on **staging**: `TOPK_HOST=topk.dev`, `TOPK_REGION=sunflower`.
This node is `eu-central-1`, the same region the blog ran TopK from, so latency is
comparable — but the blog measured **production**, we measure **staging**.

---

## 1-bis. OVERNIGHT RUN 2026-07-30/31 — supersedes much of what follows

An 8-hour unattended run completed overnight. **`results-interleaved/` replaces
`./results` for every cross-provider comparison** — it is the same-session data
§4b-ter said we lacked. Ledger of what actually happened: `stash/NIGHT_PLAN.md`.
Backend errors for server-log cross-checking: `stash/backend-errors.md`.

| what | where | status |
|---|---|---|
| Query sweep, interleaved | `results-interleaved/` | ✅ 81 files, all 27 combos at n=3 |
| Ingest rounds | `results-ingest/` | ✅ 18 files, n=3 at 100k and 1m, all 3 providers |
| Ingest @10m native | `results-ingest/` | ✅ n=3 (P3c raised it from n=1; see §4d.d) |
| Ingest @10m via shims | — | ❌ not done: ~2h pgwire + ~3h es-proxy |
| Notebook wired to both | `notebooks/bench.ipynb` | ✅ 2026-08-02, see §1-ter |

**Headlines:**

1. **§4b-ter is resolved.** All three providers measured minutes apart, 3 sizes,
   3 runs, 81 files. The cross-day ~20% drift no longer contaminates anything.
2. **Shims are cheap on reads, expensive on writes** — a genuinely new axis.
   pgwire is indistinguishable from native on queries but **8x** slower to ingest;
   es-proxy is ~2x on queries and **14x** on ingest. Measured at n=3 with
   run-to-run sd under 1%.
3. **Transport errors are native-gRPC-only.** 54 errors in 1,789,563 queries
   (0.0030%), every one from the native SDK, zero from either shim against the
   same backend and collections. No latency bias (`lat_samples == oks` verified
   in all 81 files).
4. **Ingest is no longer n=1** at 100k and 1m — the gap flagged in §7 step 3.
5. **Two `topk-sql` provider bugs found and fixed** (see §4c.6). The provider had
   never written a document before tonight.

**Do not** re-derive §4b's percentages from `./results`; use `results-interleaved/`.

---

## 1-ter. NOTEBOOK (2026-08-02)

`notebooks/bench.ipynb` is wired to the overnight data and runs clean end to end
(13 cells, 14 figures, ~20 s). Cell 1 globs the directories rather than listing files,
so re-running a size or provider is picked up with no notebook edit:

- `QUERY_FILES` ← `results-interleaved/*.parquet` (81)
- `INGEST_FILES` ← `results-ingest/*_ingest_*.parquet` (21)

⚠️ The `_ingest_` in that glob is load-bearing. `*_rw_*` query files also carry
`bench.ingest.*` metrics, so a plain `*.parquet` silently contaminates all three
ingest charts with read-write-mode writes.

Cell 1 prints a runs-per-cell coverage table on load. Read it — a missing combination
otherwise just disappears from a facet with no error.

**Known gap, accepted 2026-08-02:** ingest at 10m exists for `topk` only, so the 10m
facet of Total Ingest Latency / Ingest Throughput / Freshness Quantiles shows a single
bar. Every other chart — concurrency, filtering overhead, int_filter, keyword_filter,
recall, read/write — is complete for all three providers at all three sizes, n=3.
Closing it costs ~2 h pgwire + ~3 h es-proxy; the user chose to ship with the gap
rather than spend the staging time. Ingest cross-provider claims at 10m are therefore
**not** supported by data; the 8x/14x write penalty in §1-bis headline 2 rests on 100k
and 1m only.

Verification against the ledger: 100k total ingest latency from the notebook is
native 7.1 s / pgwire 58.1 s / es-proxy 101.9 s, matching §4d exactly.

`plotly` was missing from `.venv` and was installed (`uv pip install plotly`).
Exports land in `notebooks/topk-bench-results/` (15 JSON files, gitignored).

## 2. NOTHING RUNNING (as of 13:57, 07-30)

Box is idle, load ~0.03. Both jobs finished:

- `/tmp/rerun_es2.sh` — done 12:55. All 27 post-fix topk-es files, count guard clean on
  both ends. This is the data §4b is built on.
- `/tmp/idle_recheck.sh` — done 13:46. qps @100k ×3 for all three providers on the idle
  box, in `./results-idle/` (9 files). Intended to test the pgwire ceiling; ended up
  surfacing the between-day drift instead — see **§4b-ter**, the most consequential finding
  of the afternoon.

`./results-idle/` is deliberately a separate directory: `shim_compare.py` globs
`./results` by prefix, so pooling them would silently average two conditions together.

**Next thing to run: the interleaved rerun (§7 step 7).** Approved, not started.

**Deleted 07-30:** `/tmp/rerun_es.sh`'s output (thread-local bug, §4c.5) and
`results/pre_orjson/`. Neither should be reconstructed; the numbers worth keeping from
both are recorded in §4b-bis.

**Stale orphans: killed.** PIDs 548414–548502 (30-day-old `bench-verify2` ingest probe)
are gone. 548495 was not merely hung — it was pegging **99.9% of one core for 30 days**,
a permanent 25% tax on a 4-core box, present for every measurement taken before 11:13.
It was present for every measurement taken before 11:13 — including all the native and
`topk_sql` numbers in §4a/§4b. Note §4b-ter found the day-to-day backend drift to be the
*larger* effect of the two.

---

## 3. Data collected (`./results/*.parquet`, gitignored)

Naming: `<provider>_<mode>_<size>_r<N>.parquet`. Each run writes its own file so runs
accumulate and can be aggregated.

| provider | ingest | qps | filter | rw |
|---|---|---|---|---|
| `topk` (native proto) | 3 (1 per size) | 9 ✅ | 9 ✅ | 9 ✅ |
| `topk_sql` | — (not needed) | 9 ✅ | 9 ✅ | 9 ✅ |
| `topk_es` | — (not needed) | 9 ✅ | 9 ✅ | 9 ✅ |

**Complete as of 12:55 on 07-30.** All `topk_es` files are from the post-fix rerun; the
15 pre-fix files (thread-local bug) were deleted 07-30 — see §4b-bis.

⚠️ **But these are cross-day** — proto 07-28, sql 07-29, es 07-30 — and staging drifts
~20% between days (§4b-ter). Fine for absolute latencies and for the rw/recall findings;
**not** fine for cross-provider percentages. The interleaved rerun (§7 step 7) replaces
this table.

Second directory: **`./results-idle/`** (9 files) holds the idle-box qps @100k control
runs (§2) — all three providers measured in one session. Keep it separate from
`./results`: different conditions, not poolable.

"9" = 3 sizes (100k/1m/10m) × 3 runs. Sizes are 100k / 1m / 10m docs, MS MARCO,
768-dim.

**Ingest is only n=1 per size** — never repeated (needs a collection rebuild;
10m takes ~15 min). Every query metric is n=3.

**No ingest is needed for the shims**: the shims share state with native, so
`topk-sql` and `topk-es` query the collections native already populated —
including 10m. Verified: `SELECT _id FROM "x-100k"` returns natively-ingested rows,
and ES `count` reports exactly 100000.

---

## 4. Findings

### 4a. Native TopK: Jul 2026 vs Dec 2025

After 3 runs per config with drop-worst-then-mean (the blog's own method), the
regression count fell **16/34 → 7/34**. Most apparent regressions were single-run noise.

**Evaporated under repetition** (do not trust single-run latency deltas on ~10 ms values):
- the whole apparent "10m concurrency regression" (QPS c2/c4/c8 −6/−17/−21%,
  p99 lat +13/+40/+39%) — I reported this as a real finding at n=1; it was noise
- 100k latency oddities (kw 100% +73%, kw 1% +37%, c4 +47%)

**Survived — two clusters:**

1. **Read-write under concurrent writes** (the strongest result in the rerun):
   latency **+1078% / +777% / +146%**, QPS **−44% / −41% / −15%** at 100k/1m/10m.
   Key diagnostic: rw-on latency is ~161 ms @100k and ~154 ms @1m — *near-identical
   despite 10× the data* — while rw-off scales normally (~9 ms both). Size-independence
   rules out contention/saturation and points at a **fixed timing constant**
   (flush interval, retry backoff, lock timeout) introduced since December.

   ✅ **CONFIRMED SERVER-SIDE 07-30, across all three protocols** (n=3 each, same
   backend and collections):

   | rw=true p99 | proto (gRPC) | pgwire | es-proxy |
   |---|---|---|---|
   | 100k | 159 ms | 135 ms | 170 ms |
   | 1m | 154 ms | 139 ms | 164 ms |
   | 10m | 168 ms | 158 ms | 179 ms |

   ~135–180 ms regardless of size **and regardless of protocol**, while rw=false scales
   normally at every size (proto 8/9/25 ms). This retires the two live objections:
   - **Not a client artifact.** Three independent client stacks — Rust gRPC, psycopg over
     pgwire, Python HTTP+JSON — land on the same floor. Worth stating explicitly because
     size-independence *alone* is not sufficient evidence: it produced a false positive on
     es-proxy the same week (§4c.5). Three protocols agreeing is what makes it solid.
   - **Not the orphan core** (§2). The es-proxy rw runs were measured after the orphan was
     killed, on an idle box, and still show ~170 ms.

   ⛔ **Staging-vs-production is the only remaining explanation, and it will not be
   tested.** User ruled production out entirely on 07-30 — sunflower staging only, no prod
   runs, no prod ingest. Treat this ambiguity as permanent and write it up as a caveat;
   do not carry a prod comparison as pending work.

   Incidental, unexplained: **pgwire beats native gRPC under write contention** — −6 to
   −15% latency and +15 to +30% QPS, consistently across all three sizes at n=3.
2. **Ingest + freshness degrade with scale** ⚠️ **still n=1**:
   - throughput: +0.5% @100k → **−13.9% @1m** → **−22.3% @10m** (monotonic)
   - ingest time: −0.5% / +16.2% / +28.6%
   - freshness p99 @10m: 630 ms → **6428 ms (+920%)**; p90 374 → 2162 ms
   - Verified *not* a harness artifact: writer starvation was 0.1–3.7%, and
     `avg_batch × 8 writers ÷ p50_batch_latency` reproduces observed throughput, so it
     is bounded purely by per-batch upsert latency (1220 → 1321 → 1401 ms by size).
   - The S3 download is **not** inside the timing window (`open_file` completes before
     metrics start) — do not cite it as a confound.

**Clean at every size:** recall (identical to 3 decimals everywhere), all filtered-query
latency/QPS (improved), all concurrency QPS/latency.

### 4b. Shims vs native proto — COMPLETE (qps + filters + rw, 3 sizes, n=3)

Same data, same backend, same collections. All ES numbers here are post-§4c.5-fix.
Reproduce with `.venv/bin/python stash/shim_compare.py`.

**Headline: pgwire is free, es-proxy costs ~6 ms per request.**

Cleanest measurement — 30 identical vectors, warm connections, single-threaded, x-100k,
idle box, all three drivers in one process:

| | p50 | vs proto |
|---|---|---|
| proto (gRPC) | 7.92 ms | — |
| **pgwire** | 7.13 ms | **−0.8 ms** |
| **es-proxy** | 14.15 ms | **+6.2 ms** |

pgwire is *faster* than native gRPC here. That is within noise, so read it as
"indistinguishable from native", not "faster" — but there is no translation penalty to
speak of. It only becomes visible at 10m (+2–3 ms, +10–19%).

**Where the es-proxy 6 ms is NOT.** Four candidates ruled out by direct measurement:

| candidate | measurement | verdict |
|---|---|---|
| Python es client | raw `http.client` w/ pre-encoded body 12.75 ms vs `elasticsearch-py` 14.05 ms | only 1.3 ms |
| HTTP/TLS framing | es-proxy trivial-request floor **2.81 ms** vs pgwire floor **3.40 ms** | es-proxy floor is *cheaper* |
| payload size | es body 14,902 B vs sql literal 14,834 B, identical float precision | identical |
| engine time | shim self-reports `took=1 ms` | negligible |

So the ~6 ms sits inside the proxy's own per-query request path. Pinning it down further
needs server-side profiling of topk-es, which cannot be done from this node.

⚠️ Do **not** restate this as "most of the ES cost is the Python client" — that was an
intermediate hypothesis this session, drawn from `took=1 ms`, and the measurement above
disproved it (1.3 of 6.2 ms).

**Not perfectly fixed.** The penalty grows with size and concurrency: ~6–7 ms @100k/1m
single-threaded, ~9–10 ms @10m, ~15 ms @10m c8 (94.5 → 109.5). Relative impact therefore
*falls* as queries get more expensive: es-proxy p99 Δ is +82…+108% @100k but only
+16…+40% @10m. Its QPS gap also narrows with concurrency (−35% → −17% @10m), i.e. the
overhead pipelines rather than serializing.

**Correctness: settled.** Recall matches native to three decimals in **all 18 filter
configurations** — both shims, both filter types, every selectivity, all three sizes.
Largest disagreement anywhere is 0.001.

~~**Open — possible client-side ceiling in pgwire.**~~ ✅ **RESOLVED 07-30 — it was not
real.** The apparent ~650 QPS c8 stall (648 @100k, 653 @1m, against proto's 599 → 1057
over c4 → c8) came from comparing sql measured 07-29 against proto measured 07-28.
Re-measured in the same session, sql c8 reaches **787 QPS** and sits **−5.6%** vs proto,
not −38.7%. See **§4b-ter**.

⚠️ **All percentages in this section are cross-day and carry ~20% of drift.** Read §4b-ter
before quoting any of them. The absolute latencies and the single-process probe above are
unaffected; the QPS deltas are.

### 4b-bis. Older topk-es generations — DELETED, numbers preserved here

There were three generations of ES data. Only the current one remains on disk:

| where | when | provider state | status |
|---|---|---|---|
| ~~`results/pre_orjson/`~~ (27 files) | 07-29 | stdlib json, shared client | 🗑️ **deleted 07-30** |
| ~~`stash/tainted-es-20260730/`~~ (15 files) | 07-30 09:28–11:14 | orjson + **thread-local client** | 🗑️ **deleted 07-30** (was invalid, §4c.5) |
| `results/` (27 files) | 07-30 11:16–12:55 | orjson + shared client | ✅ current |

**`pre_orjson` needed no regeneration** — the 07-30 run covers exactly the same matrix
(qps/filters/rw × 100k/1m/10m × n=3, 27 files), so the current set already *is* the
regenerated version. Verified by comparing coverage before deleting.

**§4b's withdrawn ES table came from `pre_orjson/`.** Verified before deletion: its 10m
qps values (38.5/53.5/72.0/142.0, QPS 30.3/51.0/79.8/93.4) reproduced that table exactly.
That is why the withdrawn numbers were never reproducible from the current code.

**What the serializer was worth, measured in the harness** (qps @100k; note this also
spans the orphan kill, but the effect is far too large to be a 25% CPU change).
⚠️ The source files are gone, so these figures now live *only* here and cannot be
re-derived:

| conc | pre_orjson QPS | current QPS |
|---|---|---|
| 1 | 69.2 | 76.8 |
| 2 | 133.4 | 149.4 |
| 4 | 144.3 | 286.6 |
| 8 | **142.8** | **462.7** |

The stdlib-json run **plateaus at ~143 QPS from c4 onward** — it stops scaling with
concurrency altogether. At 10m the two runs are nearly identical (30.3 vs 31.8 QPS at c1),
because a slower query dilutes any fixed client-side cost.

**Generalisable tell:** a QPS curve that goes flat across rising concurrency means the
*client* has saturated, not the backend. It appeared here for stdlib json (~143), and it
is exactly what pgwire's ~650 plateau looks like.

### 4b-ter. ⚠️ METHODOLOGY: staging drifts ~20% between days — §4b percentages are NOT publishable

Discovered 07-30 by re-running qps @100k ×3 for all three providers on the idle box
(`results-idle/`) and comparing against the originals.

**proto measured today is ~21% SLOWER than proto measured 07-28 — despite having an extra
core free.** That is the opposite of the expected direction, so it is not the orphan.

Per-run QPS @100k c8:

```
topk      07-28:  801.4  1053.9  1060.8     <- 32% spread within one session
topk      07-30:  821.6   826.2   840.5     <- +-2%
topk_sql  07-29:  625.1   656.9   639.2
topk_sql  07-30:  785.3   788.4   768.1
```

Within-session reproducibility is fine (±2%), and the es-proxy control — both columns
measured the same day, two hours apart — drifts only 1–5%. So this is a **between-day**
effect on staging, roughly **20%**.

**Why it poisons §4b:** proto was measured 07-28, sql 07-29, es 07-30. Every
cross-provider percentage in §4b therefore carries ~20% of day-effect on top of signal.

**Aggregation makes it worse.** Drop-worst-then-mean at n=3 is just "mean of the best 2".
Given 801/1054/1061 it returns 1057, so a single lucky run sets the result. This is the
blog's own method, kept for comparability — but it is fragile at n=3 and biases toward
whichever day happened to run hot.

**Casualty: the pgwire "~650 QPS ceiling" was not real.** On the idle box sql c8 reached
787 QPS (p99 23.5 → 17.0 ms). Measured in the *same session*, sql c8 is **−5.6%** vs
proto, not −38.7%. Strike that outlier from §4b.

**Same-session comparison — the only trustworthy percentages we currently have**
(all three providers, one sitting, qps @100k, QPS):

| conc | proto | pgwire | Δ | es-proxy | Δ |
|---|---|---|---|---|---|
| 1 | 136.9 | 155.6 | **+13.7%** | 72.8 | −46.8% |
| 2 | 273.6 | 311.7 | +13.9% | 146.3 | −46.5% |
| 4 | 539.2 | 587.5 | +9.0% | 282.9 | −47.5% |
| 8 | 833.4 | 786.9 | −5.6% | 457.5 | −45.1% |

Notice how much more coherent this is than the cross-day version: pgwire lands slightly
*faster* than native except at c8, and es-proxy sits at a flat ~−46% rather than
wandering between −52% and −56%.

**Survives the drift unharmed** — cite these freely:
- the single-process probe (proto 7.92 / pgwire 7.13 / es-proxy 14.15 ms) — all three
  measured back-to-back in one process, so day effects cancel
- recall exactness (§4b)
- the rw regression (§4a.1) — a 20× effect dwarfs a 20% wobble
- es-proxy vs pgwire at ~2× — same reasoning

**Does not survive:** any §4b percentage smaller than ~25%.

**Fix:** one interleaved run — all three providers, same session, per size, rather than
one provider per day. See §7 next step 7.

### 4d. OVERNIGHT INTERLEAVED RESULTS (2026-07-30/31) — the publishable set

All from `results-interleaved/` (queries) and `results-ingest/` (ingest). Every
provider measured minutes apart, n=3, one session. These supersede §4b's
percentages. Reproduce by pointing `shim_compare.py` at `results-interleaved`
(it hardcodes `../results` in `RESULTS`).

**a. QPS vs native, same session.** The pgwire penalty *shrinks* with size; the
es-proxy penalty does not.

| conc | 100k sql / es | 1m sql / es | 10m sql / es |
|---|---|---|---|
| 1 | −16.8% / −46.9% | −5.4% / −32.3% | −5.6% / −45.4% |
| 2 | −17.4% / −48.2% | −6.8% / −32.1% | −4.3% / −29.8% |
| 4 | −10.9% / −44.0% | −0.3% / −26.7% | −2.9% / −27.5% |
| 8 | −13.8% / −47.6% | −4.4% / −22.4% | −0.9% / −28.4% |

At 10m c8 pgwire is within 1% of native. At 100k it is 11–17% behind — the
opposite of §4b's "pgwire is free at 100k, costs at 10m", which was a cross-day
artifact. Absolute p50s: native 6.0/8.3/25.3 ms at c1 by size.

**b. Read-write under writes — §4a.1 CONFIRMED at n=3, same session.**

| size | native off→on p99 | pgwire off→on | es-proxy off→on |
|---|---|---|---|
| 100k | 7.3 → 155.3 ms | 8.7 → 181.3 | 13.0 → 232.3 |
| 1m | 9.7 → 156.0 | 10.0 → 176.3 | 15.0 → 171.3 |
| 10m | 26.7 → 199.3 | 27.3 → 198.7 | 34.3 → 188.0 |

rw=off scales normally with size; rw=on sits at ~155–230 ms regardless of size
**or protocol**. QPS falls 57–82%. The fixed-timing-constant reading stands.

⚠️ **§4a.1's incidental "pgwire beats native gRPC under write contention"
(−6 to −15% latency, +15 to +30% QPS) DOES NOT REPRODUCE.** Same-session, pgwire
is *worse* than native under writes at every size (p99 181 vs 155 @100k; QPS 25.6
vs 51.9). That observation was cross-day noise — strike it.

**c. Ingest — the shims are cheap on reads and expensive on writes.** n=3,
run-to-run sd < 1%.

| size | native | pgwire | es-proxy |
|---|---|---|---|
| 100k | 7.1 s / 47.3 MB/s | 58.1 s / 5.8 | 101.9 s / 3.3 |
| 1m | 73.6 s / 46.0 | 676.8 s / 5.0 | 1053.0 s / 3.2 |

pgwire **8x** and es-proxy **14x** native. This is a new axis: pgwire is
indistinguishable from native on queries yet 8x slower to ingest. Note this is
*after* the 73x fix in §4c.6 — the remaining gap is the shim's, not the client's.
Freshness p99 does **not** follow write cost: es-proxy (601/537 ms) matches native
(509/554) while pgwire is ~3x worse (1306/1631).

**d. Native ingest by size — n=3 at EVERY size. §4a.2 is refined, not just
confirmed: the degradation is a freshness *tail* effect, not a throughput story.**

| | 100k | 1m | 10m | 10m vs 100k |
|---|---|---|---|---|
| throughput | 47.34 MB/s | 45.95 | 42.12 | **−11%** (sd < 1 MB/s at every size) |
| batch p50 | 1120 ms | 1149 | 1250 | +12% |
| freshness p50 | 222 ms | 225 | 281 | **+27%** |
| freshness p90 | 380 ms | 355 | 10,431 | **×27** |
| freshness p99 | 509 ms | 554 | 23,622 | **×46** |

This is the sharpest result of the night. Throughput barely moves and is highly
reproducible (sd < 1 MB/s). Freshness **p50 barely moves either** — 222 → 281 ms.
The entire effect is in the tail: p90 jumps 27x and p99 46x. So at 10m most writes
are visible promptly and a minority take tens of seconds. §4a.2's headline
"freshness p99 +920%" is real but describes only the tail.

⚠️ **The tail is also unstable**: per-run p99 @10m was 24,911 / 34,069 / 11,887 ms
— a 2.9x spread, against throughput's sub-1% spread in the same three runs. Any
single-run freshness p99 at 10m is nearly meaningless. Quote the range, not a
point.

Independent corroboration: after each 10m ingest the collection count needed
minutes to converge to 10,000,000 (§4c.7) — the same write-to-visible lag seen
from the other side.

**e. Errors are native-gRPC-only.** 54 in 1,789,563 queries (0.0030%), all from
the native SDK, zero from either shim against the same backend and collections.
Two signatures: `Cancelled/connection closed` and `h2 protocol error`. Verified
`lat_samples == oks` in all 81 files, so no file carries biased latency. Full
timestamps in `stash/backend-errors.md`.

**Not done:** ingest at 10m through the shims (~2 h pgwire + ~3 h es-proxy by
extrapolation). Everything else in the plan completed.

### 4c. Bugs found

1. **`topk-es`: `_bulk` rejects `PUT` with 405.** Real Elasticsearch routes both PUT and
   POST. The official Python client sends **PUT**, so `client.bulk()` *and*
   `helpers.bulk()` — the standard ingest path — fail out of the box. Reproduced on
   es-py 8.19.3 and 9.4.1, so not a client-version issue; raw `POST` works at every
   content-type. Worked around in the provider via `perform_request("POST", ...)`.
   **Worth filing against topk-es.**
2. **`topk-es`: no index enumeration.** `*`, `_all`, `_cat/indices`, `_stats` are all
   parsed as index *names* and rejected as invalid. `list_collections()` falls back to
   the native SDK (housekeeping only, never on a benchmarked path).
3. **TopK ingest can silently drop docs** (deprioritized by user): with topk_sdk 0.7.1 at
   concurrency 8, one 100k ingest landed only 52,000/100,000 — all batches ACKed, 0
   errors, all 100k `upserted_docs` recorded, yet a *contiguous* id block (~12k–58k)
   never persisted and did not self-heal. A re-ingest landed all 100k. **This produced a
   bogus "recall −58%"** in the first comparison; once complete, recall matched baseline
   exactly. Always verify `count == dataset size` before trusting query numbers.
6. **OUR bug x2, not TopK's: `topk_sql` had never written a document.** Found by
   `stash/smoke_ingest.py` on 2026-07-31 before any long run was queued behind it.
   - **upsert ran at 4 docs/s** (native 295, es 94 on the same data and session).
     A/B of four strategies: `executemany()` 4.0, explicit `conn.pipeline()` 4.0,
     **multi-row VALUES 236.9**, `COPY` unsupported by the shim (parse error).
     psycopg pipelines `executemany` only when the server supports it; this shim
     does not, so every row cost a full ~250 ms round-trip. Fixed by sending one
     multi-row statement per chunk of 2000 rows (~30 MB, measured working,
     2639 docs/s at that width). Result 4 -> 292 docs/s. At the old rate a 100k
     ingest would have taken 6.9 h and 1m would have taken 69 h.
   - **`DEALLOCATE ALL` rejected by the shim**, so psycopg's pool reset blew up
     `delete_collection()` once any statement had been prepared. Fixed with
     `prepare_threshold=None`. Real PostgreSQL supports it — third shim gap after
     §4c.1 and §4c.2, worth filing.
   - The pool change touches the query path too, so it was A/B'd before committing
     5 h of benchmarking to it: p50 6.21 ms with prepares off vs 6.97 ms on — no
     penalty. Parity re-checked after both fixes: 100% on all 15 cases.

7. **Count guard at 10m needs settle-and-retry — it produced a false failure.**
   A native 10m ingest read 9,938,000/10,000,000 immediately on completion and
   10,000,000 six minutes later. This is NOT §4c.3: that drop never self-healed
   and left a contiguous id block permanently absent. `stash/ingest_rounds.sh` now
   retries for up to 5 min before declaring failure. A genuine drop does not
   converge, so retrying cannot mask one. The same run logged exactly 1 batch
   error in 4,994 requests — the only ingest-side error of the night, against 18
   clean runs at 100k and 1m.

4. **`src/s3.rs` dropped the session token** → any STS/SSO credential failed with a bare
   `"service error"`. Fixed (committed).
5. **OUR bug, not TopK's: thread-local ES client caused a TLS handshake per query.**
   `topk_es.py` was changed at 09:28 to keep "one client per thread" via
   `threading.local()`, on the theory that per-thread connection pools remove pool-lock
   contention. The harness dispatches every query through
   `tokio::task::spawn_blocking` (`src/provider.rs:138`), and that pool churns threads
   rather than pinning them, so the thread-local missed on nearly every call:
   **58 clients constructed for ~60 queries at concurrency=1**. Each miss paid a fresh
   TLS handshake — ~70 ms against the shim, where TCP connect is ~2 ms and the query
   itself ~14 ms.
   - Signature to recognise next time: a **size-independent latency floor**. Recorded ES
     p50 was 84 ms @100k and 106 ms @10m — 100× the data for 26% more latency — plus QPS
     pinned near 1/handshake (~12–17) regardless of concurrency, while latency grew
     linearly with concurrency. Same "fixed timing constant" reasoning as §4a.1.
   - A/B over the same 10 s window: thread-local **5–6 QPS / p99 285 ms**, shared client
     **70 QPS / p99 19 ms**. After the fix, with the orphan core freed: **78 QPS @c1,
     444 QPS @c8** — up to ~26×.
   - Fixed by reverting to a single shared client with `connections_per_node=16`
     (elastic_transport's pool is already thread-safe). The provider now carries a
     comment explaining why this must not be "optimised" back.
   - `topk_sql` is unaffected — it uses a shared `psycopg_pool.ConnectionPool`, and its
     files predate this. Native `topk` never goes through Python.

---

## 5. Code state

**Branch `jergus/local-bench-runner`, pushed to `topk-io/bench` at `cf492d3` (07-30).
Working tree clean. 3 commits ahead of `main`:**
- `fcbe5c9` `fix(s3): support temporary/STS credentials` — read `AWS_SESSION_TOKEN`
  (was hard-coded `None`). Independently useful.
- `8aa0f5f` `add local.py` + gitignore `.env`
- `cf492d3` `add topk-sql and topk-es providers for shim-vs-proto benchmarking` —
  both providers, `__init__` exports, `--provider` flag, deps, lockfile.

**Draft PR: NOT YET OPENED.** `gh` is not installed on this node and no GitHub API token
is available (SSH push works, API does not). Body is written and ready at
`stash/PR_BODY.md`; open it by hand at:

```
https://github.com/topk-io/bench/compare/main...jergus/local-bench-runner?expand=1
```

Tick **"Create draft pull request"**. Alternatively install `gh` and
`gh auth login`, then `gh pr create --draft --body-file stash/PR_BODY.md`.

**In `/stash` (gitignored):**
- `baseline-2025-12/*.json` — 15 files, the December TopK numbers, scraped from the blog's
  client-side data at `https://www.topk.io/topk-bench-results/*.json` (the page renders
  charts from these; a plain fetch only shows "Loading chart data...")
- `compare.py` — aggregation + diff vs baseline. Paths anchored to the file, not cwd.
- `compare_table.py` — side-by-side table; `--regressions` for the filtered view
- `shim_compare.py` — shim-vs-**today's-native** comparison (the one to use for §4b)
- `parity_check.py` — cross-provider correctness gate (see §7)
- `topk-es-bugs.md` — the two shim bug reports, filed by user 07-30; kept for repro steps
- `PR_BODY.md` — draft PR description, ready to paste

---

## 6. Environment / setup gotchas

- **`.env`** holds `TOPK_API_KEY`, `TOPK_REGION=sunflower`, `TOPK_HOST=topk.dev`, and AWS
  creds. Gitignored (added this session — it was previously at risk of being committed).
  AWS creds are **temporary/SSO** and expire; refresh with
  `aws --profile public configure export-credentials --format env-no-export`.
  `get-session-token` does *not* work on that profile ("Cannot call GetSessionToken with
  session credentials"), and role-chained profiles cap at 1 h regardless.
- **`AWS_REGION=us-east-2`** — the `topk-bench` bucket lives in us-east-2. `src/s3.rs`
  builds the endpoint from `AWS_REGION` and cannot redirect, so a wrong region fails with
  an opaque `"service error"`. This cost significant debugging time.
- **Datasets are public over plain HTTPS**, so no AWS creds are needed if the cache is
  pre-populated: `pull_file()` returns early when `{cache_dir}/{key}` exists, before the
  S3 client is constructed. Cache: `/tmp/topk-bench/`.
- **Downloads: use `s5cmd`, not curl.** Bucket is us-east-2, node is eu-central-1
  (~105 ms RTT); a single TCP stream got 25 MiB/s, while
  `s5cmd --no-sign-request cp -c 32 -p 64` got **128 MiB/s** (54 GB in ~7 min).
  `~/.local/bin/s5cmd` is installed.
- **Disk was resized** 80 GB → 256 GB (`vol-04f6ce53e5258cd63`, acct 508153278359,
  eu-central-1) because `docs-10m.parquet` is 54.3 GB and only 33 GB was free. Done live
  with `growpart /dev/nvme0n1 1` + `resize2fs /dev/nvme0n1p1`, no reboot. 149 GB free now.
- **Build is maturin/uv**: `uv sync`, then `uv run` or `.venv/bin/python`.
  `polars` was added to the venv manually for the compare scripts.
- **`uv sync` resolves `elasticsearch>=8.15` to 9.x**, replacing any manual 8.x pin. The
  `_bulk` fix is version-independent, so this is fine — but re-run the parity check after
  any sync.

---

## 7. How to resume

```bash
cd /home/ubuntu/Code/bench
set -a; source .env; set +a
```

**Nothing is running** as of 13:57 on 07-30, and the working tree is clean. Cold-start
checklist, in order:

1. **Expired AWS creds are EXPECTED and FINE — do not go refresh them.** The ones in
   `.env` expired 2026-07-28T12:17. Every dataset is already cached in `/tmp/topk-bench/`
   (all six files, 100k/1m/10m × docs+queries, ~70 GB), and `pull_file()` returns early
   when the cached file exists, *before* the S3 client is constructed. `local.py`'s
   `_require_env()` only checks the vars are present, not valid. Every run on 07-30 was
   done on expired creds. Only refresh if `rerun_interleaved.sh`'s cache precheck fails.
2. **Collection count guard** — cheap, and non-negotiable after §4c.3 (ingest has silently
   dropped docs before). `rerun_interleaved.sh` does this automatically on both ends.
3. **Correctness gate** — run before trusting any new numbers:
   ```bash
   .venv/bin/python stash/parity_check.py --size 100k
   ```
   One identical vector through all three providers, five filter selectivities, top-10
   overlap. Expect **100% on nearly everything**, with an occasional 90% at `int 1%` —
   that is ordinary ANN variance (native recall there is 0.991) and it moves between
   providers run to run: on 07-29 topk-es showed it, on 07-30 topk-sql did. Not a
   regression. **Near-zero** overlap for topk-sql would be real — it means the SQL
   `ORDER BY` direction is inverted; see `_ORDER` in `topk_sql.py`.

**Compare results:**
```bash
.venv/bin/python stash/shim_compare.py                 # shims vs TODAY's native (§4b)
.venv/bin/python stash/compare_table.py                # native vs December (§4a)
.venv/bin/python stash/compare_table.py --regressions  # only what got worse
```

**Run a benchmark by hand:**
```bash
.venv/bin/python local.py qps --provider topk-sql --size 1m   # modes: ingest|qps|filters|rw|all
.venv/bin/python local.py all --provider topk --size 100k --runs 3
```
⚠️ For anything cross-provider, use `stash/rerun_interleaved.sh` instead — measuring one
provider at a time is what produced the §4b-ter problem.

### Immediate next steps

1. ~~TOOLING GAP~~ **DONE.** `stash/shim_compare.py` diffs `topk_sql`/`topk_es` against
   **today's native** numbers (not December's), reusing `compare.py`'s aggregation.
   Verified working. `compare.py`/`compare_table.py` remain December-only, by design.
2. ~~Full three-provider comparison~~ **DONE.** Complete across qps/filters/rw × 3 sizes,
   n=3. Results in §4b; rw confirmation in §4a.1.
3. ~~Repeat ingest at 1m ×3~~ **SKIPPED — user's call, 07-30.** Ingest therefore stays at
   **n=1 and is the one unrepeated finding in the whole rerun**. Given that most apparent
   regressions dissolved under repetition, treat §4a.2 as provisional in any writeup and
   say so explicitly rather than quietly reporting it alongside the n=3 results.
4. ~~File the two `topk-es` bugs~~ **DONE — filed by user, 07-30.** Drafts retained in
   `stash/topk-es-bugs.md` for reference (repro steps, version scope, workarounds).
5. ~~Isolate the read-write regression against production~~ ⛔ **RULED OUT — user, 07-30.
   Sunflower staging only; no prod runs, no prod ingest.** Do not re-propose this. The
   staging-vs-production ambiguity in §4a.1 is permanent; write it up as a caveat.
7. 🔴 **RERUN EVERYTHING, INTERLEAVED — top priority, user approved 07-30.** The
   ~20% between-day drift (§4b-ter) makes the current §4b percentages unpublishable, because
   each provider was measured on a different day. The fix is ordering, not more runs:
   loop **provider innermost** so all three are measured minutes apart under identical
   backend conditions.

   **Ready to launch — `stash/rerun_interleaved.sh`.** Written and dry-run verified
   07-30 (cache check + both count guards pass, clean exit). It is in `/stash` rather than
   `/tmp` so it survives a reboot.

   ```bash
   cd /home/ubuntu/Code/bench
   setsid nohup ./stash/rerun_interleaved.sh > /tmp/interleaved.log 2>&1 < /dev/null &

   pgrep -x -f "/bin/bash ./stash/rerun_interleaved.sh"
   grep -aE '^###|INTERLEAVED DONE' /tmp/interleaved.log | tail -5
   ```

   Env-var knobs, all optional: `MODES` `SIZES` `RUNS` `PROVIDERS` `BENCH_RESULTS_DIR`.
   - full sweep ≈ 6–8 h (default)
   - `MODES="qps rw"` ≈ 2 h if that is too much
   - `RUNS=5` is worth considering: drop-worst-then-mean is "best of 2" at n=3, which is
     precisely how one lucky run set proto's 07-28 number

   Baked in already: provider as the innermost loop (the whole point — the header warns
   against hoisting it back out), a fresh `./results-interleaved` output dir so the
   cross-day data in `./results` is never pooled in, count guards on both ends, and a
   dataset-cache precheck.

   Afterwards, point `shim_compare.py` at the new directory — it currently hardcodes
   `../results` in `RESULTS`.
6. **Decide whether the orphan core changes anything.** All native and `topk_sql` numbers
   were taken with one of four cores permanently consumed (§2). It was constant across
   every run *and* across the December-vs-July comparison's July side, so relative deltas
   should hold — but the §4a.1 read-write finding is the one conclusion where a starved
   client could plausibly matter. A cheap check: repeat `rw` at 100k ×3 now that the core
   is free and see whether the +1078% survives.

### Process hygiene (three self-inflicted stalls this session)

- `pkill -f 'docs-10m.parquet'` matched **its own shell command** and killed it.
- `while pgrep -f 'repeats.sh'` in `/tmp/after_repeats.sh` matched **its own filename**
  (substring), so the script waited on itself forever and deadlocked a queued job behind it.
- A provider "optimisation" was landed at 09:28 and a multi-hour rerun kicked off at
  09:28:48 — **without a before/after measurement**. It cost ~1.75 h of machine time and
  produced 15 unusable files. Any change to a provider's connection handling needs a
  60-second A/B against the harness before a long run is queued behind it.

Kill and wait by **explicit PID** or `pgrep -x`, never a substring pattern that matches
the watcher itself.

### User preferences

- **Never publish Artifacts or send data off-machine.** Terminal output only. This was
  violated once (a comparison table published to claude.ai) and the user reacted strongly;
  artifacts also cannot be deleted by the assistant once published.
- Prefers branches over committing to `main` (`jergus/<topic>` convention).
