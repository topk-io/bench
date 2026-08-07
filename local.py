#!/usr/bin/env python3
"""
Local, Modal-free benchmark runner for TopK only.

Mirrors the run_*_bench functions in bench.py but:
  - drops Modal (runs in-process on this node)
  - runs only the `topk` provider
  - writes metrics to local ./results/*.parquet instead of S3

Datasets are still read from the public s3://topk-bench/ bucket, which the
Rust S3 client signs with the AWS_* env vars (any valid creds work; the
bucket is public). TopK creds come from TOPK_API_KEY / TOPK_REGION.

Both are loaded from ./.env (see load_env below).

Usage:
    python run_local.py ingest  --size 100k
    python run_local.py qps      --size 100k
    python run_local.py filters  --size 100k
    python run_local.py rw       --size 100k
    python run_local.py all      --size 100k --size 1m      # ingest+qps+filters+rw
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---- minimal .env loader (no external dependency) ------------------------
def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

import topk_bench as tb  # noqa: E402

# ---- config (matches bench.py: eu topk => batch_size=2000, concurrency=8) --
CACHE_DIR = os.environ.get("BENCH_CACHE_DIR", "/tmp/topk-bench")
COLLECTION_PREFIX = os.environ.get("BENCH_COLLECTION_PREFIX", "x")
# One directory per launch, named for when it started. Sorts chronologically because
# it sorts lexically, and a run can never land loose in results/ again.
SESSION = os.environ.get("BENCH_SESSION") or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
RESULTS_DIR = os.environ.get("BENCH_RESULTS_DIR", f"results/{SESSION}")
# Documents per logical batch. Overridable so the write path can be swept: each
# provider re-splits this differently on the wire -- topk-es caps bodies at 512 KB
# (MAX_BULK_BYTES) and sends ~15 HTTP requests per 2000-doc batch, while topk-sql
# and native send one. Batch size is therefore the knob that separates per-request
# cost from per-document cost.
BATCH_SIZE = int(os.environ.get("BENCH_BATCH_SIZE", "2000"))

# Result-set sizes for the k sweep. This is the bytes-OUT axis: response size is
# k x bytes-per-hit, and it is the only axis that separates a client's per-byte cost
# from its per-request cost. Every other mode pins top_k=10, so without this the whole
# benchmark sits at a single point in payload space.
K_SWEEP = [int(k) for k in os.environ.get("BENCH_K_SWEEP", "1,10,100,1000").split(",")]
CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "8"))

# The concurrency ladder qps walks. Two of the four clients are a service hop as well as
# a client -- es-proxy and pgwire are one pod each -- so where a curve stops scaling is a
# property of the hop, not of the protocol, and it cannot be seen from a ladder that
# stops at 8.
CONCURRENCY_STEPS = [int(c) for c in
                     os.environ.get("BENCH_CONCURRENCY_STEPS", "1,2,4,8").split(",")]

# All three hit the same TopK backend: `topk` over the native proto/gRPC SDK,
# `topk-sql` over the PostgreSQL wire protocol, `topk-es` over the
# Elasticsearch-compatible HTTP shim. Comparing them isolates protocol and
# shim overhead rather than engine differences.
PROVIDERS = {
    "topk": lambda: tb.TopKProvider(),
    "topk-rs": lambda: tb.TopKRsProvider(),
    "topk-sql": lambda: tb.TopKSQLProvider(),
    # Both ES arms are the same class pointed at different clusters. Endpoints are
    # stated here, once, and EsProvider takes no environment fallback -- resolution at
    # the edge is the whole point of the split.
    "topk-es": lambda: tb.EsProvider(
        url=f"https://{os.environ['TOPK_REGION']}.es.{os.environ.get('TOPK_HOST', 'topk.io')}",
        api_key=os.environ["TOPK_API_KEY"], name="topk-es"),
    "es-real": lambda: tb.EsProvider(
        url=os.environ["ES_B_URL"], api_key=os.environ["ES_B_TOKEN"], name="es-real"),
}
PROVIDER = "topk"


def _require_env() -> None:
    missing = [k for k in ("TOPK_API_KEY", "TOPK_REGION") if not os.environ.get(k)]
    aws = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
           if not os.environ.get(k)]
    if missing:
        sys.exit(f"[run_local] missing TopK creds: {', '.join(missing)} (add to .env)")
    if aws:
        sys.exit(f"[run_local] missing AWS creds for s3://topk-bench reads: "
                 f"{', '.join(aws)} (add to .env)")


def docs(size: str) -> str:
    return f"s3://topk-bench/docs-{size}.parquet"


def queries(size: str) -> str:
    return f"s3://topk-bench/queries-{size}.parquet"


def out(mode: str, size: str) -> str:
    """Next free run slot: results/topk_<mode>_<size>_r<N>.parquet.

    Runs accumulate instead of overwriting, so repeated runs can be aggregated
    (drop-worst + mean) the way the published benchmark did.
    """
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    slug = PROVIDER.replace("-", "_")
    n = 1
    while Path(f"{RESULTS_DIR}/{slug}_{mode}_{size}_r{n}.parquet").exists():
        n += 1
    return f"{RESULTS_DIR}/{slug}_{mode}_{size}_r{n}.parquet"


def collection(size: str) -> str:
    return f"{COLLECTION_PREFIX}-{size}"


def provider():
    return PROVIDERS[PROVIDER]()


# ---- benchmarks (1:1 with bench.py) --------------------------------------
def run_ingest(size: str, **_) -> None:
    print(f"[ingest] topk ({size})...", flush=True)
    tb.ingest(
        provider=provider(),
        config=tb.IngestConfig(
            size=size,
            cache_dir=CACHE_DIR,
            collection=collection(size),
            input=docs(size),
            batch_size=BATCH_SIZE,
            concurrency=CONCURRENCY,
            mode="ingest",
        ),
    )
    dst = out("ingest", size)
    tb.write_metrics(dst)
    print(f"[ingest] -> {dst}", flush=True)


def run_qps(size: str, timeout: int = 30, warmup: bool = True) -> None:
    p = provider()
    if warmup:
        print(f"[qps] warmup topk ({size})...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=1, queries=queries(size), timeout=timeout * 2,
            top_k=10, int_filter=None, keyword_filter=None,
            warmup=True, mode="qps"))
    for c in CONCURRENCY_STEPS:
        print(f"[qps] topk ({size}) concurrency={c}...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=c, queries=queries(size), timeout=timeout,
            top_k=10, int_filter=None, keyword_filter=None,
            warmup=False, mode="qps"))
    dst = out("qps", size)
    tb.write_metrics(dst)
    print(f"[qps] -> {dst}", flush=True)


def run_filters(size: str, timeout: int = 30, warmup: bool = True) -> None:
    p = provider()
    if warmup:
        print(f"[filter] warmup topk ({size})...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=1, queries=queries(size), timeout=timeout * 2,
            top_k=10, int_filter=10000, keyword_filter="10000",
            warmup=True, mode="filter"))
    for i, kw in [(None, None), (10000, None), (1000, None), (100, None),
                  (None, "10000"), (None, "01000"), (None, "00100")]:
        print(f"[filter] topk ({size}) int={i} kw={kw}...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=1, queries=queries(size), timeout=timeout,
            top_k=10, int_filter=i, keyword_filter=kw, mode="filter"))
    dst = out("filter", size)
    tb.write_metrics(dst)
    print(f"[filter] -> {dst}", flush=True)


def run_ksweep(size: str, timeout: int = 30, warmup: bool = True) -> None:
    """Sweep result-set size at fixed concurrency.

    Concurrency is pinned to 1 deliberately: this measures latency as a function of
    bytes returned, and mixing in queueing effects would confound the two. The
    concurrency axis is `qps`, and the two are meant to be read separately.
    """
    p = provider()
    if warmup:
        print(f"[ksweep] warmup ({size})...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=1, queries=queries(size), timeout=timeout * 2,
            top_k=10, int_filter=None, keyword_filter=None,
            warmup=True, mode="ksweep"))
    for k in K_SWEEP:
        # Every k gets the same wall-clock by default, but large k is slower per query,
        # so the tail of the sweep collected an order of magnitude fewer samples than
        # the head -- k=1000 landed around ten, which is not a p99. Stretch the window
        # with k so each point has a comparable number of samples behind it. Capped at
        # 4x: past that the sweep costs more than the resolution is worth.
        t = int(timeout * min(4, max(1, k / 25)))
        print(f"[ksweep] ({size}) top_k={k} for {t}s...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=1, queries=queries(size), timeout=t,
            top_k=k, int_filter=None, keyword_filter=None,
            warmup=False, mode="ksweep"))
    dst = out("ksweep", size)
    tb.write_metrics(dst)
    print(f"[ksweep] -> {dst}", flush=True)


def run_rw(size: str, timeout: int = 30, warmup: bool = True) -> None:
    p = provider()
    if warmup:
        print(f"[rw] warmup topk ({size})...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=1, queries=queries(size), timeout=timeout * 2,
            top_k=10, int_filter=None, keyword_filter=None,
            warmup=True, read_write=False, mode="rw"))
    for rw in [False, True]:
        print(f"[rw] topk ({size}) read_write={rw}...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=1, queries=queries(size), timeout=timeout,
            top_k=10, int_filter=None, keyword_filter=None,
            warmup=False, read_write=rw, mode="rw"))
    dst = out("rw", size)
    tb.write_metrics(dst)
    print(f"[rw] -> {dst}", flush=True)


def run_get(size: str, timeout: int = 30, warmup: bool = True) -> None:
    """Point lookups by _id, across the same concurrency ladder as qps.

    A get does almost no server work, so the number is client plus hop. That is the
    quantity the pgwire and es-proxy arms add and the one every other mode measures with
    a vector search sitting on top of it.
    """
    p = provider()
    if warmup:
        print(f"[get] warmup topk ({size})...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=1, queries=queries(size), timeout=timeout * 2,
            top_k=10, int_filter=None, keyword_filter=None,
            warmup=True, mode="get"))
    for c in CONCURRENCY_STEPS:
        print(f"[get] topk ({size}) concurrency={c}...", flush=True)
        tb.query(provider=p, config=tb.QueryConfig(
            size=size, collection=collection(size), cache_dir=CACHE_DIR,
            concurrency=c, queries=queries(size), timeout=timeout,
            top_k=10, int_filter=None, keyword_filter=None,
            warmup=False, mode="get"))
    dst = out("get", size)
    tb.write_metrics(dst)
    print(f"[get] -> {dst}", flush=True)


BENCHES = {
    "ingest": run_ingest,
    "qps": run_qps,
    "get": run_get,
    "filters": run_filters,
    "rw": run_rw,
    "ksweep": run_ksweep,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Local TopK-only benchmark runner")
    ap.add_argument("mode", choices=[*BENCHES, "all"])
    ap.add_argument("--size", action="append", dest="sizes",
                    help="100k | 1m | 10m (repeatable). Default: 100k")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="docs per logical ingest batch (default 2000)")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="topk",
                    help="which TopK interface to drive (default: topk = native proto)")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat each benchmark N times; each run lands in its "
                         "own _r<N>.parquet so runs can be aggregated")
    args = ap.parse_args()
    if args.batch_size:
        globals()['BATCH_SIZE'] = args.batch_size

    global PROVIDER
    PROVIDER = args.provider

    _require_env()
    sizes = args.sizes or ["100k"]
    warmup = not args.no_warmup
    order = ["ingest", "qps", "filters", "rw"] if args.mode == "all" else [args.mode]

    for run in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"===== run {run}/{args.runs} =====", flush=True)
        for size in sizes:
            for mode in order:
                BENCHES[mode](size, timeout=args.timeout, warmup=warmup)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
