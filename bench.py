import uuid
import modal


app = modal.App("topk-bench")

image = modal.Image.debian_slim().pip_install("topk-bench==0.1.0")

cache = modal.Volume.from_name("topk-bench-cache", create_if_missing=True)
cache_dir = "/tmp/topk-bench"
collection_prefix = "x"


@app.local_entrypoint()
def ingest(
    size: str = None,
    provider: str = None,
):
    sizes = list_sizes(only=size)
    providers = list_providers(only=provider)
    benchmark_id = uuid.uuid4()

    run(
        runner(region).spawn(
            run_ingest_bench,
            benchmark_id=benchmark_id,
            provider_name=provider,
            size=size,
            batch_size=batch_size,
            concurrency=concurrency,
        )
        for size in sizes
        for region, provider, (batch_size, concurrency) in providers
    )


@app.local_entrypoint()
def qps(
    size: str = None,
    provider: str = None,
    timeout: int = 30,
    warmup: bool = True,
):
    sizes = list_sizes(only=size)
    providers = list_providers(only=provider)
    benchmark_id = uuid.uuid4()

    run(
        runner(region).spawn(
            run_qps_bench,
            benchmark_id=benchmark_id,
            provider_name=provider,
            size=size,
            timeout=timeout,
            warmup=warmup,
        )
        for size in sizes
        for region, provider, _ in providers
    )


@app.local_entrypoint()
def filters(
    size: str = None,
    provider: str = None,
    timeout: int = 30,
    warmup: bool = True,
):
    sizes = list_sizes(only=size)
    providers = list_providers(only=provider)
    benchmark_id = uuid.uuid4()

    run(
        runner(region).spawn(
            run_filter_bench,
            benchmark_id=benchmark_id,
            provider_name=provider,
            size=size,
            timeout=timeout,
            warmup=warmup,
        )
        for size in sizes
        for region, provider, _ in providers
    )


@app.local_entrypoint()
def rw(
    size: str = None,
    provider: str = None,
    timeout: int = 30,
    warmup: bool = True,
):
    sizes = list_sizes(only=size)
    providers = list_providers(only=provider)
    benchmark_id = uuid.uuid4()

    run(
        runner(region).spawn(
            run_rw_bench,
            benchmark_id=benchmark_id,
            provider_name=provider,
            size=size,
            timeout=timeout,
            warmup=warmup,
        )
        for size in sizes
        for region, provider, _ in providers
    )


@app.local_entrypoint()
def cleanup(
    provider: str = None,
    wet: bool = False,
):
    providers = list_providers(only=provider)

    run(
        runner(region).spawn(
            cleanup_provider,
            provider_name=provider,
            wet=wet,
        )
        for region, provider, _ in providers
    )


### Benchmarks ###


def run_ingest_bench(
    size: str,
    provider_name: str,
    batch_size: int,
    concurrency: int,
    benchmark_id: str,
):
    import topk_bench as tb

    provider_client = create_provider(provider_name)

    tb.ingest(
        provider=provider_client,
        config=tb.IngestConfig(
            size=size,
            cache_dir=cache_dir,
            #
            collection=f"{collection_prefix}-{size}",
            input=f"s3://topk-bench/docs-{size}.parquet",
            batch_size=batch_size,
            concurrency=concurrency,
            #
            mode="ingest",
        ),
    )

    tb.write_metrics(
        f"s3://my-results-bucket/{benchmark_id}/{provider_name}_ingest_{size}.parquet"
    )


def run_qps_bench(
    size: str,
    provider_name: str,
    timeout: int,
    benchmark_id: str,
    warmup: bool = True,
):
    import topk_bench as tb

    provider_client = create_provider(provider_name)

    if warmup:
        print(f"BENCH] Warming up {provider_name} ({size})...")
        tb.query(
            provider=provider_client,
            config=tb.QueryConfig(
                size=size,
                collection=f"{collection_prefix}-{size}",
                cache_dir=cache_dir,
                #
                concurrency=1,
                #
                queries=f"s3://topk-bench/queries-{size}.parquet",
                timeout=timeout * 2,
                #
                top_k=10,
                int_filter=None,
                keyword_filter=None,
                warmup=True,
                #
                mode="qps",
            ),
        )

    print(f"BENCH] Benchmarking {provider_name} ({size})...")
    for concurrency in [1, 2, 4, 8]:
        tb.query(
            provider=provider_client,
            config=tb.QueryConfig(
                size=size,
                collection=f"{collection_prefix}-{size}",
                cache_dir=cache_dir,
                #
                concurrency=concurrency,
                #
                queries=f"s3://topk-bench/queries-{size}.parquet",
                timeout=timeout,
                #
                top_k=10,
                int_filter=None,
                keyword_filter=None,
                warmup=False,
                #
                mode="qps",
            ),
        )

    tb.write_metrics(
        f"s3://my-results-bucket/{benchmark_id}/{provider_name}_qps_{size}.parquet"
    )


def run_filter_bench(
    size: str,
    provider_name: str,
    timeout: int,
    benchmark_id: str,
    warmup: bool,
):
    import topk_bench as tb

    provider = create_provider(provider_name)

    if warmup:
        print(f"BENCH] Warming up {provider_name} ({size})...")
        tb.query(
            provider=provider,
            config=tb.QueryConfig(
                size=size,
                collection=f"{collection_prefix}-{size}",
                cache_dir=cache_dir,
                #
                concurrency=1,
                #
                queries=f"s3://topk-bench/queries-{size}.parquet",
                timeout=timeout * 2,
                #
                top_k=10,
                int_filter=10000,  # 100%
                keyword_filter="10000",  # 100%
                #
                warmup=True,
                mode="filter",
            ),
        )

    print(f"BENCH] Benchmarking {provider_name} ({size})...")
    for query_config in [
        tb.QueryConfig(
            size=size,
            collection=f"{collection_prefix}-{size}",
            cache_dir=cache_dir,
            #
            concurrency=1,
            #
            queries=f"s3://topk-bench/queries-{size}.parquet",
            timeout=timeout,
            #
            top_k=10,
            int_filter=i,
            keyword_filter=kw,
            #
            mode="filter",
        )
        for i, kw in [
            (None, None),
            (10000, None),
            (1000, None),
            (100, None),
            (None, "10000"),
            (None, "01000"),
            (None, "00100"),
        ]
    ]:
        tb.query(
            provider=provider,
            config=query_config,
        )

    tb.write_metrics(
        f"s3://my-results-bucket/{benchmark_id}/{provider_name}_filter_{size}.parquet"
    )


def run_rw_bench(
    size: str,
    provider_name: str,
    timeout: int,
    benchmark_id: str,
    warmup: bool = True,
):
    import topk_bench as tb

    provider = create_provider(provider_name)

    if warmup:
        print(f"BENCH] Warming up {provider_name} ({size})...")
        tb.query(
            provider=provider,
            config=tb.QueryConfig(
                size=size,
                collection=f"{collection_prefix}-{size}",
                cache_dir=cache_dir,
                #
                concurrency=1,
                #
                queries=f"s3://topk-bench/queries-{size}.parquet",
                timeout=timeout * 2,
                #
                top_k=10,
                int_filter=None,
                keyword_filter=None,
                #
                warmup=True,
                read_write=False,
                mode="rw",
            ),
        )

    print(f"BENCH] Benchmarking {provider_name} ({size})...")
    for query_config in [
        tb.QueryConfig(
            size=size,
            collection=f"{collection_prefix}-{size}",
            cache_dir=cache_dir,
            #
            concurrency=1,
            #
            queries=f"s3://topk-bench/queries-{size}.parquet",
            timeout=timeout,
            #
            top_k=10,
            int_filter=None,
            keyword_filter=None,
            #
            warmup=False,
            read_write=rw,
            mode="rw",
        )
        for rw in [False, True]
    ]:
        tb.query(
            provider=provider,
            config=query_config,
        )

    tb.write_metrics(
        f"s3://my-results-bucket/{benchmark_id}/{provider_name}_rw_{size}.parquet"
    )


### Sizes ###


def list_sizes(only: str = None):
    sizes = ["100k", "1m", "10m"]

    if only is not None:
        return [s for s in sizes if s == only]

    return sizes


### Providers ###


def list_providers(only: str = None):
    providers = [
        ("eu", "topk", (2000, 8)),
        ("eu", "milvus", (2000, 8)),
        ("eu", "turbopuffer", (2000, 8)),
        ("eu", "qdrant", (2000, 4)),
        ("us", "pinecone", (500, 12)),
    ]

    if only is not None:
        return [(r, p, b) for r, p, b in providers if p == only]

    return providers


def create_provider(name: str):
    import topk_bench as tb

    if name == "milvus":
        return tb.MilvusProvider()
    elif name == "topk":
        return tb.TopKProvider()
    elif name == "turbopuffer":
        return tb.TurbopufferProvider()
    elif name == "qdrant":
        return tb.QdrantProvider()
    elif name == "pinecone":
        return tb.PineconeProvider()
    else:
        raise ValueError(f"Invalid provider: {name}")


def cleanup_provider(provider_name: str, wet: bool = False):
    provider = create_provider(provider_name)

    for collection in provider.list_collections():
        if wet:
            print(f"Deleting {collection=} in {provider_name=}...")
            provider.delete_collection(collection)
        else:
            print(f"DRY RUN] Would delete {collection=} in {provider_name=}.")


### Runners ###


def run(tasks):
    errors = []
    tasks = list(tasks)

    print(f"Running {len(tasks)} tasks...")
    for t in tasks:
        try:
            t.get()  # awaits the task
        except Exception as e:
            print(f"Task failed: {e}")
            errors.append(e)

    if errors:
        print(f"{len(errors)} tasks failed.")
        raise Exception(f"{len(errors)} tasks failed: {errors}")


def runner(region):
    if region == "eu":
        return eu_runner
    elif region == "us":
        return us_runner
    else:
        raise ValueError(f"Invalid region: {region}")


@app.function(
    image=image,
    region="eu-central-1",
    volumes={cache_dir: cache},
    secrets=[modal.Secret.from_name("topk-bench")],
    timeout=4 * 60 * 60,  # 4 hours
)
def eu_runner(fn: str, **kwargs):
    fn(**kwargs)


@app.function(
    image=image,
    region="us-east-1",
    volumes={cache_dir: cache},
    secrets=[modal.Secret.from_name("topk-bench")],
    timeout=4 * 60 * 60,  # 4 hours
)
def us_runner(fn: str, **kwargs):
    fn(**kwargs)
