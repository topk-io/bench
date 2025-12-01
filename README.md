# TopK Bench

A reproducible benchmarking tool for evaluating vector databases under realistic workloads. TopK Bench provides standardized datasets, query sets, and evaluation logic to compare vector database performance across ingestion, concurrency scaling, filtering, and recall.

For high-level benchmark results and analysis, see the [TopK Bench blog post](https://topk.io/blog/the-benchmark).

## Overview

TopK Bench evaluates vector databases across four core benchmarks:

1. **Ingest Performance**: Total ingestion time and throughput as systems build indexes from scratch (100k → 10M vectors)
2. **Concurrency Scaling**: How query throughput (QPS) and latency change as client-side concurrency increases (1, 2, 4, 8 workers)
3. **Filtering Performance**: Impact of metadata and keyword filters on QPS and latency at different selectivities (100%, 10%, 1%)
4. **Recall at Scale**: Recall accuracy across dataset sizes and filter selectivities

Additional properties evaluated:

- **Freshness**: Time from write acknowledgment to query visibility
- **Read-write Performance**: Query behavior during concurrent writes

## Dataset

### Schema

The benchmark uses datasets derived from MS MARCO passages and queries, with embeddings generated using [`nomic-ai/modernbert-embed-base`](https://huggingface.co/nomic-ai/modernbert-embed-base).

#### Document Schema

Each document contains the following fields:

- `id: u32` - Unique identifier in the range `0 … 9_999_999`
- `text: str` - Text passages from MS MARCO
- `dense_embedding: list[f32]` - 768-dimensional embedding vector generated from the `text` field
- `int_filter: u32` - Integer field sampled uniformly from `0 ... 9_999`, used for controlled selectivity filtering
- `keyword_filter: str` - String field containing keywords with known distribution, used for controlled selectivity filtering

#### Query Schema

Each query set contains 1,000 queries with the following fields:

- `text: str` - Query text from MS MARCO
- `dense: list[f32]` - 768-dimensional embedding vector generated from the `text` field
- `recall` - Mapping from `(int_filter, keyword_filter)` pairs to lists of relevant document IDs (ground truth)

### Selectivity

The dataset is designed to enable controlled selectivity testing through filter predicates:

#### Integer Filter Selectivity

The `int_filter` field allows selecting specific percentages of the dataset:

- `int_filter < 10_000` → selects **100%** of documents
- `int_filter < 1_000` → selects **10%** of documents
- `int_filter < 100` → selects **1%** of documents

#### Keyword Filter Selectivity

The `keyword_filter` field contains tokens with known distribution:

- `text_match(keyword_filter, "10000")` → selects **100%** of documents (p=100%)
- `text_match(keyword_filter, "01000")` → selects **10%** of documents (p=10%)
- `text_match(keyword_filter, "00100")` → selects **1%** of documents (p=1%)

### Dataset Sizes

Three dataset sizes are available:

- **100k**: 100,000 documents
- **1m**: 1,000,000 documents
- **10m**: 10,000,000 documents

### Ground Truth

Ground truth nearest neighbors are pre-computed using exact search in an offline setting, ensuring accurate recall evaluation. The dataset includes true nearest neighbors up to `top_k=100`, allowing recall evaluation at different k values.

### Availability

Datasets are publicly available on S3:

- Documents: `s3://topk-bench/docs-{100k,1m,10m}.parquet`
- Queries: `s3://topk-bench/queries-{100k,1m,10m}.parquet`

## Installation

Install TopK Bench:

```bash
pip install topk-bench
```

TopK Bench is written in Rust via PyO3, providing high-performance benchmarking capabilities.

## Usage

TopK Bench is a Python library for benchmarking vector databases. The core API provides functions for ingesting data, running queries, and collecting metrics.

### Quick Start

```python
import topk_bench as tb

# Create a provider client
provider = tb.TopKProvider()  # or MilvusProvider(), PineconeProvider(), etc.

# Ingest documents
tb.ingest(
    provider=provider,
    config=tb.IngestConfig(
        size="1m",
        collection="bench-1m",
        input="s3://topk-bench/docs-1m.parquet",
        batch_size=2000,
        concurrency=8,
    ),
)

# Run queries
tb.query(
    provider=provider,
    config=tb.QueryConfig(
        size="1m",
        collection="bench-1m",
        queries="s3://topk-bench/queries-1m.parquet",
        concurrency=4,
        timeout=30,
        top_k=10,
    ),
)

# Write metrics
tb.write_metrics("results/metrics.parquet")
```

### API Reference

#### `topk_bench.ingest()`

Ingest documents into a vector database collection.

```python
import topk_bench as tb

tb.ingest(
    provider=provider_client,
    config=tb.IngestConfig(
        size="1m",  # Dataset size: "100k", "1m", "10m"
        cache_dir="/tmp/topk-bench",
        collection="bench-1m",
        input="s3://topk-bench/docs-1m.parquet",
        batch_size=2000,  # Provider-specific
        concurrency=8,    # Provider-specific
        mode="ingest",
    ),
)
```

#### `topk_bench.query()`

Execute queries against a collection.

```python
tb.query(
    provider=provider_client,
    config=tb.QueryConfig(
        size="1m",
        collection="bench-1m",
        cache_dir="/tmp/topk-bench",
        concurrency=4,  # 1, 2, 4, or 8
        queries="s3://topk-bench/queries-1m.parquet",
        timeout=30,  # seconds
        top_k=10,
        int_filter=1000,      # None or selectivity value
        keyword_filter="01000",  # None or keyword token
        warmup=False,
        mode="qps",  # "qps", "filter", or "rw"
        read_write=False,  # For rw mode
    ),
)
```

#### `topk_bench.write_metrics()`

Write collected metrics to S3.

```python
tb.write_metrics(
    f"s3://bucket/results/{benchmark_id}/{provider}_qps_{size}.parquet"
)
```

### Supported Providers

See the `providers` directory for supported providers and their implementations.

## Example Deployment: Modal

The `bench.py` file includes a Modal setup that provides CLI entry points for running benchmarks at scale. See `bench.py` for the complete implementation.
