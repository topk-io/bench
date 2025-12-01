use std::collections::HashMap;
use std::time::{Duration, Instant};

use async_channel::{Receiver, Sender};
use colored::Colorize;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use rand::prelude::*;
use tokio::sync::mpsc;
use tokio::{signal::ctrl_c, task::JoinSet};
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

use crate::data::{load_from_path, parse_from_batch, Document, Query};
use crate::ingest::{print_writer_stats, spawn_writers};
use crate::provider::PyProvider;
use crate::query::recall::calculate_recall;
use crate::s3::ensure_file;
use crate::telemetry::metrics::{consume_metrics, snapshot_metrics, Metric, Recorder};

mod config;
pub use config::QueryConfig;

mod recall;

pub async fn start(config: QueryConfig, provider: PyProvider) -> anyhow::Result<()> {
    let provider_name = provider.name().await?;
    info!(?config, ?provider_name, "Starting query bench");

    let run_id = uuid::Uuid::new_v4().to_string();

    let (metrics_tx, metrics_rx) = mpsc::unbounded_channel::<Metric>();
    let metrics_task_handle = tokio::spawn(consume_metrics(metrics_rx));

    let m = Recorder::new(
        metrics_tx,
        [
            ("run_id", run_id.clone()),
            ("provider", provider.name().await?),
            ("collection", config.collection.clone()),
            ("queries", config.queries.clone()),
            ("top_k", config.top_k.to_string()),
            ("concurrency", config.concurrency.to_string()),
            ("size", config.size.clone()),
            ("timeout", config.timeout.to_string()),
            (
                "int_filter",
                config.int_filter.map(|v| v.to_string()).unwrap_or_default(),
            ),
            (
                "keyword_filter",
                config.keyword_filter.clone().unwrap_or_default(),
            ),
            ("warmup", config.warmup.to_string()),
            ("read_write", config.read_write.to_string()),
            ("mode", config.mode.to_string()),
        ],
    );

    let mut tasks = JoinSet::new();

    // Generate queries
    let (queries_tx, queries_rx) = async_channel::bounded::<Query>(1000);
    let qtx = queries_tx.clone();

    // Run query workers
    tasks.spawn(spawn_workers(
        config.clone(),
        provider.clone(),
        m.clone(),
        queries_rx,
        false,
    ));

    let cancel_token = CancellationToken::new();
    let cancel_token_clone = cancel_token.clone();

    if config.read_write {
        let (writes_tx, writes_rx) = async_channel::bounded::<Vec<Document>>(100);
        let file_path = ensure_file(
            format!("s3://topk-bench/docs-{}.parquet", config.size),
            config.cache_dir.clone(),
        )
        .await?;
        let fp = file_path.clone();

        tasks.spawn_blocking(move || {
            // Spawn continuous batch producer task that loops the file
            let file = std::fs::File::open(&fp)?;
            let mut batch_reader = ParquetRecordBatchReader::try_new(file, 1)?;

            loop {
                if cancel_token_clone.is_cancelled() {
                    return anyhow::Ok(());
                }

                while let Some(batch) = batch_reader.next() {
                    if cancel_token_clone.is_cancelled() {
                        break;
                    }

                    match &parse_from_batch(batch?)[..] {
                        [] => anyhow::bail!("No documents in batch"),
                        [document] => queries_tx.send_blocking(Query {
                            dense: document
                                .dense_embedding
                                .clone()
                                .expect("Dense embedding not found"),
                            recall: HashMap::new(),
                        })?,
                        _ => anyhow::bail!("Multiple documents in batch"),
                    }
                }
            }
        });

        let cancel_token = cancel_token.clone();
        tasks.spawn_blocking(move || {
            let file = std::fs::File::open(&file_path)?;
            let mut batch_reader = ParquetRecordBatchReader::try_new(file, 100)?;

            loop {
                if cancel_token.is_cancelled() {
                    return anyhow::Ok(());
                }

                while let Some(batch) = batch_reader.next() {
                    if cancel_token.is_cancelled() {
                        break;
                    }

                    let documents = parse_from_batch(batch?);
                    let documents = documents
                        .into_iter()
                        .map(|mut doc| {
                            doc.tag = Some(format!("tag-{}", rand::rng().random_range(0..1000)));
                            doc
                        })
                        .collect();

                    writes_tx.send_blocking(documents)?;
                }
            }
        });

        tasks.spawn(spawn_writers(
            provider.clone(),
            config.collection.clone(),
            1,
            m.clone(),
            writes_rx,
        ));
    } else {
        let queries = load_from_path(&config.queries, &config.cache_dir).await?;
        tasks.spawn(random_query_generator(queries, queries_tx));
    }

    tasks.spawn(report_metrics(
        run_id.clone(),
        format!("{}@{}", provider_name, config.size),
        config.read_write,
    ));

    let start = Instant::now();
    tokio::select! {
        _ = ctrl_c() => {
            info!("Ctrl-C received, aborting.");
            return Ok(());
        }
        _ = tokio::time::sleep(Duration::from_secs(config.timeout)) => {
            info!("Queries completed in {:.2}s", start.elapsed().as_secs_f64());
        }
    }

    qtx.close();
    cancel_token.cancel();

    tasks.abort_all();
    while let Some(_) = tasks.join_next().await {
        //
    }

    if config.mode == "filter" && !config.warmup {
        measure_recall(
            provider.clone(),
            {
                let mut c = config;
                c.concurrency = 8; // For recall, it doesn't matter how many workers are used
                c
            },
            m.clone(),
            run_id.clone(),
        )
        .await?;
    }
    metrics_task_handle.abort();

    Ok(())
}

async fn measure_recall(
    provider: PyProvider,
    config: QueryConfig,
    m: Recorder,
    run_id: String,
) -> anyhow::Result<()> {
    info!("Measuring recall...");

    let queries = load_from_path(&config.queries, &config.cache_dir).await?;

    let (queries_tx, queries_rx) = async_channel::bounded::<Query>(1_000);

    // Send queries to the workers
    let generator = tokio::spawn(async move {
        for query in queries {
            queries_tx.send(query).await?;
        }
        anyhow::Ok(())
    });
    let reporter = report_metrics(
        run_id.clone(),
        format!("{}@{}", provider.name().await?, config.size),
        config.read_write,
    );
    let workers = spawn_workers(
        config.clone(),
        provider.clone(),
        m.clone(),
        queries_rx,
        true,
    );

    tokio::select! {
        _ = workers => {}
        _ = reporter => {}
    }
    generator.abort();
    generator.await??;

    Ok(())
}

// Spawn query generator task
async fn random_query_generator(queries: Vec<Query>, tx: Sender<Query>) -> anyhow::Result<()> {
    loop {
        let random_query = queries
            .choose(&mut rand::rng())
            .expect("Failed to choose query")
            .clone();

        tx.send(random_query).await?;
    }
}

async fn spawn_workers(
    config: QueryConfig,
    provider: PyProvider,
    m: Recorder,
    queries: Receiver<Query>,
    recall: bool,
) -> anyhow::Result<()> {
    // Spawn worker tasks
    let mut workers = JoinSet::new();

    for _ in 0..config.concurrency {
        let queries = queries.clone();
        let config = config.clone();
        let provider = provider.clone();
        let m = m.clone();

        workers.spawn(async move {
            loop {
                let ss = Instant::now();
                let query = match queries.recv().await {
                    Ok(query) => query,
                    Err(_) => break,
                };
                m.record(
                    "bench.query.recv_latency_ms",
                    ss.elapsed().as_millis() as f64,
                );

                loop {
                    let start = Instant::now();

                    match provider
                        .query(
                            config.collection.clone(),
                            query.dense.clone(),
                            config.top_k,
                            config.int_filter.clone(),
                            config.keyword_filter.clone(),
                        )
                        .await
                    {
                        Ok(res) => {
                            if recall {
                                let recall = calculate_recall(res, query.clone(), &config)
                                    .expect("failed to calculate recall");
                                m.record("bench.query.recall", recall as f64);
                            } else {
                                let duration = start.elapsed().as_millis();
                                m.record("bench.query.oks", 1.0);
                                m.record("bench.query.latency_ms", duration as f64);
                            }

                            break;
                        }
                        Err(error) => {
                            m.record("bench.query.errors", 1.0);
                            error!(?error, "Failed to query documents");

                            // Sleep & retry
                            let jitter = rand::rng().random_range(10..100);
                            tokio::time::sleep(Duration::from_millis(jitter)).await;
                        }
                    }
                }
            }
        });
    }

    // Poll the JoinSet directly with cancellation support
    while let Some(res) = workers.join_next().await {
        res?;
    }

    Ok(())
}

// metrics reporter task
async fn report_metrics(run_id: String, prefix: String, writes: bool) -> anyhow::Result<()> {
    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    ticker.tick().await;

    loop {
        ticker.tick().await;

        let stats = snapshot_metrics(&run_id).await;

        // Check if metrics exist (not just if they're zero)
        if stats.is_empty() {
            println!("{prefix}] Waiting for metrics...");
            continue;
        }

        let oks_total = stats.total("bench.query.oks");
        let errors_total = stats.total("bench.query.errors");
        let requests_total = oks_total + errors_total;

        let availability = if requests_total > 0.0 {
            (1.0 - (errors_total / requests_total)) * 100.0
        } else {
            100.0
        };

        println!(
            "{:>16}] {}, Throughput: {}, Latency: {}, {}, Recall: {}{}",
            prefix,
            // Availability
            match availability {
                a if a == 100.0 => format!("100%").green().bold(),
                a if a > 99.0 => format!("{:.2}%", a).yellow().bold(),
                a => format!("{:.2}%", a).red().bold(),
            },
            // Throughput
            format!("{} queries/s", stats.instantaneous_rate("bench.query.oks"))
                .blue()
                .bold(),
            // Latency
            format!("avg={:.2}ms", stats.avg("bench.query.latency_ms"))
                .yellow()
                .bold(),
            format!(
                "p99={:.2}ms",
                stats.quantile("bench.query.latency_ms", 0.99)
            )
            .magenta()
            .bold(),
            // Recall
            format!("avg={:.2}", stats.avg("bench.query.recall"))
                .yellow()
                .bold(),
            // Recv
            {
                let recv_max = stats.quantile("bench.query.recv_latency_ms", 1.0);
                if recv_max == 0.0 {
                    "".to_string()
                } else {
                    format!(", Skew max={:.2}ms", recv_max).bold().to_string()
                }
            },
        );

        if writes {
            print_writer_stats(&stats, prefix.clone())
        }
    }
}
