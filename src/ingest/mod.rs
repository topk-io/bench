use std::{
    fs::File,
    time::{Duration, Instant},
};

use async_channel::{Receiver, Sender};
use colored::Colorize;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use rand::prelude::*;
use tokio::{
    signal::ctrl_c,
    sync::mpsc,
    task::{JoinHandle, JoinSet},
};
use tracing::{error, info};

use crate::{
    data::{parse_from_batch, Document},
    provider::PyProvider,
    s3::open_file,
    telemetry::{
        metrics::{consume_metrics, snapshot_metrics, Metric, Recorder},
        Snapshot,
    },
};

mod config;
pub use config::IngestConfig;

pub async fn start(provider: PyProvider, config: IngestConfig) -> anyhow::Result<()> {
    let run_id = uuid::Uuid::new_v4().to_string();

    let (metrics_tx, metrics_rx) = mpsc::unbounded_channel::<Metric>();

    let provider_name = provider.name().await?;
    let m = Recorder::new(
        metrics_tx,
        [
            ("provider", provider_name.clone()),
            ("collection", config.collection.clone()),
            ("batch_size", config.batch_size.to_string()),
            ("concurrency", config.concurrency.to_string()),
            ("input", config.input.clone()),
            ("size", config.size.clone()),
            ("run_id", run_id.clone()),
            ("mode", config.mode.clone()),
        ],
    );

    // Load dataset
    let file = open_file(&config.input, config.cache_dir.clone()).await?;

    info!(?config, "Benchmarking {provider_name}");
    provider.setup(config.collection.clone()).await?;

    // Spawn batch producer
    let (tx, rx) = async_channel::bounded::<Vec<Document>>(100);
    spawn_batch_producer(file, config.batch_size, tx);

    let mut tasks = JoinSet::new();

    // Spawn writers
    tasks.spawn(spawn_writers(
        provider.clone(),
        config.collection.clone(),
        config.concurrency,
        m.clone(),
        rx,
    ));

    // Spawn metrics reporter
    tasks.spawn(spawn_metrics_reporter(
        run_id.clone(),
        format!("{}@{}", provider_name, config.size),
    ));

    // Consume metrics
    tasks.spawn(consume_metrics(metrics_rx));

    // Control-C
    tasks.spawn(async {
        ctrl_c().await?;
        info!("Ctrl-C received, aborting ingest");
        Ok(())
    });

    let start = Instant::now();
    while let Some(_) = tasks.join_next().await {
        tasks.abort_all();
        break;
    }
    info!("Ingest completed in {:.2}s", start.elapsed().as_secs_f64());

    provider.close().await?;

    Ok(())
}

// Spawn batch producer task
pub fn spawn_batch_producer(
    file: File,
    batch_size: usize,
    tx: Sender<Vec<Document>>,
) -> JoinHandle<anyhow::Result<()>> {
    tokio::task::spawn_blocking(move || {
        let mut batch_reader = ParquetRecordBatchReader::try_new(file, batch_size)?;

        while let Some(batch) = batch_reader.next() {
            let documents = parse_from_batch(batch?);

            // Use send_blocking since we're in a blocking task
            tx.send_blocking(documents)?;
        }

        Ok(())
    })
}

// Spawn writer tasks
pub async fn spawn_writers(
    provider: PyProvider,
    collection: String,
    concurrency: usize,
    m: Recorder,
    rx: Receiver<Vec<Document>>,
) -> anyhow::Result<()> {
    let mut writers = JoinSet::<anyhow::Result<()>>::new();

    for _ in 0..concurrency {
        let collection = collection.clone();
        let rx = rx.clone();
        let provider = provider.clone();
        let m = m.clone();

        writers.spawn(async move {
            // Spawn freshness tasks
            let mut freshness_tasks = JoinSet::new();

            // Writer task
            loop {
                let recv_start = Instant::now();
                let documents = match rx.recv().await {
                    Ok(documents) => documents,
                    Err(_) => break, // Channel closed
                };
                m.record(
                    "bench.ingest.recv_latency_ms",
                    recv_start.elapsed().as_millis() as f64,
                );

                let doc_count = documents.len();
                let provider = provider.clone();

                // Upsert loop
                loop {
                    let documents = documents.clone();

                    // Calculate encoded size from parsed documents
                    let byte_size: usize = documents.iter().map(|doc| doc.approx_size()).sum();

                    // Calculate max ID from batch
                    let max_id = documents
                        .iter()
                        .map(|doc| doc.id.parse::<u64>().expect("Failed to parse ID as u64"))
                        .max()
                        .expect("Failed to find max ID")
                        .to_string();

                    let s = Instant::now();
                    let result = provider.upsert(collection.clone(), documents).await;

                    m.record("bench.ingest.requests", 1.0);
                    match result {
                        Ok(_) => {
                            m.record("bench.ingest.oks", 1.0);
                            m.record("bench.ingest.upserted_docs", doc_count as f64);
                            m.record("bench.ingest.upserted_bytes", byte_size as f64);
                            m.record("bench.ingest.latency_ms", s.elapsed().as_millis() as f64);

                            // After a successful upsert, measure the freshness of the document.
                            freshness_tasks.spawn(measure_freshness(
                                m.clone(),
                                provider.clone(),
                                collection.clone(),
                                max_id,
                            ));

                            break;
                        }
                        Err(error) => {
                            m.record("bench.ingest.errors", 1.0);

                            // TODO: use signal to propagate to the `tokio::select!` block
                            if error.to_string().contains("KeyboardInterrupt") {
                                info!("Keyboard interrupt received, aborting writers");
                                break;
                            } else {
                                error!(?error, "Failed to upsert documents");
                            }

                            // Sleep
                            let jitter = rand::rng().random_range(10..100);
                            tokio::time::sleep(Duration::from_millis(jitter)).await;
                        }
                    }
                }
            }

            // Wait for freshness tasks
            while let Some(res) = freshness_tasks.join_next().await {
                res??;
            }

            Ok(())
        });
    }

    // Spawn writer clients
    while let Some(res) = writers.join_next().await {
        res??;
    }

    Ok(())
}

// metrics reporter task
async fn spawn_metrics_reporter(run_id: String, prefix: String) -> anyhow::Result<()> {
    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    // Skip the immediate first tick to align with 1-second boundaries
    ticker.tick().await;

    loop {
        // Sleep for 1 second
        ticker.tick().await;

        // Get current stats
        let stats = snapshot_metrics(&run_id).await;

        // Check if metrics exist (not just if they're zero)
        if stats.is_empty() {
            println!("{}] Waiting for metrics...", prefix);
            continue;
        }

        print_writer_stats(&stats, prefix.clone())
    }
}

pub fn print_writer_stats(stats: &Snapshot, prefix: String) {
    let requests_total = stats.total("bench.ingest.requests");
    let errors_total = stats.total("bench.ingest.errors");
    let availability = if requests_total > 0.0 {
        (1.0 - errors_total / requests_total) * 100.0
    } else {
        100.0
    };

    println!(
        "{prefix:>16}] {} {} Throughput: {}, Latency: {}, {}{}{}",
        // Availability
        match availability {
            a if a == 100.0 => format!("100%").green().bold(),
            a if a > 99.0 => format!("{:.2}%", a).yellow().bold(),
            a if a.is_nan() => format!("...").bold(),
            a => format!("{:.2}%", a).red().bold(),
        },
        // Total
        match stats.total("bench.ingest.upserted_bytes") {
            b if b < 1024.0 => format!("{:.2} B", b),
            b if b < (1024.0 * 1024.0) => format!("{:.2} KB", b / 1024.0),
            b => format!("{:.2} MB", b / (1024.0 * 1024.0)),
        }
        .bold(),
        // Throughput
        match stats.instantaneous_rate("bench.ingest.upserted_bytes") {
            b if b < 1024.0 => format!("{:.2} B/s", b),
            b if b < 1024.0 * 1024.0 => format!("{:.2} KB/s", b / 1024.0),
            b => format!("{:.2} MB/s", b / (1024.0 * 1024.0)),
        }
        .magenta()
        .bold(),
        // Latency
        format!(
            "p50={:.2}ms",
            stats.quantile("bench.ingest.latency_ms", 0.50)
        )
        .yellow()
        .bold(),
        format!(
            "p99={:.2}ms",
            stats.quantile("bench.ingest.latency_ms", 0.99)
        )
        .magenta()
        .bold(),
        // Freshness
        {
            let freshness_max = stats.quantile("bench.ingest.freshness_latency_ms", 1.0);
            if freshness_max == 0.0 {
                "".to_string()
            } else {
                format!(", Freshness max={:.2}ms", freshness_max)
                    .bold()
                    .to_string()
            }
        },
        // Recv
        {
            let recv_max = stats.quantile("bench.ingest.recv_latency_ms", 1.0);
            if recv_max == 0.0 {
                "".to_string()
            } else {
                format!(", Skew max={:.2}ms", recv_max).bold().to_string()
            }
        },
    );
}

/// Measure the freshness of a document by querying it until it is found.
async fn measure_freshness(
    m: Recorder,
    provider: PyProvider,
    collection: String,
    id: String,
) -> anyhow::Result<()> {
    let start = Instant::now();

    loop {
        // TODO: latency of `query_by_id`
        let s = Instant::now();
        let doc = provider.query_by_id(collection.clone(), id.clone()).await?;
        m.record(
            "bench.ingest.query_by_id_latency_ms",
            s.elapsed().as_millis() as f64,
        );

        if doc.is_some() {
            break;
        }

        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    m.record(
        "bench.ingest.freshness_latency_ms",
        start.elapsed().as_millis() as f64,
    );

    Ok(())
}
