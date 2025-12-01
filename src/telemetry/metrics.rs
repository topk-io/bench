use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::RwLock;

use crate::telemetry::snapshot::Snapshot;

static METRICS: Lazy<RwLock<Vec<Metric>>> = Lazy::new(|| RwLock::new(Vec::new()));

pub async fn snapshot_metrics(run_id: &str) -> Snapshot {
    let guard = METRICS.read().await;
    let metrics = guard
        .iter()
        .filter(|m| m.metadata.get("run_id").expect("run_id is required") == run_id)
        .cloned()
        .collect();

    Snapshot { metrics }
}

pub async fn consume_metrics(mut rx: UnboundedReceiver<Metric>) -> anyhow::Result<()> {
    while let Some(metric) = rx.recv().await {
        let mut metrics = METRICS.write().await;
        metrics.push(metric);
        // Explicitly drop the guard
        drop(metrics);
    }

    Ok(())
}

pub async fn flush_metrics() -> Vec<Metric> {
    let mut guard = METRICS.write().await;
    let metrics = guard.drain(..).collect();
    metrics
}

#[derive(Debug, Clone)]
pub struct Metric {
    /// Name of the metric
    pub name: String,
    /// Value of the metric
    pub value: f64,
    /// Timestamp of the metric
    pub timestamp: DateTime<Utc>,
    /// Metadata for the metric
    pub metadata: Arc<HashMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct Recorder {
    /// Sender for the metrics
    tx: UnboundedSender<Metric>,
    /// Metadata for the metrics
    metadata: Arc<HashMap<String, String>>,
}

impl Recorder {
    pub fn new(
        tx: UnboundedSender<Metric>,
        metadata: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            tx,
            metadata: Arc::new(
                metadata
                    .into_iter()
                    .map(|(k, v)| (k.into(), v.into()))
                    .collect(),
            ),
        }
    }

    pub fn record(&self, name: &str, value: f64) {
        self.tx
            .send(Metric {
                name: name.to_string(),
                value,
                timestamp: Utc::now(),
                metadata: self.metadata.clone(),
            })
            .unwrap();
    }
}
