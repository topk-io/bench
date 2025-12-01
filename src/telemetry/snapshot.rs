use chrono::Utc;

use crate::telemetry::metrics::Metric;

pub struct Snapshot {
    pub metrics: Vec<Metric>,
}

impl Snapshot {
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    pub fn total(&self, name: &str) -> f64 {
        self.metrics
            .iter()
            .filter(|m| m.name == name)
            .map(|m| m.value)
            .sum()
    }

    pub fn instantaneous_rate(&self, name: &str) -> f64 {
        let now = Utc::now();
        self.metrics
            .iter()
            .filter(|m| m.name == name)
            .filter(|m| (now - m.timestamp).num_milliseconds() <= 1000)
            .map(|m| m.value)
            .sum()
    }

    pub fn avg(&self, name: &str) -> f64 {
        let mut count = 0usize;
        let mut total = 0.0;
        for m in self.metrics.iter().filter(|m| m.name == name) {
            total += m.value;
            count += 1;
        }
        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    pub fn quantile(&self, name: &str, quantile: f64) -> f64 {
        let mut values: Vec<f64> = self
            .metrics
            .iter()
            .filter(|m| m.name == name)
            .map(|m| m.value)
            .collect();
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let len = values.len();
        let idx = ((quantile * (len as f64 - 1.0)).round() as usize).min(len - 1);
        values[idx]
    }
}
