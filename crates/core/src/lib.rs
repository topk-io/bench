//! The measurement harness: dataset loading, the ingest and query drivers, metrics.
//!
//! Deliberately free of any host-language binding. Each binding crate supplies a
//! `Provider` implementation and its own entry points, so the driver loop, the clock,
//! the concurrency model and the metrics are literally the same code for every client
//! we compare. That is what makes the numbers comparable; a second timing loop is where
//! that guarantee would quietly die.

pub mod data;
pub mod ingest;
pub mod native;
pub mod provider;
pub mod query;
pub mod s3;
pub mod telemetry;

pub use provider::Provider;
