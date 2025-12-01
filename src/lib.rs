use colored::control;
use once_cell::sync::Lazy;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::sync::Mutex;
use tokio::runtime::Runtime;

mod ingest;
mod query;

mod data;
mod provider;
mod s3;
mod telemetry;

pub(crate) static RUNTIME: Lazy<Mutex<Option<Runtime>>> = Lazy::new(|| {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create runtime");
    Mutex::new(Some(runtime))
});

#[pymodule]
fn topk_bench(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Force colored output
    control::set_override(true);

    m.add_class::<data::Document>()?;
    m.add_class::<provider::Provider>()?;
    m.add_class::<query::QueryConfig>()?;
    m.add_class::<ingest::IngestConfig>()?;

    m.add_function(wrap_pyfunction!(ingest_fn, m)?)?;
    m.add_function(wrap_pyfunction!(query_fn, m)?)?;
    m.add_function(wrap_pyfunction!(write_metrics, m)?)?;

    // Install telemetry
    py.allow_threads(|| {
        let runtime_guard = RUNTIME.lock().unwrap();
        if let Some(ref runtime) = *runtime_guard {
            runtime.block_on(async move { telemetry::install() })
        } else {
            panic!("Runtime was shut down");
        }
    })
    .map_err(|e| PyValueError::new_err(format!("Failed to init telemetry: {e}")))?;

    // Register cleanup function to shut down Tokio runtime before Python finalizes
    // This prevents GIL errors from Tokio threads trying to access Python during shutdown
    let cleanup_fn = wrap_pyfunction!(shutdown_runtime, m)?;
    let atexit = py.import("atexit")?;
    atexit.call_method1("register", (cleanup_fn,))?;

    Ok(())
}

/// Shutdown the Tokio runtime before Python finalizes.
/// This prevents Tokio threads from trying to access Python after it has started finalizing.
#[pyfunction]
fn shutdown_runtime(py: Python<'_>) {
    // Tokio threads might try to access Python during shutdown, so we release the GIL first
    py.allow_threads(|| {
        if let Ok(mut runtime_guard) = RUNTIME.lock() {
            let _runtime = runtime_guard.take();
            // Runtime is dropped here
        }
    });
}

#[pyfunction(name = "ingest")]
#[pyo3(signature = (provider, config))]
pub(crate) fn ingest_fn(
    py: Python<'_>,
    provider: provider::PyProvider,
    config: ingest::IngestConfig,
) -> PyResult<()> {
    py.allow_threads(|| {
        let runtime_guard = RUNTIME.lock().unwrap();
        if let Some(ref runtime) = *runtime_guard {
            runtime.block_on(async move { ingest::start(provider, config).await })
        } else {
            Err(anyhow::anyhow!("Runtime was shut down"))
        }
    })
    .map_err(|e| PyValueError::new_err(format!("Failed to ingest: {e}")))?;

    Ok(())
}

#[pyfunction(name = "query")]
#[pyo3(signature = (provider, config))]
pub(crate) fn query_fn(
    py: Python<'_>,
    provider: provider::PyProvider,
    config: query::QueryConfig,
) -> PyResult<()> {
    py.allow_threads(|| {
        let runtime_guard = RUNTIME.lock().unwrap();
        if let Some(ref runtime) = *runtime_guard {
            runtime.block_on(async move { query::start(config, provider).await })
        } else {
            Err(anyhow::anyhow!("Runtime was shut down"))
        }
    })
    .map_err(|e| PyValueError::new_err(format!("Failed to query: {e:?}")))?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (path,))]
pub(crate) fn write_metrics(py: Python<'_>, path: &str) -> PyResult<()> {
    py.allow_threads(|| {
        let runtime_guard = RUNTIME.lock().unwrap();
        if let Some(ref runtime) = *runtime_guard {
            runtime.block_on(telemetry::export(&path))
        } else {
            Err(anyhow::anyhow!("Runtime was shut down"))
        }
    })
    .map_err(|e| PyValueError::new_err(format!("Failed to write metrics: {e:?}")))?;

    Ok(())
}
