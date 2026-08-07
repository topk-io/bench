use std::sync::Arc;

use async_trait::async_trait;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use topk_bench_core::{data::Document, native::NativeProvider, Provider};

/// The base class Python providers subclass. Carries no behaviour; it exists so the
/// module exports a name to inherit from.
#[pyclass(name = "Provider", subclass)]
#[derive(Debug, Clone)]
pub struct ProviderBase {}

#[pymethods]
impl ProviderBase {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let _ = (args, kwargs); // Suppress unused variable warnings
        Ok(Self {})
    }
}

/// Drives a provider written in Python.
#[derive(Debug, Clone)]
pub struct PyProvider {
    py: Arc<Py<PyAny>>,
}

/// A provider argument coming from Python.
///
/// A newtype because neither `Arc` nor `Provider` is local to this crate, so
/// `FromPyObject` cannot be implemented on the combination directly.
pub struct ProviderArg(pub Arc<dyn Provider>);

impl FromPyObject<'_> for ProviderArg {
    fn extract_bound(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // A provider marked __native__ is built in Rust from the environment; everything
        // else is driven as a Python object.
        let native = obj
            .getattr("__native__")
            .and_then(|v| v.extract::<bool>())
            .unwrap_or(false);
        if native {
            let p = NativeProvider::from_env()
                .map_err(|e| PyValueError::new_err(format!("topk-rs provider: {e}")))?;
            return Ok(ProviderArg(Arc::new(p)));
        }
        Ok(ProviderArg(Arc::new(PyProvider {
            py: Arc::new(obj.as_any().clone().into()),
        })))
    }
}

#[async_trait]
impl Provider for PyProvider {
    async fn name(&self) -> anyhow::Result<String> {
        let provider = self.py.clone();

        Ok(run_py(move |py| -> PyResult<String> {
            let name = provider.call_method0(py, "name")?;
            let name = name.extract(py)?;
            Ok(name)
        })
        .await?)
    }

    async fn setup(&self, collection: String) -> anyhow::Result<()> {
        let provider = self.py.clone();

        run_py(move |py| provider.call_method1(py, "setup", (collection,))).await?;

        Ok(())
    }

    async fn upsert(
        &self,
        collection: String,
        docs: Vec<Document>,
    ) -> anyhow::Result<Option<u64>> {
        let provider = self.py.clone();

        let wire = run_py(move |py| -> PyResult<Option<u64>> {
            let res = provider.call_method1(py, "upsert", (collection, docs))?;
            // None, or a provider that returns nothing, means "not reported".
            Ok(res.extract::<Option<u64>>(py).unwrap_or(None))
        })
        .await?;

        Ok(wire)
    }

    async fn freshness_probe(
        &self,
        collection: String,
        id: String,
    ) -> anyhow::Result<Option<Document>> {
        let provider = self.py.clone();

        let document = run_py(move |py| {
            let result = provider.call_method1(py, "freshness_probe", (collection, id))?;
            let result = result.downcast_bound::<PyList>(py)?;
            let result = Vec::<Document>::extract_bound(result)?;

            match &result[..] {
                [] => Ok(None),
                [doc] => Ok(Some(doc.clone())),
                _ => Err(PyValueError::new_err(format!(
                    "expected 1 document, got {}",
                    result.len()
                ))),
            }
        })
        .await?;

        Ok(document)
    }

    async fn point_get(
        &self,
        collection: String,
        id: String,
    ) -> anyhow::Result<Option<Document>> {
        let provider = self.py.clone();

        let document = run_py(move |py| {
            // Mirrors the trait's default: a provider whose client has no key lookup
            // simply does not define point_get, and falls back to the freshness probe.
            // Its `get` column is then not the same operation as the providers that do.
            let method = match provider.bind(py).hasattr("point_get")? {
                true => "point_get",
                false => "freshness_probe",
            };
            let result = provider.call_method1(py, method, (collection, id))?;
            let result = result.downcast_bound::<PyList>(py)?;
            let result = Vec::<Document>::extract_bound(result)?;

            match &result[..] {
                [] => Ok(None),
                [doc] => Ok(Some(doc.clone())),
                _ => Err(PyValueError::new_err(format!(
                    "expected 1 document, got {}",
                    result.len()
                ))),
            }
        })
        .await?;

        Ok(document)
    }

    async fn query(
        &self,
        collection: String,
        vector: Vec<f32>,
        top_k: u32,
        int_filter: Option<u32>,
        keyword_filter: Option<String>,
    ) -> anyhow::Result<Vec<Document>> {
        let provider = self.py.clone();

        let documents = run_py(move |py| {
            let result = provider.call_method1(
                py,
                "query",
                (collection, vector, top_k, int_filter, keyword_filter),
            )?;
            let result = result.downcast_bound::<PyList>(py)?;
            Vec::<Document>::extract_bound(result)
        })
        .await?;

        Ok(documents)
    }

    async fn close(&self) -> anyhow::Result<()> {
        let provider = self.py.clone();

        run_py(move |py| provider.call_method0(py, "close")).await?;

        Ok(())
    }
}

/// Spawn a blocking task that acquires the Python GIL to execute Python code.
///
/// Tokio <> GIL Interaction:
/// - This function is called from async code running on the Tokio runtime
/// - tokio::task::spawn_blocking() spawns a thread from the runtime's blocking thread pool
/// - Python::with_gil() acquires the GIL in that thread to safely call Python code
/// - This works because the GIL is released before block_on() in the caller (see ingest.rs)
///
/// Why this works:
/// - The GIL is not held by the thread blocked in block_on() (released via allow_threads)
/// - spawn_blocking threads can acquire the GIL when Python::with_gil() is called
/// - No deadlock because the GIL is available for acquisition
async fn run_py<F, R>(f: F) -> PyResult<R>
where
    F: FnOnce(Python<'_>) -> PyResult<R> + Send + 'static,
    R: Send + 'static,
{
    tokio::task::spawn_blocking(move || Python::with_gil(move |py| f(py)))
        .await
        .map_err(|e| PyValueError::new_err(format!("Failed to run Python code: {e}")))?
}
