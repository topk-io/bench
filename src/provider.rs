use std::sync::Arc;

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};

use crate::data::Document;

#[pyclass(subclass)]
#[derive(Debug, Clone)]
pub struct Provider {}

#[pymethods]
impl Provider {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let _ = (args, kwargs); // Suppress unused variable warnings
        Ok(Self {})
    }
}

#[derive(Debug, Clone)]
pub struct PyProvider {
    py: Arc<Py<PyAny>>,
}

impl FromPyObject<'_> for PyProvider {
    fn extract_bound(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyProvider {
            py: Arc::new(obj.as_any().clone().into()),
        })
    }
}

impl PyProvider {
    pub async fn name(&self) -> PyResult<String> {
        let provider = self.py.clone();

        run_py(move |py| -> PyResult<String> {
            let name = provider.call_method0(py, "name")?;
            let name = name.extract(py)?;
            Ok(name)
        })
        .await
    }

    pub async fn setup(&self, collection: String) -> PyResult<()> {
        let provider = self.py.clone();

        run_py(move |py| provider.call_method1(py, "setup", (collection,))).await?;

        Ok(())
    }

    /// Upsert a batch. Returns the number of bytes the provider actually encoded and
    /// sent, when it reports one.
    ///
    /// This is deliberately separate from `bench.ingest.upserted_bytes`, which is
    /// `Document::approx_size()` over the parsed documents and is therefore identical
    /// for every provider. That value is goodput; this one is what crossed the wire.
    /// Their ratio is the protocol's encoding tax -- for a 768-float vector as JSON it
    /// is roughly 5x, and without this metric that number can only be estimated.
    ///
    /// Providers that return None simply record no wire-bytes metric.
    pub async fn upsert(
        &self,
        collection: String,
        docs: Vec<Document>,
    ) -> PyResult<Option<u64>> {
        let provider = self.py.clone();

        let wire = run_py(move |py| -> PyResult<Option<u64>> {
            let res = provider.call_method1(py, "upsert", (collection, docs))?;
            // None, or a provider that returns nothing, means "not reported".
            Ok(res.extract::<Option<u64>>(py).unwrap_or(None))
        })
        .await?;

        Ok(wire)
    }

    pub async fn query_by_id(&self, collection: String, id: String) -> PyResult<Option<Document>> {
        let provider = self.py.clone();

        let document = run_py(move |py| {
            let result = provider.call_method1(py, "query_by_id", (collection, id))?;
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

    pub async fn query(
        &self,
        collection: String,
        vector: Vec<f32>,
        top_k: u32,
        int_filter: Option<u32>,
        keyword_filter: Option<String>,
    ) -> PyResult<Vec<Document>> {
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

    pub async fn close(&self) -> PyResult<()> {
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


/// The provider the harness actually drives.
///
/// `Py` calls into a Python object over PyO3 -- that is every provider except one.
/// `Native` is `topk-rs`, which has no Python on the hot path and therefore measures the
/// protocol rather than the client language.
#[derive(Debug, Clone)]
pub enum AnyProvider {
    Py(PyProvider),
    Native(crate::native::NativeProvider),
}

impl AnyProvider {
    pub async fn name(&self) -> PyResult<String> {
        match self {
            Self::Py(p) => p.name().await,
            Self::Native(p) => p.name().await,
        }
    }

    pub async fn setup(&self, collection: String) -> PyResult<()> {
        match self {
            Self::Py(p) => p.setup(collection).await,
            Self::Native(p) => p.setup(collection).await,
        }
    }

    pub async fn upsert(&self, collection: String, docs: Vec<Document>) -> PyResult<Option<u64>> {
        match self {
            Self::Py(p) => p.upsert(collection, docs).await,
            Self::Native(p) => p.upsert(collection, docs).await,
        }
    }

    pub async fn query_by_id(&self, collection: String, id: String) -> PyResult<Option<Document>> {
        match self {
            Self::Py(p) => p.query_by_id(collection, id).await,
            Self::Native(p) => p.query_by_id(collection, id).await,
        }
    }

    pub async fn query(
        &self,
        collection: String,
        vector: Vec<f32>,
        top_k: u32,
        int_filter: Option<u32>,
        keyword_filter: Option<String>,
    ) -> PyResult<Vec<Document>> {
        match self {
            Self::Py(p) => {
                p.query(collection, vector, top_k, int_filter, keyword_filter)
                    .await
            }
            // the SDK's topk stage takes u64
            Self::Native(p) => {
                p.query(collection, vector, top_k as u64, int_filter, keyword_filter)
                    .await
            }
        }
    }

    pub async fn close(&self) -> PyResult<()> {
        match self {
            Self::Py(p) => p.close().await,
            Self::Native(p) => p.close().await,
        }
    }
}

impl FromPyObject<'_> for AnyProvider {
    fn extract_bound(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // A provider marked __native__ is built in Rust from the environment; everything
        // else is driven as a Python object.
        let native = obj
            .getattr("__native__")
            .and_then(|v| v.extract::<bool>())
            .unwrap_or(false);
        if native {
            let p = crate::native::NativeProvider::from_env()
                .map_err(|e| PyValueError::new_err(format!("topk-rs provider: {e}")))?;
            return Ok(AnyProvider::Native(p));
        }
        Ok(AnyProvider::Py(PyProvider::extract_bound(obj)?))
    }
}
