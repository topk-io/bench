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

    pub async fn upsert(&self, collection: String, docs: Vec<Document>) -> PyResult<()> {
        let provider = self.py.clone();

        run_py(move |py| provider.call_method1(py, "upsert", (collection, docs))).await?;

        Ok(())
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
