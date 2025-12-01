use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
#[derive(Clone, Debug)]
pub struct QueryConfig {
    pub collection: String,
    pub queries: String,
    pub top_k: u32,
    pub int_filter: Option<u32>,
    pub keyword_filter: Option<String>,
    pub concurrency: usize,
    pub size: String,
    pub timeout: u64,
    pub warmup: bool,
    pub read_write: bool,
    pub mode: String,
    pub cache_dir: String,
}

#[pymethods]
impl QueryConfig {
    #[new]
    #[pyo3(signature = (collection, queries, top_k, concurrency, size, timeout, mode, cache_dir, int_filter=None, keyword_filter=None, read_write=false, warmup=false))]
    fn new(
        collection: String,
        queries: String,
        top_k: u32,
        concurrency: usize,
        size: String,
        timeout: u64,
        mode: String,
        cache_dir: String,
        int_filter: Option<u32>,
        keyword_filter: Option<String>,
        read_write: bool,
        warmup: bool,
    ) -> PyResult<Self> {
        if !["100k", "1m", "10m"].contains(&size.as_str()) {
            return Err(PyValueError::new_err(format!("Invalid size: {}", size)));
        }

        Ok(Self {
            collection,
            queries,
            top_k,
            int_filter,
            keyword_filter,
            concurrency,
            size,
            timeout,
            mode,
            cache_dir,
            read_write,
            warmup,
        })
    }
}
