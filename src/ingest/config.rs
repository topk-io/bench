use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct IngestConfig {
    pub collection: String,
    pub batch_size: usize,
    pub concurrency: usize,
    pub input: String,
    pub mode: String,
    pub size: String,
    pub cache_dir: String,
}

#[pymethods]
impl IngestConfig {
    #[new]
    fn new(
        collection: String,
        batch_size: usize,
        concurrency: usize,
        input: String,
        mode: String,
        size: String,
        cache_dir: String,
    ) -> Self {
        Self {
            collection,
            batch_size,
            concurrency,
            input,
            mode,
            size,
            cache_dir,
        }
    }
}
