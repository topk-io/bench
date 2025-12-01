use arrow::datatypes::Int32Type;
use arrow_array::{
    types::Float64Type, Array, LargeListArray, LargeStringArray, PrimitiveArray, RecordBatch,
};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Document {
    #[pyo3(get, set)]
    pub id: String,

    #[pyo3(get, set)]
    pub text: String,

    #[pyo3(get, set)]
    pub int_filter: u32,

    #[pyo3(get, set)]
    pub keyword_filter: String,

    // Only set when upserting. We don't fetch raw vectors during queries.
    #[pyo3(get, set)]
    pub dense_embedding: Option<Vec<f32>>,

    #[pyo3(get, set)]
    pub tag: Option<String>,
}

impl Document {
    /// Approximate size of the document in bytes.
    pub fn approx_size(&self) -> usize {
        self.id.len()
            + self.text.len()
            + self.int_filter.to_le_bytes().len()
            + self.keyword_filter.len()
            + self
                .dense_embedding
                .as_ref()
                .map(|v| v.len() * std::mem::size_of::<f32>())
                .unwrap_or(0)
    }
}

#[pymethods]
impl Document {
    #[new]
    #[pyo3(signature = (id, text, int_filter, keyword_filter, dense_embedding=None, tag=None))]
    fn new(
        id: String,
        text: String,
        int_filter: u32,
        keyword_filter: String,
        dense_embedding: Option<Vec<f32>>,
        tag: Option<String>,
    ) -> Self {
        Self {
            id,
            text,
            int_filter,
            keyword_filter,
            dense_embedding,
            tag,
        }
    }
}

pub fn parse_from_batch(batch: RecordBatch) -> Vec<Document> {
    let id = batch
        .column_by_name("id")
        .expect("id column not found")
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .expect("id column is not a LargeStringArray");

    let text = batch
        .column_by_name("text")
        .expect("text column not found")
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .expect("text column is not a LargeStringArray");

    let mut dense = {
        let list = batch
            .column_by_name("dense")
            .expect("dense column not found")
            .as_any()
            .downcast_ref::<LargeListArray>()
            .expect("dense column is not LargeList<Float64>");

        let mut out = Vec::with_capacity(list.len());
        for i in 0..list.len() {
            if list.is_null(i) {
                out.push(Vec::new());
                continue;
            }
            let sub = list.value(i); // each row’s vector
            let floats = sub
                .as_any()
                .downcast_ref::<PrimitiveArray<Float64Type>>()
                .expect("inner type not Float64Array");
            let vec: Vec<f32> = floats.values().iter().map(|v| *v as f32).collect();
            out.push(vec);
        }
        out
    };

    let int_filter = batch
        .column_by_name("int_filter")
        .expect("int_filter column not found")
        .as_any()
        .downcast_ref::<PrimitiveArray<Int32Type>>()
        .expect("int_filter column is not a Int32Array");

    let keyword_filter = batch
        .column_by_name("keyword_filter")
        .expect("keyword_filter column not found")
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .expect("keyword_filter column is not a LargeStringArray");

    let mut rows = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let id = id.value(i).to_string();
        let text = text.value(i).to_string();
        let dense_embedding = std::mem::take(&mut dense[i]);
        let int_filter = int_filter.value(i) as u32;
        let keyword_filter = keyword_filter.value(i).to_string();

        rows.push(Document {
            id,
            text,
            dense_embedding: Some(dense_embedding),
            int_filter,
            keyword_filter,
            tag: None,
        });
    }

    rows
}
