use napi_derive::napi;
use topk_bench_core::data::Document;

/// A document as JavaScript sees it.
///
/// Separate from the core `Document` because JS has no f32: the embedding widens to
/// f64 crossing the boundary and narrows coming back. That conversion is not harness
/// overhead to be optimised away -- topk-js takes the same widening from any caller, so
/// it is part of what this client costs.
#[napi(object)]
pub struct DocumentJs {
    pub id: String,
    pub text: String,
    pub int_filter: u32,
    pub keyword_filter: String,
    pub dense_embedding: Option<Vec<f64>>,
    pub tag: Option<String>,
}

impl From<Document> for DocumentJs {
    fn from(d: Document) -> Self {
        Self {
            id: d.id,
            text: d.text,
            int_filter: d.int_filter,
            keyword_filter: d.keyword_filter,
            dense_embedding: d
                .dense_embedding
                .map(|v| v.into_iter().map(|f| f as f64).collect()),
            tag: d.tag,
        }
    }
}

impl From<DocumentJs> for Document {
    fn from(d: DocumentJs) -> Self {
        Self {
            id: d.id,
            text: d.text,
            int_filter: d.int_filter,
            keyword_filter: d.keyword_filter,
            dense_embedding: d
                .dense_embedding
                .map(|v| v.into_iter().map(|f| f as f32).collect()),
            tag: d.tag,
        }
    }
}
