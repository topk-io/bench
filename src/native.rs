use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use std::collections::HashMap;

use topk_rs::proto::v1::control::{
    FieldIndex, FieldSpec, KeywordIndexType, VectorDistanceMetric,
};
use topk_rs::proto::v1::data::{Document as TopkDoc, Value};
use topk_rs::query::{field, fns, select};
use topk_rs::{Client, ClientConfig};

use crate::data::Document;

/// Talks to TopK over the native proto/gRPC SDK, from Rust.
///
/// Every other provider is a Python object the harness calls into over PyO3, so
/// `topk-py` measures the Python client as much as the protocol. This one has no Python
/// on the hot path: same proto, same backend, no client-language cost. It is the floor
/// the other clients should be read against.
///
/// It mirrors `python/topk_bench/providers/topk.py` deliberately -- same select list,
/// same filters, and `topk(vector_distance, k, asc=false)`, matching the Python
/// binding's `asc=False` default so the score sorts descending as a similarity. Any
/// divergence would make topk-rs quietly incomparable to topk-py, which is the one
/// comparison it exists to enable; preflight's cross-provider canary is the guard.
#[derive(Clone)]
pub struct NativeProvider {
    client: Client,
}

// topk_rs::Client is not Debug, and AnyProvider derives it.
impl std::fmt::Debug for NativeProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NativeProvider")
    }
}

impl NativeProvider {
    pub fn from_env() -> anyhow::Result<Self> {
        let api_key = std::env::var("TOPK_API_KEY")?;
        let region = std::env::var("TOPK_REGION")?;
        let mut config = ClientConfig::new(api_key, region);
        if let Ok(host) = std::env::var("TOPK_HOST") {
            config = config.with_host(host);
        }
        if let Ok(https) = std::env::var("TOPK_HTTPS") {
            config = config.with_https(!matches!(https.as_str(), "0" | "false" | "False"));
        }
        Ok(Self {
            client: Client::new(config),
        })
    }

    pub async fn name(&self) -> PyResult<String> {
        Ok("topk-rs".to_string())
    }

    pub async fn setup(&self, collection: String) -> PyResult<()> {
        // Mirrors `TopKProvider.setup` field for field. A no-op would work for the read
        // collections, which are created once and reused -- but the write sweep ingests
        // into bs-*, and whichever provider runs first has to create it. Diverging here
        // would mean topk-rs measured a differently-indexed collection than the others.
        let schema = HashMap::from([
            ("text".to_string(), FieldSpec::text(true)),
            (
                "dense_embedding".to_string(),
                FieldSpec::f32_vector(768, false)
                    .with_index(FieldIndex::vector(VectorDistanceMetric::Cosine)),
            ),
            ("int_filter".to_string(), FieldSpec::integer(true)),
            (
                "keyword_filter".to_string(),
                FieldSpec::text(true).with_index(FieldIndex::keyword(KeywordIndexType::Text)),
            ),
        ]);

        match self.client.collections().create(collection, schema, None).await {
            Ok(_) => Ok(()),
            Err(topk_rs::Error::CollectionAlreadyExists) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("setup: {e}"))),
        }
    }

    pub async fn upsert(&self, collection: String, docs: Vec<Document>) -> PyResult<Option<u64>> {
        let out: Vec<TopkDoc> = docs
            .into_iter()
            .map(|d| {
                let mut fields: Vec<(String, Value)> = vec![
                    ("_id".to_string(), d.id.into()),
                    ("text".to_string(), d.text.into()),
                    ("int_filter".to_string(), d.int_filter.into()),
                    ("keyword_filter".to_string(), d.keyword_filter.into()),
                ];
                if let Some(v) = d.dense_embedding {
                    fields.push(("dense_embedding".to_string(), v.into()));
                }
                TopkDoc::from(fields)
            })
            .collect();

        self.client
            .collection(&collection)
            .upsert(out)
            .await
            .map_err(|e| PyValueError::new_err(format!("upsert: {e}")))?;

        // The proto encoding happens inside the SDK; the harness cannot see that size
        // from here, so no wire_bytes is reported.
        Ok(None)
    }

    pub async fn query_by_id(&self, collection: String, id: String) -> PyResult<Option<Document>> {
        let q = select([("text", field("text"))]).filter(field("_id").eq(id));
        let docs = self
            .client
            .collection(&collection)
            .query(q, None, None)
            .await
            .map_err(|e| PyValueError::new_err(format!("query_by_id: {e}")))?;
        Ok(docs.into_iter().next().map(to_document))
    }

    pub async fn query(
        &self,
        collection: String,
        vector: Vec<f32>,
        top_k: u64,
        int_filter: Option<u32>,
        keyword_filter: Option<String>,
    ) -> PyResult<Vec<Document>> {
        let mut q = select([
            ("text", field("text")),
            ("int_filter", field("int_filter")),
            ("keyword_filter", field("keyword_filter")),
        ])
        .select([(
            "vector_distance",
            fns::vector_distance("dense_embedding", vector),
        )]);

        if let Some(v) = int_filter {
            q = q.filter(field("int_filter").lte(v));
        }
        if let Some(kw) = keyword_filter {
            q = q.filter(field("keyword_filter").match_all(kw));
        }

        // asc=false matches the Python binding's default for .topk()
        let q = q.topk(field("vector_distance"), top_k, false);

        let docs = self
            .client
            .collection(&collection)
            .query(q, None, None)
            .await
            .map_err(|e| PyValueError::new_err(format!("query: {e}")))?;
        Ok(docs.into_iter().map(to_document).collect())
    }

    pub async fn close(&self) -> PyResult<()> {
        Ok(())
    }
}

fn to_document(d: TopkDoc) -> Document {
    let get_str = |k: &str| -> String {
        d.fields
            .get(k)
            .and_then(|v| v.as_string())
            .unwrap_or_default()
            .to_string()
    };
    Document {
        id: get_str("_id"),
        text: get_str("text"),
        int_filter: d
            .fields
            .get("int_filter")
            .and_then(|v| v.as_u32())
            .unwrap_or_default(),
        keyword_filter: get_str("keyword_filter"),
        dense_embedding: None,
        tag: None,
    }
}
