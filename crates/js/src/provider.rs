use async_trait::async_trait;
use napi::bindgen_prelude::{Function, JsObjectValue, Promise};
use napi::threadsafe_function::ThreadsafeFunction;
use topk_bench_core::{data::Document, Provider};

use crate::doc::DocumentJs;

/// A threadsafe handle to one method of the JS provider.
///
/// `false` is CalleeHandled: these are ordinary functions returning a Promise, not
/// error-first callbacks. Every call is queued onto the single JS thread and resolved
/// through the returned Promise, which is also the shape of the measurement -- a Node
/// client marshals on one thread no matter how many requests are in flight.
type Method<A, R> = ThreadsafeFunction<A, Promise<R>, A, napi::Status, false>;

pub struct JsProvider {
    name: String,
    setup: Method<String, ()>,
    upsert: Method<(String, Vec<DocumentJs>), Option<u32>>,
    query_by_id: Method<(String, String), Vec<DocumentJs>>,
    query: Method<(String, Vec<f64>, u32, Option<u32>, Option<String>), Vec<DocumentJs>>,
    close: Method<(), ()>,
}

impl std::fmt::Debug for JsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JsProvider({})", self.name)
    }
}

/// Pull the provider's methods off a JS object once, at startup.
///
/// The methods must already be bound -- see `bindProvider` in index.js. Binding here
/// would mean a `this` round trip through napi on every call, which lands in the
/// measured path for no benefit.
pub fn from_js_object(obj: &napi::bindgen_prelude::Object, name: String) -> napi::Result<JsProvider> {
    macro_rules! method {
        ($key:literal) => {{
            let f: Function<_, _> = obj.get_named_property($key)?;
            f.build_threadsafe_function().callee_handled::<false>().build()?
        }};
    }

    Ok(JsProvider {
        name,
        setup: method!("setup"),
        upsert: method!("upsert"),
        query_by_id: method!("queryById"),
        query: method!("query"),
        close: method!("close"),
    })
}

#[async_trait]
impl Provider for JsProvider {
    async fn name(&self) -> anyhow::Result<String> {
        Ok(self.name.clone())
    }

    async fn setup(&self, collection: String) -> anyhow::Result<()> {
        self.setup.call_async(collection).await?.await?;
        Ok(())
    }

    async fn upsert(
        &self,
        collection: String,
        docs: Vec<Document>,
    ) -> anyhow::Result<Option<u64>> {
        // JS numbers are f64; u32 is what napi will hand back losslessly for a byte
        // count of this size, and a provider that reports nothing returns undefined.
        let docs: Vec<DocumentJs> = docs.into_iter().map(Into::into).collect();
        let wire = self.upsert.call_async((collection, docs)).await?.await?;
        Ok(wire.map(|w| w as u64))
    }

    async fn query_by_id(
        &self,
        collection: String,
        id: String,
    ) -> anyhow::Result<Option<Document>> {
        let docs = self.query_by_id.call_async((collection, id)).await?.await?;
        match docs.len() {
            0 => Ok(None),
            1 => Ok(docs.into_iter().next().map(Into::into)),
            n => Err(anyhow::anyhow!("expected 1 document, got {n}")),
        }
    }

    async fn query(
        &self,
        collection: String,
        vector: Vec<f32>,
        top_k: u32,
        int_filter: Option<u32>,
        keyword_filter: Option<String>,
    ) -> anyhow::Result<Vec<Document>> {
        // The vector widens to f64 because that is the only number JS has. This is a
        // real cost of the JS client, not an artefact of the harness -- topk-js takes
        // the same widening from any caller.
        let vector: Vec<f64> = vector.into_iter().map(|v| v as f64).collect();
        let docs = self
            .query
            .call_async((collection, vector, top_k, int_filter, keyword_filter))
            .await?
            .await?;
        Ok(docs.into_iter().map(Into::into).collect())
    }

    async fn close(&self) -> anyhow::Result<()> {
        self.close.call_async(()).await?.await?;
        Ok(())
    }
}
