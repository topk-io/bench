//! Node binding: the entry points a JS driver calls, plus the bridge to a JS provider
//! object. All measurement lives in `topk_bench_core`, exactly as for the Python
//! binding -- same driver loop, same clock, same metrics.

use std::sync::Arc;

use napi::bindgen_prelude::Object;
use napi::bindgen_prelude::PromiseRaw;
use napi::Env;
use napi_derive::napi;
use topk_bench_core::{ingest, query, telemetry};

mod doc;
mod provider;

/// Mirrors `IngestConfig`; a plain JS object rather than a wrapped handle, since it is
/// built once per run and its conversion cost is not on any measured path.
#[napi(object)]
pub struct IngestConfigJs {
    pub collection: String,
    pub batch_size: u32,
    pub concurrency: u32,
    pub input: String,
    pub mode: String,
    pub size: String,
    pub cache_dir: String,
}

#[napi(object)]
pub struct QueryConfigJs {
    pub collection: String,
    pub queries: String,
    pub top_k: u32,
    pub int_filter: Option<u32>,
    pub keyword_filter: Option<String>,
    pub concurrency: u32,
    pub size: String,
    pub timeout: u32,
    pub warmup: bool,
    pub read_write: bool,
    pub mode: String,
    pub cache_dir: String,
}

#[napi]
pub fn install_telemetry() -> napi::Result<()> {
    telemetry::install().map_err(err)
}

#[napi(ts_return_type = "Promise<void>")]
pub fn ingest<'env>(env: &'env Env, provider: Object, config: IngestConfigJs) -> napi::Result<PromiseRaw<'env, ()>> {
    // The Object is JS-thread-bound; the provider built from it is not. Everything after
    // this line is Send, which is what lets the shared driver run it unchanged.
    let p = Arc::new(provider::from_js_object(&provider, "topk-js".to_string())?);
    let cfg = ingest::IngestConfig {
        collection: config.collection,
        batch_size: config.batch_size as usize,
        concurrency: config.concurrency as usize,
        input: config.input,
        mode: config.mode,
        size: config.size,
        cache_dir: config.cache_dir,
    };
    env.spawn_future(async move { ingest::start(p, cfg).await.map_err(err) })
}

#[napi(ts_return_type = "Promise<void>")]
pub fn query<'env>(env: &'env Env, provider: Object, config: QueryConfigJs) -> napi::Result<PromiseRaw<'env, ()>> {
    let p = Arc::new(provider::from_js_object(&provider, "topk-js".to_string())?);
    let cfg = query::QueryConfig {
        collection: config.collection,
        queries: config.queries,
        top_k: config.top_k,
        int_filter: config.int_filter,
        keyword_filter: config.keyword_filter,
        concurrency: config.concurrency as usize,
        size: config.size,
        timeout: config.timeout as u64,
        warmup: config.warmup,
        read_write: config.read_write,
        mode: config.mode,
        cache_dir: config.cache_dir,
    };
    env.spawn_future(async move { query::start(cfg, p).await.map_err(err) })
}

#[napi(ts_return_type = "Promise<void>")]
pub fn write_metrics<'env>(env: &'env Env, path: String) -> napi::Result<PromiseRaw<'env, ()>> {
    env.spawn_future(async move { telemetry::export(&path).await.map_err(err) })
}

fn err(e: anyhow::Error) -> napi::Error {
    napi::Error::from_reason(format!("{e:?}"))
}
