use async_trait::async_trait;

use crate::data::Document;

/// What the drivers need from a client, and nothing more.
///
/// Was an enum over the two known providers until the crate split, which made that
/// impossible: the variants named types that now live in crates depending on this one.
/// A trait is what the shape wanted anyway -- adding a client is now a new impl in its
/// own crate rather than an arm in a match here.
///
/// `Debug` is required so a driver can name the provider in an error without a separate
/// accessor; implementations wrapping a non-Debug client write it by hand.
#[async_trait]
pub trait Provider: Send + Sync + std::fmt::Debug {
    async fn name(&self) -> anyhow::Result<String>;

    async fn setup(&self, collection: String) -> anyhow::Result<()>;

    /// Upsert a batch. Returns the number of bytes the provider actually encoded and
    /// sent, when it reports one.
    ///
    /// Deliberately separate from `bench.ingest.upserted_bytes`, which is
    /// `Document::approx_size()` over the parsed documents and is therefore identical
    /// for every provider. That value is goodput; this one is what crossed the wire.
    /// Their ratio is the protocol's encoding tax -- for a 768-float vector as JSON it
    /// is roughly 5x, and without this metric that number can only be estimated.
    ///
    /// Providers that return None simply record no wire-bytes metric.
    async fn upsert(&self, collection: String, docs: Vec<Document>)
        -> anyhow::Result<Option<u64>>;

    /// Poll for a document after writing it, to time write-to-visible.
    ///
    /// Whatever read is cheapest and most likely to observe a fresh write. It is not a
    /// benchmark of anything on its own -- `point_get` is. Keeping them separate matters:
    /// on engines where a filtered query serves from an in-memory cache and a key lookup
    /// does not, one method cannot be both the freshness probe and the point-lookup
    /// measurement without the `get` mode silently comparing different operations across
    /// clients.
    async fn freshness_probe(&self, collection: String, id: String)
        -> anyhow::Result<Option<Document>>;

    /// Fetch one document by id, using the client's real point-get where it has one.
    ///
    /// Defaults to `freshness_probe` so a provider written before this split still
    /// works; a provider whose SDK exposes a key lookup should override it, and the
    /// `get` mode is only comparable across the providers that do.
    async fn point_get(&self, collection: String, id: String)
        -> anyhow::Result<Option<Document>> {
        self.freshness_probe(collection, id).await
    }

    async fn query(
        &self,
        collection: String,
        vector: Vec<f32>,
        top_k: u32,
        int_filter: Option<u32>,
        keyword_filter: Option<String>,
    ) -> anyhow::Result<Vec<Document>>;

    async fn close(&self) -> anyhow::Result<()>;
}
