const topk = require('topk-js')
const { select, field } = topk.query
// vectorDistance lives on the top-level query_fn export, not under query.
const fn = topk.query_fn

/// The topk-js client, driven by the shared Rust harness.
///
/// Mirrors `python/topk_bench/providers/topk.py` deliberately: same select list, same
/// filters, same topk stage. Any divergence would make topk-js quietly incomparable to
/// the other clients, which is the one thing it exists to be compared against --
/// preflight's cross-provider canary is the guard.
class TopKJsProvider {
  constructor() {
    this.client = new topk.Client({
      apiKey: process.env.TOPK_API_KEY,
      region: process.env.TOPK_REGION,
      host: process.env.TOPK_HOST || 'topk.io',
      https: process.env.TOPK_HTTPS !== '0',
    })
  }

  async setup(collection) {
    try {
      await this.client.collections().create(collection, {
        text: topk.schema.text().required(),
        dense_embedding: topk.schema
          .f32Vector({ dimension: 768 })
          .index(topk.schema.vectorIndex({ metric: 'cosine' })),
        int_filter: topk.schema.int().required(),
        keyword_filter: topk.schema
          .text()
          .required()
          .index(topk.schema.keywordIndex()),
      })
    } catch (e) {
      if (!/already exists/i.test(String(e))) throw e
    }
  }

  async upsert(collection, docs) {
    await this.client.collection(collection).upsert(
      docs.map((d) => ({
        _id: d.id,
        text: d.text,
        int_filter: d.intFilter,
        keyword_filter: d.keywordFilter,
        ...(d.denseEmbedding ? { dense_embedding: topk.data.f32Vector(d.denseEmbedding) } : {}),
      })),
    )
    // The proto encoding happens inside the SDK, so the harness cannot see the wire
    // size from here; undefined means "not reported".
    return undefined
  }

  async freshnessProbe(collection, id) {
    const rows = await this.client
      .collection(collection)
      .query(select({ text: field('text') }).filter(field('_id').eq(id)))
    return rows.map(toDocument)
  }

  // A real key lookup, not the filtered query above. Different access path: the query
  // path serves from an in-memory cache and this does not.
  async pointGet(collection, id) {
    const rows = await this.client
      .collection(collection)
      .get([id], ['text', 'int_filter', 'keyword_filter'])
    return Object.values(rows).map(toDocument)
  }

  async query(collection, vector, topK, intFilter, keywordFilter) {
    let q = select({
      text: field('text'),
      int_filter: field('int_filter'),
      keyword_filter: field('keyword_filter'),
      vector_distance: fn.vectorDistance('dense_embedding', topk.data.f32Vector(vector)),
    })

    if (intFilter !== null && intFilter !== undefined) {
      q = q.filter(field('int_filter').lte(intFilter))
    }
    if (keywordFilter !== null && keywordFilter !== undefined) {
      q = q.filter(field('keyword_filter').matchAll(keywordFilter))
    }

    q = q.topk(field('vector_distance'), topK)

    const rows = await this.client.collection(collection).query(q)
    return rows.map(toDocument)
  }

  async close() {}
}

/// The harness boundary with no client behind it.
///
/// Ingest hands every document across napi as 768 f64s onto the single JS thread, which
/// reads never do -- a read carries ten documents back. This arm is what that marshalling
/// costs before topk-js is involved at all, so a topk-js ingest number can be read as
/// client cost rather than binding cost.
class NullProvider {
  async setup() {}
  async upsert() {
    return undefined
  }
  // The driver polls this after every upsert to time freshness, and nothing was actually
  // written, so an empty result would spin that poll until its deadline on every batch.
  async freshnessProbe(collection, id) {
    return [{ id, text: '', intFilter: 0, keywordFilter: '' }]
  }
  async pointGet(collection, id) {
    return [{ id, text: '', intFilter: 0, keywordFilter: '' }]
  }
  async query() {
    return []
  }
  async close() {}
}

function toDocument(row) {
  return {
    id: row._id ?? '',
    text: row.text ?? '',
    intFilter: row.int_filter ?? 0,
    keywordFilter: row.keyword_filter ?? '',
    // napi maps Option<T> to undefined; null is a type error at the boundary.
    denseEmbedding: undefined,
    tag: undefined,
  }
}

module.exports = { TopKJsProvider, NullProvider }
