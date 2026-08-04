const addon = require('./topk-bench.node')

/// Bind a provider's methods so the addon can hold them as plain functions.
///
/// The Rust side pulls each method off the object once and calls it without a `this`;
/// binding here rather than there keeps a napi round trip off the measured path.
function bindProvider(p) {
  return {
    setup: p.setup.bind(p),
    upsert: p.upsert.bind(p),
    queryById: p.queryById.bind(p),
    query: p.query.bind(p),
    close: p.close.bind(p),
  }
}

module.exports = { ...addon, bindProvider }
