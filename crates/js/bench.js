#!/usr/bin/env node
// Driver for the JS client. Mirrors local.py's qps and ksweep modes so the two produce
// the same parquet layout, which is what lets one notebook read both.
const path = require('path')
const addon = require(path.join(__dirname, 'topk-bench.node'))
const { TopKJsProvider } = require('./provider')

function bindProvider(p) {
  // The addon pulls each method off the object once and calls it without a `this`.
  //
  // Multi-argument methods arrive as a single array rather than spread positional
  // arguments, so they are destructured here. Doing it in the adapter rather than in
  // the provider keeps the provider looking like the code a topk-js user would
  // actually write, which is the thing being measured.
  return {
    setup: (collection) => p.setup(collection),
    upsert: (args) => p.upsert(...args),
    queryById: (args) => p.queryById(...args),
    query: (args) => p.query(...args),
    close: () => p.close(),
  }
}

const CACHE_DIR = process.env.BENCH_CACHE_DIR || '/tmp/topk-bench'
const RESULTS_DIR = process.env.BENCH_RESULTS_DIR || './results'
const PREFIX = process.env.BENCH_COLLECTION_PREFIX || 'x'

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 ? process.argv[i + 1] : dflt
}

async function main() {
  const mode = process.argv[2] || 'qps'
  const size = arg('size', '100k')
  const timeout = parseInt(arg('timeout', '30'), 10)
  const collection = `${PREFIX}-${size}`
  const queries = `s3://topk-bench/queries-${size}.parquet`

  addon.installTelemetry()
  const provider = bindProvider(new TopKJsProvider())

  const run = (cfg) =>
    addon.query(provider, {
      collection,
      queries,
      cacheDir: CACHE_DIR,
      size,
      // napi maps Option<T> to undefined; null is a type error at the boundary.
      intFilter: undefined,
      keywordFilter: undefined,
      readWrite: false,
      ...cfg,
    })

  // Warmup is recorded but tagged, exactly as the Python driver does; the notebook
  // drops those rows at load.
  await run({ topK: 10, concurrency: 1, timeout: timeout * 2, warmup: true, mode })

  if (mode === 'qps') {
    for (const c of [1, 2, 4, 8]) {
      console.log(`[qps] topk-js (${size}) concurrency=${c}...`)
      await run({ topK: 10, concurrency: c, timeout, warmup: false, mode })
    }
  } else if (mode === 'ksweep') {
    const ks = (process.env.BENCH_K_SWEEP || '1,10,100,1000').split(',').map(Number)
    for (const k of ks) {
      // Same rule as local.py: stretch the window with k so each point has a
      // comparable number of samples behind its p99.
      const t = Math.floor(timeout * Math.min(4, Math.max(1, k / 25)))
      console.log(`[ksweep] topk-js (${size}) top_k=${k} for ${t}s...`)
      await run({ topK: k, concurrency: 1, timeout: t, warmup: false, mode })
    }
  } else {
    throw new Error(`unknown mode: ${mode}`)
  }

  const dst = nextSlot(RESULTS_DIR, mode, size)
  await addon.writeMetrics(dst)
  console.log(`[${mode}] -> ${dst}`)
}

function nextSlot(dir, mode, size) {
  const fs = require('fs')
  fs.mkdirSync(dir, { recursive: true })
  let n = 1
  while (fs.existsSync(`${dir}/topk_js_${mode}_${size}_r${n}.parquet`)) n++
  return `${dir}/topk_js_${mode}_${size}_r${n}.parquet`
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
