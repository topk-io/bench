#!/usr/bin/env node
// Driver for the JS client. Mirrors local.py's qps and ksweep modes so the two produce
// the same parquet layout, which is what lets one notebook read both.
const path = require('path')
const addon = require(path.join(__dirname, 'topk-bench.node'))
const { TopKJsProvider, NullProvider } = require('./provider')

const PROVIDERS = { 'topk-js': TopKJsProvider, 'js-null': NullProvider }

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
  const name = arg('provider', 'topk-js')
  const collection = `${PREFIX}-${size}`
  const queries = `s3://topk-bench/queries-${size}.parquet`

  addon.installTelemetry()
  const provider = bindProvider(new PROVIDERS[name]())

  const run = (cfg) =>
    addon.query(provider, {
      collection,
      queries,
      cacheDir: CACHE_DIR,
      size,
      providerName: name,
      // napi maps Option<T> to undefined; null is a type error at the boundary.
      intFilter: undefined,
      keywordFilter: undefined,
      readWrite: false,
      ...cfg,
    })

  if (mode === 'ingest') {
    // One batch size per invocation, like local.py -- the sweep is the caller's loop,
    // so a batch point can be rerun without redoing the rest.
    const batchSize = parseInt(arg('batch-size', process.env.BENCH_BATCH_SIZE || '2000'), 10)
    const concurrency = parseInt(arg('concurrency', process.env.BENCH_CONCURRENCY || '8'), 10)
    console.log(`[ingest] ${name} (${size}) batch=${batchSize} concurrency=${concurrency}...`)
    await addon.ingest(provider, {
      collection,
      batchSize,
      concurrency,
      input: `s3://topk-bench/docs-${size}.parquet`,
      mode,
      size,
      cacheDir: CACHE_DIR,
      providerName: name,
    })
    const dst = nextSlot(RESULTS_DIR, name, mode, size)
    await addon.writeMetrics(dst)
    console.log(`[${mode}] -> ${dst}`)
    return
  }

  // Warmup is recorded but tagged, exactly as the Python driver does; the notebook
  // drops those rows at load.
  if (!process.argv.includes('--no-warmup')) {
    await run({ topK: 10, concurrency: 1, timeout: timeout * 2, warmup: true, mode })
  }

  if (mode === 'qps' || mode === 'get') {
    const steps = (process.env.BENCH_CONCURRENCY_STEPS || '1,2,4,8').split(',').map(Number)
    for (const c of steps) {
      console.log(`[${mode}] ${name} (${size}) concurrency=${c}...`)
      await run({ topK: 10, concurrency: c, timeout, warmup: false, mode })
    }
  } else if (mode === 'rw') {
    for (const rw of [false, true]) {
      console.log(`[rw] ${name} (${size}) read_write=${rw}...`)
      await run({ topK: 10, concurrency: 1, timeout, warmup: false, readWrite: rw, mode })
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

  const dst = nextSlot(RESULTS_DIR, name, mode, size)
  await addon.writeMetrics(dst)
  console.log(`[${mode}] -> ${dst}`)
}

function nextSlot(dir, name, mode, size) {
  const fs = require('fs')
  fs.mkdirSync(dir, { recursive: true })
  const slug = name.replace(/-/g, '_')
  let n = 1
  while (fs.existsSync(`${dir}/${slug}_${mode}_${size}_r${n}.parquet`)) n++
  return `${dir}/${slug}_${mode}_${size}_r${n}.parquet`
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
