#!/usr/bin/env node
// Parity canary for the JS client.
//
// preflight.py gates a sweep on every client returning the same documents for the same
// query, but topk-js has no Python provider object to call -- it only exists behind the
// node driver. So preflight pipes {collection, vector, k} in here and reads ids back.
//
//   echo '{"collection":"x-100k","vector":[...],"k":10}' | node canary.js
const { TopKJsProvider } = require('./provider')

async function main() {
  const chunks = []
  for await (const c of process.stdin) chunks.push(c)
  const { collection, vector, k } = JSON.parse(chunks.join(''))
  const docs = await new TopKJsProvider().query(collection, vector, k, null, null)
  process.stdout.write(JSON.stringify(docs.map((d) => d.id)))
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
