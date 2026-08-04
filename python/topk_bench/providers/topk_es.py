import os

import orjson
from elastic_transport import JsonSerializer
from elasticsearch import Elasticsearch, NotFoundError

from ..topk_bench import Document, Provider

# TopK's Elasticsearch shim exposes an ES-compatible HTTP API, so the official
# client works unmodified. Point ES_URL at the shim (:9200 by default); the
# repo's docker-compose runs a real Elasticsearch on :9201 for conformance
# runs, so the same provider can benchmark either backend.
#
# Supported surface (topk-es/src/api): search, msearch, bulk, index, doc, mget,
# count, mapping, refresh, source, aggs. Query DSL: bool (must/filter/must_not/
# should), term, terms, range, match, multi_match, match_all, ids, prefix,
# regexp, exists, semantic. Vector search is a top-level `knn` clause.

DIM = 768



class OrjsonSerializer(JsonSerializer):
    """JSON via orjson instead of the stdlib.

    This is not a micro-optimisation. Every query body carries a 768-float
    query_vector, and stdlib json.dumps is a C call that holds the GIL
    uninterruptibly for ~0.43 ms to encode it. The harness drives 8 concurrent
    threads through one provider, so those holds convoy: measured throughput on
    a 100k collection went 151 -> 382 QPS (p50 49.6 -> 18.2 ms) purely by
    swapping the serializer. Raw CPU time understates the cost badly, because
    GIL contention is not linear in the time held.
    """

    def dumps(self, data):
        if isinstance(data, (str, bytes)):
            return data
        return orjson.dumps(data)

    def loads(self, data):
        return orjson.loads(data)


class TopKESProvider(Provider):
    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        num_candidates: int | None = None,
        request_timeout: int = 60,
    ):
        region = os.environ["TOPK_REGION"]
        base = os.environ.get("TOPK_HOST", "topk.io")

        self._url = url or os.environ.get("ES_URL") or f"https://{region}.es.{base}"
        self._api_key = api_key or os.environ.get("TOPK_API_KEY")
        self._request_timeout = request_timeout

        # `num_candidates` is Option<u64> in the shim: omitting it sends None
        # and the engine takes its own path, which is what keeps this
        # equivalent to the native provider. Set it only to sweep the
        # recall/latency knob deliberately — a fixed value here would make the
        # ES-vs-proto comparison measure our tuning rather than the shim.
        self.num_candidates = num_candidates

        # ONE shared client, deliberately.
        #
        # Do not "optimise" this into a thread-local client per worker. That
        # was tried, and it is catastrophic here: the Rust harness dispatches
        # every query through tokio::task::spawn_blocking (src/provider.rs),
        # whose pool churns threads rather than pinning them. A thread-local
        # therefore misses on nearly every call -- an instrumented run at
        # concurrency=1 constructed 58 clients for ~60 queries -- and each miss
        # pays a fresh TLS handshake. That handshake is ~70 ms against the
        # shim (TCP connect is only ~2 ms), so it dwarfs the ~14 ms query and
        # pins throughput near 1/handshake regardless of concurrency or
        # collection size. Measured A/B over the same 10 s window:
        # thread-local 5-6 QPS / p99 285 ms, shared client 70 QPS / p99 19 ms.
        #
        # elastic_transport's own connection pool is thread-safe and already
        # does the pooling; size it past the harness's max concurrency (8) so
        # workers never queue on a connection.
        self._client = Elasticsearch(
            self._url,
            api_key=self._api_key,
            request_timeout=self._request_timeout,
            connections_per_node=16,
            serializers={"application/json": OrjsonSerializer()},
        )

    @property
    def client(self) -> Elasticsearch:
        return self._client

    def name(self) -> str:
        return "topk-es"

    def setup(self, collection: str):
        if self.client.indices.exists(index=collection):
            return

        self.client.indices.create(
            index=collection,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "dense_embedding": {
                        "type": "dense_vector",
                        "dims": DIM,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "int_filter": {"type": "integer"},
                    # `text` (not `keyword`) so the field is tokenized and a
                    # match query with operator=and can require every token,
                    # mirroring native match_all semantics.
                    "keyword_filter": {"type": "text"},
                }
            },
        )

    def query_by_id(self, collection: str, id: str):
        try:
            res = self.client.get(
                index=collection,
                id=id,
                source_includes=["text", "int_filter", "keyword_filter"],
            )
        except NotFoundError:
            return []
        return [to_document(res["_id"], res.get("_source", {}))]

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        int_filter: int | None,
        keyword_filter: str | None,
    ) -> list[Document]:
        filters = []
        if int_filter is not None:
            filters.append({"range": {"int_filter": {"lte": int_filter}}})
        if keyword_filter is not None:
            # operator=and => every token must be present (native match_all).
            filters.append(
                {
                    "match": {
                        "keyword_filter": {"query": keyword_filter, "operator": "and"}
                    }
                }
            )

        knn = {"field": "dense_embedding", "query_vector": vector, "k": top_k}
        if filters:
            knn["filter"] = filters
        if self.num_candidates is not None:
            knn["num_candidates"] = self.num_candidates

        res = self.client.search(
            index=collection,
            knn=knn,
            size=top_k,
            source_includes=["text", "int_filter", "keyword_filter"],
        )
        return [to_document(h["_id"], h.get("_source", {})) for h in res["hits"]["hits"]]

    def _post_bulk(self, collection: str, body: str):
        """POST one bulk request.

        The official client (and helpers.bulk) issues `PUT /<index>/_bulk`.
        Real Elasticsearch routes both PUT and POST, but the shim only accepts
        POST and answers PUT with a bare 405, so the client's normal bulk path
        cannot be used. Going through perform_request pins the method.
        """
        res = self.client.perform_request(
            "POST",
            f"/{collection}/_bulk",
            headers={
                "content-type": "application/x-ndjson",
                "accept": "application/json",
            },
            body=body,
        )
        if res.body.get("errors"):
            first = next(
                (
                    item[op]["error"]
                    for item in res.body.get("items", [])
                    for op in item
                    if "error" in item[op]
                ),
                "unknown error",
            )
            raise RuntimeError(f"bulk failed: {first}")
        return res

    def _bulk(self, collection: str, lines: list[dict]):
        """Encode and POST the whole batch as one bulk request.

        This deliberately does NOT split. The provider used to chunk on an internal
        MAX_BULK_BYTES, which meant batch size was controlled in two places at once:
        the harness asked for 2000 documents and the provider quietly turned that into
        ~15 HTTP requests. `bench.ingest.requests` counts the logical batch, so the
        amplification was invisible in the metrics and led to a wrong conclusion about
        where the write penalty came from.

        Now `--batch-size` is the only knob and one logical batch is one request, so
        request count is whatever the caller asked for. The caller is responsible for
        keeping batch x document size under the server's body limit (64 MiB as of
        2026-08-02); exceeding it fails loudly rather than being silently papered over.

        orjson rather than the stdlib: a batch is thousands of 768-float vectors, and
        stdlib encoding would hold the GIL far longer. The transport wants str for
        x-ndjson, and decoding is cheap next to encoding.
        """
        body = b"".join(orjson.dumps(line) + b"\n" for line in lines)
        self._post_bulk(collection, body.decode())
        return len(body)

    def upsert(self, collection: str, docs: list[Document]):
        # NB: no refresh here, deliberately. The freshness benchmark measures
        # write-to-visible by polling query_by_id; forcing a refresh would
        # fabricate that number and make it incomparable to the native
        # provider, which has no such control.
        lines = []
        for doc in docs:
            lines.append({"index": {"_id": doc.id}})
            lines.append(
                {
                    "text": doc.text,
                    "dense_embedding": doc.dense_embedding or [],
                    "int_filter": doc.int_filter,
                    "keyword_filter": doc.keyword_filter,
                }
            )
        # Returned to the harness as bench.ingest.wire_bytes. Against
        # bench.ingest.upserted_bytes (source size, identical for every provider)
        # this gives the protocol's encoding tax as a measurement rather than a
        # constant calibrated by hand.
        return self._bulk(collection, lines)

    def delete_by_id(self, collection: str, ids: list[str]):
        self._bulk(collection, [{"delete": {"_id": i}} for i in ids])

    def delete_collection(self, collection: str):
        self.client.indices.delete(index=collection, ignore_unavailable=True)

    def list_collections(self):
        """Enumerate collections.

        The shim implements no index-enumeration endpoint: `*`, `_all`,
        `_cat/indices` and `_stats` are all parsed as index names and rejected
        as invalid. Enumeration is only used for housekeeping (cleanup), never
        on a benchmarked path, so fall back to the native SDK -- it is the same
        backend and the same state.
        """
        try:
            return list(self.client.indices.get(index="*").keys())
        except Exception:
            import topk_sdk as t

            client = t.Client(
                api_key=os.environ["TOPK_API_KEY"],
                region=os.environ["TOPK_REGION"],
                host=os.environ.get("TOPK_HOST", "topk.io"),
                https=True,
            )
            return [c.name for c in client.collections().list()]

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass


def to_document(id: str, source: dict) -> Document:
    return Document(
        id=str(id),
        text=source.get("text", ""),
        int_filter=source.get("int_filter", 0),
        keyword_filter=source.get("keyword_filter", ""),
    )
