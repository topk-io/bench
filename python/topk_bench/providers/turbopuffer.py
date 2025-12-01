import turbopuffer
import os
from ..topk_bench import Document, Provider


class TurbopufferProvider(Provider):
    def __init__(self, api_key: str | None = None, region: str | None = None):
        self.client = turbopuffer.Turbopuffer(
            api_key=api_key or os.environ["TURBOPUFFER_API_KEY"],
            region=region or os.environ["TURBOPUFFER_REGION"],
        )

    def name(self) -> str:
        return "turbopuffer"

    def setup(self, namespace: str):
        if self.client.namespace(namespace).exists():
            return

        # In Turbopuffer, namespaces are created implicitly when the first document is upserted.
        # However, we need the namespace to be already created, as we want to send empty queries
        # to warm up the connection (eg. ping). We use the same collection for warm up and real
        # queries to make sure the provider has a chance to set up the namespace.
        doc = Document(
            id="__bootstrap__",
            text="Hello, world!",
            dense_embedding=[0.1] * 768,
            int_filter=1,
            keyword_filter="Hello",
        )
        self.upsert(namespace, [doc])
        self.delete_by_id(namespace, ids=[doc.id])

    def query_by_id(self, namespace: str, id: str):
        result = self.client.namespace(namespace).query(
            rank_by=("id", "desc"),
            filters=("id", "Eq", id),
            top_k=1,
            include_attributes=["text", "int_filter", "keyword_filter"],
        )

        return [to_document(r) for r in (result.rows or [])]

    def delete_by_id(self, namespace: str, ids: list[str]):
        self.client.namespace(namespace).write(
            deletes=ids,
        )

    def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int,
        int_filter: int | None = None,
        keyword_filter: str | None = None,
    ) -> list[Document]:
        filters = []
        if int_filter:
            filters.append(("int_filter", "Lte", int_filter))
        if keyword_filter:
            filters.append(("keyword_filter", "ContainsAllTokens", keyword_filter))

        result = self.client.namespace(namespace).query(
            rank_by=("vector", "ANN", vector),
            top_k=top_k,
            filters=None if len(filters) == 0 else ("And", tuple(filters)),
            include_attributes=["text", "int_filter", "keyword_filter"],
        )
        return [to_document(r) for r in (result.rows or [])]

    def upsert(self, namespace: str, docs: list[Document]):
        self.client.namespace(namespace).write(
            upsert_rows=[from_document(doc) for doc in docs],
            distance_metric="cosine_distance",
            schema={
                "text": {"type": "string"},
                "int_filter": {"type": "int"},
                "keyword_filter": {"type": "string", "full_text_search": True},
            },
        )

    def delete_collection(self, namespace: str):
        self.client.namespace(namespace).delete_all()

    def list_collections(self):
        return [ns.id for ns in self.client.namespaces()]

    def close(self):
        self.client.close()


def to_document(row: turbopuffer.types.namespace_query_response.Row) -> Document:
    return Document(
        id=row.id,
        text=row.text,
        int_filter=row.int_filter,
        keyword_filter=row.keyword_filter,
    )


def from_document(doc: Document) -> turbopuffer.types.namespace_query_response.Row:
    return turbopuffer.types.namespace_query_response.Row(
        id=doc.id,
        text=doc.text,
        vector=doc.dense_embedding if doc.dense_embedding is not None else [],
        int_filter=doc.int_filter,
        keyword_filter=doc.keyword_filter,
    )
