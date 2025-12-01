import os
import topk_sdk as t
import topk_sdk.query as tq
import topk_sdk.error as te
import topk_sdk.schema as ts
from ..topk_bench import Document, Provider


class TopKProvider(Provider):
    def __init__(
        self,
        api_key: str | None = None,
        region: str | None = None,
        host: str | None = None,
        https: bool = True,
    ):
        self.client = t.Client(
            api_key=api_key or os.environ["TOPK_API_KEY"],
            region=region or os.environ["TOPK_REGION"],
            host=host or os.environ.get("TOPK_HOST", "topk.io"),
            https=https or bool(os.environ.get("TOPK_HTTPS", "1") == "1"),
        )

    def name(self) -> str:
        return "topk"

    def setup(self, collection: str):
        try:
            self.client.collections().create(
                collection,
                schema={
                    "text": ts.text().required(),
                    "dense_embedding": ts.f32_vector(dimension=768).index(
                        ts.vector_index(metric="cosine")
                    ),
                    "int_filter": ts.int().required(),
                    "keyword_filter": ts.text().required().index(ts.keyword_index()),
                },
            )
        except te.CollectionAlreadyExistsError:
            pass

    def query_by_id(self, collection: str, id: str):
        results = self.client.collection(collection).query(
            tq.select("text", "int_filter", "keyword_filter")
            .filter(tq.field("_id").eq(id))
            .limit(1)
        )

        return [to_document(row) for row in results]

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        int_filter: int | None,
        keyword_filter: str | None,
    ) -> list[Document]:
        query = tq.select(
            "text",
            "int_filter",
            "keyword_filter",
            vector_distance=tq.fn.vector_distance("dense_embedding", vector),
        )

        if int_filter is not None:
            query = query.filter(tq.field("int_filter").lte(int_filter))

        if keyword_filter is not None:
            query = query.filter(tq.field("keyword_filter").match_all(keyword_filter))

        query = query.topk(tq.field("vector_distance"), top_k)

        results = self.client.collection(collection).query(query)

        return [to_document(row) for row in results]

    def upsert(self, collection: str, docs: list[Document]):
        self.client.collection(collection).upsert([from_document(doc) for doc in docs])

    def delete_by_id(self, collection: str, ids: list[str]):
        self.client.collection(collection).delete(ids)

    def delete_collection(self, collection: str):
        self.client.collections().delete(collection)

    def list_collections(self):
        return [collection.name for collection in self.client.collections().list()]

    def close(self):
        pass


def to_document(row: dict) -> Document:
    return Document(
        id=row["_id"],
        text=row["text"],
        int_filter=row["int_filter"],
        keyword_filter=row["keyword_filter"],
    )


def from_document(doc: Document) -> dict:
    return {
        "_id": doc.id,
        "text": doc.text,
        "dense_embedding": doc.dense_embedding,
        "int_filter": doc.int_filter,
        "keyword_filter": doc.keyword_filter,
    }
