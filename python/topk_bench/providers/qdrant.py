import os
from qdrant_client import QdrantClient, models
from ..topk_bench import Document, Provider


class QdrantProvider(Provider):
    def __init__(self, url: str | None = None, api_key: str | None = None):
        self.client = QdrantClient(
            url=url or os.environ["QDRANT_URL"],
            api_key=api_key or os.environ["QDRANT_API_KEY"],
        )

    def name(self) -> str:
        return "qdrant"

    def setup(self, collection: str):
        if collection not in [
            c.name for c in self.client.get_collections().collections
        ]:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                ),
            )

            self.client.create_payload_index(
                collection_name=collection,
                field_name="int_filter",
                field_schema=models.PayloadSchemaType.INTEGER,
            )

            self.client.create_payload_index(
                collection_name=collection,
                field_name="keyword_filter",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def query_by_id(self, collection: str, id: str):
        result = self.client.retrieve(
            collection_name=collection,
            ids=[int(id)],
            with_payload=True,
        )
        return [to_document(row) for row in result]

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        int_filter: int | None,
        keyword_filter: str | None,
    ) -> list[Document]:
        # Build filter
        filters = []
        if int_filter is not None:
            filters.append(
                models.FieldCondition(
                    key="int_filter",
                    range=models.Range(lte=int_filter),
                )
            )
        if keyword_filter is not None:
            filters.append(
                models.FieldCondition(
                    key="keyword_filter",
                    match=models.MatchValue(value=keyword_filter),
                )
            )
        qfilter = None
        if filters:
            qfilter = models.Filter(must=filters)

        result = self.client.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=qfilter,
        )
        return [to_document(point) for point in result.points]

    def upsert(self, collection: str, docs: list[Document]):
        try:
            self.client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=int(doc.id),
                        vector=doc.dense_embedding
                        if doc.dense_embedding is not None
                        else [],
                        payload={
                            "text": doc.text,
                            "int_filter": doc.int_filter,
                            "keyword_filter": doc.keyword_filter.split(" ")
                            if isinstance(doc.keyword_filter, str)
                            else doc.keyword_filter,
                        },
                    )
                    for doc in docs
                ],
                wait=True,
            )
        except Exception as e:
            # print the full error (error message would not get printed otherwise)
            print(e)
            raise e

    def delete_by_id(self, collection: str, ids: list[str]):
        self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[int(id) for id in ids]),
        )

    def delete_collection(self, collection: str):
        self.client.delete_collection(collection_name=collection)

    def list_collections(self):
        return [c.name for c in self.client.get_collections().collections]

    def close(self):
        pass


def to_document(point) -> Document:
    """Convert Qdrant point to Document."""
    payload = point.payload or {}
    # Handle keyword_filter - Qdrant stores it as a list, but we need it as a string
    keyword_filter = payload.get("keyword_filter")
    if isinstance(keyword_filter, list):
        # Join list items with space to convert back to string
        keyword_filter = " ".join(str(k) for k in keyword_filter)
    elif keyword_filter is None:
        keyword_filter = ""
    else:
        # Already a string, keep as-is
        keyword_filter = str(keyword_filter)

    return Document(
        id=str(point.id),
        text=payload.get("text", ""),
        int_filter=payload.get("int_filter", 0),
        keyword_filter=keyword_filter,
    )
