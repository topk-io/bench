import os
from pymilvus import DataType, MilvusClient
from ..topk_bench import Document, Provider


class MilvusProvider(Provider):
    def __init__(self, uri: str | None = None, token: str | None = None):
        self.client = MilvusClient(
            uri=uri or os.environ["MILVUS_URI"],
            token=token or os.environ["MILVUS_TOKEN"],
        )

    def name(self) -> str:
        return "milvus"

    def setup(self, collection: str):
        schema = MilvusClient.create_schema(
            enable_dynamic_field=False,
        )

        schema.add_field(
            "id",
            DataType.VARCHAR,
            max_length=256,
            is_primary=True,
        )
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=4096,
            enable_analyzer=True,
            enable_match=True,
        )
        schema.add_field(
            "dense_embedding",
            DataType.FLOAT_VECTOR,
            dim=768,
            enable_index=True,
        )
        schema.add_field(
            "int_filter",
            DataType.INT64,
            enable_index=True,
        )
        schema.add_field(
            "keyword_filter",
            DataType.VARCHAR,
            enable_analyzer=True,
            enable_match=True,
            max_length=256,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_embedding",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            param={"nlist": 1024},
        )
        index_params.add_index(
            field_name="keyword_filter",
            index_params={"index_type": "INVERTED"},
        )

        self.client.create_collection(
            collection_name=sanitize_collection(collection),
            schema=schema,
            index_params=index_params,
        )

        self.client.load_collection(
            collection_name=sanitize_collection(collection),
        )

    def query_by_id(self, collection: str, id: str):
        result = self.client.query(
            collection_name=sanitize_collection(collection),
            ids=[id],
            output_fields=["text", "int_filter", "keyword_filter"],
        )
        return [to_document_from_query(hit) for hit in result]

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
            filters.append(f"int_filter <= {int_filter}")
        if keyword_filter is not None:
            filters.append(f"TEXT_MATCH(keyword_filter, '{keyword_filter}')")

        # Search
        results = self.client.search(
            collection_name=sanitize_collection(collection),
            data=[vector],
            anns_field="dense_embedding",
            limit=top_k,
            filter=" and ".join(filters) if filters else None,
            output_fields=["text", "int_filter", "keyword_filter"],
        )

        # Convert
        return [to_document_from_search(hit) for hits in results for hit in hits]

    def upsert(self, collection: str, docs: list[Document]):
        self.client.upsert(
            collection_name=sanitize_collection(collection),
            data=[
                {
                    "id": doc.id,
                    "text": doc.text,
                    "dense_embedding": doc.dense_embedding
                    if doc.dense_embedding is not None
                    else [],
                    "int_filter": doc.int_filter,
                    "keyword_filter": doc.keyword_filter,
                }
                for doc in docs
            ],
        )

    def delete_by_id(self, collection: str, ids: list[str]):
        self.client.delete(
            collection_name=sanitize_collection(collection),
            filter=f"id in {ids}",
        )

    def delete_collection(self, collection: str):
        self.client.drop_collection(
            collection_name=sanitize_collection(collection),
        )

    def list_collections(self):
        return self.client.list_collections()

    def close(self):
        pass


def sanitize_collection(collection: str) -> str:
    return collection.replace("-", "_")


def to_document_from_query(entity: dict) -> Document:
    """Convert Milvus query result entity to Document."""
    return Document(
        id=str(entity.get("id", "")),
        text=entity.get("text", ""),
        int_filter=entity.get("int_filter", 0),
        keyword_filter=entity.get("keyword_filter", ""),
    )


def to_document_from_search(hit: dict) -> Document:
    """Convert Milvus search result hit to Document."""
    entity = hit.get("entity", {})
    return Document(
        id=str(hit.get("id", "")),
        text=entity.get("text", ""),
        int_filter=entity.get("int_filter", 0),
        keyword_filter=entity.get("keyword_filter", ""),
    )
