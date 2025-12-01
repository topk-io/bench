import os
from pinecone import QueryResponse, ServerlessSpec
from pinecone.grpc import GRPCIndex, PineconeGRPC
from ..topk_bench import Document, Provider


class PineconeProvider(Provider):
    def __init__(
        self,
        api_key: str | None = None,
        cloud: str | None = None,
        region: str | None = None,
    ):
        api_key = api_key or os.environ["PINECONE_API_KEY"]
        cloud = cloud or os.environ.get("PINECONE_CLOUD", "aws")
        region = region or os.environ.get("PINECONE_REGION", "us-east-1")
        self.client = PineconeGRPC(api_key=api_key)
        self.cloud = cloud
        self.region = region
        self._index_cache: dict[str, GRPCIndex] = {}

    def _get_index(self, collection: str):
        """Get or create cached index instance."""
        if collection not in self._index_cache:
            self._index_cache[collection] = self.client.Index(collection)
        return self._index_cache[collection]

    def name(self) -> str:
        return "pinecone"

    def setup(self, collection: str):
        if self.client.has_index(collection):
            return

        self.client.create_index(
            name=collection,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self.cloud,
                region=self.region,
            ),
        )

    def query_by_id(self, collection: str, id: str):
        index = self._get_index(collection)

        results = index.fetch(ids=[id])

        return [to_document(vector) for vector in results.vectors.values()]

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        int_filter: int | None,
        keyword_filter: str | None,
    ) -> list[Document]:
        index = self._get_index(collection)

        # Build filter
        filt = {}
        if int_filter is not None:
            filt["int_filter"] = {"$lte": int_filter}
        if keyword_filter is not None:
            filt["keyword_filter"] = {"$in": [keyword_filter]}

        results: QueryResponse = index.query(
            vector=vector,
            top_k=top_k,
            filter=None if not filt else filt,
            include_metadata=True,
        )

        return [to_document(match) for match in results["matches"]]

    def upsert(self, collection: str, docs: list[Document]):
        index = self._get_index(collection)

        index.upsert(
            vectors=[
                (
                    doc.id,
                    doc.dense_embedding,
                    {
                        "text": doc.text,
                        "int_filter": doc.int_filter,
                        "keyword_filter": doc.keyword_filter.split(" ")
                        if isinstance(doc.keyword_filter, str)
                        else doc.keyword_filter,
                    },
                )
                for doc in docs
            ]
        )

    def delete_by_id(self, collection: str, ids: list[str]):
        index = self._get_index(collection)
        index.delete(ids=ids)

    def delete_collection(self, collection: str):
        self.client.delete_index(collection)

    def list_collections(self):
        return [index.name for index in self.client.list_indexes()]

    def close(self):
        pass


def to_document(result: dict) -> Document:
    return Document(
        id=result["id"],
        text=result["metadata"]["text"],
        int_filter=int(result["metadata"]["int_filter"]),
        keyword_filter=" ".join(result["metadata"]["keyword_filter"]),
    )
