from abc import ABC, abstractmethod

class Document:
    id: str
    text: str
    dense_embedding: list[float]
    int_filter: int
    keyword_filter: str

class Provider(ABC):
    @abstractmethod
    def setup(self, collection: str):
        pass

    @abstractmethod
    def freshness_probe(self, collection: str, id: str):
        """Poll after a write to time write-to-visible. Not a benchmark on its own."""
        pass

    def point_get(self, collection: str, id: str):
        """Fetch by id using the client's real key lookup, where it has one.

        Optional: a provider that omits it falls back to `freshness_probe`, and its
        `get` results are then not the same operation as providers that define it.
        """
        pass

    @abstractmethod
    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        int_filter: int | None,
        keyword_filter: str | None,
    ):
        pass

    @abstractmethod
    def upsert(self, collection: str, docs: list[dict]):
        pass

    @abstractmethod
    def delete_by_id(self, collection: str, ids: list[str]):
        pass

    @abstractmethod
    def delete_collection(self, collection: str):
        pass

def write_metrics(path: str):
    pass
