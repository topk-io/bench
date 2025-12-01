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
    def query_by_id(self, collection: str, id: str):
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
