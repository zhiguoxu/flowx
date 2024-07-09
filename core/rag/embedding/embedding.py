from abc import abstractmethod, ABC
from typing import List


class Embedding(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...

    @abstractmethod
    def embed_documents(self, text_list: List[str]) -> List[List[float]]:
        ...
