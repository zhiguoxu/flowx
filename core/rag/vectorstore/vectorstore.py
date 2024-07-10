from abc import ABC, abstractmethod
from typing import List, TypeVar, Any, Tuple

from core.rag.document.document import Document

VectorStoreT = TypeVar("VectorStoreT", bound="VectorStore")


class VectorStore(ABC):
    @abstractmethod
    def add_texts(self,
                  texts: List[str],
                  metadatas: List[dict] | None = None,
                  ids: List[str] | None = None) -> List[str]:
        ...

    def add_documents(self, documents: List[Document], ids: List[str] | None = None) -> List[str]:
        if not ids:
            ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, ids)

    def search(self,
               query: str | None = None,
               embedding: List[float] | None = None,
               k: int = 4,
               **kwargs: Any) -> List[Document]:
        ...

    def search_with_score(self,
                          query: str | None = None,
                          embedding: List[float] | None = None,
                          k: int = 4,
                          **kwargs: Any
                          ) -> List[Tuple[Document, float]]:  # [document, similarity score]
        ...

    def mmr_search(self,
                   query: str | None = None,
                   embedding: List[float] | None = None,
                   k: int = 5,
                   fetch_k: int = 20,
                   lambda_mult: float = 0.5,
                   **kwargs: Any) -> List[Document]:
        """
        Return documents using maximal marginal relevance (MMR).
        MMR balances similarity to the query and diversity among results.
        Args:
            query: Query str.
            embedding: Query embedding.
            k: Number of documents to return (default: 5).
            fetch_k: Number of documents to fetch for MMR (default: 20).
            lambda_mult: Controls diversity (0 = max diversity, 1 = min diversity, default: 0.5).
        Returns: List of selected documents and scores.
        """

    def mmr_search_with_score(self,
                              query: str | None = None,
                              embedding: List[float] | None = None,
                              k: int = 5,
                              fetch_k: int = 20,
                              lambda_mult: float = 0.5,
                              **kwargs: Any
                              ) -> List[Tuple[Document, float]]:  # [document, similarity score]
        ...
