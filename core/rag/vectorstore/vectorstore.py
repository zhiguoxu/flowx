from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TypeVar, Any, Tuple, TYPE_CHECKING, Literal

from core.rag.document.document import Document

from core.utils.utils import filter_kwargs_by_method, filter_kwargs_by_pydantic

VectorStoreT = TypeVar("VectorStoreT", bound="VectorStore")

if TYPE_CHECKING:
    from core.rag.retrivers.vectorstore_retriever import VectorStoreRetriever


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self,
                      documents: List[Document],
                      ids: List[str] | None = None,
                      batch_size: int = 1_000) -> List[str]:
        ...

    def search(self,
               query: str | None = None,
               embedding: List[float] | None = None,
               top_k: int = 4,
               score_threshold: float | None = None,
               **kwargs: Any) -> List[Document]:
        kwargs = filter_kwargs_by_method(self.mmr_search_with_score, {**locals(), **kwargs}, exclude={"kwargs"})
        docs_and_scores = self.search_with_score(**kwargs)
        return [doc for doc, _ in docs_and_scores]

    @abstractmethod
    def search_with_score(self,
                          query: str | None = None,
                          embedding: List[float] | None = None,
                          top_k: int = 4,
                          score_threshold: float | None = None,
                          **kwargs: Any
                          ) -> List[Tuple[Document, float]]:  # [document, similarity score]
        ...

    def mmr_search(self,
                   query: str | None = None,
                   embedding: List[float] | None = None,
                   top_k: int = 5,
                   fetch_k: int = 20,
                   lambda_mult: float = 0.5,
                   **kwargs: Any) -> List[Document]:
        """
        Return documents using maximal marginal relevance (MMR).
        MMR balances similarity to the query and diversity among results.
        Args:
            query: Query str.
            embedding: Query embedding.
            top_k: Number of documents to return (default: 5).
            fetch_k: Number of documents to fetch for MMR (default: 20).
            lambda_mult: Controls diversity (0 = max diversity, 1 = min diversity, default: 0.5).
        Returns: List of selected documents and scores.
        """
        kwargs = filter_kwargs_by_method(self.mmr_search_with_score, {**locals(), **kwargs}, exclude={"kwargs"})
        docs_and_scores = self.mmr_search_with_score(**kwargs)
        return [doc for doc, _ in docs_and_scores]

    @abstractmethod
    def mmr_search_with_score(self,
                              query: str | None = None,
                              embedding: List[float] | None = None,
                              top_k: int = 5,
                              fetch_k: int = 20,
                              lambda_mult: float = 0.5,
                              **kwargs: Any
                              ) -> List[Tuple[Document, float]]:  # [document, similarity score]
        ...

    @abstractmethod
    def delete(self, ids: List[str] | None = None, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def delete_all(self):
        ...

    def as_retriever(self,
                     search_type: Literal["similarity", "mmr"] = "similarity",
                     top_k: int = 4,
                     score_threshold: float | None = None,
                     # mmr search kwargs
                     fetch_k: int = 20,
                     lambda_mult: float = 0.5,
                     **extra_search_kwargs: Any) -> VectorStoreRetriever:
        from core.rag.retrivers.vectorstore_retriever import VectorStoreRetriever
        init_kwargs = filter_kwargs_by_pydantic(VectorStoreRetriever, locals())
        return VectorStoreRetriever(vectorstore=self, **init_kwargs)
