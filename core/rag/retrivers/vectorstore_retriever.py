from typing import Literal, List, Dict, Any

from pydantic import Field

from core.rag.document.document import Document
from core.rag.retrivers.retriever import Retriever
from core.rag.vectorstore.vectorstore import VectorStore


class VectorStoreRetriever(Retriever):
    vectorstore: VectorStore
    search_type: Literal["similarity", "mmr"] = "similarity"

    # search kwargs
    top_k: int = 4
    score_threshold: float | None = None
    extra_search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # mmr search kwargs
    fetch_k: int = 20,
    lambda_mult: float = 0.5

    def invoke(self, query: str) -> List[Document]:

        if self.search_type == "similarity":
            return self.vectorstore.search(query, **self.search_kwargs)
        elif self.search_type == "mmr":
            return self.vectorstore.mmr_search(query, **{**self.search_kwargs,
                                                         "fetch_k": self.fetch_k,
                                                         "lambda_mult": self.lambda_mult})
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

    @property
    def search_kwargs(self) -> Dict[str, Any]:
        return dict(self.extra_search_kwargs, top_k=self.top_k, score_threshold=self.score_threshold)

    class Config:
        arbitrary_types_allowed = True
