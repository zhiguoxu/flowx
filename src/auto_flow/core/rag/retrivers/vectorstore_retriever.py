from typing import Literal, List, Dict, Any, Tuple

from pydantic import Field

from auto_flow.core.rag.document.document import Document
from auto_flow.core.rag.retrivers.retriever import Retriever
from auto_flow.core.rag.vectorstore.vectorstore import VectorStore


class VectorStoreRetriever(Retriever):
    vectorstore: VectorStore
    search_type: Literal["similarity", "mmr"] = "similarity"

    # search kwargs
    top_k: int = 4
    score_threshold: float | None = None
    extra_search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # mmr search kwargs
    fetch_k: int = 20
    lambda_mult: float = 0.5

    def invoke_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        search_kwargs = dict(self.extra_search_kwargs,
                             top_k=self.top_k,
                             score_threshold=self.score_threshold)
        if self.search_type == "similarity":
            return self.vectorstore.search_with_score(query, **search_kwargs)
        elif self.search_type == "mmr":
            return self.vectorstore.mmr_search_with_score(
                query, **{**search_kwargs,
                          "fetch_k": self.fetch_k,
                          "lambda_mult": self.lambda_mult})
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

    class Config:
        arbitrary_types_allowed = True
