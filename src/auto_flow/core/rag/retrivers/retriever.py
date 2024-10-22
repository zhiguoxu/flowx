from abc import ABC, abstractmethod
from typing import List, Tuple

from auto_flow.core.callbacks.run_stack import current_run
from auto_flow.core.flow.flow import Flow
from auto_flow.core.rag.document.document import Document


class Retriever(Flow[str, List[Document]], ABC):
    def invoke(self, query: str) -> List[Document]:
        doc_and_scores = self.invoke_with_scores(query)
        current_run().extra_data["score_dict"] = {doc.id: score for doc, score in doc_and_scores}
        return [item[0] for item in doc_and_scores]

    @abstractmethod
    def invoke_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        ...


RetrieverLike = Retriever | Flow[str, List[Document]]
