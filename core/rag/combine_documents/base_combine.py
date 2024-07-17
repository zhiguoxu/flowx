from abc import abstractmethod
from typing import Dict, Any, List, Tuple

from core.flow.flow import Flow
from core.rag.document.document import Document


class CombineDocumentsFlow(Flow[Dict[str, Any], Dict[str, Any]]):
    input_key: str = "input_documents"
    output_key: str = "output_text"

    def invoke(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        docs = inp[self.input_key]
        # Other keys are assumed to be needed for LLM generation.
        other_keys = {k: v for k, v in inp.items() if k != self.input_key}
        output, extra_return_dict = self.combine_docs(docs, **other_keys)
        extra_return_dict[self.output_key] = output
        return extra_return_dict

    @abstractmethod
    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        """Combine documents into a single string.
        Args:
            docs: List[Document], the documents to combine
            **kwargs: Other parameters to use in combining documents,
                 often other inputs to the prompt.

        Returns:
            The first element returned is the single string output.
             The second element returned is a dictionary of other keys to return.
        """
