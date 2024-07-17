from typing import List, Any, Tuple, Dict

from core.llm.llm import LLM
from core.llm.message_parser import StrOutParser
from core.llm.model_configs import get_model_config
from core.messages.chat_message import ChatMessage
from core.prompts.message_template import StrTemplate, PromptTemplate
from core.rag.combine_documents.base_combine import CombineDocumentsFlow
from core.rag.document.document import Document
from core.rag.prompts import format_document, estimate_prompt_token_length


class RefineDocumentsFlow(CombineDocumentsFlow):
    initial_prompt: PromptTemplate[str, ChatMessage] | PromptTemplate[str, List[ChatMessage]]
    """Prompt to use on initial document."""
    refine_prompt: PromptTemplate[str, ChatMessage] | PromptTemplate[str, List[ChatMessage]]
    """Prompt to use when refining."""
    llm: LLM
    document_key: str = "context"
    """The variable name in the initial_llm_flow and refine_llm_flow to put the documents(str format) in."""
    initial_answer_key: str = "initial_answer"
    """The variable name to format the initial response in when refining."""
    document_prompt: StrTemplate = StrTemplate("{text}")
    """Prompt to use to format each document, gets passed to `format_document`."""
    repack_to_max_size: bool = True
    """Repack all documents to max available chunk size."""

    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        initial_llm_flow = self.initial_prompt | self.llm | StrOutParser()
        refine_llm_flow = self.refine_prompt | self.llm | StrOutParser()
        docs, res_docs = self.split_documents(docs, True, **kwargs)

        def get_inputs(cur_docs: List[Document]) -> Dict[str, Any]:
            doc_str = "\n\n".join([format_document(doc, self.document_prompt) for doc in cur_docs])
            return {self.document_key: doc_str, **kwargs}

        initial_answer = initial_llm_flow.invoke(get_inputs(docs))
        refine_steps = [initial_answer]
        while res_docs:
            docs, res_docs = self.split_documents(docs, True, **kwargs)
            inputs = get_inputs(docs)
            inputs[self.initial_answer_key] = initial_answer
            initial_answer = refine_llm_flow.invoke(inputs)
            refine_steps.append(initial_answer)
        return initial_answer, {"intermediate_steps": refine_steps}

    def split_documents(self,
                        docs: List[Document],
                        initial: bool,
                        **kwargs: Any) -> Tuple[List[Document], List[Document]]:
        if not self.repack_to_max_size:
            return docs[0:1], docs[1:]
        length = 0
        for i, doc in enumerate(docs):
            base_inputs: dict = {
                self.document_key: format_document(doc, self.document_prompt)
            }
            inputs = {**base_inputs, **kwargs}
            length += self.estimate_prompt_token_length(initial, **inputs)
            if length > get_model_config(self.llm.model).context_window:
                k = max(1, i)
                return docs[:k], docs[k:]

        return docs, []

    def estimate_prompt_token_length(self, initial: bool, **kwargs: Any):
        prompt = self.initial_prompt if initial else self.refine_prompt
        prompt = prompt.partial_format(**kwargs)
        prompt_length = estimate_prompt_token_length(prompt, self.llm)
        if not initial:
            # Reserve【max_new_tokens】for【initial_answer】prompt argument.
            prompt_length += self.llm.max_new_tokens
        return prompt_length + self.llm.max_new_tokens
