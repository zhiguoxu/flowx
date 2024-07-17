from operator import itemgetter
from typing import Dict, Any, List

from core.flow.flow import Flow, identity
from core.llm.llm import LLM
from core.llm.message_parser import StrOutParser
from core.prompts.chat_template import ChatTemplate
from core.prompts.message_template import MessageTemplate
from core.rag.combine_documents.refine import RefineDocumentsFlow
from core.rag.document.document import Document
from core.rag.prompts import qa_prompt, qa_chat_with_history, refine_prompt as default_refine_prompt
from core.rag.retrivers.history_aware_retriever import create_history_aware_retriever
from core.rag.retrivers.retriever import Retriever, RetrieverLike


def create_rag_flow(retriever: Retriever | Flow[Dict, List[Document]],
                    combine_docs_flow: Flow[Dict[str, Any], str]
                    ) -> Flow[Dict[str, Any], Dict[str, Any]]:
    """
    Create a flow that retrieves documents and processes them.
    Args:
        retriever: Object returning a list of documents, either a Retriever
            subclass or a Flow that accepts a dictionary input.
        combine_docs_flow: Flow that takes inputs, including the retrieved
            documents and optionally an empty chat history, and produces a string.
    Returns:
        A Flow producing a dictionary with at least `context` and `answer` keys.
    Please refer to tutorials/rag_with_history.ipynb for the examples.
    """
    if not isinstance(retriever, Retriever):
        retrieval_docs: Flow[Dict, List[Document]] = retriever
    else:
        retrieval_docs = (lambda x: x["input"]) | retriever
    return (
        identity.assign(
            context=retrieval_docs.with_config(run_name="retrieve_documents")
        ).assign(answer=combine_docs_flow)
    ).with_config(run_name="retrieval_chain")


def format_docs(documents: List[Document]):
    return "\n\n".join(doc.text for doc in documents)


def create_qa_flow(retriever: RetrieverLike,
                   llm: LLM,
                   prompt: MessageTemplate | ChatTemplate = qa_prompt) -> Flow[str, str]:
    """
    Create a qa RAG flow,
    the prompt must support input key of context and input.
    """

    return (
            {"context": retriever | format_docs, "input": identity}
            | prompt
            | llm
            | StrOutParser()
    ).with_config(run_name="qa_flow")


def create_qa_flow_with_history(retriever: RetrieverLike,
                                llm: LLM,
                                prompt: ChatTemplate = qa_chat_with_history) -> Flow[Dict[str, Any], str]:
    """
    Create a qa RAG flow with chat history,
    the prompt must support input key of 'context', 'input' and 'chat_history'.
    return a flow with input of a dict which must has keys 'input' and 'chat_history'
    """

    # get retriever with dict input, the input must accept keys 'input' and 'chat_history'
    retriever_ = create_history_aware_retriever(llm, retriever)
    return (
            identity.assign(context=retriever_ | format_docs)
            | prompt
            | llm
            | StrOutParser()
    ).with_config(run_name="qa_flow_with_history")


def create_refine_flow(retriever: RetrieverLike,
                       llm: LLM,
                       initial_prompt: MessageTemplate | ChatTemplate = qa_prompt,
                       refine_prompt: MessageTemplate | ChatTemplate = default_refine_prompt) -> Flow[str, str]:
    refine_doc_flow = RefineDocumentsFlow(initial_prompt=initial_prompt,
                                          refine_prompt=refine_prompt,
                                          llm=llm)

    return (
            {"input": identity, refine_doc_flow.input_key: retriever}
            | refine_doc_flow
            | itemgetter(refine_doc_flow.output_key)
    ).with_config(run_name="refine_flow")


def create_refine_flow_with_history(retriever: RetrieverLike,
                                    llm: LLM,
                                    initial_prompt: MessageTemplate | ChatTemplate = qa_prompt,
                                    refine_prompt: MessageTemplate | ChatTemplate = default_refine_prompt
                                    ) -> Flow[Dict[str, Any], str]:
    retriever_ = create_history_aware_retriever(llm, retriever)
    refine_doc_flow = RefineDocumentsFlow(initial_prompt=initial_prompt,
                                          refine_prompt=refine_prompt,
                                          llm=llm)
    return (
            identity.assign(**{refine_doc_flow.input_key: retriever_})
            | refine_doc_flow
            | itemgetter(refine_doc_flow.output_key)
    ).with_config(run_name="refine_flow_with_history")
