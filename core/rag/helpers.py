from typing import Dict, Any, List

from core.flow.flow import Flow, identity
from core.rag.document.document import Document
from core.rag.retrivers.retriever import Retriever


def create_rag_flow(retriever: Retriever | Flow[Dict[str, Any], List[Document]],
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
        retrieval_docs: Flow[dict, List[Document]] = retriever
    else:
        retrieval_docs = (lambda x: x["input"]) | retriever
    return (
        identity.assign(
            context=retrieval_docs.with_config(run_name="retrieve_documents"),
        ).assign(answer=combine_docs_flow)
    ).with_config(run_name="retrieval_chain")
