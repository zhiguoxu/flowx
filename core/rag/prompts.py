from typing import Dict

from core.prompts.chat_template import ChatTemplate, MessagesPlaceholder
from core.prompts.message_template import MessageTemplate
from core.rag.document.document import Document

qa_prompt = MessageTemplate.user_message(
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer the question."
    "If you don't know the answer, just say that you don't know."
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \nContext: {context} \nAnswer:")

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.
{context}"""
retriever_qa_chat = ChatTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ]
)


def _get_document_info(doc: Document, prompt: MessageTemplate) -> Dict:
    base_info = {"text": doc.text, **doc.metadata}
    missing_metadata = set(prompt.input_vars).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [
            iv for iv in prompt.input_vars if iv != "text"
        ]
        raise ValueError(
            f"Document prompt requires documents to have metadata variables: "
            f"{required_metadata}. Received document with missing metadata: "
            f"{list(missing_metadata)}."
        )
    return {k: base_info[k] for k in prompt.input_vars}


def format_document(doc: Document, prompt: MessageTemplate) -> str:
    """
    Format a document into a string based on a prompt template.
    Extracts information from the document's `text` and `metadata`,
    then uses these to generate a formatted string via the provided `prompt`.
    Args:
        doc (Document): Contains `text` and `metadata` to be formatted.
        prompt (ChatTemplate): Template used to format the document's content.
    Returns:
        str: Formatted document string.
    """
    content = prompt.format(**_get_document_info(doc, prompt)).content
    assert content
    return content
