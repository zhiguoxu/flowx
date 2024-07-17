from typing import Dict

from core.llm.llm import LLM
from core.messages.chat_message import ChatMessage
from core.prompts.chat_template import ChatTemplate, MessagesPlaceholder
from core.prompts.message_template import MessageTemplate, StrTemplate, PromptTemplate
from core.rag.document.document import Document

qa_prompt = MessageTemplate.user_message(
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer the question."
    "If you don't know the answer, just say that you don't know."
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {input} \nContext: {context} \nAnswer:")

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.
{context}"""
qa_chat_with_history = ChatTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ]
)

refine_prompt = MessageTemplate.user_message(
    "You are an assistant for question-answering tasks."
    "The original query is as follows: {input}\n"
    "We have provided an initial answer: {initial_answer}\n"
    "We have the opportunity to refine the initial answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context}\n"
    "------------\n"
    "Given the new context, refine the initial answer to better answer the query. "
    "If the context isn't useful, return the initial answer.\n"
    "Refined Answer: "
)

refine_system_prompt = """You are an assistant for question-answering tasks.
We have provided an initial answer: {initial_answer}
We have the opportunity to refine the initial answer \
(only if needed) with some more context below.
------------
{context}
------------
Given the new context, refine the initial answer to better answer the query. \
If the context isn't useful, return the initial answer."""

refine_chat_with_history = ChatTemplate.from_messages(
    [
        ("system", refine_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ]
)


def format_document(doc: Document, prompt: StrTemplate) -> str:
    """
    Format a document into a string based on a prompt template.
    Extracts information from the document's `text` and `metadata`,
    then uses these to generate a formatted string via the provided `prompt`.
    Args:
        doc (Document): Contains `text` and `metadata` to be formatted.
        prompt (StrTemplate): Template used to format the document's content.
    Returns:
        str: Formatted document string.
    """

    def _get_document_info() -> Dict:
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

    return prompt.format(**_get_document_info())


def estimate_prompt_token_length(prompt: ChatMessage | PromptTemplate, llm: LLM) -> int:
    if isinstance(prompt, ChatMessage):
        token_length = llm.token_length(prompt.model_dump_json())
    elif isinstance(prompt, StrTemplate):
        token_length = llm.token_length(str(prompt.partial_vars.values()))
        token_length += llm.token_length(prompt.template)
    elif isinstance(prompt, MessageTemplate):
        token_length = estimate_prompt_token_length(prompt.template, llm)
    elif isinstance(prompt, ChatTemplate):
        token_length = 0
        for message_template in prompt.messages:
            token_length += estimate_prompt_token_length(message_template, llm)
    else:
        raise TypeError
    return token_length
