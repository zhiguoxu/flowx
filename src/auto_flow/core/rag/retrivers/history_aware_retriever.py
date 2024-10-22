from typing import Any, List, Dict

from auto_flow.core.flow.branch import BranchFlow
from auto_flow.core.flow.flow import Flow
from auto_flow.core.llm.llm import LLMInput
from auto_flow.core.llm.message_parser import StrOutParser
from auto_flow.core.messages.chat_message import ChatMessage
from auto_flow.core.prompts.chat_template import ChatTemplate, MessagesPlaceholder
from auto_flow.core.rag.document.document import Document
from auto_flow.core.rag.retrivers.retriever import RetrieverLike

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

default_contextualize_q_prompt = ChatTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def create_history_aware_retriever(llm: Flow[LLMInput, ChatMessage],
                                   retriever: RetrieverLike,
                                   prompt: ChatTemplate = default_contextualize_q_prompt
                                   ) -> Flow[Dict[str, Any], List[Document]]:
    """
    Create a chain that returns documents based on conversation history.
    If `chat_history` is absent, the `input` is directly sent to the retriever.
    If `chat_history` is present, the prompt and LLM generate a search query,
    which is then sent to the retriever.

    Args:
        llm: Language model for generating search queries from chat history.
        retriever: Retriever that returns a list of Documents from a string input.
        prompt: Prompt template for generating search queries.

    Returns:
        A Flow that takes `input`, optionally `chat_history`, and returns a list of Documents.
    """

    if "input" not in prompt.input_vars:
        raise ValueError(
            f"Expected `input` to be a prompt variable, but got {prompt.input_vars}"
        )

    return BranchFlow(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["input"]) | retriever,
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever.
        prompt | llm | StrOutParser() | retriever,
    ).with_config(run_name="chat_retriever_flow")
