from typing import Dict, Any, TypeVar

from core.flow.flow import Flow, identity
from core.llm.llm import LLMInput
from core.llm.message_parser import StrOutParser
from core.messages.chat_message import ChatMessage
from core.prompts.chat_template import ChatTemplate
from core.prompts.message_template import MessageTemplate
from core.rag.prompts import format_document

Output = TypeVar("Output", covariant=True)

DOCUMENTS_KEY = "context"


def create_stuff_documents_flow(
        llm: Flow[LLMInput, ChatMessage],
        prompt: ChatTemplate,
        *,
        output_parser: Flow[ChatMessage, Output | str] = StrOutParser(),
        document_prompt: MessageTemplate = MessageTemplate.user_message("{text}"),
        document_separator: str = "\n\n") -> Flow[Dict[str, Any], Output | str]:
    """
    Create a chain to pass a list of Documents to a model.
    Args:
        llm: LLM like flow.
        prompt: Template containing the input variable "context" for formatted documents.
        output_parser: Optional parser for the output (defaults to StrOutParser).
        document_prompt: Template for formatting each document, using "text" or metadata keys.
        document_separator: String used to separate formatted document strings.
    Returns:
        An Flow. Input must be a dictionary with a "context" key mapping to List[Document],
         plus any other variables expected in the prompt. Output type depends on the output_parser.

    Example:
        from core.rag.combine_documents.stuff import create_stuff_documents_flow
        from core.prompts.chat_template import ChatTemplate
        from core.llm.openai.openai_llm import OpenAILLM
        from core.rag.document.document import Document

        prompt = ChatTemplate.from_messages(
            [("system", "What are everyone's favorite colors:\\n\\n{context}")]
        )
        llm = OpenAILLM(model="gpt-4o")
        chain = create_stuff_documents_flow(llm, prompt)
        docs = [
            Document(text="Jesse loves red but not yellow"),
            Document(text="Jamal loves green but not as much as he loves orange")
        ]
        chain.invoke({"context": docs})
    """
    if DOCUMENTS_KEY not in prompt.input_vars:
        raise ValueError(
            f"Prompt must accept {DOCUMENTS_KEY} as an input variable."
            f"Received prompt with input variables: {prompt.input_vars}"
        )

    def format_docs(inputs: dict) -> str:
        return document_separator.join(
            format_document(doc, document_prompt) for doc in inputs[DOCUMENTS_KEY]
        )

    return (
            identity.assign(**{DOCUMENTS_KEY: format_docs}).with_config(run_name="format_inputs")
            | prompt
            | llm
            | output_parser
    ).with_config(run_name="stuff_documents_chain")  # type: ignore[return-value]
