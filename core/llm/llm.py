from __future__ import annotations

from abc import abstractmethod
from typing import Union, Sequence, Tuple, List, Iterator, Literal, Any

from pydantic import Field, BaseModel

from core.callbacks.run import current_run
from core.callbacks.trace import trace
from core.flow.flow import Flow
from core.llm.generation_args import GenerationArgs
from core.messages.chat_message import ChatMessage, ChatMessageChunk, chunk_to_message
from core.messages.utils import to_chat_message, MessageLike
from core.tool import Tool

LLMInput = Union[str, Sequence[MessageLike]]
ToolChoiceLiteral = Literal["none", "auto", "required", "any"]
ToolChoiceType = str | ToolChoiceLiteral


class LLM(Flow[LLMInput, ChatMessage]):
    generation_args: GenerationArgs = Field(default_factory=GenerationArgs)
    tools: List[Tool] | None = None
    tool_choice: ToolChoiceType | None = None

    @trace
    def invoke(self, inp: LLMInput, **kwargs: Any) -> ChatMessage:
        messages = to_chat_messages(inp)
        chat_result = self.chat(messages, **kwargs)
        current_run().update_extra_data(token_usage=chat_result.usage)
        return chat_result.messages[0]

    @trace
    def stream(self, inp: LLMInput, **kwargs: Any) -> Iterator[ChatMessageChunk]:  # type: ignore[override]
        messages = to_chat_messages(inp)
        result = self.stream_chat(messages, **kwargs)
        assert result.message_stream
        token_usage = None
        for message_chunk, usage in result.message_stream:
            yield message_chunk
            token_usage = usage
        current_run().update_extra_data(token_usage=token_usage)

    @abstractmethod
    def chat(self, messages: List[ChatMessage] | str, **kwargs: Any) -> ChatResult:
        ...

    @abstractmethod
    def stream_chat(self, messages: List[ChatMessage] | str, **kwargs: Any) -> ChatResult:
        ...

    def set_tools(self, tools: List[Tool] | None = None, tool_choice: ToolChoiceType | None = None):
        self.tools = tools
        if not tools:
            self.tool_choice = None
        else:
            self.tool_choice = tool_choice


def to_chat_messages(inp: LLMInput) -> List[ChatMessage]:
    if isinstance(inp, str):
        return [to_chat_message(inp)]

    return list(map(to_chat_message, inp))


class TokenUsage(BaseModel):
    completion_tokens: int
    """Number of tokens in the generated completion."""

    prompt_tokens: int
    """Number of tokens in the prompt."""

    total_tokens: int
    """Total number of tokens used in the request (prompt + completion)."""


class ChatResult(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    usage: TokenUsage | None = None

    message_stream: Iterator[Tuple[ChatMessageChunk, TokenUsage | None]] | None = Field(
        default=None, description="only return stream of index 0"
    )

    def merge_chunk(self) -> ChatResult:
        if not self.message_stream:
            return self

        assert len(self.messages) == 0
        message: ChatMessageChunk = ChatMessageChunk()
        for message_chunk, usage_chunk in self.message_stream:
            message += message_chunk
            if self.usage or usage_chunk:
                if self.usage and usage_chunk:
                    self.usage.completion_tokens += usage_chunk.completion_tokens
                    self.usage.prompt_tokens += usage_chunk.prompt_tokens
                    self.usage.total_tokens += usage_chunk.total_tokens
                else:
                    self.usage = self.usage or usage_chunk
        self.messages.append(chunk_to_message(message))
        self.message_stream = None
        return self

    class Config:
        arbitrary_types_allowed = True
