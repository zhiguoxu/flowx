from __future__ import annotations

from abc import abstractmethod
from typing import Union, Sequence, Tuple, List, Iterator, Literal, Any

from pydantic import Field, BaseModel

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

    def invoke(self, inp: LLMInput) -> ChatMessage:
        messages = to_chat_messages(inp)
        return self.chat(messages).messages[0]

    def stream(self, inp: LLMInput) -> Iterator[ChatMessageChunk]:  # type: ignore[override]
        messages = to_chat_messages(inp)
        result = self.stream_chat(messages)
        assert result.message_stream
        for message_chunk, _ in result.message_stream:
            yield message_chunk

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
    message_stream: Iterator[Tuple[ChatMessageChunk, TokenUsage | None]] | None = Field(
        default=None, description="only return stream of index 0"
    )
    usage: TokenUsage | None = None

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
