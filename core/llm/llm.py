from __future__ import annotations

from abc import abstractmethod
from typing import Union, Sequence, Tuple, List, Iterator, Literal, Any, Callable, TYPE_CHECKING

from pydantic import Field, BaseModel, field_validator

from core.callbacks.chat_history import BaseChatMessageHistory
from core.callbacks.run_stack import current_run
from core.callbacks.trace import trace, ENABLE_TRACE
from core.flow.flow import Flow
from core.flow.utils import ConfigurableField
from core.llm.types import TokenUsage
from core.messages.chat_message import ChatMessage, ChatMessageChunk
from core.messages.utils import to_chat_message, MessageLike
from core.prompts.message_list_template import MessageListTemplate, MessagesPlaceholder
from core.tool import Tool, to_tool, ToolLike
from core.utils.utils import filter_kwargs_by_init_or_pydantic, add

if TYPE_CHECKING:
    from core.llm.llm_with_history import LLMWithHistory

LLMInput = Union[MessageLike, Sequence[MessageLike]]
ToolChoiceLiteral = Literal["none", "auto", "required", "any"]
ToolChoice = str | ToolChoiceLiteral | bool


class LLM(Flow[LLMInput, ChatMessage]):
    max_new_tokens: int = Field(
        default=512,
        description="Number of tokens the model can output when generating a response.",
    )

    temperature: float = Field(
        default=0.1,
        description="The temperature to use during generation.",
        ge=0.0,
    )

    streaming: bool = Field(default=False, alias="stream", description="Streaming output.")

    repetition_penalty: float | None = 1

    stop: str | List[str] | None = None

    n: int = Field(default=1, description="How many chat completion choices to generate for each input message.")

    tools: List[Tool] | None = None
    tool_choice: ToolChoice | None = None

    @field_validator('tools')
    @classmethod
    def validate_tools(cls, tools: List[Tool | Callable]) -> List[Tool]:
        return [to_tool(tool) for tool in tools]

    @trace
    def invoke(self, inp: LLMInput, **kwargs: Any) -> ChatMessage:
        chat_result = self.chat(inp, **kwargs)
        if ENABLE_TRACE:
            current_run().token_usage = chat_result.usage
        return chat_result.messages[0]

    @trace
    def stream(self, inp: LLMInput, **kwargs: Any) -> Iterator[ChatMessageChunk]:  # type: ignore[override]
        result = self.stream_chat(inp, **kwargs)
        assert result.message_stream
        token_usage = None
        for message_chunk, usage in result.message_stream:
            yield message_chunk
            token_usage = usage
        if ENABLE_TRACE:
            current_run().token_usage = token_usage

    @abstractmethod
    def chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        ...

    @abstractmethod
    def stream_chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        ...

    def bin_tools(self, tools: List[ToolLike] | None = None, tool_choice: ToolChoice | None = None):
        return self.bind(tools=tools, tool_choice=tool_choice)

    def with_history(self,
                     get_session_history: Callable[..., BaseChatMessageHistory],
                     history_factory_config: Sequence[ConfigurableField] | None = None
                     ) -> LLMWithHistory[ChatMessage]:
        from core.llm.llm_with_history import LLMWithHistory
        prompt = MessageListTemplate.from_messages([
            MessagesPlaceholder(var_name="input")
        ])

        bound = prompt | self
        kwargs = filter_kwargs_by_init_or_pydantic(LLMWithHistory, locals(), exclude_none=True)
        return LLMWithHistory(**kwargs)

    class Config:
        extra = "forbid"


def to_chat_messages(inp: LLMInput) -> List[ChatMessage]:
    try:
        return [to_chat_message(inp)]  # type: ignore[arg-type]
    except ValueError as e:
        return list(map(to_chat_message, inp))


class ChatResult(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    """
    When used as LLM's output, len(messages) = LLM.n,
    and when used as Agent's output, messages messages[-1] = the final answer,
    messages[:-1] = intermedia steps, if it has.
    """

    usage: TokenUsage | None = None

    message_stream: Iterator[Tuple[ChatMessageChunk, TokenUsage | None]] | None = Field(
        default=None, description="only return stream of index 0"
    )

    message_stream_for_agent: Iterator[ChatMessageChunk | ChatMessage] | None = None
    """
    ChatMessageChunk are final answer or thoughts, ChatMessage are tool calls or observations
    ChatMessageChunk always go before ChatMessage
    """

    def merge_chunk(self) -> ChatResult:
        if self.message_stream:
            assert len(self.messages) == 0
            message_cache: ChatMessageChunk | None = None
            for message_chunk, usage_chunk in self.message_stream:
                message_cache = add(message_cache, message_chunk)
                if self.usage or usage_chunk:
                    if self.usage and usage_chunk:
                        self.usage += usage_chunk
                    else:
                        self.usage = self.usage or usage_chunk
            if message_cache:
                self.messages.append(message_cache.to_message())
            self.message_stream = None

        if self.message_stream_for_agent:
            assert len(self.messages) == 0
            message_cache = None
            for message_or_chunk in self.message_stream_for_agent:
                if isinstance(message_or_chunk, ChatMessageChunk):
                    message_cache = add(message_cache, message_or_chunk)
                else:
                    if message_cache:
                        self.messages.append(message_cache.to_message())
                        message_cache = None
                    self.messages.append(message_or_chunk)
            if message_cache:
                self.messages.append(message_cache.to_message())
            self.message_stream_for_agent = None

        return self

    class Config:
        arbitrary_types_allowed = True
