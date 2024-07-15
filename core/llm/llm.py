from __future__ import annotations

from abc import abstractmethod
from operator import itemgetter
from typing import Union, Sequence, Tuple, List, Iterator, Literal, Any, Callable, TYPE_CHECKING, Dict

from pydantic import Field, BaseModel, field_validator
from pydantic.main import Model

from core.callbacks.chat_history import BaseChatMessageHistory
from core.callbacks.run_stack import current_run
from core.callbacks.trace import trace, ENABLE_TRACE
from core.flow.flow import Flow, identity, ParallelFlow
from core.flow.utils import ConfigurableField
from core.llm.json_format_prompt import get_json_format_prompt_by_schema
from core.llm.message_parser import MessagePydanticOutParser
from core.llm.types import TokenUsage
from core.logging import get_logger
from core.messages.chat_message import ChatMessage, ChatMessageChunk, Role
from core.messages.utils import to_chat_message, MessageLike
from core.prompts.chat_template import ChatTemplate, MessagesPlaceholder
from core.tool import Tool, to_tool, ToolLike
from core.utils.utils import add, filter_kwargs_by_pydantic

if TYPE_CHECKING:
    from core.llm.llm_with_history import LLMWithHistory

LLMInput = Union[MessageLike, Sequence[MessageLike]]
ToolChoiceLiteral = Literal["none", "auto", "required", "any"]
ToolChoice = str | ToolChoiceLiteral | bool

logger = get_logger(__name__)


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

    system_prompt: str | None = None

    tools: List[Tool] | None = None
    tool_choice: ToolChoice | None = None

    parallel_tool_calls: bool | None = None

    json_mode: bool = False

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

    def try_add_system_message(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        system_prompt = self.system_prompt
        if system_prompt is None:
            return messages

        if messages[0].role == "system":
            assert messages[0].content is not None
            logger.warning(f"System prompt merge: {messages[0].content}, {system_prompt}")
            if messages[0].content.find("system_prompt") == 0:
                messages[0].content += "." + system_prompt
            return messages

        return [ChatMessage(role=Role.SYSTEM, content=system_prompt)] + list(messages)

    def with_history(self,
                     get_session_history: Callable[..., BaseChatMessageHistory],
                     history_factory_config: Sequence[ConfigurableField] | None = None
                     ) -> LLMWithHistory[ChatMessage]:
        from core.llm.llm_with_history import LLMWithHistory
        prompt = ChatTemplate.from_messages([
            MessagesPlaceholder("input")
        ])

        bound = prompt.pipe(self, main=True)
        kwargs = filter_kwargs_by_pydantic(LLMWithHistory, locals(), exclude_none=True)
        return LLMWithHistory(**kwargs)

    def with_structured_output(self,
                               schema: ToolLike, *,
                               method: Literal["function_calling", "json_mode"] = "function_calling",
                               return_type: Literal["pydantic", "dict"] = "pydantic",
                               include_raw: bool = False
                               ) -> Flow[LLMInput, Model | Dict[str, Any]]:
        """Model wrapper for returning outputs in a specified schema.
        Args:
            schema: The desired output schema, ToolLike that will be converted to Pydantic class.
            method: The method for output formatting, either "function_calling" or "json_mode".
                - "function_calling": Converts the schema to model's function and uses the function-calling API.
                - "json_mode": Uses model's JSON mode and requires formatting instructions in the model call.
            return_type: Specify the output type of "pydantic" of "dict" for the input schema.
            include_raw:
                - If False, only returns the parsed output, raising any parsing errors directly.
                - If True, The final output is always a dict with keys "raw", "parsed", and "parsing_error".
                if no parsing error occurs, returns both the raw and parsed model responses,
                or else error be caught and returned under the key 'parsing_error'.
        """

        tool = to_tool(schema)
        if method == "function_calling":
            llm = self.bind_tools([tool], tool_choice=tool.name).bind(parallel_tool_calls=False)
        elif method == "json_mode":
            system_prompt = get_json_format_prompt_by_schema(tool.args_schema)
            llm = (self.configurable_fields(system_prompt="system_prompt")
                   .with_configurable(system_prompt=system_prompt)
                   .bind(json_mode=True))
        else:
            raise ValueError(f"Unrecognized method: {method}. Expected one of 'function_calling' or 'json_mode'")

        message_parser = MessagePydanticOutParser(schemas=[tool.args_schema],
                                                  return_dict=return_type == "dict",
                                                  return_first=True)
        if include_raw:  # return dict with keys "raw", "parsed", and "parsing_error"
            base_parser: Flow = identity.assign(parsed=itemgetter("raw") | message_parser, parsing_error=None)
            none_parser: Flow = identity.assign(parsed=None)
            parser_with_fallback = base_parser.with_fallbacks([none_parser], exception_key="parsing_error")
            return ParallelFlow(raw=llm) | parser_with_fallback  # type: ignore[return-value]
        else:
            return llm | message_parser

    @property
    def tokenizer(self) -> Callable[[str], List[int]]:
        """Used to count the number of tokens in documents to constrain them to be under a certain limit."""
        raise NotImplemented

    def token_length(self, text: str):
        return len(self.tokenizer(text))

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
    When used as LLM's output, len(messages) is LLM.n,
    and when used as Agent's output, messages messages[-1] is the final answer,
    messages[:-1] is intermediate steps, if it has.
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
                self.usage = add(self.usage, usage_chunk)
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
