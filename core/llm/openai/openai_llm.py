from typing import List, Sequence, Dict, Any, Literal

from openai import OpenAI
from openai.types import ChatModel
from pydantic import Field

from core.llm.generation_args import GenerationArgs
from core.llm.llm import LLM, ChatResult, to_chat_messages
from core.llm.openai.utils import chat_result_from_openai, to_openai_message
from core.messages.chat_message import ChatMessage, Role
from core.utils.utils import filter_kwargs_by_method, filter_kwargs_by_pydantic

ToolChoiceType = str | Literal["none", "auto", "required"] | Dict[str, Any]


class OpenAILLM(LLM):
    model: str | ChatModel = "gpt-3.5-turbo"
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 2
    timeout: float = 20  # seconds
    system_prompt: str | None = Field(default=None, description="System prompt for LLM calls.")
    stream_include_usage: bool = Field(default=False, description="Refer to ChatCompletionChunk.usage and"
                                                                  " ChatCompletionStreamOptionsParam")
    tool_choice: ToolChoiceType | None = None
    # "none", "auto", "required", {"type": "function", "function": {"name": "my_function"}}

    tools: List[Dict[str, Any]] | None = None
    # [{"type": "function", "function": {"description": "xxx", "name": "yyy", "parameters": { json format }}}]
    # refer to https://platform.openai.com/docs/api-reference/chat/create for json format

    def __init__(self,
                 model: str | ChatModel = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_new_tokens: int = 512,
                 stream: bool = False,
                 repetition_penalty: float = 1,
                 stop: str | List[str] | None = None,
                 system_prompt: str | None = None,
                 api_key: str | None = None,
                 base_url: str | None = None,
                 max_retries: int = 2,
                 timeout: float = 20,
                 stream_include_usage: bool = False):

        generation_kwargs = filter_kwargs_by_pydantic(GenerationArgs, locals(), exclude_none=True)
        generation_args = GenerationArgs(**generation_kwargs)
        kwargs = filter_kwargs_by_pydantic(type(self), locals(), exclude_none=True)
        super().__init__(**kwargs)

    def chat(self, messages: List[ChatMessage] | str, **kwargs: Any) -> ChatResult:
        return self._chat_(messages, **kwargs).merge_chunk()

    def stream_chat(self, messages: List[ChatMessage] | str, **kwargs: Any) -> ChatResult:
        return self._chat_(messages, **{**kwargs, "stream": True})

    def _chat_(self, messages: List[ChatMessage] | str, **kwargs: Any) -> ChatResult:
        if isinstance(messages, str):
            messages = to_chat_messages(messages)
        messages = self.try_add_system_message(messages)
        openai_messages = list(map(to_openai_message, messages))
        resp = self.client.chat.completions.create(messages=openai_messages, **{**self.chat_kwargs, **kwargs})
        return chat_result_from_openai(resp)

    def try_add_system_message(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        if self.system_prompt is None or messages[0].role == "system":
            return messages
        return [ChatMessage(role=Role.SYSTEM, content=self.system_prompt)] + list(messages)

    @property
    def chat_kwargs(self) -> Dict[str, Any]:
        kwargs = self.generation_args.model_dump()
        kwargs.update(self.model_dump())

        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")

        repetition_penalty = kwargs.pop("repetition_penalty")
        if not self.model.startswith("gpt"):
            kwargs["extra_body"] = dict(repetition_penalty=repetition_penalty)

        if kwargs.get("stream_include_usage"):
            kwargs["stream_options"] = dict(include_usage=True)

        return filter_kwargs_by_method(OpenAI().chat.completions.create, kwargs, exclude_none=True)

    @property
    def client(self):
        return OpenAI(api_key=self.api_key,
                      base_url=self.base_url,
                      max_retries=self.max_retries,
                      timeout=self.timeout)
