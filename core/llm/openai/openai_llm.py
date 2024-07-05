from typing import List, Dict, Any, cast

from openai import OpenAI
from openai.types import ChatModel
from pydantic import Field

from core.llm.llm import LLM, ChatResult, to_chat_messages, LLMInput
from core.llm.openai.utils import chat_result_from_openai, to_openai_message, tool_to_openai, tools_to_openai, \
    tool_choice_to_openai
from core.messages.utils import remove_extra_info
from core.tool import Tool
from core.utils.utils import filter_kwargs_by_method


class OpenAILLM(LLM):
    model: str | ChatModel = "gpt-3.5-turbo"
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 2
    timeout: float = 20  # seconds
    stream_include_usage: bool = Field(
        default=True,
        description="If set, the token usage will return at the end of stream."
                    "Refer to ChatCompletionChunk.usage and ChatCompletionStreamOptionsParam for more info"
    )

    def chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        # Not specify stream=False.
        # Maybe, we want to stream in the background and merge the streamed chunks at the end.
        return self._chat_(messages, **kwargs).merge_chunk()

    def stream_chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        return self._chat_(messages, **{**kwargs, "stream": True})

    def _chat_(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        messages = to_chat_messages(messages)
        messages = remove_extra_info(messages)
        messages = self.try_add_system_message(messages)
        openai_messages = list(map(to_openai_message, messages))
        kwargs = self.get_chat_kwargs(**kwargs)
        assert not (kwargs.get("stream") and kwargs["n"] > 1)
        resp = self.client.chat.completions.create(messages=openai_messages, **kwargs)
        return chat_result_from_openai(resp)

    def get_chat_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs = {**self.model_dump(exclude_none=True, by_alias=True), **kwargs}

        kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        if "stream" not in kwargs:
            kwargs["stream"] = kwargs.pop("streaming")

        # Openai doesn't support repetition_penalty,
        # but we can extend it to other models that support it
        # with openai interface compatible server by using 'extra_body'.
        if repetition_penalty := kwargs.pop("repetition_penalty", None):
            kwargs["presence_penalty"] = repetition_penalty
            kwargs["extra_body"] = dict(repetition_penalty=repetition_penalty)  # todo remove

        if kwargs.get("stream_include_usage"):
            kwargs["stream_options"] = dict(include_usage=True)

        if tools := kwargs.pop("tools", None):
            kwargs["tools"] = tools_to_openai(tools)

            # If return direct, parallel tool calls is now allowed.
            if any(tool.return_direct for tool in (cast(List[Tool], tools))):
                kwargs["parallel_tool_calls"] = False
        if (tool_choice := kwargs.pop("tool_choice", None)) is not None:
            kwargs["tool_choice"] = tool_choice_to_openai(tool_choice, tools)

        if kwargs.pop("json_mode", False):
            kwargs["response_format"] = {"type": "json_object"}

        return filter_kwargs_by_method(OpenAI().chat.completions.create, kwargs, exclude_none=True)

    @property
    def client(self):
        return OpenAI(api_key=self.api_key,
                      base_url=self.base_url,
                      max_retries=self.max_retries,
                      timeout=self.timeout)

    @property
    def openai_tools(self) -> List[Dict[str, Any]] | None:
        return list(map(tool_to_openai, self.tools)) if self.tools else None
