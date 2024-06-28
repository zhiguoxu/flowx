from typing import List, Dict, Any, get_args

from openai import OpenAI
from openai.types import ChatModel
from pydantic import Field

from core.llm.llm import LLM, ChatResult, to_chat_messages, ToolChoiceLiteral
from core.llm.openai.utils import chat_result_from_openai, to_openai_message, tool_to_openai, \
    get_tool_choice_by_pydantic
from core.messages.chat_message import ChatMessage, Role
from core.utils.utils import filter_kwargs_by_method


class OpenAILLM(LLM):
    model: str | ChatModel = "gpt-3.5-turbo"
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = 2
    timeout: float = 20  # seconds
    system_prompt: str | None = None
    stream_include_usage: bool = Field(
        default=False,
        description="If set, the token usage will return at the end of stream."
                    "Refer to ChatCompletionChunk.usage and ChatCompletionStreamOptionsParam for more info"
    )

    def chat(self, messages: List[ChatMessage] | str, **kwargs: Any) -> ChatResult:
        return self._chat_(messages, **kwargs).merge_chunk()

    def stream_chat(self, messages: List[ChatMessage] | str, **kwargs: Any) -> ChatResult:
        return self._chat_(messages, **{**kwargs, "stream": True})

    def _chat_(self, messages: List[ChatMessage] | str, **kwargs: Any) -> ChatResult:
        if isinstance(messages, str):
            messages = to_chat_messages(messages)
        messages = self.try_add_system_message(messages)
        openai_messages = list(map(to_openai_message, messages))
        kwargs = {**self.chat_kwargs, **kwargs}
        assert not (kwargs["stream"] and kwargs["n"] > 1)
        resp = self.client.chat.completions.create(messages=openai_messages, **kwargs)
        return chat_result_from_openai(resp)

    def try_add_system_message(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        if self.system_prompt is None or messages[0].role == "system":
            return messages
        return [ChatMessage(role=Role.SYSTEM, content=self.system_prompt)] + list(messages)

    @property
    def chat_kwargs(self) -> Dict[str, Any]:
        effect_generation_args = {
            "max_new_tokens", "temperature", "streaming", "repetition_penalty", "stop", "n"
        }
        kwargs = self.model_dump(exclude_none=True, include=effect_generation_args)
        kwargs.update(self.model_dump(exclude_none=True))

        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        kwargs["stream"] = kwargs.pop("streaming")

        # Openai doesn't support repetition_penalty,
        # but we can extend it to other models that support it
        # with openai interface compatible server by using 'extra_body'.
        repetition_penalty = kwargs.pop("repetition_penalty")
        if not self.model.startswith("gpt"):
            kwargs["extra_body"] = dict(repetition_penalty=repetition_penalty)

        if kwargs.get("stream_include_usage"):
            kwargs["stream_options"] = dict(include_usage=True)

        kwargs["tools"] = self.openai_tools
        kwargs["tool_choice"] = self.openai_tool_choice

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

    @property
    def openai_tool_choice(self):
        if not self.tool_choice:
            return None

        tool_choice = "required" if self.tool_choice == "any" else self.tool_choice
        if tool_choice in get_args(ToolChoiceLiteral):
            return tool_choice

        for tool in self.tools or []:
            if tool.name == tool_choice:
                return get_tool_choice_by_pydantic(tool.args_schema)

        tool_names = list(map(lambda tool_: tool_.name, self.tools))
        raise ValueError(f"""Error tool choice: "{self.tool_choice}" not in tools: {tool_names}""")
