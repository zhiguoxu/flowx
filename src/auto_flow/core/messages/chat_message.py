from __future__ import annotations

from enum import Enum
from typing import Literal, Sequence, Any, TYPE_CHECKING, List

from pydantic import BaseModel

from auto_flow.core.utils.utils import add

if TYPE_CHECKING:
    from auto_flow.core.prompts.chat_template import ChatTemplate


class Function(BaseModel):
    name: str
    arguments: str  # json str


class ToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    @classmethod
    def from_name(cls, role_name: Any):
        if isinstance(role_name, Role):
            return role_name

        if isinstance(role_name, str):
            for name, member in cls.__members__.items():
                if name.lower() == role_name.lower():
                    return member

            if role_name.lower() == "ai":
                return Role.ASSISTANT
            if role_name.lower() == "human":
                return Role.USER

        raise ValueError(f"error role name: {role_name}")


class ChatMessage(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: Sequence[ToolCall] | None = None  # for assistant
    tool_call_id: str | None = None  # for tool
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | str | None = None
    extra_data: Any = None

    def __add__(self, other: Any) -> ChatTemplate:
        from auto_flow.core.prompts.chat_template import ChatTemplate
        return ChatTemplate(messages=[self]) + other


class FunctionChunk(BaseModel):
    name: str | None = None
    arguments: str | None = None  # json str chunk


class ToolCallChunk(BaseModel):
    index: int
    id: str | None = None
    function: FunctionChunk | None = None
    type: Literal["function"] | str | None = None

    def __add__(self, other: ToolCallChunk) -> ToolCallChunk:
        assert self.index == other.index
        function = self.function.model_copy(deep=True) if self.function else FunctionChunk()
        if other.function:
            function.name = (function.name or "") + (other.function.name or "")
            function.arguments = (function.arguments or "") + (other.function.arguments or "")
        return ToolCallChunk(index=self.index,
                             id=self.id or other.id,
                             function=function,
                             type=self.type or other.type)

    class Config:
        extra = "forbid"


class ChatMessageChunk(BaseModel):
    role: Role | None = None
    content: str | None = None
    tool_calls: Sequence[ToolCallChunk] | None = None  # for assistant
    tool_call_id: str | None = None  # for tool
    finish_reason: Literal["stop", "length", "tool_calls"] | str | None = None
    extra_data: Any = None

    def __add__(self, other: ChatMessageChunk) -> ChatMessageChunk:
        content: str | None = None
        if self.content is not None or other.content is not None:
            content = (self.content or "") + (other.content or "")
        tool_calls: List[ToolCallChunk] = list(self.tool_calls or [])
        for other_tool_call in other.tool_calls or []:
            if other_tool_call.index >= len(tool_calls):
                assert other_tool_call.index == len(tool_calls)
                tool_calls.append(other_tool_call.model_copy(deep=True))
            else:
                tool_calls[other_tool_call.index] += other_tool_call

        return ChatMessageChunk(role=self.role or other.role,
                                content=content,
                                tool_calls=tool_calls,
                                tool_call_id=self.tool_call_id or other.tool_call_id,
                                finish_reason=self.finish_reason or other.finish_reason,
                                extra_data=add(self.extra_data, other.extra_data))

    def to_message(self) -> ChatMessage:
        message_dump = self.model_dump()
        for tool_call in message_dump.get("tool_calls") or []:
            tool_call.pop("index")
        return ChatMessage(**message_dump)

    class Config:
        extra = "forbid"
