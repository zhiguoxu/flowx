from __future__ import annotations

from typing import Dict, List, Tuple, Sequence, Any, Union

from core.flow.flow import Flow
from core.messages.chat_message import ChatMessage, Role
from core.messages.utils import MessageLike, to_chat_message
from core.prompts.message_template import MessageTemplate, validate_template_vars


class MessageListTemplate(Flow[Union[str, Dict[str, Any]], List[ChatMessage]]):
    messages: List[MessageTemplateLike]

    @classmethod
    def from_messages(cls, messages: Sequence[str | Tuple[str, str] | MessageTemplateLike]) -> MessageListTemplate:
        messages_: List[MessageTemplateLike] = []
        for msg in messages:
            if isinstance(msg, ChatMessage | MessageTemplate | MessageListTemplate):
                messages_.append(msg)
            else:
                messages_.append(MessageTemplate.from_arg(msg))
        return cls(messages=messages_)

    def invoke(self, inp: str | Dict[str, Any]) -> List[ChatMessage]:
        inp = validate_template_vars(inp, self.input_vars)
        return self.format(**inp)

    def partial_format(self, **kwargs: Any):
        for msg in self.messages:
            if isinstance(msg, (MessageTemplate, MessageListTemplate)):
                msg.partial_format(**kwargs)

    def format(self, arg: str | None = None, **kwargs: Any) -> List[ChatMessage]:
        if arg is not None:
            assert len(kwargs) == 0
            kwargs = validate_template_vars(arg, self.input_vars)

        ret: List[ChatMessage] = []
        for msg in self.messages:
            if isinstance(msg, ChatMessage):
                ret.append(msg)
            elif isinstance(msg, MessageTemplate):
                ret.append(msg.format(**kwargs))
            elif isinstance(msg, MessageListTemplate):
                ret.extend(msg.format(**kwargs))
            else:
                raise TypeError(f"message type error: {msg}")
        return ret

    @property
    def input_vars(self) -> set[str]:
        ret = set()
        for msg in self.messages:
            if isinstance(msg, (MessageTemplate, MessageListTemplate)):
                ret.update(msg.input_vars)
        return ret


MessageTemplateLike = ChatMessage | MessageTemplate | MessageListTemplate

MessageListTemplate.update_forward_refs()

MessageLikeInput = Union[MessageLike, Sequence[MessageLike]]
PlaceholderInput = Union[MessageLikeInput, Dict[str, MessageLikeInput]]


class MessagesPlaceholder(Flow[PlaceholderInput, List[ChatMessage]]):
    variable_name: str
    """Name of variable to use as messages."""

    optional: bool = False
    """If True, format can be called without arguments and will return an empty list.
     If False, a var_name argument must be provided, even if it's an empty list."""

    def invoke(self, inp: PlaceholderInput) -> List[ChatMessage]:
        return self.to_chat_messages(inp)

    def format(self, arg: PlaceholderInput | None = None, **kwargs: MessageLikeInput) -> List[ChatMessage]:
        if arg is not None:
            assert len(kwargs) == 0
            return self.to_chat_messages(arg)

        return self.to_chat_messages(kwargs)

    def to_chat_messages(self, inp: PlaceholderInput) -> List[ChatMessage]:
        if isinstance(inp, Dict):  # Dict[str, MessageLikeInput]
            v = inp.get(self.variable_name)
            if self.optional:
                v = v or []
            assert v
            return self.to_chat_messages(v)

        if isinstance(inp, str) or not isinstance(inp, Sequence):
            return [to_chat_message(inp)]

        # Sequence
        try:
            if len(inp) == 2 and Role.from_name(inp[0]):
                return [to_chat_message(inp)]   # type: ignore[arg-type]
        except ValueError as e:
            ...

        # Sequence[MessageLike]
        return list(map(to_chat_message, inp))
