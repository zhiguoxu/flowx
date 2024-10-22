from __future__ import annotations

from typing import Dict, List, Tuple, Sequence, Any, Union

from auto_flow.core.messages.chat_message import ChatMessage, Role
from auto_flow.core.messages.utils import MessageLike, to_chat_message
from auto_flow.core.prompts.message_template import MessageTemplate, validate_template_vars, PromptTemplate


class ChatTemplate(PromptTemplate[str, List[ChatMessage]]):
    messages: List[MessageTemplateLike]

    @classmethod
    def from_messages(cls,
                      messages: Sequence[str | List[str] | Tuple[str, str] | MessageTemplateLike]
                      ) -> ChatTemplate:
        messages_: List[MessageTemplateLike] = []
        for msg in messages:
            if isinstance(msg, ChatMessage | MessageTemplate | ChatTemplate | MessagesPlaceholder):
                messages_.append(msg)
            elif isinstance(msg, (list, tuple)) and msg[0] == "placeholder":
                assert len(msg) >= 2
                var_name = msg[1]
                assert var_name[0] == "{" and var_name[-1] == "}", "var"
                optional = msg[2] if len(msg) > 2 and isinstance(msg[2], bool) else True  # type: ignore[misc]
                messages_.append(MessagesPlaceholder(var_name=var_name[1:-1], optional=optional))
            else:
                messages_.append(MessageTemplate.from_arg(msg))
        return cls(messages=messages_)

    def invoke(self, inp: str | Dict[str, Any]) -> List[ChatMessage]:
        inp = validate_template_vars(inp, self.input_vars)
        return self.format(**inp)

    def partial_format(self, **kwargs: Any):
        for msg in self.messages:
            if isinstance(msg, (MessageTemplate, ChatTemplate)):
                msg.partial_format(**kwargs)

    def format(self, arg: str | None = None, **kwargs: Any) -> List[ChatMessage]:
        if arg is not None:
            assert len(kwargs) == 0, \
                f"Use position argument when only has one template var, which means no other input!"
            kwargs = validate_template_vars(arg, self.input_vars)

        ret: List[ChatMessage] = []
        for msg in self.messages:
            if isinstance(msg, ChatMessage):
                ret.append(msg)
            elif isinstance(msg, MessageTemplate):
                ret.append(msg.format(**kwargs))
            elif isinstance(msg, (ChatTemplate, MessagesPlaceholder)):
                ret.extend(msg.format(**kwargs))
            else:
                raise TypeError(f"message type error: {msg}")
        return ret

    @property
    def input_vars(self) -> set[str]:
        ret = set()
        for msg in self.messages:
            if isinstance(msg, (MessageTemplate, ChatTemplate)):
                ret.update(msg.input_vars)
            elif isinstance(msg, MessagesPlaceholder):
                ret.add(msg.var_name)
        return ret

    def __add__(self, other: str | List[str] | Tuple[str, str] | MessageTemplateLike) -> ChatTemplate:
        # Allow for easy combining.
        if isinstance(other, ChatTemplate):
            return ChatTemplate(messages=self.messages + other.messages)
        elif isinstance(other, (ChatMessage, MessageTemplate, MessagesPlaceholder)):
            return ChatTemplate(messages=self.messages + [other])
        elif isinstance(other, (str, list, tuple)):
            other = ChatTemplate.from_messages(other)
            return ChatTemplate(messages=self.messages + other.messages)
        else:
            raise NotImplementedError(f"Unsupported operand type for +: {type(other)}")


MessageLikeInput = Union[MessageLike, Sequence[MessageLike]]
PlaceholderInput = Union[MessageLikeInput, Dict[str, MessageLikeInput]]


class MessagesPlaceholder(PromptTemplate[MessageLikeInput, List[ChatMessage]]):
    var_name: str
    """Name of variable to use as messages."""

    optional: bool = False
    """If True, format can be called without arguments and will return an empty list.
     If False, a var_name argument must be provided, even if it's an empty list."""

    def __init__(self, var_name: str, optional: bool = False):
        super().__init__(var_name=var_name, optional=optional)  # type: ignore[call-arg]

    def invoke(self, inp: PlaceholderInput) -> List[ChatMessage]:
        return self.to_chat_messages(inp)

    def partial_format(self, **kwargs: Any) -> MessagesPlaceholder:
        raise NotImplementedError

    def format(self, arg: MessageLikeInput | None = None, **kwargs: MessageLikeInput) -> List[ChatMessage]:
        if arg is not None:
            assert len(kwargs) == 0
            return self.to_chat_messages(arg)

        return self.to_chat_messages(kwargs)

    def to_chat_messages(self, inp: PlaceholderInput) -> List[ChatMessage]:
        if isinstance(inp, dict):  # Dict[str, MessageLikeInput] or {"role": xxx,"content": yyy}
            if len(inp) == 2 and "role" in inp and "content" in inp:
                return [to_chat_message(inp)]

            v = inp.get(self.var_name)
            if self.optional:
                v = v or []
            assert v is not None, f"Expect input key: {self.var_name}"
            return self.to_chat_messages(v)

        # Sequence
        if isinstance(inp, str) or not isinstance(inp, Sequence):
            return [to_chat_message(inp)]

        # try MessageLike
        try:
            if len(inp) == 2 and Role.from_name(inp[0]):
                return [to_chat_message(inp)]  # type: ignore[arg-type]
        except ValueError as e:
            ...

        # Sequence[MessageLike]
        return list(map(to_chat_message, inp))


MessageTemplateLike = ChatMessage | MessageTemplate | ChatTemplate | MessagesPlaceholder
