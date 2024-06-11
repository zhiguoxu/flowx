from __future__ import annotations

from typing import Dict, List, Tuple, Sequence, Any, Union

from core.flow.flow import Flow
from core.messages.chat_message import ChatMessage
from core.prompts.message_template import MessageTemplate, validate_template_vars


class MessageListTemplate(Flow[Union[str | Dict], List[ChatMessage]]):
    messages: List[MessageLike]

    @classmethod
    def from_messages(cls, messages: Sequence[str | Tuple[str, str] | MessageLike]) -> MessageListTemplate:
        messages_: List[MessageLike] = []
        for msg in messages:
            if isinstance(msg, ChatMessage | MessageTemplate | MessageListTemplate):
                messages_.append(msg)
            else:
                messages_.append(MessageTemplate.from_arg(msg))
        return cls(messages=messages_)

    def invoke(self, inp: str | Dict) -> List[ChatMessage]:
        inp = validate_template_vars(inp, self.input_vars)
        return self.format(**inp)

    def partial_format(self, **kwargs: Any):
        for msg in self.messages:
            if isinstance(msg, (MessageTemplate, MessageListTemplate)):
                msg.partial_format(**kwargs)

    def format(self, **kwargs: Any) -> List[ChatMessage]:
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


MessageLike = ChatMessage | MessageTemplate | MessageListTemplate

MessageListTemplate.update_forward_refs()
