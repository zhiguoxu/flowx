from typing import Union, List, Tuple, Dict, Any, Sequence

from core.messages.chat_message import ChatMessage, Role

MessageLike = Union[ChatMessage, List[str], Tuple[str, str], str, Dict[str, Any]]


def to_chat_message(message: MessageLike) -> ChatMessage:
    if isinstance(message, ChatMessage):
        return message

    if isinstance(message, str):
        return ChatMessage(role=Role.USER, content=message)

    if isinstance(message, Sequence):  # list, tuple
        assert len(message) == 2, f"MessageLike type error {message}"
        return ChatMessage(role=Role.from_name(message[0]), content=message[1])

    if isinstance(message, dict):
        return ChatMessage(role=Role.from_name(message.get("role")), content=message.get("content"))

    raise TypeError(f"Error type: {message}")
