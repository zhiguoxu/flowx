from typing import Union, List, Tuple, Dict, Any
import json5  # type: ignore[import-untyped]
from auto_flow.core.messages.chat_message import ChatMessage, Role

MessageLike = Union[ChatMessage, List[str], Tuple[str, str], str, Dict[str, Any]]


def to_chat_message(message: MessageLike) -> ChatMessage:
    if isinstance(message, ChatMessage):
        return message

    if isinstance(message, str):
        return ChatMessage(role=Role.USER, content=message)

    if isinstance(message, (list, tuple)):
        if len(message) != 2:
            raise ValueError(f"MessageLike type error {message}")

        return ChatMessage(role=Role.from_name(message[0]), content=message[1])

    if isinstance(message, dict):
        return ChatMessage(role=Role.from_name(message["role"]), content=message["content"])

    raise ValueError(f"Error chat message: {message}")


def remove_extra_info(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    Extra_intro may be added to the tool observation message content,
    which can be used to construct the final agent output, but cannot be used in LLM conversations,
    so they need to be removed before sending to the llm model.
    """
    ret_messages = []
    for message in messages:
        content = message.content or ""
        if message.role == Role.TOOL and "extra_info" in content:
            rest_right = ""
            if r_index := content.rfind("}") >= 0:
                content, rest_right = content[:r_index + 1], content[r_index + 1]
            try:
                content_dict = json5.loads(content)
                if content_dict.pop("extra_info", None):
                    message = message.model_copy()
                    message.content = str(content_dict) + rest_right
            except ValueError as e:
                ...
        ret_messages.append(message)

    return messages
