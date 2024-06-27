from abc import ABC, abstractmethod
from typing import List, Sequence

from pydantic import BaseModel, Field

from core.messages.chat_message import ChatMessage


class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history."""

    @abstractmethod
    def get_messages(self) -> List[ChatMessage]:
        ...

    def add_message(self, message: ChatMessage) -> None:
        if self.__class__.add_messages == BaseChatMessageHistory.add_messages:
            raise NotImplemented
        return self.add_messages([message])

    def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        for message in messages:
            self.add_message(message)


class InMemoryChatMessageHistory(BaseModel, BaseChatMessageHistory):
    """
    In memory implementation of chat message history.
    Store messages in an in memory list.
    """

    messages: List[ChatMessage] = Field(default_factory=list)

    def get_messages(self) -> List[ChatMessage]:
        return self.messages

    def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        self.messages.extend(messages)
