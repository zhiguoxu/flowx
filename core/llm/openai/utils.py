from typing import Iterator, Tuple, List

from openai import Stream
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolMessageParam, \
    ChatCompletionMessageToolCallParam
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice, ChatCompletion

from core.llm.llm import ChatResult, TokenUsage
from core.messages.chat_message import ChatMessage, Role, ToolCall, ChatMessageChunk, ToolCallChunk


def to_openai_message(message: ChatMessage) -> ChatCompletionMessageParam:
    if message.role == Role.SYSTEM:
        return ChatCompletionSystemMessageParam(role="system", content=message.content or "")
    if message.role == Role.USER:
        return ChatCompletionUserMessageParam(role="user", content=message.content or "")
    if message.role == Role.ASSISTANT:
        msg = ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
        tool_calls: List[ChatCompletionMessageToolCallParam] = []
        for tool_call in message.tool_calls or []:
            tool_calls.append(tool_call.model_dump())  # type: ignore[arg-type]
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg
    if message.role == Role.TOOL:
        assert message.tool_call_id
        return ChatCompletionToolMessageParam(role="tool",
                                              content=message.content or "",
                                              tool_call_id=message.tool_call_id)
    raise ValueError(f"message: {message}")


def message_from_openai_choice(choice: Choice) -> ChatMessage:
    message = ChatMessage(role=Role.from_name(choice.message.role),
                          content=choice.message.content,
                          finish_reason=choice.finish_reason)
    tool_calls = list(map(lambda tool_call: ToolCall(**tool_call.model_dump()), choice.message.tool_calls or []))
    if tool_calls:
        message.tool_calls = tool_calls
    return message


def message_chunk_from_openai_choice(choice_chunk: ChoiceChunk) -> ChatMessageChunk:
    role = Role.from_name(choice_chunk.delta.role) if choice_chunk.delta.role else None
    message_chunk = ChatMessageChunk(role=role,
                                     content=choice_chunk.delta.content,
                                     finish_reason=choice_chunk.finish_reason)
    tool_calls = list(map(lambda tool_call: ToolCallChunk(**tool_call.model_dump()),
                          choice_chunk.delta.tool_calls or []))
    if tool_calls:
        message_chunk.tool_calls = tool_calls
    return message_chunk


def chat_result_from_openai(chat_completion: ChatCompletion | Stream[ChatCompletionChunk]) -> ChatResult:
    result = ChatResult()

    if isinstance(chat_completion, ChatCompletion):
        for choice in chat_completion.choices:
            result.messages.append(message_from_openai_choice(choice))
            if chat_completion.usage:
                result.usage = TokenUsage(**chat_completion.usage.model_dump())
    else:
        def to_message_stream() -> Iterator[Tuple[ChatMessageChunk, TokenUsage | None]]:
            for completion_chunk in chat_completion:
                usage = TokenUsage(**completion_chunk.usage.model_dump()) if completion_chunk.usage else None
                if completion_chunk.choices:
                    chunk = message_chunk_from_openai_choice(completion_chunk.choices[0])
                else:
                    chunk = ChatMessageChunk()
                yield chunk, usage

        result.message_stream = to_message_stream()
    return result
