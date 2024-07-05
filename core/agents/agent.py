import json
import sys
import time
from collections import defaultdict
from typing import Any, Iterator, Dict, List, Tuple, cast, Callable
import json5  # type: ignore[import-untyped]
from pydantic import field_validator

from core.callbacks.run_stack import current_run
from core.callbacks.trace import ENABLE_TRACE, trace
from core.flow.config import var_local_config
from core.flow.flow import Flow, FunctionFlow, FixOutputFlow, Flowable, GeneratorFlow
from core.llm.llm import LLMInput, ChatResult, to_chat_messages, ToolChoice
from core.logging import get_logger
from core.messages.chat_message import ChatMessage, Role, ToolCall, ChatMessageChunk
from core.tool import Tool, to_tool, ToolLike
from core.utils.utils import NotGiven, NOT_GIVEN, add, is_generator

logger = get_logger(__name__)

STOP_TOOL_CALLS_PROMPT = "\nPlease stop calling the tool and summarize the final response to the user."
EXCEED_MAX_ITERATIONS = "Tool calls execution exceeded max iterations limit {}."
EXCEED_MAX_EXECUTION_TIME = "Tool calls execution exceeded max max execution time limit {} s."


class Agent(Flow[LLMInput, ChatMessage]):
    """Has the same interface as LLM"""

    llm: Flowable[LLMInput, ChatMessage]

    tools: List[Tool] | None = None
    """This tools will override llm tools"""

    tool_choice: ToolChoice | None = None
    """This tools will override llm tools"""

    max_iterations: int = 5
    """The maximum number of steps to take before ending the execution loop."""

    max_execution_time: float = sys.float_info.max
    """The maximum amount of wall clock time to spend in the execution loop."""

    return_intermediate_steps: bool = False
    """Whether to return the agent's trajectory of intermediate steps at the end in addition to the final output."""

    handle_parsing_errors: bool | str | Callable[[str, Exception], Tuple[bool, str | Dict[str, Any]]] = False
    """
    Specifies how to handle errors in tool arguments json formatting.
   - If False, raises the error.
   - If True, sends the error back to the LLM as an observation.
   - If a string, sends that string to the LLM as a an observation.
   - If a function, calls it with the invalid arguments and error,
        the result is a tuple (is_fix, fixed_argument dict or str observation),
        if is_fix=True, it return the fixed_argument dict, else return str observation.
   """

    @field_validator('tools')
    @classmethod
    def validate_tools(cls, tools: List[Tool | Callable]) -> List[Tool]:
        return [to_tool(tool) for tool in tools]

    @trace
    def invoke(self, inp: LLMInput, **kwargs: Any) -> ChatMessage:
        chat_result = self.chat(inp, **kwargs)
        if self.return_intermediate_steps and ENABLE_TRACE:
            current_run().update_extra_data(intermedia_steps=chat_result.messages[:-1])

        last_message = chat_result.messages[-1]
        if last_message and last_message.role == Role.TOOL:
            # If the last message is tool observation, it is returned direct.
            observation_data = last_message.extra_data
            if isinstance(observation_data, Flow):
                last_message = last_message.model_copy(deep=True)
                last_message.extra_data = observation_data.invoke({})
                if isinstance(last_message.extra_data, ChatMessage):
                    last_message = last_message.extra_data
        return last_message

    @trace
    def stream(self, inp: LLMInput, **kwargs: Any  # type: ignore[override]
               ) -> Iterator[ChatMessageChunk | ChatMessage]:
        chat_result = self.stream_chat(inp, **kwargs)
        intermedia_steps: List[ChatMessage] = []

        # Split output stream into intermedia_steps and final answer.
        last_message_list: List[ChatMessageChunk | ChatMessage | None] = [None]

        def final_answer_stream() -> Iterator[ChatMessageChunk]:
            assert chat_result.message_stream_for_agent
            for message_or_chunk in chat_result.message_stream_for_agent:
                last_message_list[0] = message_or_chunk
                if isinstance(message_or_chunk, ChatMessageChunk):
                    yield message_or_chunk
                else:
                    # Only tool calls request and observations are ChatMessage, which are also intermedia_steps.
                    intermedia_steps.append(message_or_chunk)

        yield from final_answer_stream()

        last_message = last_message_list[0]
        if last_message and last_message.role == Role.TOOL:
            intermedia_steps = intermedia_steps[:-1]
            # If the last message is tool observation, it is returned direct.
            observation_data = last_message.extra_data
            if isinstance(observation_data, Flow):
                def get_observation_stream() -> Iterator[ChatMessageChunk]:
                    tool_call_id = last_message.tool_call_id
                    for o in observation_data.stream({}):
                        if isinstance(o, ChatMessageChunk):
                            yield o
                        else:
                            yield ChatMessageChunk(role=Role.TOOL, tool_call_id=tool_call_id, extra_data=o)
                        tool_call_id = None

                yield from get_observation_stream()
            else:
                yield last_message

        if self.return_intermediate_steps and ENABLE_TRACE:
            current_run().update_extra_data(intermedia_steps=intermedia_steps)

    def chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        # If stream = False, _run_loop only return iter of ChatMessage.
        message_stream = self._run_loop(messages, False, **kwargs)
        return ChatResult(messages=cast(List[ChatMessage], list(message_stream)))

    def stream_chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        return ChatResult(message_stream_for_agent=self._run_loop(messages, True, **kwargs))

    def _run_loop(self,
                  messages: LLMInput,
                  stream: bool,
                  tools: List[Tool] | NotGiven | None = NOT_GIVEN,
                  tool_choice: ToolChoice | NotGiven | None = NOT_GIVEN,
                  **kwargs: Any) -> Iterator[ChatMessage | ChatMessageChunk]:
        """Return ChatMessageChunk for streaming final answer or thoughts, ChatMessage for tool calls and observation"""

        messages = to_chat_messages(messages)

        # Prepare tools.
        if tools is NOT_GIVEN:
            if "tools" in self.model_fields_set:
                tools = self.tools
            else:
                tools = getattr(self.llm, "tools", NOT_GIVEN)
        if tool_choice is NOT_GIVEN:
            if "tool_choice" in self.model_fields_set:
                tool_choice = self.tool_choice
            else:
                tool_choice = getattr(self.llm, "tool_choice", NOT_GIVEN)

        # Prepare llm with local_config.
        local_config = var_local_config.get()
        llm = self.llm.with_config(local_config) if local_config else self.llm

        loop_count = 0
        time_elapsed = 0.0
        start_time = time.time()
        has_observation = True
        disable_tool_calls = False
        return_direct = False
        # Prevents calling the same tools with the same args.
        used_tool_calls: Dict[str, set[str]] = defaultdict(set)
        while (has_observation and
               not return_direct and
               loop_count <= self.max_iterations and
               time_elapsed <= self.max_execution_time):
            loop_count += 1
            has_observation = False
            logger.debug(f"\n\n-------------- Turn {loop_count}  ---------------\n")

            # 1. Check and disable tool calls, and we hope it will never trigger tool calls in the following loop.
            if loop_count == self.max_iterations or disable_tool_calls:
                messages[-1].content = (messages[-1].content or "") + STOP_TOOL_CALLS_PROMPT
                tools = None
                tool_choice = None

            # 2. Prepare llm kwargs and bind it.
            llm_kwargs = dict(kwargs)
            if tools is not NOT_GIVEN:
                llm_kwargs["tools"] = tools
            if tool_choice is not NOT_GIVEN:
                llm_kwargs["tool_choice"] = tool_choice
            callable_tools = None if tools is NOT_GIVEN else cast(List[Tool] | None, tools)
            llm_with_kwargs = llm.bind(**llm_kwargs)

            # 3. Run one step by calling llm (expect tool calls or final answer).
            if not stream:
                invoke_response = llm_with_kwargs.invoke(messages)
                step_output = self._run_step(invoke_response, used_tool_calls, callable_tools)
            else:
                stream_response = llm_with_kwargs.stream(messages)
                step_output = self._stream_run_step(
                    stream_response, used_tool_calls, callable_tools)  # type: ignore[assignment, arg-type]

            # 4. Yield new messages and collect message for next turn.
            step_message_cache = None
            for step_message, repeated_tool_call in step_output:
                # 4.1. yield new messages.
                disable_tool_calls = disable_tool_calls or repeated_tool_call
                is_intermediate = step_message.tool_calls or step_message.role == Role.TOOL

                # Check if tool return direct.
                if step_message.tool_calls:
                    return_direct = any(_is_tool_return_direct(tool_call.function.name, callable_tools)
                                        for tool_call in step_message.tool_calls)
                    if return_direct and len(step_message.tool_calls) != 1:
                        raise RuntimeError("Tool return direct don't support parallel tool calls")

                if self.return_intermediate_steps or not is_intermediate or return_direct:
                    yield step_message

                # 4.2. collect next turn messages.
                if isinstance(step_message, ChatMessageChunk):
                    # Thought or final answer, no need to add to messages.
                    step_message_cache = add(step_message_cache, step_message)
                else:  # tool calls or observations
                    if step_message_cache:
                        assert step_message.tool_calls, \
                            "The first ChatMessage after ChatMessageChunk must have tool_calls."
                        # step_message_cache is thoughts, not final answer, when we get tool_calls,
                        # so add it to the tool_calls message.
                        step_message = step_message.model_copy()
                        step_message.content = (step_message.content or "") + (step_message_cache.content or "")
                        step_message_cache = None
                    messages.append(step_message)
                has_observation = has_observation or step_message.tool_calls is not None

            time_elapsed = time.time() - start_time

        if has_observation and not return_direct:
            content = (EXCEED_MAX_ITERATIONS.format(loop_count) if loop_count > self.max_iterations else
                       EXCEED_MAX_EXECUTION_TIME.format(time_elapsed))
            yield ChatMessage(role=Role.ASSISTANT, content=content)

    def _run_step(self,
                  response_message: ChatMessage,
                  used_tool_calls: Dict[str, set[str]],
                  tools: List[Tool] | None
                  ) -> Iterator[Tuple[ChatMessage, bool]]:  # [, is repeat tool call]
        # Yield message llm just output.
        yield response_message, False

        # Process tool_calls and get observation, then yield it.
        for tool_call in response_message.tool_calls or []:
            yield self._process_tool_call(tool_call, used_tool_calls, tools)

        if not response_message.tool_calls:
            logger.debug(f"Final answer: {response_message.content}")

    def _stream_run_step(self,
                         response_stream_message: Iterator[ChatMessageChunk],
                         used_tool_calls: Dict[str, set[str]],
                         tools: List[Tool] | None
                         ) -> Iterator[Tuple[ChatMessageChunk | ChatMessage, bool]]:  # [, is repeated tool call]
        """Return ChatMessageChunk for streaming final answer or thoughts,
        and return ChatMessage for tool calls and observation"""

        chunk_message_cache: ChatMessageChunk | None = None
        for chunk_message in response_stream_message:
            tool_calls = chunk_message.tool_calls
            if not tool_calls:
                # Yield streaming content of final answer or thoughts.
                if chunk_message.content:
                    yield chunk_message, False
                    logger.debug(f"Final answer(thoughts) chunk: {chunk_message.content}")
            else:
                # Collect tool calls request message.
                chunk_message_cache = add(chunk_message_cache, chunk_message)

        if chunk_message_cache:
            # Yield tool calls request message.
            message = chunk_message_cache.to_message()
            yield message, False
            # Process tool_calls and get observation, then yield it.
            for tool_call in message.tool_calls or []:
                yield self._process_tool_call(tool_call, used_tool_calls, tools)

    def _process_tool_call(self,
                           tool_call: ToolCall,
                           used_tool_calls: Dict[str, set[str]],
                           tools: List[Tool] | None
                           ) -> Tuple[ChatMessage, bool]:  # [, is repeated tool call]

        function_call = tool_call.function
        logger.debug(f"Function Call request: {function_call.model_dump()}")

        try:
            function_args = json5.loads(function_call.arguments)
        except ValueError as e:
            function_args = function_call.arguments
            if isinstance(self.handle_parsing_errors, bool):
                if not self.handle_parsing_errors:
                    raise ValueError(
                        "An output parsing error occurred."
                        "To retry, set `handle_parsing_errors=True` in Agent."
                        f"Error details: {str(e)}, invalid function_args: {function_args}"
                    )
                observation = "Invalid or incomplete response."
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                is_fixed, observation = self.handle_parsing_errors(function_args, e)  # type: ignore[assignment]
                if is_fixed:  # Give it a chance to fix the invalid arguments.
                    assert isinstance(observation, dict)
                    function_args = observation
            else:
                raise ValueError(f"Got unexpected type of `handle_parsing_errors`: {self.handle_parsing_errors}")

            if not isinstance(function_args, dict):
                FixOutputFlow(output=observation).invoke(function_args)  # for trace
                logger.error(f"Function call arguments json format error: {function_args}, {observation}, {e}")
                return ChatMessage(role=Role.TOOL, tool_call_id=tool_call.id, content=observation), False

        tool_response = _dispatch_tool_call(function_call.name, function_args, tools)
        logger.debug(f"Function Call observation: {tool_response}")
        observation_msg = ChatMessage(role=Role.TOOL, tool_call_id=tool_call.id)
        if isinstance(tool_response, str):
            observation_msg.content = tool_response
        else:
            observation_msg.extra_data = tool_response

        # If got the same function, then return disable tool calls.
        dumped_arguments = json.dumps(function_args, sort_keys=True)
        repeated_tool_call = function_call.arguments in used_tool_calls[dumped_arguments]
        used_tool_calls[function_call.name].add(dumped_arguments)
        return observation_msg, repeated_tool_call


def _dispatch_tool_call(function_name: str, function_args: dict, tools: List[Tool] | None) -> str | Flow:
    for tool in tools or []:
        if tool.name == function_name:
            local_config = var_local_config.get()
            if tool.return_direct:  # Only return direct support generator function.
                if is_generator(tool.function):  # Support tool output of stream.
                    def transformer(inp: Iterator) -> Iterator:
                        yield from tool.function(**function_args)

                    return GeneratorFlow(transformer).with_config(local_config)

            func_flow: Flow = FunctionFlow(lambda inp: tool(**inp), name=tool.name).with_config(local_config)
            output = func_flow.invoke(function_args)
            return output if tool.return_direct else str(output)

    return f"Tool `{function_name}` not found. Please use a provided tool."


def _is_tool_return_direct(function_name: str, tools: List[Tool] | None):
    for tool in tools or []:
        if tool.name == function_name:
            return tool.return_direct
    return False
