import json
import sys
import time
from collections import defaultdict
from typing import Any, Iterator, Dict, List, Tuple, cast, Callable
import json5  # type: ignore[import-untyped]
from core.callbacks.run_stack import current_run
from core.callbacks.trace import ENABLE_TRACE, trace
from core.flow.config import var_local_config
from core.flow.flow import Flow, to_flow, FunctionFlow, FixOutputFlow
from core.llm.llm import LLMInput, LLM, ChatResult, to_chat_messages, ToolChoice
from core.logging import get_logger
from core.messages.chat_message import ChatMessage, Role, ToolCall, ChatMessageChunk
from core.tool import Tool
from core.utils.utils import NotGiven, NOT_GIVEN, add

logger = get_logger(__name__)

STOP_TOOL_CALLS_PROMPT = "\nPlease stop calling the tool and summarize the final response to the user."
EXCEED_MAX_ITERATIONS = "Tool calls execution exceeded max iterations limit {}."


class Agent(Flow[LLMInput, ChatMessage]):
    """Has the same interface as LLM"""

    llm: LLM

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

    handle_parsing_errors: bool | str | Callable[[str, Exception], str] = False
    """
    Specifies how to handle errors from the agent's tool arguments parser
   - If False, raises the error.
   - If True, sends the error back to the LLM as an observation.
   - If a string, sends that string to the LLM as a an observation.
   - If a function, calls it with the invalid arguments and error and sends to the LLM as an observation.
   """

    @trace
    def invoke(self, inp: LLMInput, **kwargs: Any) -> ChatMessage:
        chat_result = self.chat(inp, **kwargs)
        if self.return_intermediate_steps and ENABLE_TRACE:
            current_run().update_extra_data(intermedia_steps=chat_result.messages[:-1])
        return chat_result.messages[-1]

    @trace
    def stream(self, inp: LLMInput, **kwargs: Any  # type: ignore[override]
               ) -> Iterator[ChatMessageChunk | ChatMessage]:
        chat_result = self.stream_chat(inp, **kwargs)
        intermedia_steps: List[ChatMessage] = []

        # Split intermedia_steps and final answer.
        def final_answer_stream() -> Iterator[ChatMessageChunk]:
            assert chat_result.message_stream_for_agent
            for message_or_chunk in chat_result.message_stream_for_agent:
                if isinstance(message_or_chunk, ChatMessageChunk):
                    yield message_or_chunk
                else:
                    # Only tool calls request and observations are ChatMessage, which are also intermedia_steps.
                    intermedia_steps.append(message_or_chunk)

        yield from final_answer_stream()
        if self.return_intermediate_steps and ENABLE_TRACE:
            current_run().update_extra_data(intermedia_steps=intermedia_steps)

    def chat(self, messages: LLMInput, **kwargs: Any) -> ChatResult:
        # If stream = False, _run_loop only return iter of ChatMessage.
        response_messages = cast(List[ChatMessage], list(self._run_loop(messages, False, **kwargs)))
        return ChatResult(messages=response_messages)

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
        if tools is NOT_GIVEN and "tools" in self.model_fields_set:
            tools = self.tools
        if tool_choice is NOT_GIVEN and "tool_choice" in self.model_fields_set:
            tool_choice = self.tool_choice

        # Prepare llm with local_config.
        local_config = var_local_config.get()
        llm = self.llm.with_config(local_config) if local_config else self.llm

        loop_count = 0
        time_elapsed = 0.0
        start_time = time.time()
        has_observation = False
        disable_tool_calls = False
        # Prevents calling the same tools with the same args.
        used_tool_calls: Dict[str, set[str]] = defaultdict(set)
        while has_observation and loop_count <= self.max_iterations and time_elapsed <= self.max_execution_time:
            loop_count += 1
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
                if self.return_intermediate_steps or not is_intermediate:
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

        if has_observation:
            yield ChatMessage(role=Role.ASSISTANT, content=EXCEED_MAX_ITERATIONS.format(loop_count))

    def _run_step(self,
                  response_message: ChatMessage,
                  used_tool_calls: Dict[str, set[str]],
                  tools: List[Tool] | None
                  ) -> Iterator[Tuple[ChatMessage, bool]]:  # [, is repeat tool call]
        # Yield llm just output message.
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
                observation = "Invalid or incomplete response, the response's tool arguments must be a valid json str."
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(function_args, e)
            else:
                raise ValueError(f"Got unexpected type of `handle_parsing_errors`: {self.handle_parsing_errors}")

            FixOutputFlow(output=observation).invoke(function_args)  # for trace
            logger.error(f"Function call arguments json format error: {function_args}, {observation}, {e}")
            return ChatMessage(role=Role.TOOL, tool_call_id=tool_call.id, content=observation), False

        tool_response = self._tool_dispatcher(function_call.name, function_args, tools)
        logger.debug(f"Function Call observation: {tool_response}")
        observation_msg = ChatMessage(role=Role.TOOL, tool_call_id=tool_call.id, content=tool_response)

        # If got the same function, then return disable tool calls.
        dumped_arguments = json.dumps(function_args, sort_keys=True)
        repeated_tool_call = function_call.arguments in used_tool_calls[dumped_arguments]
        used_tool_calls[function_call.name].add(dumped_arguments)
        return observation_msg, repeated_tool_call

    def _tool_dispatcher(self, function_name: str, function_args: dict, tools: List[Tool] | None) -> str:
        for tool in tools or self.tools or self.llm.tools or []:
            if tool.name == function_name:
                local_config = var_local_config.get()
                func_flow: Flow = FunctionFlow(lambda inp: tool(**inp), name=tool.name)
                if local_config:
                    func_flow = func_flow.with_config(local_config)
                return str(func_flow.invoke(function_args))

        return f"Tool `{function_name}` not found. Please use a provided tool."
