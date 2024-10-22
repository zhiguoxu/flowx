from __future__ import annotations

import inspect
from typing import Callable, List, Sequence, Any, Dict, Union, TypeVar

from pydantic import model_validator
from typing_extensions import Self

from auto_flow.core.callbacks.chat_history import BaseChatMessageHistory
from auto_flow.core.callbacks.run import Run
from auto_flow.core.flow.config import get_cur_config, var_local_config, FlowConfig
from auto_flow.core.flow.flow import BindingFlowBase, Flowable, to_flow, Flow
from auto_flow.core.flow.utils import ConfigurableField
from auto_flow.core.llm.llm import to_chat_messages
from auto_flow.core.messages.chat_message import ChatMessage
from auto_flow.core.messages.utils import MessageLike

Input = Union[MessageLike, Dict[str, Any]]
Output = TypeVar("Output", covariant=True, bound=Union[ChatMessage, Dict[str, Any]])


class LLMWithHistory(BindingFlowBase[Input, Output]):
    """Flow that manages chat message history for another Flow."""

    bound: Flowable[Input, Output]
    """The flow is constructed like 'MessageListTemplate | LLM'"""

    get_session_history: Callable[..., BaseChatMessageHistory]
    """The method to get chat history by session."""

    history_messages_key: str | None = None
    """The key for history message replacement key in chat prompt."""

    input_messages_key: str | None = None
    """The key for input message replacement key in chat prompt."""

    output_messages_key: str | None = None
    """Normally the output of bound is ChatMessage,
     bug if output is a dict, output_message = output[output_messages_key]."""

    history_factory_config: Sequence[ConfigurableField] = (ConfigurableField(id="session_id", annotation=str),)
    """ConfigurableField define the arguments of get_session_history."""

    @model_validator(mode='after')
    def reset_bound_with_history(self) -> Self:
        start_flow: Flow = to_flow(
            lambda inp_: inp_ if isinstance(inp_, dict) else {self.input_messages_key or "input": inp_}
        )
        if self.history_messages_key:
            # assign history value
            history_flow: Flow = to_flow(lambda _: self.get_history().get_messages())
            history_flow = start_flow.assign(**{self.history_messages_key: history_flow})
        elif self.input_messages_key:
            # replace input[input_messages_key] = history messages + input[input_messages_key]
            def get_input_messages_by_key(inp: Dict[str, Any]) -> List[ChatMessage]:
                assert self.input_messages_key is not None
                input_messages = to_chat_messages(inp[self.input_messages_key])
                history_messages = self.get_history().get_messages()
                return history_messages + input_messages

            history_flow = start_flow.assign(**{self.input_messages_key: get_input_messages_by_key})
        else:
            # If no history_messages_key and no input_messages_key, the input must be MessageLike.
            def get_input_messages(inp: MessageLike) -> List[ChatMessage]:
                return self.get_history().get_messages() + to_chat_messages(inp)

            history_flow = to_flow(get_input_messages)

        self.bound = history_flow.pipe(self.bound, main=True).with_listeners(on_end=self.save_history)
        return self

    def get_history(self) -> BaseChatMessageHistory:
        # Use config.configurable for history cache.
        history_configurable_key = "message_history"
        local_config = var_local_config.get() or FlowConfig()
        if history_configurable_key in local_config.configurable:
            return local_config.configurable[history_configurable_key]

        expected_keys = {field_spec.id for field_spec in self.history_factory_config}
        configurable = get_cur_config().configurable
        missing_keys = expected_keys - set(configurable.keys())
        if missing_keys:
            example_configurable = {missing_key: "[your-value-here]" for missing_key in missing_keys}
            raise ValueError(
                f"Missing keys {sorted(missing_keys)} in config.configurable "
                f"Expected keys are {sorted(expected_keys)}."
                f"You can pass the configurable vars by flow.with_configurable({example_configurable})"
            )
        if len(expected_keys) == 1:
            # If arity = 1, then invoke function by positional arguments
            return self.get_session_history(configurable[expected_keys.pop()])

        # otherwise verify that names of keys patch and invoke by named arguments
        parameter_names = inspect.signature(self.get_session_history).parameters.keys()
        if expected_keys != set(parameter_names):
            raise ValueError(
                f"Expected keys {sorted(expected_keys)} do not match parameter "
                f"names {sorted(parameter_names)} of get_session_history."
            )

        kwargs = {key: configurable[key] for key in expected_keys}
        history = self.get_session_history(**kwargs)
        local_config.configurable[history_configurable_key] = history
        return history

    def save_history(self, run: Run):
        output = run.output
        assert output
        if self.output_messages_key:
            output = output[self.output_messages_key]
        assert isinstance(output, ChatMessage)
        # If it is intermediate message, don't save it.
        # Agent will handler the intermediate messages and invoke llm in the next round
        # with intermediate messages, and all the message will be saved finally.
        if output.tool_calls:
            return

        history = self.get_history()
        inp = run.input
        # Extract the real input that is MessageLike.
        if isinstance(inp, dict):
            input_key = self.input_messages_key or "input"
            if input_key in inp:
                inp = inp[input_key]
        history.add_messages(to_chat_messages(inp) + [output])
