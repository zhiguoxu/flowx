from __future__ import annotations

import inspect
from typing import Callable, List, Sequence, Any, Dict, Union, TypeVar

from pydantic import model_validator
from typing_extensions import Self

from core.callbacks.chat_history import BaseChatMessageHistory
from core.callbacks.run import Run
from core.flow.config import get_cur_config, var_local_config, FlowConfig
from core.flow.flow import BindingFlowBase, FlowBase, to_flow, Flow
from core.flow.utils import ConfigurableField
from core.messages.chat_message import ChatMessage
from core.messages.utils import to_chat_message, MessageLike

Input = Union[MessageLike, Dict[str, Any]]
# Output = Union[ChatMessage, Dict[str, Any]]
Output = TypeVar("Output", covariant=True)


class LLMWithHistory(BindingFlowBase[Input, Output]):
    """
    Flow that manages chat message history for another Flow.
    This flow's input type is Dict[str, Any] | LLMInput, the inner bound input type is LLMInput.
    If input_messages_key or history_messages_key is given, the input must be a dict.
    If output_messages_key is give, the inner bound's output must be a dict.
    """

    bound: FlowBase[Input, Output]
    """The flow constructed like 'flow = MessageListTemplate | LLM'"""

    get_session_history: Callable[..., BaseChatMessageHistory]

    input_messages_key: str | None = None
    """If input is a dict, input messages = input[input_messages_key]."""

    output_messages_key: str | None = None
    """Normally the output of bound is ChatMessage,
     bug if output is a dict, output message = output[input_messages_key]."""

    history_messages_key: str | None = None
    """The key for history message replacement key in chat prompt."""

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
            def get_input_messages(inp: Dict[str, Any]) -> List[ChatMessage]:
                assert self.input_messages_key is not None
                input_message = to_chat_message(inp[self.input_messages_key])
                history_messages = self.get_history().get_messages()
                return history_messages + [input_message]

            history_flow = start_flow.assign(**{self.input_messages_key: get_input_messages})
        else:
            # If no history_messages_key and no input_messages_key, the input must be MessageLike.
            def get_input_message(inp: MessageLike):
                return self.get_history().get_messages() + [to_chat_message(inp)]

            history_flow = to_flow(get_input_message)
        self.bound = (history_flow | self.bound).with_listeners(on_end=self.save_history)

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
        history = self.get_history()
        inp = run.input
        # Extract the real input that is MessageLike.
        if isinstance(inp, dict):
            input_key = self.input_messages_key or "input"
            if input_key in inp:
                inp = inp[input_key]
        history.add_messages([to_chat_message(inp), output])
