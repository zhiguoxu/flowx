from __future__ import annotations

from itertools import tee
from typing import TypeVar, Callable, Iterator, Any

from core.flow.utils import merge_iterator, is_generator

Output = TypeVar("Output", covariant=True)


def trace(func: Callable[..., Output | Iterator[Any]]):
    def wrapper(self: Any,  # Flow
                inp: Any,
                **kwargs: Any) -> Output:
        from core.callbacks.callback_manager import callback_manager

        if not callback_manager.on_flow_start(self, inp):
            return func(self, inp, **kwargs)  # type: ignore[return-value] # no trace

        try:
            output = func(self, inp, **kwargs)
            check_run_stack(self)

            callback_manager.on_flow_end(output)
            assert not isinstance(output, Iterator)
            return output
        except BaseException as e:
            check_run_stack(self)
            callback_manager.on_flow_error(e)
            raise

    def stream_wrapper(self: Any,  # Flow
                       inp: Any,
                       **kwargs: Any) -> Iterator[Output]:
        from core.callbacks.callback_manager import callback_manager

        if isinstance(inp, Iterator):
            inp, input_for_trace = tee(inp, 2)
            input_for_trace = merge_iterator(input_for_trace)
        else:
            input_for_trace = inp

        if not callback_manager.on_flow_start(self, input_for_trace):
            yield from func(self, inp, **kwargs)  # type: ignore[misc, return-value] # no trace

        try:
            stream_out = func(self, inp, **kwargs)
            check_run_stack(self)

            assert isinstance(stream_out, Iterator)
            output, output_for_trace = tee(stream_out, 2)
            yield from output
            callback_manager.on_flow_end(merge_iterator(output_for_trace))
        except BaseException as e:
            check_run_stack(self)
            callback_manager.on_flow_error(e)
            raise

    return stream_wrapper if is_generator(func) else wrapper


def check_run_stack(flow: Any):
    from core.callbacks.run import current_flow
    assert current_flow() == flow, f"Flow stack error: flow【{flow}】run in error stack【{current_flow()}】"
