from __future__ import annotations

from itertools import tee
from typing import TypeVar, Callable, Iterator, Any, TYPE_CHECKING

from auto_flow.core.exceptions import RunStackException
from auto_flow.core.utils.iter import merge_iterator
from auto_flow.core.utils.utils import is_generator, env_is_set

if TYPE_CHECKING:
    from auto_flow.core.flow.flow import Flow

Output = TypeVar("Output", covariant=True)

# Trace is enabled by default.
# Disable trace maybe helpful when debugging the code, because it will reduce the deep of call stack.
ENABLE_TRACE = env_is_set("FLOW_ENABLE_TRACE", False)


def trace(func: Callable[..., Output]) -> Callable[..., Output]:
    """Trace decorator only for Flow function with the first argument the input of flow."""

    if not ENABLE_TRACE:
        return func

    def wrapper(flow: Flow, inp: Any, **kwargs: Any) -> Output:
        from auto_flow.core.callbacks.callback_manager import callback_manager
        from auto_flow.core.callbacks.run_stack import check_cur_flow

        if not callback_manager.on_flow_start(flow, inp, **kwargs):
            return func(flow, inp, **kwargs)  # type: ignore[return-value] # no trace

        try:
            output = func(flow, inp, **kwargs)
            check_cur_flow(flow)

            callback_manager.on_flow_end(output)
            assert not isinstance(output, Iterator)
            return output
        except BaseException as e:
            if isinstance(e, RunStackException):
                raise
            check_cur_flow(flow)
            callback_manager.on_flow_error(e)
            raise

    def stream_wrapper(flow: Any, inp: Any, **kwargs: Any) -> Output:  # type: ignore[misc]
        from auto_flow.core.callbacks.callback_manager import callback_manager
        from auto_flow.core.callbacks.run_stack import check_cur_flow

        if isinstance(inp, Iterator):
            inp, input_for_trace = tee(inp, 2)
            input_for_trace = merge_iterator(input_for_trace)
        else:
            input_for_trace = inp

        if not callback_manager.on_flow_start(flow, input_for_trace, **kwargs):
            yield from func(flow, inp, **kwargs)  # type: ignore[misc, return-value] # no trace
        else:
            try:
                stream_out = func(flow, inp, **kwargs)
                check_cur_flow(flow)

                assert isinstance(stream_out, Iterator)
                output, output_for_trace = tee(stream_out, 2)
                yield from output
                callback_manager.on_flow_end(merge_iterator(output_for_trace))
            except BaseException as e:
                if isinstance(e, RunStackException):
                    raise
                check_cur_flow(flow)
                callback_manager.on_flow_error(e)
                raise

    return stream_wrapper if is_generator(func) else wrapper
