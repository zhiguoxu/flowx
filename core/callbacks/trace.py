from __future__ import annotations

from itertools import tee
from typing import TypeVar, Callable, Iterator, Any

from core.errors import RunStackError
from core.flow.flow_config import var_flow_config
from core.utils.iter import merge_iterator
from core.utils.utils import is_generator, env_is_set, filter_config_by_method

Output = TypeVar("Output", covariant=True)

# Trace is enabled by default.
# Disable trace maybe helpful when debugging the code, because it will reduce the deep of call stack.
ENABLE_TRACE = env_is_set("FLOW_ENABLE_TRACE", True)


def trace(func: Callable[..., Output | Iterator[Any]]):
    if not ENABLE_TRACE:
        return func

    def wrapper(self: Any,  # Flow
                inp: Any,
                **kwargs: Any) -> Output:
        from core.callbacks.callback_manager import callback_manager

        kwargs.setdefault("config", var_flow_config.get())
        func_kwargs = filter_config_by_method(kwargs, func)
        if not callback_manager.on_flow_start(self, inp, **kwargs):
            return func(self, inp, **func_kwargs)  # type: ignore[return-value] # no trace

        try:
            output = func(self, inp, **func_kwargs)
            check_run_stack(self)

            callback_manager.on_flow_end(output)
            assert not isinstance(output, Iterator)
            return output
        except BaseException as e:
            if isinstance(e, RunStackError):
                raise
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

        kwargs.setdefault("config", var_flow_config.get())
        func_kwargs = filter_config_by_method(kwargs, func)
        if not callback_manager.on_flow_start(self, input_for_trace, **kwargs):
            yield from func(self, inp, **func_kwargs)  # type: ignore[misc, return-value] # no trace
        else:
            try:
                stream_out = func(self, inp, **func_kwargs)
                check_run_stack(self)

                assert isinstance(stream_out, Iterator)
                output, output_for_trace = tee(stream_out, 2)
                yield from output
                callback_manager.on_flow_end(merge_iterator(output_for_trace))
            except BaseException as e:
                if isinstance(e, RunStackError):
                    raise
                check_run_stack(self)
                callback_manager.on_flow_error(e)
                raise

    return stream_wrapper if is_generator(func) else wrapper


def check_run_stack(flow: Any):
    if not ENABLE_TRACE:
        return

    from core.callbacks.run import current_flow
    if current_flow() != flow:
        raise RunStackError(f"Run stack error, flow run in error stack:【{flow}】in【{current_flow()}】")
