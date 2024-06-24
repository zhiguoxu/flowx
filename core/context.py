import functools
from contextvars import copy_context, ContextVar, Context
from typing import Any, Union, Callable, TypeVar, ParamSpec, Generator, Iterable, Iterator, cast, Dict, Tuple
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager

from core.flow.flow_config import var_flow_config, FlowConfig, var_local_config
from core.flow.flow import Flow
from core.utils.utils import is_generator

Output = TypeVar("Output")

var_context_cache: ContextVar[Dict[str, Tuple[Context, int]]] = ContextVar("context_cache", default=dict())


@contextmanager
def new_context(obj: Flow) -> Iterator[Context]:
    # 1. create new context or inc reference
    cache = cast(Dict[str, Tuple[Context, int]], var_context_cache.get())
    if obj.id not in cache.keys():
        ctx, count = copy_context().copy(), 0
    else:
        ctx, count = cache[obj.id]
    cache[obj.id] = ctx, count + 1

    try:
        yield cast(Context, ctx if count == 0 else None)  # prevent re-enter the same context
    finally:
        # 2. release context or dec reference
        ctx, count = cache[obj.id]
        if count == 1:
            cache.pop(obj.id)
        else:
            cache[obj.id] = ctx, count - 1


def flow_context(func: Callable[..., Output | Iterator[Output]]) -> Union[Callable, Callable[[Callable], Callable]]:
    def set_new_context_config(flow: Flow, local_config: FlowConfig | None) -> None:
        from core.flow.flow import BindingFlow
        # set var_flow_config (inheritable)
        config_bound_in_flow: FlowConfig | Dict = (flow.config or {}) if isinstance(flow, BindingFlow) else {}
        new_config = var_flow_config.get().merge(config_bound_in_flow)
        var_flow_config.set(new_config)

        # set var_local_config (not inheritable)
        var_local_config.set(local_config)

    @functools.wraps(func)
    def wrapper(self: Flow, inp: Any, local_config: FlowConfig | None = None, /, **kwargs_: Any) -> Output:

        def run(first_enter: bool = True) -> Output:
            if first_enter:
                set_new_context_config(self, local_config)
            return cast(Output, func(self, inp, **kwargs_))

        with new_context(self) as context:
            if context is None:
                # prevent re-enter the same context
                assert local_config is None, "local_config only pass first call"
                return run(False)
            return context.run(run)

    @functools.wraps(func)
    def stream_wrapper(self: Flow,
                       inp: Any,
                       local_config: FlowConfig | None = None, /,
                       **kwargs_: Any) -> Iterator[Output]:
        def run_stream(first_enter: bool = True) -> Iterator[Output]:
            if first_enter:
                set_new_context_config(self, local_config)
            yield from cast(Iterator[Output], func(self, inp, **kwargs_))

        with new_context(self) as context:
            if context is None:
                # prevent re-enter the same context
                assert local_config is None, "local_config only pass first call"
                yield from run_stream(False)
            else:
                iterator = run_stream()
                while True:
                    try:
                        yield context.run(next, iterator)  # type: ignore[arg-type]
                    except StopIteration:
                        break

    return stream_wrapper if is_generator(func) else wrapper  # type: ignore[return-value]


P = ParamSpec("P")
T = TypeVar("T")


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that copies the context to the child thread."""

    def submit(self, func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        return super().submit(
            cast(Callable[..., T], copy_context().run), func, *args, **kwargs)

    def map(self,
            fn: Callable[..., T],
            *iterables: Iterable[Any],
            timeout: float | None = None,
            chunksize: int = 1
            ) -> Iterator[T]:
        contexts = [copy_context() for _ in range(len(iterables[0]))]  # type: ignore

        return super().map(
            lambda *args: contexts.pop().run(fn, *args),
            *iterables,
            timeout=timeout,
            chunksize=chunksize,
        )


@contextmanager
def get_executor(
) -> Generator[Executor, None, None]:
    with ContextThreadPoolExecutor(
            max_workers=var_flow_config.get().max_concurrency
    ) as executor:
        yield executor
