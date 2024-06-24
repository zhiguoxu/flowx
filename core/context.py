import functools
from contextvars import copy_context, ContextVar, Context
from typing import Any, Union, Callable, TypeVar, ParamSpec, Generator, Iterable, Iterator, cast, Dict, Tuple
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager

from core.flow.flow_config import var_flow_config, FlowConfig
from core.flow.flow import Flow
from core.utils.utils import is_generator, accepts_config

Output = TypeVar("Output")

var_context_cache: ContextVar[Dict[str, Tuple[Context, int]]] = ContextVar("context_cache", default=dict())


@contextmanager
def new_context(obj: Flow) -> Iterator[Context]:
    cache = cast(Dict[str, Tuple[Context, int]], var_context_cache.get())
    if obj.id not in cache.keys():
        ctx, count = copy_context().copy(), 0
    else:
        ctx, count = cache[obj.id]
    cache[obj.id] = ctx, count + 1
    yield cast(Context, ctx if count == 0 else None)  # prevent re-enter the same context
    ctx, count = cache[obj.id]
    if count == 1:
        cache.pop(obj.id)
    else:
        cache[obj.id] = ctx, count - 1


def flow_context(*args: Any,
                 config: FlowConfig | Dict | None = None,
                 **kwargs: Any  # config fields in kwargs format
                 ) -> Union[Callable, Callable[[Callable], Callable]]:

    def decorator(func: Callable[..., Output | Iterator[Output]]) -> Callable[..., Output | Iterator[Output]]:

        def set_new_context_config(flow: Flow) -> None:
            from core.flow.flow import BindingFlow
            config_bound_in_flow: FlowConfig | Dict = (flow.config or {}) if isinstance(flow, BindingFlow) else {}
            new_config = var_flow_config.get().merge(config or {}, kwargs, config_bound_in_flow)
            var_flow_config.set(new_config)

        @functools.wraps(func)
        def wrapper(self: Flow, *args_: Any, **kwargs_: Any) -> Output:

            def run() -> Output:
                set_new_context_config(self)
                # prepare config argument for func calling
                if accepts_config(func):
                    kwargs_.setdefault("config", var_flow_config.get())
                else:
                    kwargs_.pop("config", None)
                return cast(Output, func(self, *args_, **kwargs_))

            with new_context(self) as context:
                if context is None:
                    # prevent re-enter the same context
                    return run()
                return context.run(run)

        @functools.wraps(func)
        def stream_wrapper(self: Flow, *args_: Any, **kwargs_: Any) -> Iterator[Output]:
            def run_stream() -> Iterator[Output]:
                set_new_context_config(self)
                # prepare config argument for func calling
                if accepts_config(func):
                    kwargs_.setdefault("config", var_flow_config.get())
                else:
                    kwargs_.pop("config", None)
                yield from cast(Iterator[Output], func(self, *args_, **kwargs_))

            with new_context(self) as context:
                if context is None:
                    # prevent re-enter the same context
                    yield from run_stream()
                else:
                    iterator = run_stream()

                    while True:
                        try:
                            yield context.run(next, iterator)  # type: ignore[arg-type]
                        except StopIteration:
                            break

        return stream_wrapper if is_generator(func) else wrapper  # type: ignore[return-value]

    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])

    return decorator


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
