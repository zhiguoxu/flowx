import functools
from contextvars import copy_context, ContextVar, Context
from typing import Any, Union, Callable, TypeVar, ParamSpec, Generator, Iterable, Iterator, cast, Dict, Tuple
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager

from core.flow.flow_config import var_flow_config, FlowConfig

Output = TypeVar("Output")

var_context_cache: ContextVar[Dict[str, Tuple[Context, int]]] = ContextVar("context_cache", default=dict())


@contextmanager
def new_context(obj: "Flow") -> Iterator[Context]:  # type: ignore[name-defined]
    from core.flow.flow import Flow
    obj = cast(Flow, obj)
    cache = cast(Dict[str, Tuple[Context, int]], var_context_cache.get())
    if obj.id not in cache.keys():
        ctx, count = copy_context().copy(), 0
    else:
        ctx, count = cache[obj.id]
    cache[obj.id] = ctx, count + 1
    yield cast(Context, ctx)
    ctx, count = cache[obj.id]
    if count == 1:
        cache.pop(obj.id)
    else:
        cache[obj.id] = ctx, count - 1


def flow_context(*args: Any,
                 config: FlowConfig | Dict | None = None,
                 **kwargs: Any  # config fields in kwargs format
                 ) -> Union[Callable, Callable[[Callable], Callable]]:
    def decorator(func: Callable[..., Output]) -> Callable[..., Output]:
        @functools.wraps(func)
        def wrapper(self, *args_: Any, **kwargs_: Any) -> Output:

            def run() -> Output:
                if config:
                    new_config = var_flow_config.get().merge(config).merge(kwargs)
                else:
                    new_config = var_flow_config.get().merge(kwargs)
                var_flow_config.set(new_config)
                return func(self, *args_, **kwargs_)

            with new_context(self) as context:
                return context.run(run)

        return wrapper

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