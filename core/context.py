import functools
from contextvars import copy_context
from typing import Any, Union, Callable, TypeVar, ParamSpec, Generator, Iterable, Iterator, cast, Dict
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager

from core.flow.flow_config import var_flow_config, FlowConfig

Output = TypeVar("Output")


def flow_context(*args: Any,
                 config: FlowConfig | Dict | None = None,
                 **kwargs: Any  # config in kwargs format
                 ) -> Union[Callable, Callable[[Callable], Callable]]:
    def decorator(func: Callable[..., Output]) -> Callable[..., Output]:
        @functools.wraps(func)
        def wrapper(*args_: Any, **kwargs_: Any) -> Output:

            def run() -> Output:
                if config:
                    new_config = var_flow_config.get().merge(config).merge(kwargs)
                else:
                    new_config = var_flow_config.get().merge(kwargs)
                var_flow_config.set(new_config)
                return func(*args_, **kwargs_)

            return copy_context().copy().run(run)

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
