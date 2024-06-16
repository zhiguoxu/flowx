import inspect
from contextlib import contextmanager
from typing import Iterator, TypeVar, Any, Callable, TypeGuard, cast, AsyncIterator

from core.flow.flow_config import var_flow_config

T = TypeVar("T")


def merge_iterator(iterator: Iterator[T]) -> T:
    """return (has_value, merged_value)"""

    init_v = object()
    final_v: T | object = init_v
    for i in iterator:
        if final_v == init_v:
            final_v = i
        else:
            try:
                final_v += i  # type: ignore
            except TypeError:
                final_v = i
    assert final_v != init_v
    return cast(T, final_v)


def is_generator(func: Any) -> TypeGuard[Callable[..., Iterator]]:
    return (
            inspect.isgeneratorfunction(func)
            or (hasattr(func, "__call__") and inspect.isgeneratorfunction(func.__call__))
    )


def is_async_generator(func: Any) -> TypeGuard[Callable[..., AsyncIterator]]:
    return (
            inspect.isasyncgenfunction(func)
            or (hasattr(func, "__call__") and inspect.isasyncgenfunction(func.__call__))
    )


@contextmanager
def recurse_flow(flow: Any, inp: Any):
    config = var_flow_config.get()
    if config.recursion_limit <= 0:
        raise RecursionError(
            f"Recursion limit reached when invoking {flow} with input {inp}."
        )
    config.recursion_limit -= 1
    yield
    config.recursion_limit += 1
