import inspect
from typing import Iterator, TypeVar, Tuple, Any, Callable, TypeGuard, cast, AsyncIterator

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


def isgeneratorfunction(func: Any) -> TypeGuard[Callable[..., Iterator]]:
    return inspect.isgeneratorfunction(func)


def is_async_generator(func: Any) -> TypeGuard[Callable[..., AsyncIterator]]:
    return (
        inspect.isasyncgenfunction(func)
        or (hasattr(func, "__call__") and inspect.isasyncgenfunction(func.__call__))
    )
