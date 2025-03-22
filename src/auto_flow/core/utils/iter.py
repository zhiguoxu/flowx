import itertools
from threading import Lock
from typing import TypeVar, Iterator, cast, AsyncIterator


# https://www.kingname.info/2019/11/06/thread-safe-in-tee/


class SafeTee:
    def __init__(self, tee_obj, lock):
        self.tee_obj = tee_obj
        self.lock = lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.tee_obj)

    def __copy__(self):
        return SafeTee(self.tee_obj.__copy__(), self.lock)


def safe_tee(iterable, n=2):
    """tuple of n independent thread-safe iterators"""
    lock = Lock()
    return tuple(SafeTee(tee_obj, lock) for tee_obj in itertools.tee(iterable, n))


T = TypeVar("T")


def merge_iterator(iterator: Iterator[T]) -> T:
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
    assert final_v != init_v, "merge_iterator's input is empty iterator"
    return cast(T, final_v)


async def async_merge_iterator(iterator: AsyncIterator[T]) -> T:
    init_v = object()
    final_v: T | object = init_v
    async for i in iterator:
        if final_v == init_v:
            final_v = i
        else:
            try:
                final_v += i  # type: ignore
            except TypeError:
                final_v = i
    assert final_v != init_v, "merge_iterator's input is empty iterator"
    return cast(T, final_v)
