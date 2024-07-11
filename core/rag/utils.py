from itertools import islice
from typing import TypeVar, Sequence, Any, Union, Iterable, Iterator, List, Tuple


def infer_torch_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


T = TypeVar("T")
OneOrMany = Union[T | Sequence[T]]


def to_list(obj: Any) -> list:
    if isinstance(obj, str):
        return [obj]
    return list(obj) if isinstance(obj, Sequence) else [obj]


def batch(size: int, iterable: Iterable[T]) -> Iterator[List[T]]:
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


T2 = TypeVar("T2")
T3 = TypeVar("T3")


def batch2(size: int,
           iterable: Iterable[T],
           iterable2: Iterable[T2] | None
           ) -> Iterator[Tuple[List[T], List[T2] | None]]:
    it = iter(iterable)
    it2 = iter(iterable2) if iterable2 else None
    while True:
        chunk = list(islice(it, size))
        chunk2 = list(islice(it2, size)) if it2 else None
        if not chunk:
            return
        yield chunk, chunk2


def batch3(size: int,
           iterable: Iterable[T],
           iterable2: Iterable[T2] | None,
           iterable3: Iterable[T3] | None
           ) -> Iterator[Tuple[List[T], List[T2] | None, List[T3] | None]]:
    it = iter(iterable)
    it2 = iter(iterable2) if iterable2 else None
    it3 = iter(iterable3) if iterable3 else None
    while True:
        chunk = list(islice(it, size))
        chunk2 = list(islice(it2, size)) if it2 else None
        chunk3 = list(islice(it3, size)) if it3 else None
        if not chunk:
            return
        yield chunk, chunk2, chunk3
