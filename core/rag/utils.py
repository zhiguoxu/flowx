from typing import TypeVar, Sequence, Any, Union


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
