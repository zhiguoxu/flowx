import asyncio
from contextlib import contextmanager
from contextvars import copy_context
from functools import partial
from typing import Any, ParamSpec, TypeVar, Callable, cast

from pydantic import BaseModel

from auto_flow.core.flow.config import var_flow_config
from auto_flow.core.utils.utils import NOT_GIVEN


@contextmanager
def recurse_flow(flow: Any, inp: Any):
    config = var_flow_config.get()
    if config.recursion_limit <= 0:
        raise RecursionError(
            f"Recursion limit reached when invoking {flow} with input {inp}."
        )
    config.recursion_limit -= 1
    try:
        yield
    finally:
        config.recursion_limit += 1


class ConfigurableField(BaseModel):
    id: str
    description: str | None = None
    annotation: Any = None
    default: Any = NOT_GIVEN


P = ParamSpec("P")
T = TypeVar("T")


async def run_in_executor(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            raise RuntimeError from exc

    return await asyncio.get_running_loop().run_in_executor(
        None,
        cast(Callable[..., T], partial(copy_context().run, wrapper))
    )
