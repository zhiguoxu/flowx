from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from typing import Generator, ParamSpec, TypeVar, Callable, cast, Iterable, Any, Iterator, Dict, List

from pydantic import BaseModel, Field


class FlowConfig(BaseModel):
    recursion_limit: int = Field(default=20)
    """Maximum number of times a call can recurse."""

    max_concurrency: int | None = Field(default=None)
    """Maximum number of parallel calls to make."""

    tags: List[str] = Field(default_factory=list)

    def merge(self, other: FlowConfig | Dict) -> FlowConfig:
        if isinstance(other, FlowConfig):
            other = other.model_dump(exclude_unset=True)

        data = self.model_dump(exclude_unset=True)
        for key, value in other.items():
            old_v = data[key]
            if isinstance(value, list) and isinstance(old_v, list):
                old_v.extend(value)
            data[key] = value
        return FlowConfig(**data)

    def patch(self, other: FlowConfig | Dict) -> FlowConfig:
        if isinstance(other, FlowConfig):
            other = other.model_dump(exclude_unset=True)

        data = self.model_dump(exclude_unset=True)
        data.update(other)
        return FlowConfig(**data)


var_flow_config = ContextVar("flow_config", default=FlowConfig())

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
