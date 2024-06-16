from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from concurrent.futures import wait, FIRST_COMPLETED
from typing import TypeVar, Generic, Iterator, Callable, cast, Mapping, Any, Awaitable, AsyncIterator, Union, \
    List, Dict, Type, Tuple

from pydantic import BaseModel, Field, field_validator
from tenacity import stop_after_attempt, wait_exponential_jitter, retry_if_exception_type, Retrying

from core.context import get_executor, flow_context
from core.flow.addable_dict import AddableDict
from core.flow.flow_config import FlowConfig
from core.flow.utils import merge_iterator, is_async_generator, is_generator, recurse_flow
from core.utils.iter import safe_tee
from core.utils.utils import filter_kwargs_by_pydantic, filter_kwargs_by_init_or_pydantic

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)
Other = TypeVar("Other")


class Flow(BaseModel, Generic[Input, Output], ABC):
    id: str = str(uuid.uuid4())
    name: str | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.invoke = flow_context(cls.invoke)
        cls.stream = flow_context(cls.stream)
        cls.transform = flow_context(cls.transform)

    @abstractmethod
    def invoke(self, inp: Input) -> Output:
        ...

    def stream(self, inp: Input) -> Iterator[Output]:
        yield self.invoke(inp)

    def transform(self, inp: Iterator[Input]) -> Iterator[Output]:
        yield from self.stream(merge_iterator(inp))

    def __or__(self, other: FlowLikeRight) -> SequenceFlow[Input, Other]:
        return SequenceFlow(self, other)

    def __ror__(self, other: FlowLikeLeft) -> SequenceFlow[Other, Output]:
        return to_flow(other).__or__(self)  # type: ignore[return-value]

    def pipe(self, *others: FlowLikeRight, name: str | None = None) -> SequenceFlow[Input, Other]:
        return SequenceFlow(self, *others, name=name)

    def bind(self, **kwargs: Any) -> BindingFlow[Input, Output]:
        return BindingFlow(bound=self, kwargs=kwargs)

    def with_config(self, config: FlowConfig | None = None, **kwargs: Any) -> BindingFlow[Input, Output]:
        return BindingFlow(
            bound=self,
            config=(config or FlowConfig()).merge(kwargs)
        )

    def with_retry(self, *,
                   retry_exception_types: Type[BaseException] | Tuple[Type[BaseException], ...] = Exception,
                   is_wait_exponential_jitter: bool = True,
                   max_attempt: int = 3
                   ) -> RetryFlow[Input, Output]:
        kwargs = filter_kwargs_by_init_or_pydantic(RetryFlow, locals())
        return RetryFlow(bound=self, **kwargs)


class FunctionFlow(Flow[Input, Output]):
    func: Callable[[Input], Output] | Callable[[Input], Iterator[Output]]

    def __init__(self, func: Callable[[Input], Output] | Callable[[Input], Iterator[Output]], name: str | None = None):
        if not name and func.__name__ != "<lambda>":
            name = func.__name__
        super().__init__(func=func, name=name)  # type: ignore[call-arg]

    def invoke(self, inp: Input) -> Output:
        output = self.func(inp)
        if is_generator(self.func):
            assert isinstance(output, Iterator)
            output = cast(Output, merge_iterator(output))

        if isinstance(output, Flow):
            with recurse_flow(self, inp):
                output = output.invoke(inp)

        return cast(Output, output)

    def stream(self, inp: Input) -> Iterator[Output]:
        output = self.func(inp)
        if is_generator(self.func):
            assert isinstance(output, Iterator)
            for o in output:
                if isinstance(o, Flow):
                    with recurse_flow(self, inp):
                        yield from o.stream(inp)
                else:
                    yield o
        elif isinstance(output, Flow):
            with recurse_flow(self, inp):
                yield from output.stream(inp)
        else:
            yield cast(Output, output)


class SequenceFlow(Flow[Input, Output]):
    steps: List[Flow]

    def __init__(self, *steps: FlowLike, **kwargs: Any):
        flow_steps: List[Flow] = list(map(to_flow, steps))
        kwargs = filter_kwargs_by_pydantic(SequenceFlow, locals(), exclude={"steps"}, exclude_none=True)
        super().__init__(steps=flow_steps, **kwargs)  # type: ignore[call-arg]

    def invoke(self, inp: Input) -> Output:
        for step in self.steps:
            inp = step.invoke(inp)
        return cast(Output, input)

    def stream(self, inp: Input) -> Iterator[Output]:
        yield from self.transform(iter([inp]))

    def transform(self, inp: Iterator[Input]) -> Iterator[Output]:
        for step in self.steps:
            inp = step.transform(inp)
        yield from cast(Iterator[Output], inp)

    def __or__(self, other: FlowLikeRight) -> SequenceFlow[Input, Other]:
        if isinstance(other, SequenceFlow):
            return SequenceFlow(*self.steps, *other.steps, name=self.name or other.name)
        else:
            return SequenceFlow(*self.steps, other, name=self.name)


class ParallelFlow(Flow[Input, Dict[str, Any]]):
    steps: Mapping[str, Flow[Input, Any]]

    def __init__(self,
                 steps: Mapping[str, FlowLike] | None = None,
                 **kwargs: FlowLike):
        merged_steps = {**(steps or {})}
        merged_steps.update(kwargs)
        steps = {k: to_flow(v) for k, v in merged_steps.items()}
        super().__init__(steps=steps)  # type: ignore[call-arg]

    def invoke(self, inp: Input) -> Dict[str, Any]:
        with get_executor() as executor:
            futures = [
                executor.submit(step.invoke, inp)
                for key, step in self.steps.items()
            ]
            return {key: future.result() for key, future in zip(self.steps.keys(), futures)}

    def stream(self, inp: Input) -> Iterator[Dict[str, Any]]:
        yield from self.transform(iter([inp]))

    def transform(self, inp: Iterator[Input]) -> Iterator[Dict[str, Any]]:
        input_copies = list(safe_tee(inp, len(self.steps)))
        with get_executor() as executor:
            # Create the transform() generator for each step
            named_generators = [
                (
                    name,
                    step.transform(input_copies.pop())
                )
                for name, step in self.steps.items()
            ]
            # Start the first iteration of each generator
            futures = {
                executor.submit(next, generator): (step_name, generator)
                for step_name, generator in named_generators
            }
            # Yield chunks from each iterator as they become available,
            # and start the next iteration of that iterator when it yields a chunk.
            while futures:
                completed_futures, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    step_name, generator = futures.pop(future)
                    try:
                        yield AddableDict({step_name: future.result()})
                        futures[executor.submit(next, generator)] = (step_name, generator)
                    except StopIteration:
                        pass


class GeneratorFlow(Flow[Input, Output]):
    generator: Callable[[Iterator[Input]], Iterator[Output]] | None = None
    a_generator: Callable[[AsyncIterator[Input]], AsyncIterator[Output]] | None = None

    def __init__(self,
                 generator: Union[
                     Callable[[Iterator[Input]], Iterator[Output]],
                     Callable[[AsyncIterator[Input]], AsyncIterator[Output]],
                 ]):
        try:
            name = generator.__name__
        except AttributeError as e:
            ...

        if is_generator(generator):
            ...
        elif is_async_generator(generator):
            a_generator = generator
            generator = None  # type: ignore[assignment]
        else:
            raise TypeError(f"GeneratorFlow do not support: {type(generator)}")

        kwargs = filter_kwargs_by_pydantic(GeneratorFlow, locals())
        super().__init__(**kwargs)

    def invoke(self, inp: Input) -> Output:
        return merge_iterator(self.stream(inp))

    def stream(self, inp: Input) -> Iterator[Output]:
        return self.transform(iter([inp]))

    def transform(self, inp: Iterator[Input]) -> Iterator[Output]:
        # todo support self.a_transform
        assert self.generator
        yield from self.generator(inp)


class BindingFlow(Flow[Input, Output]):
    bound: Flow[Input, Output]
    kwargs: Mapping[str, Any] = Field(default_factory=dict)
    config: FlowConfig = Field(default_factory=FlowConfig)

    def __init__(self,
                 bound: Flow[Input, Output],
                 kwargs: Mapping[str, Any] | None = None,
                 config: FlowConfig | None = None):
        if isinstance(bound, BindingFlow):
            kwargs_ = dict(bound.kwargs)
            kwargs_.update(kwargs or {})
            kwargs = kwargs_
            config = bound.config.patch(config or {})
            bound = bound.bound

        init_kwargs = filter_kwargs_by_pydantic(type(self), locals(), exclude_none=True)
        super().__init__(**init_kwargs)  # type: ignore[call-arg]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.bound, name)

    def invoke(self, inp: Input) -> Output:
        return self.bound.invoke(inp, **self.kwargs)  # type: ignore[call-arg]
        # Not every invoke accept **kwargs, so if you bind kwargs, it must be accepted by inner flow.


class RetryFlow(Flow[Input, Output]):
    bound: Flow[Input, Output]

    retry_exception_types: Type[BaseException] | Tuple[Type[BaseException], ...] = Exception
    """Retries if an exception has been raised of one or more types."""

    is_wait_exponential_jitter: bool = True
    """Whether to use wait strategy that applies exponential backoff and jitter."""

    max_attempt: int = 3
    """Stop when the previous attempt >= max_attempt."""

    @classmethod
    @field_validator('bound')
    def check_bound(cls, bound: Flow[Input, Output]):
        if isinstance(bound, cls):
            bound = bound.bound
        return bound

    @property
    def retrying_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(reraise=True)
        if self.max_attempt:
            kwargs["stop"] = stop_after_attempt(self.max_attempt)
        if self.is_wait_exponential_jitter:
            kwargs["wait"] = wait_exponential_jitter()
        if self.retry_exception_types:
            kwargs["retry"] = retry_if_exception_type(self.retry_exception_types)

        return kwargs

    def invoke(self, inp: Input) -> Output:
        for attempt in Retrying(**self.retrying_kwargs):
            with attempt:
                result = self.bound.invoke(inp)
        return result


FlowLike_ = Union[
    Flow[Input, Output],
    Callable[[Input], Output],
    Callable[[Input], Awaitable[Output]],
    Callable[[Iterator[Input]], Iterator[Output]],
    Callable[[AsyncIterator[Input]], AsyncIterator[Output]]
]

FlowLike = FlowLike_ | Mapping[str, FlowLike_]

FlowLikeRight_ = Union[
    Flow[Output, Other],
    Callable[[Output], Other],
    Callable[[Output], Awaitable[Other]],
    Callable[[Iterator[Output]], Iterator[Other]],
    Callable[[AsyncIterator[Output]], AsyncIterator[Other]]
]

FlowLikeRight = FlowLikeRight_ | Mapping[str, FlowLikeRight_]

FlowLikeLeft_ = Union[
    Flow[Other, Input],
    Callable[[Other], Input],
    Callable[[Other], Awaitable[Input]],
    Callable[[Iterator[Other]], Iterator[Input]],
    Callable[[AsyncIterator[Other]], AsyncIterator[Input]]
]

FlowLikeLeft = FlowLikeLeft_ | Mapping[str, FlowLikeLeft_]


def to_flow(flow_like: FlowLike) -> Flow[Input, Output]:
    if isinstance(flow_like, Flow):
        return flow_like
    elif is_generator(flow_like) or is_async_generator(flow_like):
        return GeneratorFlow(flow_like)
    elif callable(flow_like):
        return FunctionFlow(func=cast(Callable[[Input], Output], flow_like))
    elif isinstance(flow_like, dict):
        return cast(Flow[Input, Output], ParallelFlow(flow_like))
    else:
        raise TypeError(
            f"to_flow got an unsupported type: {type(flow_like)}"
        )
