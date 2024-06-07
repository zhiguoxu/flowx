from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import wait, FIRST_COMPLETED
from typing import TypeVar, Generic, Iterator, Callable, cast, Mapping, Any, Awaitable, AsyncIterator, Union, \
    List, Dict

from pydantic import BaseModel

from core.flow.addable_dict import AddableDict
from core.flow.flow_config import var_flow_config, get_executor
from core.flow.utils import merge_iterator, isgeneratorfunction, is_async_generator
from core.utils.iter import safe_tee
from core.utils.utils import filter_kwargs_by_pydantic

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)
Other = TypeVar("Other")


class Flow(BaseModel, Generic[Input, Output], ABC):
    name: str | None = None

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


class FunctionFlow(Flow[Input, Output]):
    func: Callable[[Input], Output] | Callable[[Input], Iterator[Output]]

    def invoke(self, inp: Input) -> Output:
        output = self.func(inp)
        if isgeneratorfunction(self.func):
            assert isinstance(output, Iterator)
            output = cast(Output, merge_iterator(output))

        return cast(Output, output)

    def stream(self, inp: Input) -> Iterator[Output]:
        output = self.func(inp)
        if isgeneratorfunction(self.func):
            assert isinstance(output, Iterator)
            yield from output
        elif isinstance(output, Flow):
            config = var_flow_config.get()
            if config.recursion_limit <= 0:
                raise RecursionError(
                    f"Recursion limit reached when invoking {self} with input {inp}."
                )
            config.recursion_limit -= 1
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

    def __init__(self, /, steps: Mapping[str, FlowLike | Mapping[str, FlowLike]] | None = None,
                 **kwargs: FlowLike | Mapping[str, FlowLike]):
        merged_steps = {**(steps or {})}
        merged_steps.update(kwargs)
        super().__init__(steps={k: to_flow(v) for k, v in merged_steps.items()})  # type: ignore

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
    elif is_async_generator(flow_like) or isgeneratorfunction(flow_like):
        # return RunnableGenerator(thing)
        return cast(Flow[Input, Output], flow_like)  # todo
    elif callable(flow_like):
        return FunctionFlow(func=cast(Callable[[Input], Output], flow_like))
    elif isinstance(flow_like, dict):
        return cast(Flow[Input, Output], flow_like)  # todo
        # return cast(Runnable[Input, Output], RunnableParallel(thing))
    else:
        raise TypeError(
            f"to_flow got an unsupported type: {type(flow_like)}"
        )
