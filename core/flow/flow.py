from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from concurrent.futures import wait, FIRST_COMPLETED
from contextlib import contextmanager
from typing import TypeVar, Generic, Iterator, Callable, cast, Mapping, Any, Awaitable, AsyncIterator, Union, \
    List, Dict, Type, Tuple, TYPE_CHECKING, Sequence

from pydantic import BaseModel, Field, model_validator, GetCoreSchemaHandler, field_validator
from pydantic_core import core_schema
from typing_extensions import Self

from core.callbacks.listeners_callback import ListenersCallback
from core.callbacks.run import Run
from core.callbacks.run_cache import var_run_cache
from core.callbacks.trace import trace
from core.flow.addable_dict import AddableDict
from core.flow.config import FlowConfig, get_cur_config, var_local_config
from core.flow.utils import recurse_flow, ConfigurableField
from core.logging import get_logger
from core.utils.iter import safe_tee, merge_iterator
from core.utils.utils import filter_kwargs_by_pydantic, is_generator, is_async_generator, to_pydantic_obj_with_init, \
    NOT_GIVEN, NotGiven

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)
Other = TypeVar("Other")

logger = get_logger(__name__)

if TYPE_CHECKING:
    from core.tool import ToolLike
    from core.llm.llm import ToolChoice
    from core.flow.retry import RetryFlow
    from core.flow.fallbacks import FlowWithFallbacks


def empty_flow_context(arg: Callable[..., Output]) -> Callable[..., Output]:
    return arg


class Flowable(Generic[Input, Output], ABC):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        # https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.any_schema()

    @abstractmethod
    @empty_flow_context
    def invoke(self, inp: Input) -> Output:
        """
        Normally the inp contain all info we need, but if you bind extra kwargs by calling Flow.bind(k=v),
        the bound kwargs will pass to inner Flow.invoke, and inner Flow.invoke must accept it.
        The same applies to stream and transform method.
        """

    @abstractmethod
    @empty_flow_context
    def stream(self, inp: Input) -> Iterator[Output]:
        ...

    @abstractmethod
    @empty_flow_context
    def transform(self, inp: Iterator[Input]) -> Iterator[Output]:
        ...

    @abstractmethod
    def __or__(self, other: FlowLike[Output, Other]) -> Flowable[Input, Other]:
        ...

    @abstractmethod
    def __ror__(self, other: FlowLike[Other, Input]) -> Flowable[Other, Output]:
        ...

    @abstractmethod
    def pipe(self,
             *others: FlowLike[Any, Other],
             main: bool = False,  # is others[0] main step
             name: str | None = None) -> SequenceFlow[Input, Other]:
        ...

    @abstractmethod
    def bind(self, **kwargs: Any) -> Flowable[Input, Output]:
        ...

    @abstractmethod
    def bind_tools(self, tools: List[ToolLike], tool_choice: ToolChoice | None | NotGiven = NOT_GIVEN):
        ...

    @abstractmethod
    def with_config(self,
                    config: FlowConfig | None = None,
                    inheritable: bool = True,
                    **kwargs: Any) -> Flowable[Input, Output]:
        ...

    @abstractmethod
    def configurable_fields(self, **kwargs: str | ConfigurableField) -> Flowable[Input, Output]:
        """Configure particular flow fields at runtime."""

    @abstractmethod
    def with_configurable(self, **kwargs: Any) -> Flowable[Input, Output]:
        """Set the configurable arguments specified by configurable_fields.
        It is another way of pass arguments to invoke."""

    @abstractmethod
    def with_retry(self, *,
                   retry_if_exception_type: Type[BaseException] | Tuple[Type[BaseException], ...] = Exception,
                   wait_exponential_jitter: bool = True,
                   max_attempt: int = 3) -> Flowable[Input, Output]:
        """Add retry config to a flow."""

    @abstractmethod
    def with_fallbacks(self,
                       fallbacks: Sequence[Flow[Input, Output]], *,
                       exceptions_to_handle: Type[BaseException] | Tuple[Type[BaseException], ...] = Exception,
                       exception_key: str | None = None,
                       ) -> FlowWithFallbacks[Input, Output]:
        """Add fallbacks to a flow."""

    @abstractmethod
    def pick(self, keys: List[str]) -> Flowable[Input, Dict[str, Any]]:
        """Pick keys from the dict output of this flow."""

    @abstractmethod
    def drop(self, keys: List[str]) -> Flowable[Input, Dict[str, Any]]:
        """Drop keys of the dict output of this flow."""

    @abstractmethod
    def assign(self, **kwargs: FlowLike[Input, Any]) -> Flowable[Input, Dict[str, Any]]:
        """Assigns new fields to the dict output of this flow."""

    @abstractmethod
    def with_listeners(self,
                       on_start: Callable[[Run], None] | None = None,
                       on_end: Callable[[Run], None] | None = None,
                       on_error: Callable[[Run], None] | None = None) -> Flowable[Input, Output]:
        """Bind lifecycle listeners to a Flow."""

    @abstractmethod
    @contextmanager
    def get_run(self) -> Iterator[Run]:
        """"Prepare an empty run before flow start, then we can get the run out of running context, example:
        llm = OpenAILLM()
        with llm.get_run() as llm_run:
            llm.invoke("你好")
            print(llm_run.token_usage)
        """


class Flow(BaseModel, Flowable[Input, Output], ABC):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None
    class_type: Type = Flowable

    def __init_subclass__(cls, **kwargs):
        from core.context import flow_context
        super().__init_subclass__(**kwargs)

        # Replace stream and transform with default implement, if they are not override.
        # In this way, we can keep the stream and transform's interface to simple one input,
        # and the subclass's override versions don't have to include extra arguments if it doesn't need.
        if cls.stream == Flow.stream:
            cls.stream = cls._default_stream
        if cls.transform == Flow.transform:
            cls.transform = cls._default_transform

        cls.invoke = cast(Callable[..., Output], flow_context(cls.invoke))
        cls.stream = cast(Callable[..., Iterator[Output]], flow_context(cls.stream))
        cls.transform = cast(Callable[..., Iterator[Output]], flow_context(cls.transform))

    @model_validator(mode="after")
    def set_class_type(self) -> Self:
        self.class_type = self.__class__
        return self

    def stream(self, inp: Input) -> Iterator[Output]:
        raise NotImplemented

    def transform(self, inp: Iterator[Input]) -> Iterator[Output]:
        raise NotImplemented

    def _default_stream(self, inp: Input, **kwargs: Any) -> Iterator[Output]:
        if self.transform is Flow.transform:
            yield self.invoke(inp, **kwargs)
        else:
            yield from self.transform(iter([inp]), **kwargs)

    def _default_transform(self, inp: Iterator[Input], **kwargs: Any) -> Iterator[Output]:
        yield from self.stream(merge_iterator(inp), **kwargs)

    def __or__(self, other: FlowLike[Any, Other]) -> SequenceFlow[Input, Other]:
        return SequenceFlow(self, other)

    def __ror__(self, other: FlowLike[Other, Any]) -> SequenceFlow[Other, Output]:
        return to_flow(other).__or__(self)  # type: ignore[return-value]

    def pipe(self,
             *others: FlowLike[Any, Other],
             main: bool = False,  # is others[0] main step
             name: str | None = None) -> SequenceFlow[Input, Other]:
        other_steps, main_step, name = self._prepare_pipe(*others, main=main, name=name)
        return SequenceFlow(self, *other_steps, main_step=main_step, name=name)

    def _prepare_pipe(self,
                      *others: FlowLike[Any, Other],
                      main: bool,  # is others[0] main step
                      name: str | None = None
                      ) -> Tuple[List[Flowable], Flowable | None, str | None]:  # [steps, name, main_step]
        other_flows = [to_flow(other) for other in others]
        main_step: Flowable | None = None
        if main:
            first_other = other_flows[0]
            main_step = first_other
            if isinstance(first_other, SequenceFlow):
                main_step = first_other.main_step or first_other.steps[0]

        name = name or self.name
        other_steps = []
        for other in other_flows:
            if isinstance(other, SequenceFlow):
                name = name or other.name
                other_steps += other.steps
                if other.main_step is not None:
                    assert main_step is None or main_step is other.main_step, "Pipe with more than 1 main steps!"
                    main_step = other.main_step
            else:
                other_steps.append(other)
        return other_steps, main_step, name

    def bind(self, **kwargs: Any) -> BindingFlow[Input, Output]:
        return BindingFlow(bound=self, kwargs=kwargs)

    def bind_tools(self, tools: List[ToolLike], tool_choice: ToolChoice | None | NotGiven = NOT_GIVEN):
        from core.tool import to_tool
        kwargs = dict(tools=[to_tool(tool) for tool in tools])
        if tool_choice is not NOT_GIVEN:
            kwargs["tool_choice"] = tool_choice  # type: ignore[assignment]
        return self.bind(**kwargs)

    def with_config(self,
                    config: FlowConfig | None = None,
                    inheritable: bool = False,
                    **kwargs: Any) -> Flow[Input, Output]:
        if not config and not kwargs:
            return self

        config = (config or FlowConfig()).merge(kwargs)
        if inheritable:
            return BindingFlow(self, config=config)
        else:
            return BindingFlow(self, local_config=config)

    def configurable_fields(self, **kwargs: str | ConfigurableField) -> BindingFlow[Input, Output]:
        return BindingFlow(self, fields=kwargs)

    def with_configurable(self, **kwargs: Any) -> Flow[Input, Output]:
        return self.with_config(configurable=kwargs, inheritable=True)

    def with_retry(self, *,
                   retry_if_exception_type: Type[BaseException] | Tuple[Type[BaseException], ...] = Exception,
                   wait_exponential_jitter: bool = True,
                   max_attempt: int = 3
                   ) -> RetryFlow[Input, Output]:
        from core.flow.retry import RetryFlow
        kwargs = filter_kwargs_by_pydantic(RetryFlow, locals())
        return RetryFlow(bound=self, **kwargs)

    def with_fallbacks(self,
                       fallbacks: Sequence[Flow[Input, Output]], *,
                       exceptions_to_handle: Type[BaseException] | Tuple[Type[BaseException], ...] = Exception,
                       exception_key: str | None = None,
                       ) -> FlowWithFallbacks[Input, Output]:
        from core.flow.fallbacks import FlowWithFallbacks
        kwargs = filter_kwargs_by_pydantic(FlowWithFallbacks, locals())
        return FlowWithFallbacks(bound=self, **kwargs)

    def pick(self, keys: str | List[str]) -> SequenceFlow[Input, Dict[str, Any]]:
        keys = [keys] if isinstance(keys, str) else keys
        return self | PickFlow(keys=keys)

    def drop(self, keys: str | List[str]) -> SequenceFlow[Input, Dict[str, Any]]:
        keys = [keys] if isinstance(keys, str) else keys
        return self | PickFlow(drop_keys=keys)

    def assign(self, **kwargs: FlowLike[Input, Any]) -> Flow[Input, Dict[str, Any]]:
        return self | ParallelFlow(steps=kwargs, steps_without_key=[identity.drop(list(kwargs.keys()))])

    def with_listeners(self,
                       on_start: Callable[[Run], None] | None = None,
                       on_end: Callable[[Run], None] | None = None,
                       on_error: Callable[[Run], None] | None = None) -> Flow[Input, Output]:
        return self.with_config(callbacks=[ListenersCallback(on_start=on_start, on_end=on_end, on_error=on_error)])

    @contextmanager
    def get_run(self) -> Iterator[Run]:
        run = Run(flow=self, config=FlowConfig(), input=None)
        var_run_cache.get()[self.id] = run
        try:
            yield run
        finally:
            var_run_cache.get().pop(self.id)

    def _get_local_config(self, extra_local_config: Dict | FlowConfig | None = None) -> FlowConfig | None:
        local_config = var_local_config.get()
        if local_config is None:
            if extra_local_config is None:
                return None
            if isinstance(extra_local_config, FlowConfig):
                return extra_local_config
            return FlowConfig(**extra_local_config)  # type: ignore[arg-type]
        else:
            if extra_local_config is None:
                return local_config
            else:
                return local_config.merge(extra_local_config)


class FunctionFlow(Flow[Input, Output]):
    func: Callable[..., Output] | Callable[..., Iterator[Output]]

    def __init__(self,
                 func: Union[
                     Callable[[Input], Output],
                     Callable[[Input], Iterator[Output]]
                 ],
                 name: str | None = None):
        if not name and hasattr(func, "__name__") and func.__name__ != "<lambda>":
            name = func.__name__
        super(Flow, self).__init__(func=func, name=name)

    @trace
    def invoke(self, inp: Input, **kwargs: Any) -> Output:
        output = self.func(inp, **kwargs)
        if is_generator(self.func):
            assert isinstance(output, Iterator)
            output = cast(Output, merge_iterator(output))

        if isinstance(output, Flow):
            # check the recurse limitation
            with recurse_flow(self, inp):
                output = output.invoke(inp)

        return cast(Output, output)

    @trace
    def stream(self, inp: Input, **kwargs: Any) -> Iterator[Output]:
        output = self.func(inp, **kwargs)
        if is_generator(self.func):
            assert isinstance(output, Iterator)

            for o in output:
                if isinstance(o, Flow):
                    # check the recurse limitation
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
    steps: List[Flowable]
    main_step: Flowable | None = None
    """The flow that will accept bound kwargs. If it is None, the first step will accept bound kwargs."""

    def __init__(self, *steps: FlowLike[Any, Any], main_step: Flowable | None = None, name: str | None = None):
        flow_steps: List[Flow] = list(map(to_flow, steps))
        super(Flow, self).__init__(steps=flow_steps, main_step=main_step, name=name)

    @trace
    def invoke(self, inp: Input, **kwargs: Any) -> Output:
        for step in self.steps:
            if self._is_main_step(step):
                inp = step.invoke(inp, **kwargs)  # Kwargs are only bound to main_step or the first step.
            else:
                inp = step.invoke(inp)
        return cast(Output, inp)

    @trace
    def transform(self, inp: Iterator[Input], **kwargs: Any) -> Iterator[Output]:
        for step in self.steps:
            if self._is_main_step(step):
                inp = step.transform(inp, **kwargs)
            else:
                inp = step.transform(inp)
        yield from cast(Iterator[Output], inp)

    def __or__(self, other: FlowLike[Any, Other]) -> SequenceFlow[Input, Other]:
        if isinstance(other, SequenceFlow):
            return SequenceFlow(*self.steps, *other.steps, name=self.name or other.name)
        else:
            return SequenceFlow(*self.steps, other, name=self.name)

    def pipe(self,
             *others: FlowLike[Any, Other],
             main: bool = False,  # is others[0] main step
             name: str | None = None) -> SequenceFlow[Input, Other]:
        other_steps, main_step, name = self._prepare_pipe(*others, main=main, name=name or self.name)
        assert self.main_step is None or main_step is None, "Pipe with more than 1 main steps!"
        return SequenceFlow(*self.steps, *other_steps, main_step=main_step, name=name)

    def _is_main_step(self, step: Flowable):
        return (self.main_step is None and step is self.steps[0]) or (step is self.main_step)


class ParallelFlow(Flow[Input, Dict[str, Any]]):
    steps: Mapping[str, Flowable[Input, Any]]
    steps_without_key: List[Flowable[Input, Dict[str, Any]]]

    # Pydantic will create ValidationError when we initialize ParallelFlow using FunctionFlow as init params!
    # because FunctionFlow[Input, int] is not subclass of Flow[Input, Any] (Flow is subclass if BaseModel),
    # so we need to use FlowBase in steps's type definition.

    def __init__(self,
                 steps: Mapping[str, FlowLike[Input, Any]] | None = None,
                 steps_without_key: List[FlowLike[Input, Dict[str, Any]]] | None = None,
                 name: str | None = None,
                 **kwargs: FlowLike[Input, Any]):
        merged_steps = {**(steps or {}), **kwargs}
        steps = {k: to_flow(v) for k, v in merged_steps.items()}
        steps_without_key = [to_flow(step) for step in steps_without_key or []]
        super(Flow, self).__init__(steps=steps, steps_without_key=steps_without_key, name=name)

    @trace
    def invoke(self, inp: Input) -> Dict[str, Any]:
        from core.context import get_executor
        with get_executor() as executor:
            futures = [executor.submit(step.invoke, inp) for key, step in self.steps.items()]
            result = {
                key: future.result()
                for key, future in zip(self.steps.keys(), futures)
            }
            futures_without_key = [executor.submit(step.invoke, inp) for step in self.steps_without_key]
            for future in futures_without_key:
                result.update(**future.result())
            return result

    @trace
    def transform(self, inp: Iterator[Input]) -> Iterator[AddableDict]:
        from core.context import get_executor
        input_copies = list(safe_tee(inp, len(self.steps) + len(self.steps_without_key)))
        with get_executor() as executor:
            # Create the transform() generator for each step
            named_generators = [
                                   (name, step.transform(input_copies.pop()))
                                   for name, step in self.steps.items()
                               ] + [
                                   ("", step.transform(input_copies.pop()))
                                   for step in self.steps_without_key
                               ]
            # Start the first iteration of each generator
            futures = {
                executor.submit(next, generator): (step_name, generator)
                for step_name, generator in named_generators
            }
            # Yield chunks from each iterator as they become available,
            # and start the next iteration of that iterator after it yields a chunk.
            while futures:
                completed_futures, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    step_name, generator = futures.pop(future)
                    try:
                        if step_name:
                            yield AddableDict({step_name: future.result()})
                        else:
                            result = future.result()
                            if result:
                                yield AddableDict({**result})
                        futures[executor.submit(next, generator)] = (step_name, generator)
                    except StopIteration:
                        pass


class GeneratorFlow(Flow[Input, Output]):
    """Like FunctionFlow, but inner func accept Iterator"""

    generator: Callable[..., Iterator[Output]] | None = None
    a_generator: Callable[..., AsyncIterator[Output]] | None = None

    def __init__(self,
                 generator: Union[
                     Callable[..., Iterator[Output]],  # [Iterator[Input]]
                     Callable[..., AsyncIterator[Output]],  # [AsyncIterator[Input]]
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

    @trace
    def invoke(self, inp: Input, **kwargs: Any) -> Output:
        assert self.generator
        return merge_iterator(self.generator(iter([inp]), **kwargs))

    @trace
    def transform(self, inp: Iterator[Input], **kwargs: Any) -> Iterator[Output]:
        # todo support self.a_transform
        assert self.generator
        yield from self.generator(inp, **kwargs)


class BindingFlowBase(Flow[Input, Output], ABC):
    bound: Flowable[Input, Output]

    def invoke(self, inp: Input, **kwargs: Any) -> Output:
        # Transmit the local_config (share it with bound),
        # the local config may be set by its caller by caller.invoke(inp, local_config).
        # Also transmit the kwargs to inner bound.
        return self.bound.invoke(inp, var_local_config.get(), **kwargs)

    def stream(self, inp: Input, **kwargs: Any) -> Iterator[Output]:
        yield from self.bound.stream(inp, var_local_config.get(), **kwargs)

    def transform(self, inp: Iterator[Input], **kwargs) -> Iterator[Output]:
        yield from self.bound.transform(inp, var_local_config.get(), **kwargs)

    @contextmanager
    def get_run(self) -> Iterator[Run]:
        with self.bound.get_run() as run:
            yield run

    def __getattr__(self, name: str) -> Any:
        return getattr(self.bound, name)


class BindingFlow(BindingFlowBase[Input, Output]):
    kwargs: Dict[str, Any] = Field(default_factory=dict)  # local kwargs pass to bound
    config: FlowConfig | None = None  # inheritable config
    local_config: FlowConfig | None = None  # local config don't affect children flow
    fields: Dict[str, ConfigurableField] = Field(default_factory=dict)

    def __init__(self,
                 bound: Flowable[Input, Output],
                 kwargs: Dict[str, Any] | None = None,
                 config: FlowConfig | None = None,
                 local_config: FlowConfig | None = None,
                 fields: Mapping[str, str | ConfigurableField] | None = None):
        fields = fields or {}
        fields = {
            k: v if isinstance(v, ConfigurableField) else ConfigurableField(id=v)
            for k, v in fields.items()
        }

        if isinstance(bound, BindingFlow):
            kwargs = bound.kwargs | (kwargs or {})
            if bound.config:
                config = bound.config.patch(config or {})
            if bound.local_config:
                local_config = bound.local_config.patch(local_config or {})
            fields = bound.fields | (fields or {})
            bound = bound.bound

        # Set name by the bound flow that is not instance of BindingFlowBase.
        bound_for_name = bound
        while isinstance(bound_for_name, BindingFlowBase):
            bound_for_name = bound_for_name.bound
        name = cast(Flow, bound_for_name).name

        init_kwargs = filter_kwargs_by_pydantic(self, locals(), exclude_none=True)
        super().__init__(**init_kwargs)

    def invoke(self, inp: Input, **kwargs: Any) -> Output:
        return self._get_bound().invoke(inp, self._get_local_config(), **{**self.kwargs, **kwargs})
        # Not every invoke accept **kwargs, so if you bind kwargs, it must be accepted by inner flow.

    def stream(self, inp: Input, **kwargs: Any) -> Iterator[Output]:
        yield from self._get_bound().stream(inp, self._get_local_config(), **{**self.kwargs, **kwargs})

    def transform(self, inp: Iterator[Input], **kwargs: Any) -> Iterator[Output]:
        yield from self._get_bound().transform(inp, self._get_local_config(), **{**self.kwargs, **kwargs})

    def with_retry(self, **kwargs: Any) -> BindingFlow[Input, Output]:  # type: ignore[override]
        return BindingFlow(
            bound=self.bound.with_retry(**kwargs),
            kwargs=self.kwargs,
            config=self.config,
            local_config=self.local_config
        )

    def _get_local_config(self, extra_local_config: Dict | FlowConfig | None = None) -> FlowConfig | None:
        assert extra_local_config is None
        return super()._get_local_config(self.local_config)

    def _get_bound(self) -> Flowable[Input, Output]:
        configurable = get_cur_config().configurable
        update_fields = {}
        for k, field in self.fields.items():
            value = configurable.get(field.id, field.default)
            if value is not NOT_GIVEN:
                update_fields[k] = value

        assert isinstance(self.bound, Flow)

        if not update_fields:
            return self.bound

        if self.bound.__class__.__init__ is BaseModel.__init__:
            return self.bound.model_copy(update=update_fields, deep=True)

        return to_pydantic_obj_with_init(self.bound.__class__,
                                         {**self.bound.model_dump(exclude_unset=True), **update_fields})

    def __getattr__(self, name: str) -> Any:
        if name in self.kwargs:
            return self.kwargs[name]

        return getattr(self.bound, name)


class PickFlow(Flow[Dict[str, Any], Dict[str, Any]]):
    keys: List[str] = Field(default_factory=list)
    drop_keys: List[str] = Field(default_factory=list)

    def invoke(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        assert isinstance(inp, dict), "The input of PickFlow must be a dict."
        return self._pick(inp)

    def transform(self, inp: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for item in inp:
            yield AddableDict(self._pick(item))

    def _pick(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v
            for k, v in inp.items() if (not self.keys or k in self.keys) and k not in self.drop_keys
        }


class IdentityFlow(Flow[Other, Other]):
    def invoke(self, inp: Other) -> Other:
        return inp

    def transform(self, inp: Iterator[Other]) -> Iterator[Other]:
        yield from inp

    def assign(self, **kwargs: FlowLike[Input, Any]) -> ParallelFlow[Input]:
        return ParallelFlow(steps=kwargs, steps_without_key=[identity.drop(list(kwargs.keys()))])


class FixOutputFlow(Flow[Any, Output]):
    """Flow return fix output"""
    output: Output

    @trace
    def invoke(self, inp: Any) -> Output:
        return self.output


identity = IdentityFlow[Any]()

FlowLike_ = Union[
    Flowable[Input, Output],
    Callable[[Input], Output],
    Callable[[Input], Awaitable[Output]],
    Callable[[Iterator[Input]], Iterator[Output]],
    Callable[[AsyncIterator[Input]], AsyncIterator[Output]]
]

FlowLike = Union[
    FlowLike_[Input, Output],
    Mapping[str, FlowLike_[Input, Output]],
    Callable[..., Output],
    None, int, str, float
]


def to_flow(flow_like: FlowLike[Input, Output]) -> Flow[Input, Output]:
    if isinstance(flow_like, Flow):
        return flow_like
    elif is_generator(flow_like) or is_async_generator(flow_like):
        return GeneratorFlow(flow_like)
    elif callable(flow_like):
        return FunctionFlow(func=cast(Callable[[Input], Output], flow_like))
    elif isinstance(flow_like, dict):
        return cast(Flow[Input, Output], ParallelFlow(flow_like))
    elif flow_like is None or isinstance(flow_like, (int, str, float)):
        return FixOutputFlow(output=flow_like)  # type: ignore[arg-type]
    else:
        raise TypeError(f"to_flow got an unsupported type: {type(flow_like)}")
