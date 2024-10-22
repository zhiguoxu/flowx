import itertools
from typing import TypeVar, Sequence, Type, Tuple, Iterator, Any

from auto_flow.core.flow.config import var_local_config
from auto_flow.core.flow.flow import BindingFlowBase, Flowable

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)


class FlowWithFallbacks(BindingFlowBase[Input, Output]):
    fallbacks: Sequence[Flowable[Input, Output]]

    exceptions_to_handle: Type[BaseException] | Tuple[Type[BaseException], ...] = Exception
    """Exceptions on which fallbacks should be tried; others raise immediately."""

    exception_key: str | None = None
    """
    If it is specified, last handled exceptions will be passed to next fallbacks under this key in the input,
    and the base flow and its fallbacks must accept a dict as input.
    """

    def invoke(self, inp: Input, **kwargs: Any) -> Output:
        first_error = None
        last_error = None
        for flow in itertools.chain([self.bound], self.fallbacks):
            if self.exception_key and last_error is not None:
                inp[self.exception_key] = last_error
            try:
                return flow.invoke(inp, var_local_config.get(), **kwargs)
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
                last_error = e
        assert first_error is not None
        raise first_error

    def stream(self, inp: Input, **kwargs: Any) -> Iterator[Output]:
        yield from self._stream_of_transform(inp, True, **kwargs)

    def transform(self, inp: Iterator[Input], **kwargs: Any) -> Iterator[Output]:
        yield from self._stream_of_transform(inp, False, **kwargs)

    def _stream_of_transform(self, inp: Input | Iterator[Input], is_stream: bool, **kwargs: Any) -> Iterator[Output]:
        first_error = None
        last_error = None
        stream = None
        for flow in itertools.chain([self.bound], self.fallbacks):
            if self.exception_key and last_error is not None:
                inp[self.exception_key] = last_error
            try:
                if is_stream:
                    stream = flow.stream(inp, var_local_config.get(), **kwargs)
                else:
                    stream = flow.transform(inp, var_local_config.get(), **kwargs)
                yield next(stream)
                first_error = None
                break
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
                last_error = e

        if first_error is not None:
            raise first_error

        assert stream is not None
        yield from stream
