import itertools
from typing import TypeVar, Sequence, Type, Tuple, Iterator

from core.flow.config import var_local_config
from core.flow.flow import BindingFlowBase, Flowable

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

    def invoke(self, inp: Input) -> Output:
        first_error = None
        last_error = None
        for flow in itertools.chain([self.bound], self.fallbacks):
            if self.exception_key and last_error is not None:
                inp[self.exception_key] = last_error
            try:
                return flow.invoke(inp, var_local_config.get())
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
                last_error = e
        assert first_error is not None
        raise first_error

    def stream(self, inp: Input) -> Iterator[Output]:
        yield from self._stream_of_transform(inp, True)

    def transform(self, inp: Iterator[Input]) -> Iterator[Output]:
        yield from self._stream_of_transform(inp, False)

    def _stream_of_transform(self, inp: Input | Iterator[Input], is_stream: bool) -> Iterator[Output]:
        first_error = None
        last_error = None
        stream = None
        for flow in itertools.chain([self.bound], self.fallbacks):
            if self.exception_key and last_error is not None:
                inp[self.exception_key] = last_error
            try:
                if is_stream:
                    stream = flow.stream(inp, var_local_config.get())
                else:
                    stream = flow.transform(inp, var_local_config.get())
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
