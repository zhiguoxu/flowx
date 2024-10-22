from typing import TypeVar, Type, Dict, Any, Iterator, Tuple

from pydantic import field_validator
from tenacity import stop_after_attempt, wait_exponential_jitter, retry_if_exception_type, Retrying

from auto_flow.core.flow.config import var_local_config
from auto_flow.core.flow.flow import BindingFlowBase, Flowable
from auto_flow.core.logging import get_logger

logger = get_logger(__name__)

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)


class RetryFlow(BindingFlowBase[Input, Output]):
    retry_if_exception_type: Type[BaseException] | Tuple[Type[BaseException], ...] = Exception
    """Retries if an exception has been raised of one or more types."""

    wait_exponential_jitter: bool = True
    """Whether to use wait strategy that applies exponential backoff and jitter."""

    max_attempt: int = 3
    """Stop when the previous attempt >= max_attempt."""

    @field_validator('bound')  # The decorator's order is important here!
    @classmethod
    def validate_bound(cls, bound: Flowable[Input, Output]) -> Flowable:
        if isinstance(bound, cls):
            bound = bound.bound
        return bound

    @property
    def retrying_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(reraise=True)
        if self.max_attempt:
            kwargs["stop"] = stop_after_attempt(self.max_attempt)
        if self.wait_exponential_jitter:
            kwargs["wait"] = wait_exponential_jitter()
        if self.retry_if_exception_type:
            kwargs["retry"] = retry_if_exception_type(self.retry_if_exception_type)

        return kwargs

    def invoke(self, inp: Input, **kwargs: Any) -> Output:
        for attempt in Retrying(**self.retrying_kwargs):
            with attempt:
                attempt_number = attempt.retry_state.attempt_number
                local_config = var_local_config.get()
                if attempt_number > 1:
                    local_config = self._get_local_config({"tags": [f"retry:attempt:{attempt_number}"]})
                result = self.bound.invoke(inp, local_config, **kwargs)
        return result

    def stream(self, inp: Input, **kwargs: Any) -> Iterator[Output]:
        logger.warning("RetryFlow doesn't work in stream.")
        yield from self.bound.stream(inp, var_local_config.get(), **kwargs)

    def transform(self, inp: Iterator[Input], **kwargs: Any) -> Iterator[Output]:
        logger.warning("RetryFlow doesn't work in transform.")
        yield from self.bound.transform(inp, var_local_config.get(), **kwargs)
