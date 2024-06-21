import abc
from typing import Any

from core.flow.flow import Flow
from core.flow.flow_config import FlowConfig


class CallbackHandler(abc.ABC):
    @abc.abstractmethod
    def on_flow_start(self, flow: Flow, inp: Any, config: FlowConfig, **kwargs: Any) -> bool:
        ...

    @abc.abstractmethod
    def on_flow_end(self, output: Any) -> None:
        ...

    @abc.abstractmethod
    def on_flow_error(self, e: BaseException) -> None:
        ...
