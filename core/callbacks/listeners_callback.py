from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel

from core.callbacks.callback_handler import CallbackHandler
from core.callbacks.run import Run

if TYPE_CHECKING:
    from core.flow.flow import Flow


class ListenersCallback(BaseModel, CallbackHandler):
    on_start: Callable[[Run], None] | None = None
    on_end: Callable[[Run], None] | None = None
    on_error: Callable[[Run], None] | None = None

    def on_flow_start(self, flow: "Flow", inp: Any, **kwargs: Any) -> bool:
        from core.callbacks.run_stack import current_run
        if self.on_start:
            self.on_start(current_run())
        return True

    def on_flow_end(self, output: Any) -> None:
        from core.callbacks.run_stack import current_run
        if self.on_end:
            self.on_end(current_run())

    def on_flow_error(self, e: BaseException) -> None:
        from core.callbacks.run_stack import current_run
        if self.on_error:
            self.on_error(current_run())
