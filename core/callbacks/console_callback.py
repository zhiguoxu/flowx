from typing import Any, TYPE_CHECKING

from core.callbacks.callback_handler import CallbackHandler
from core.callbacks.run_stack import current_run_list, current_run
from core.flow.config import get_cur_config

if TYPE_CHECKING:
    from core.flow.flow import Flow


class ConsoleCallback(CallbackHandler):
    def on_flow_start(self, flow: "Flow", inp: Any, **kwargs: Any) -> bool:
        config = get_cur_config()
        print(f"{get_breadcrumbs()}: on_flow_start: {inp}"
              f"{f', {kwargs}' if kwargs else ''}"
              f"{f', {config.configurable}' if config.configurable else ''}")
        return True

    def on_flow_end(self, output: Any) -> None:
        extra_data = current_run().extra_data
        print(f"{get_breadcrumbs()}: on_flow_end: {output}{f', {extra_data}' if extra_data else ''}")

    def on_flow_error(self, e: BaseException) -> None:
        print(f"{get_breadcrumbs()}: on_flow_error: {e}")


def get_breadcrumbs() -> str:
    return ">".join([run.flow.name or str(type(run.flow))[8:-2] for run in current_run_list()])
