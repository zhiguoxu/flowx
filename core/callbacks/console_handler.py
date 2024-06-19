from typing import Any

from core.callbacks.callback_handler import CallbackHandler
from core.callbacks.run import current_run_list
from core.flow.flow import Flow


class ConsoleHandler(CallbackHandler):
    # todo
    def on_flow_start(self, flow: Flow, inp: Any) -> bool:
        print(f"{get_breadcrumbs()}: on_flow_start: {inp}")
        return True

    def on_flow_end(self, output: Any) -> None:
        print(f"{get_breadcrumbs()}: on_flow_end: {output}")

    def on_flow_error(self, e: BaseException) -> None:
        print(f"{get_breadcrumbs()}: on_flow_error: {e}")


def get_breadcrumbs() -> str:
    # todo
    return ">".join([run.flow.name or str(type(run.flow))[8:-2] for run in current_run_list()])
