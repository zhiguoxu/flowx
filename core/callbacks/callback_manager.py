from typing import List, Any
from core.callbacks.callback_handler import CallbackHandler
from core.callbacks.console_handler import ConsoleHandler
from core.callbacks.run import Run, current_flow, push_run_stack, current_run, pop_run_stack, is_run_stack_empty, \
    current_config
from core.flow.flow import Flow
from core.flow.flow_config import var_cur_config
from core.logging import get_logger

logger = get_logger(__name__)


class CallbackManager(CallbackHandler):
    def __init__(self) -> None:
        self.handlers: List[CallbackHandler] = []

    def on_flow_start(self, flow: Flow, inp: Any, **kwargs: Any) -> bool:
        if not is_run_stack_empty() and current_flow() is flow.id:  # prevent re-enter stack
            logger.warning(f"Flow re-enter on_flow_start, please check and remove extra @trace. Flow:【{flow}】")
            return False

        run = Run(flow=flow, input=inp, config=var_cur_config.get())
        push_run_stack(run)
        self.handler_event("on_flow_start", flow, inp=inp, **kwargs)
        return True

    def on_flow_end(self, output: Any) -> None:
        current_run().output = output
        self.handler_event("on_flow_end", output=output)
        pop_run_stack()

    def on_flow_error(self, e: BaseException) -> None:
        current_run().error = e
        self.handler_event("on_flow_error", e=e)
        pop_run_stack()

    def handler_event(self, event_name: str, *args, **kwargs) -> None:
        verbose = current_config().verbose
        for handler in self.handlers:
            if not verbose and isinstance(handler, ConsoleHandler):
                continue
            getattr(handler, event_name)(*args, **kwargs)

    def add_handler(self, handler: CallbackHandler) -> None:
        if handler not in self.handlers:
            self.handlers.append(handler)

    def remove_handler(self, handler: CallbackHandler) -> None:
        self.handlers.remove(handler)


def init_callback_manager():
    cb = CallbackManager()
    cb.add_handler(ConsoleHandler())
    # todo add send to server
    return cb


# global
callback_manager = init_callback_manager()
