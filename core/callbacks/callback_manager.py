from typing import List, Any
from core.callbacks.callback_handler import CallbackHandler
from core.callbacks.console_handler import ConsoleHandler
from core.callbacks.run import Run, current_flow, push_run_stack, current_run, pop_run_stack, is_run_stack_empty
from core.flow.flow import Flow
from core.logging import get_logger

logger = get_logger(__name__)


class CallbackManager(CallbackHandler):
    def __init__(self) -> None:
        self.handlers: List[CallbackHandler] = []

    def on_flow_start(self, flow: Flow, inp: Any) -> bool:
        if not is_run_stack_empty() and current_flow().id == flow.id:  # prevent re-enter stack
            logger.warning("Flow re-enter on_flow_start, please check and remove extra @trace.")
            return False

        run = Run(flow=flow, config=flow.effect_config.model_dump(), input=inp)
        push_run_stack(run)
        self.handler_event("on_flow_start", flow, inp=inp)
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
        verbose = current_flow().effect_config.verbose
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
    # todo
    cb = CallbackManager()
    cb.add_handler(ConsoleHandler())
    return cb


# global
callback_manager = init_callback_manager()
