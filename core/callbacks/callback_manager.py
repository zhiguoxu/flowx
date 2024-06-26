import time
from typing import List, Any, TYPE_CHECKING
from core.callbacks.callback_handler import CallbackHandler
from core.callbacks.console_callback import ConsoleCallback
from core.callbacks.run import Run
from core.callbacks.run_stack import current_flow, push_run_stack, current_run, pop_run_stack, is_run_stack_empty, \
    current_config
from core.flow.config import get_cur_config
from core.logging import get_logger

if TYPE_CHECKING:
    from core.flow.flow import Flow

logger = get_logger(__name__)


class CallbackManager(CallbackHandler):
    def __init__(self) -> None:
        self.handlers: List[CallbackHandler] = []

    def on_flow_start(self, flow: "Flow", inp: Any, **kwargs: Any) -> bool:
        if not is_run_stack_empty() and current_flow() is flow.id:  # prevent re-enter stack
            logger.warning(f"Flow re-enter on_flow_start, please check and remove extra @trace. Flow:【{flow}】")
            return False

        run = Run(flow=flow, input=inp, extra_data=dict(kwargs), config=get_cur_config())
        push_run_stack(run)
        self.handler_event("on_flow_start", flow, inp=inp, **kwargs)
        return True

    def on_flow_end(self, output: Any) -> None:
        run = current_run()
        run.output = output
        run.end_time = time.time()
        self.handler_event("on_flow_end", output=output)
        pop_run_stack()

    def on_flow_error(self, e: BaseException) -> None:
        run = current_run()
        run.error = e
        run.end_time = time.time()
        self.handler_event("on_flow_error", e=e)
        pop_run_stack()

    def handler_event(self, event_name: str, *args, **kwargs) -> None:
        verbose = current_config().verbose
        for handler in self.handlers + get_cur_config().callbacks:
            if not verbose and isinstance(handler, ConsoleCallback):
                continue
            getattr(handler, event_name)(*args, **kwargs)

    def add_handler(self, handler: CallbackHandler) -> None:
        if handler not in self.handlers:
            self.handlers.append(handler)

    def remove_handler(self, handler: CallbackHandler) -> None:
        self.handlers.remove(handler)


def init_callback_manager():
    cb = CallbackManager()
    cb.add_handler(ConsoleCallback())
    # todo add send to server
    return cb


# global
callback_manager = init_callback_manager()
