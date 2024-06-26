from contextvars import ContextVar
from typing import List, Any

from pydantic import BaseModel, Field

from core.callbacks.run import Run
from core.callbacks.trace import ENABLE_TRACE
from core.errors import RunStackError
from core.flow.config import FlowConfig
from core.flow.flow import Flow


class RunStack(BaseModel):
    stack: List[Run] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.stack) == 0

    def push(self, run: Run) -> None:
        self.stack.append(run)

    def pop(self) -> Run:
        if not self.is_empty():
            return self.stack.pop()
        raise RunStackError("pop from empty stack")

    def peek(self) -> Run:
        if not self.is_empty():
            return self.stack[-1]
        raise RunStackError("peek from empty stack")

    def size(self) -> int:
        return len(self.stack)


var_run_stack = ContextVar("var_run_stack", default=RunStack())


def push_run_stack(run: Run) -> None:
    run_stack = var_run_stack.get()
    if run_stack.is_empty():
        run_stack.push(run)
    else:
        assert run_stack.peek().flow is not run.flow, f"Flow has re-enter the stack: {run.flow}"
        parent_thread_id = run_stack.peek().thread_id
        if parent_thread_id == run.thread_id:
            run_stack.push(run)
        else:
            run_stack = RunStack(stack=run_stack.stack)  # Copy stack in new thread for parallel.
            run_stack.push(run)
            var_run_stack.set(run_stack)


def pop_run_stack() -> Run:
    return var_run_stack.get().pop()


def current_run_list() -> List[Run]:
    assert ENABLE_TRACE, "Please enable trace by set env 'FLOW_ENABLE_TRACE' and use @trace!"
    return var_run_stack.get().stack


def current_run() -> Run:
    assert ENABLE_TRACE, "Please enable trace by set env 'FLOW_ENABLE_TRACE' and use @trace!"
    return var_run_stack.get().peek()


def current_flow() -> Flow:
    return current_run().flow


def current_config() -> FlowConfig:
    return current_run().config


def is_run_stack_empty():
    return var_run_stack.get().is_empty()


def check_cur_flow(flow: Any):
    if current_flow() != flow:
        raise RunStackError(f"Run stack error, flow run in error stack:【{flow}】in【{current_flow()}】")
