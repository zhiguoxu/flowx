from contextlib import contextmanager
from typing import Any

from core.flow.flow_config import var_flow_config


@contextmanager
def recurse_flow(flow: Any, inp: Any):
    config = var_flow_config.get()
    if config.recursion_limit <= 0:
        raise RecursionError(
            f"Recursion limit reached when invoking {flow} with input {inp}."
        )
    config.recursion_limit -= 1
    yield
    config.recursion_limit += 1
