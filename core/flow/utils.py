from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

from core.flow.flow_config import var_flow_config
from core.utils.utils import NOT_GIVEN


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


class ConfigurableField(BaseModel):
    name: str
    description: str | None = None
    annotation: Any = None
    default: Any = NOT_GIVEN
