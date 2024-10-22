from contextvars import ContextVar
from typing import Dict

from auto_flow.core.callbacks.run import Run

# type of [flow_id, Run | None]
var_run_cache = ContextVar[Dict[str, Run | None]]("var_run_cache", default={})
