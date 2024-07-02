from contextvars import ContextVar
from typing import Dict

from core.callbacks.run import Run

# type of [flow_id, Run | None]
var_run_cache = ContextVar[Dict[str, Run | None]]("var_run_cache", default={})
