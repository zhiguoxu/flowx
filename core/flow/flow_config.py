from __future__ import annotations

from contextvars import ContextVar
from typing import Dict, List, Any

from pydantic import BaseModel, Field

from core.utils.utils import get_model_field_type


class FlowConfig(BaseModel):
    recursion_limit: int = Field(default=20)
    """Maximum number of times a call can recurse. (must inheritable)"""

    max_concurrency: int | None = Field(default=None)  # can not be local
    """Maximum number of parallel calls to make. (must inheritable)"""

    tags: List[str] = Field(default_factory=list)

    verbose: bool = False
    """If you want to use verbose, make sure enable trace by setting environ var【FLOW_ENABLE_TRACE】"""

    configurable: Dict[str, Any] = Field(default_factory=dict)
    """
    Bound arguments to the fields of Flow obj,
    the【configurable】's visibility will be transmitted to sub flow with context var of FlowConfig,
    so it is somehow like global variables work in Flow scope.
    """

    def merge(self, *others: FlowConfig | Dict) -> FlowConfig:
        others_dict = [other.model_dump(exclude_unset=True)
                       if isinstance(other, FlowConfig) else other
                       for other in others]

        data = self.model_dump(exclude_unset=True)
        for other in others_dict:
            for key, value in other.items():
                field_type = get_model_field_type(self, key)
                if field_type is list:
                    value = (data.get(key) or []) + value
                elif field_type is set:
                    value = (data.get(key) or set()) | value
                data[key] = value

        return FlowConfig(**data)

    def patch(self, other: FlowConfig | Dict) -> FlowConfig:
        if isinstance(other, FlowConfig):
            other = other.model_dump(exclude_unset=True)

        return self.model_copy(update=other, deep=True)


var_flow_config = ContextVar("flow_config", default=FlowConfig())
