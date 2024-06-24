from __future__ import annotations

from contextvars import ContextVar
from typing import Dict, List, Any, Mapping, get_type_hints

from pydantic import BaseModel, Field


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
    The arguments bound to the fields of Flow obj,
    the【configurable】's visibility will be transmitted to sub flow with context var of FlowConfig,
    so it is somehow like global variables work in Flow scope.
    """

    def merge(self, *others: FlowConfig | Dict) -> FlowConfig:
        others_dict = [
            other.model_dump(exclude_unset=True)
            if isinstance(other, FlowConfig) else other
            for other in others
        ]

        data = self.model_dump(exclude_unset=True)
        for other in others_dict:
            for key, value in other.items():
                field_type = get_type_hints(self.__class__)[key]
                origin_type = field_type.__origin__ if hasattr(field_type, "__origin__") else None
                if origin_type in (list, tuple):
                    value = data.get(key, origin_type()) + value
                elif origin_type in (set, dict):
                    value = data.get(key, origin_type()) | value
                elif origin_type is Mapping:
                    value = dict(data.get(key, {})) | value
                data[key] = value

        return FlowConfig(**data)

    def patch(self, other: FlowConfig | Dict) -> FlowConfig:
        if isinstance(other, FlowConfig):
            other = other.model_dump(exclude_unset=True)

        return self.model_copy(update=other, deep=True)

    class Config:
        extra = "forbid"


var_flow_config = ContextVar("flow_config", default=FlowConfig())
