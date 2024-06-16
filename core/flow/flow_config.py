from __future__ import annotations

from contextvars import ContextVar
from typing import Dict, List, Set

from pydantic import BaseModel, Field

from core.utils.utils import get_model_field_type


class FlowConfig(BaseModel):
    recursion_limit: int = Field(default=20)
    """Maximum number of times a call can recurse."""

    max_concurrency: int | None = Field(default=None)
    """Maximum number of parallel calls to make."""

    tags: List[str] = Field(default_factory=list)

    def merge(self, other: FlowConfig | Dict) -> FlowConfig:
        if isinstance(other, FlowConfig):
            other = other.model_dump(exclude_unset=True)

        data = self.model_dump(exclude_unset=True)
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
