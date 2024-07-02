from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict

from pydantic import BaseModel, Field

from core.flow.config import FlowConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.flow.flow import Flow


class Run(BaseModel):
    start_time: float = Field(default_factory=time.time)
    end_time: float = 0
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    flow: Flow
    config: FlowConfig
    input: Any
    output: Any = None
    error: BaseException | None = None
    thread_id: int = Field(default_factory=threading.get_ident)
    extra_data: Dict[str, Any] = Field(default_factory=dict)

    def update_extra_data(self, **kwargs: Any):
        self.extra_data.update(kwargs)

    class Config:
        arbitrary_types_allowed = True
