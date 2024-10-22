import json
import os
from typing import Dict

from pydantic import BaseModel

from auto_flow.core.logging import get_logger

logger = get_logger(__name__)


class ModelConfig(BaseModel):
    context_window: int
    is_chat_model: bool
    support_tool_calls: bool


current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "model_configs.json")) as fp:
    model_configs: Dict[str, ModelConfig] = {}
    configs: Dict[str, dict] = json.load(fp)
    for model_name, config in configs.items():
        model_configs[model_name] = ModelConfig(**config)


def get_model_config(model: str) -> ModelConfig:
    if model_config := model_configs.get(model):
        return model_config

    logger.warning(f"Model {model} is not found in model_config.json")
    return model_configs["default"]
