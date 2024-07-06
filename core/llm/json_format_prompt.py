import json
from typing import Type

from pydantic import BaseModel

JSON_FORMAT_PROMPT = """You are a helpful assistant designed to output JSON"""

JSON_FORMAT_PROMPT_WITH_SCHEMA = """You are a helpful assistant designed to output JSON with schema:
```
{schema}
```"""


def get_json_format_prompt_by_schema(model_type: Type[BaseModel]):
    schema = {k: v for k, v in model_type.model_json_schema().items()}
    if "title" in schema:
        del schema["title"]
    if "type" in schema:
        del schema["type"]

    return JSON_FORMAT_PROMPT_WITH_SCHEMA.format(schema=json.dumps(schema))
