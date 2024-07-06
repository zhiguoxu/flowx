import json
from typing import Dict, Any, List, Type, Union, Iterator

from pydantic.main import BaseModel

from core.flow.flow import Flow
from core.messages.chat_message import ChatMessage, ChatMessageChunk
from core.utils.parse_json import parse_partial_json

Output = Union[BaseModel, List[BaseModel], Dict[str, Any], List[Dict[str, Any]]]


class MessagePydanticOutParser(Flow[ChatMessage, Output]):
    schemas: List[Type[BaseModel]]
    """A pydantic class describe the output data schema"""

    return_dict: bool = False
    """
    If True, return the the dict with data unverified,
    If False, return pydantic object of the input schema.
    """

    return_first: bool = False
    """
    If True, return the the first
    If False, return list.
    """

    partial_parse: bool = False
    """Whether can parse partial json which may be missing closing braces."""

    strict: bool = False
    """Whether to allow non-JSON-compliant strings."""

    def invoke(self, inp: ChatMessage) -> Output:
        # parse message.content
        if not inp.tool_calls:
            assert inp.content is not None
            output_dict = self.parse_json_str(inp.content)
            if self.return_dict:
                return output_dict if self.return_first else [output_dict]
            assert len(self.schemas) == 1
            model = self.schemas[0](**output_dict)
            return model if self.return_first else [model]

        # parse message.tool_calls
        objs = []
        for tool_call in inp.tool_calls:
            args_dict = self.parse_json_str(tool_call.function.arguments)
            if self.return_dict:
                objs.append(args_dict)
            else:
                for schema in self.schemas:
                    if schema.__name__ == tool_call.function.name:
                        objs.append(schema(**args_dict))  # type: ignore[arg-type]
        return objs[0] if self.return_first else objs

    def parse_json_str(self, json_str: str) -> Dict[str, Any]:
        if self.partial_parse:
            return parse_partial_json(json_str)
        return json.loads(json_str, strict=self.strict)


class MessageStrOutParser(Flow[ChatMessage, str]):
    def invoke(self, inp: ChatMessage) -> str:
        return inp.content or ""

    def transform(self, inp: Iterator[ChatMessageChunk]) -> Iterator[str]:  # type: ignore[override]
        for chunk in inp:
            yield chunk.content or ""
