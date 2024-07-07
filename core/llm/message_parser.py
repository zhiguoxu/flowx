import json
from json import JSONDecodeError
from typing import Dict, Any, List, Type, Union, Iterator

import jsonpatch  # type: ignore[import-untyped]
from pydantic.main import BaseModel

from core.flow.flow import Flow
from core.messages.chat_message import ChatMessage, ChatMessageChunk
from core.utils.parse_json import parse_partial_json
from core.utils.utils import add

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

    def transform(self, inp: Iterator[ChatMessageChunk]) -> Iterator[Dict[str, Any]]:  # type: ignore[override]
        assert self.return_dict, "Message output transform parser only support dict output."
        assert self.return_first, "Message output transform parser only support return first"

        def get_str_stream() -> Iterator[str]:
            for message_chunk in inp:
                if not message_chunk.tool_calls:
                    # parse message.content
                    if message_chunk.content:
                        yield message_chunk.content
                else:
                    # parse message.tool_calls
                    for tool_call in message_chunk.tool_calls:
                        if tool_call.function and tool_call.function.arguments:
                            yield tool_call.function.arguments

        yield from json_transform_parser(get_str_stream(), self.strict)

    def parse_json_str(self, json_str: str) -> Dict[str, Any]:
        if self.partial_parse:
            return parse_partial_json(json_str)
        return json.loads(json_str, strict=self.strict)


class MessageStrOutParser(Flow[ChatMessage, str]):
    def invoke(self, inp: ChatMessage) -> str:
        return inp.content or ""

    def transform(self, inp: Iterator[ChatMessageChunk]) -> Iterator[str]:  # type: ignore[override]
        for chunk in inp:
            if chunk.content:
                yield chunk.content


def json_transform_parser(inp: Iterator[str], strict: bool = False) -> Iterator[Dict[str, Any]]:
    prev_parsed = None
    acc = None
    for chunk in inp:
        acc = add(acc, chunk)
        assert isinstance(acc, str)
        if not acc.startswith("{"):
            index = acc.find("{")
            if index < 0:
                continue
            acc = acc[index:]
        try:
            obj, end = json.JSONDecoder(strict=strict).raw_decode(acc)
            yield jsonpatch.make_patch(prev_parsed, obj).patch
            acc = acc[end:]
            prev_parsed = None
        except JSONDecodeError as e:
            parsed = parse_partial_json(acc, strict)
            if parsed is not None and parsed != prev_parsed:
                yield jsonpatch.make_patch(prev_parsed, parsed).patch
            prev_parsed = parsed
