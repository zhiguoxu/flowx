from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from string import Formatter
from typing import Dict, Tuple, Any, Union, List, TYPE_CHECKING, TypeVar

from pydantic import Field

from core.flow.flow import Flow
from core.logging import get_logger
from core.messages.chat_message import ChatMessage, Role

if TYPE_CHECKING:
    from core.prompts.chat_template import ChatTemplate

logger = get_logger(__name__)

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)
T = TypeVar("T")


class PromptTemplate(Flow[Union[Input, Dict], Output], ABC):
    @abstractmethod
    def partial_format(self: T, **kwargs: Any) -> T:
        ...

    @abstractmethod
    def format(self, arg: Input | None = None, **kwargs: Any) -> Output:
        ...


class StrTemplate(PromptTemplate[str, str]):
    partial_vars: Dict[str, Any] = Field(default_factory=dict)
    template: str

    def __init__(self, template: str):
        super().__init__(template=template)  # type: ignore[call-arg]

    def invoke(self, inp: str | Dict) -> str:
        inp = validate_template_vars(inp, self.input_vars)
        return self.format(**inp)

    def partial_format(self, **kwargs: Any) -> StrTemplate:
        ret = self.model_copy(deep=True)
        input_vars = self.input_vars
        kwargs = {k: v for k, v in kwargs.items() if k in input_vars}
        ret.partial_vars.update(kwargs)
        return ret

    def format(self, arg: str | None = None, **kwargs: Any) -> str:
        if arg is not None:
            assert len(kwargs) == 0
            kwargs = validate_template_vars(arg, self.input_vars)
        for k, v in itertools.chain(kwargs.items(), self.partial_vars.items()):
            if k in self.input_vars and not isinstance(v, (str, int, float)):
                logger.warning(f"Format str template with value of complex type with key = '{k}', value = {v}")

        return self.template.format(**kwargs, **self.partial_vars)

    @property
    def input_vars(self) -> set[str]:
        return {var_name for _, var_name, _, _ in Formatter().parse(self.template) if var_name is not None}


class MessageTemplate(PromptTemplate[str, ChatMessage]):
    role: Role
    template: StrTemplate

    @classmethod
    def ai_message(cls, template: str):
        return cls(role=Role.ASSISTANT, template=StrTemplate(template))

    @classmethod
    def user_message(cls, template: str):
        return cls(role=Role.USER, template=StrTemplate(template))

    @classmethod
    def from_arg(cls, arg: str | List[str] | Tuple[str, str] | Dict[str, Any]):
        if isinstance(arg, str):
            return cls(role=Role.USER, template=StrTemplate(arg))
        if isinstance(arg, dict):
            return cls(role=Role.from_name(arg["role"]), template=arg["content"])
        assert isinstance(arg, (list, tuple)) and len(arg) == 2
        return cls(role=Role.from_name(arg[0]), template=StrTemplate(arg[1]))

    def invoke(self, inp: str | Dict) -> ChatMessage:
        inp = validate_template_vars(inp, self.input_vars)
        return self.format(**inp)

    def partial_format(self, **kwargs: Any) -> MessageTemplate:
        ret = self.model_copy(deep=True)
        ret.template = ret.template.partial_format(**kwargs)
        return ret

    def format(self, arg: str | None = None, **kwargs: Any) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.template.format(arg, **kwargs))

    @property
    def input_vars(self) -> set[str]:
        return self.template.input_vars

    def __add__(self, other: Any) -> ChatTemplate:
        from core.prompts.chat_template import ChatTemplate
        return ChatTemplate(messages=[self]) + other


def validate_template_vars(inp: str | Dict, template_vars: set[str]) -> Dict:
    if isinstance(inp, dict):
        return inp

    assert len(template_vars) == 1, "Expect str type input only when has only one template input var!"

    var_name = next(iter(template_vars))
    return {var_name: inp}
