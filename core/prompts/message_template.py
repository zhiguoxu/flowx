from __future__ import annotations

from string import Formatter
from typing import Dict, Tuple, Any, Union, List

from pydantic import Field

from core.flow.flow import Flow
from core.messages.chat_message import ChatMessage, Role


class MessageTemplate(Flow[Union[str, Dict], ChatMessage]):
    role: Role
    template: str
    partial_vars: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ai_message(cls, template: str):
        return cls(role=Role.ASSISTANT, template=template)

    @classmethod
    def user_message(cls, template: str):
        return cls(role=Role.USER, template=template)

    @classmethod
    def from_arg(cls, arg: str | List[str] | Tuple[str, str] | Dict[str, Any]):
        if isinstance(arg, str):
            return cls(role=Role.USER, template=arg)
        if isinstance(arg, dict):
            return cls(role=Role.from_name(arg["role"]), template=arg["content"])
        assert isinstance(arg, (list, tuple)) and len(arg) == 2
        return cls(role=Role.from_name(arg[0]), template=arg[1])

    def invoke(self, inp: str | Dict) -> ChatMessage:
        inp = validate_template_vars(inp, self.input_vars)
        return self.format(**inp)

    def partial_format(self, **kwargs: Any) -> MessageTemplate:
        ret = self.model_copy(deep=True)
        input_vars = self.input_vars
        kwargs = {k: v for k, v in kwargs.items() if k in input_vars}
        ret.partial_vars.update(kwargs)
        return ret

    def format(self, arg: str | None = None, **kwargs: Any) -> ChatMessage:
        if arg is not None:
            assert len(kwargs) == 0
            kwargs = validate_template_vars(arg, self.input_vars)
        return ChatMessage(role=self.role, content=self.template.format(**kwargs, **self.partial_vars))

    @property
    def input_vars(self) -> set[str]:
        return {var_name for _, var_name, _, _ in Formatter().parse(self.template) if var_name is not None}


def validate_template_vars(inp: str | Dict, template_vars: set[str]) -> Dict:
    if isinstance(inp, dict):
        return inp

    assert len(template_vars) == 1, "Expect str type input only when has only one template input var!"

    var_name = next(iter(template_vars))
    return {var_name: inp}
