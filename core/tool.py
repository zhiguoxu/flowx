from __future__ import annotations

from inspect import signature
from typing import Callable, Type, Any, TypeVar, Generic, cast

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from core.flow.flow import Flow
from core.utils.utils import filter_kwargs_by_pydantic, is_pydantic_class

ToolOutput = TypeVar("ToolOutput")


class Tool(BaseModel, Generic[ToolOutput]):
    function: Callable[..., ToolOutput]
    args_schema: Type[BaseModel]
    return_direct: bool = False  # todo use openai's parallel_tool_calls

    def __init__(self,
                 function: Callable[..., ToolOutput],
                 args_schema: Type[BaseModel] | None = None,
                 name: str | None = None,
                 description: str | None = None,
                 return_direct: bool = False):
        args_schema = args_schema or create_schema_from_function(function)
        args_schema.__name__ = name or args_schema.__name__
        args_schema.__doc__ = description or function.__doc__ or args_schema.__doc__
        kwargs = filter_kwargs_by_pydantic(self, locals())
        super().__init__(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        return self.function(*args, **kwargs)

    @property
    def name(self):
        return self.args_schema.__name__


def tool(*args: str | Callable[..., ToolOutput] | Flow[Any, ToolOutput],
         args_schema: Type[BaseModel] | None = None,
         return_direct: bool = False) -> Tool[ToolOutput] | Callable[[Callable[..., ToolOutput]], Tool[ToolOutput]]:
    tool_name: str | None = None

    def make_tool(func: Callable[..., ToolOutput]) -> Tool[ToolOutput]:
        name = tool_name or func.__name__
        if func.__doc__ is None:
            raise ValueError(f"Function【{name}】must have a docstring as it's description.")

        return Tool(function=func,
                    args_schema=args_schema,
                    name=name,
                    return_direct=return_direct)

    if len(args) == 2 and isinstance(args[0], str):
        arg_1 = args[1]
        if isinstance(arg_1, Flow):
            def invoke_wrapper(*args_: Any, **kwargs: Any) -> ToolOutput:
                if args_:
                    assert len(args_) == 1 and len(kwargs) == 0
                    return arg_1.invoke(args_[0])
                return arg_1.invoke(kwargs)

            return make_tool(invoke_wrapper)
        assert callable(arg_1)
        return make_tool(arg_1)

    if len(args) == 1 and isinstance(args[0], str):
        tool_name = args[0]
        return make_tool

    if len(args) == 1 and callable(args[0]):
        return make_tool(args[0])

    if len(args) == 0:
        return make_tool

    raise TypeError(f"Arguments type error: {args}")


def create_schema_from_function(func: Callable[..., Any]) -> Type[BaseModel]:
    """Create schema from function."""

    fields = {}
    params = signature(func).parameters
    for param_name in params:
        param_type = params[param_name].annotation
        param_default = params[param_name].default

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            # Required field
            fields[param_name] = (param_type, FieldInfo())
        elif isinstance(param_default, FieldInfo):
            # Field with pydantic.Field as default value
            fields[param_name] = (param_type, param_default)
        else:
            fields[param_name] = (param_type, FieldInfo(default=param_default))

    return create_model(func.__name__, **fields)  # type: ignore[call-overload]


ToolLike = Tool | Callable | Type[BaseModel]


def to_tool(tool_like: ToolLike) -> Tool:
    if isinstance(tool_like, Tool):
        return tool_like

    if callable(tool_like):
        return cast(Tool, tool(tool_like))

    if is_pydantic_class(tool_like):
        def not_implemented(**kwargs):
            raise NotImplemented

        return Tool(args_schema=tool_like, function=not_implemented)
