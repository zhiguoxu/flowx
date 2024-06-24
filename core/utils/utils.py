import inspect
import os
from typing import Callable, List, Dict, Any, Type, TypeGuard, Iterator, AsyncIterator, get_type_hints, \
    Mapping, Literal

from overrides import override
from pydantic.main import Model, BaseModel


def get_method_parameters(obj: Callable[..., Any],
                          exclude: set[str] | None = None,
                          exclude_self_cls: bool = True) -> List[str]:
    params = [param for param in inspect.signature(obj).parameters.keys()]
    exclude = exclude or set()
    if exclude_self_cls:
        exclude.update({"self", "cls"})
    return list(filter(lambda item: item not in exclude, params) if exclude else params)


def filter_kwargs_by_method(obj: Callable[..., Any],
                            kwargs: Dict[str, Any],
                            exclude: set[str] | None = None,
                            exclude_none: bool = False) -> Dict[str, Any]:
    params = get_method_parameters(obj)
    exclude = exclude or set()
    return {
        k: v for k, v in kwargs.items()
        if k in params and (k not in exclude) and (not exclude_none or v is not None)
    }


def filter_kwargs_by_pydantic(model_type: Type[Model] | Model,
                              kwargs: Dict[str, Any],
                              exclude: set[str] | None = None,
                              exclude_none: bool = False) -> Dict[str, Any]:
    params = model_type.__fields__.keys()  # type: ignore
    exclude = exclude or set()
    return {
        k: v
        for k, v in kwargs.items()
        if k in params and (k not in exclude) and (not exclude_none or v is not None)
    }


def filter_kwargs_by_init_or_pydantic(model_type: Type[Model] | Model,
                                      kwargs: Dict[str, Any],
                                      exclude: set[str] | None = None,
                                      exclude_none: bool = False) -> Dict[str, Any]:
    if not isinstance(model_type, type):
        model_type = model_type.__class__

    if model_type.__init__ is not BaseModel.__init__:
        return filter_kwargs_by_method(model_type.__init__, kwargs, exclude=exclude, exclude_none=exclude_none)

    return filter_kwargs_by_pydantic(model_type, kwargs, exclude=exclude, exclude_none=exclude_none)


def to_pydantic_obj(model_type: Type, obj: Any) -> Any:
    if hasattr(model_type, "__origin__") and hasattr(model_type, "__args__"):
        type_args = model_type.__args__
        model_type = model_type.__origin__

        if model_type in (list, tuple, set):
            return model_type(to_pydantic_obj(type_args[0], item) for item in obj)

        if model_type in (dict, Mapping):
            return {k: to_pydantic_obj(type_args[1], v) for k, v in obj.items()}

    # convert dict to BaseModel
    if ((isinstance(obj, dict) and "class_type" in obj) or
            (hasattr(model_type, "__mro__") and
             BaseModel in model_type.__mro__ and
             not isinstance(obj, BaseModel))):
        assert isinstance(obj, Dict)
        class_type = obj.get("class_type", model_type)
        fields_type_hints = get_type_hints(class_type)

        kwargs = {}
        for key, value in obj.items():
            if sub_type := fields_type_hints.get(key):
                value = to_pydantic_obj(sub_type, value)
            kwargs[key] = value

        init_kwargs = filter_kwargs_by_method(class_type.__init__, kwargs)
        if not init_kwargs:
            return class_type(**kwargs)

        # 1. init by __init__;
        pydantic_obj = class_type(**init_kwargs)
        # 2. set the rest fields.
        kwargs = {k: v for k, v in kwargs.items() if k not in init_kwargs}
        copy_if_unset(kwargs, pydantic_obj)
        return pydantic_obj

    return obj


def copy_if_unset(src: Dict, dst: Model | Dict, deep: bool = True) -> None:
    if isinstance(dst, dict):
        assert isinstance(src, dict)
        for key, value in src.items():
            if key not in dst or dst[key] is NotGiven:
                dst[key] = value
            elif deep:
                if isinstance(value, BaseModel):
                    value = value.model_dump()
                dst_value = dst[key]
                if isinstance(dst_value, BaseModel):
                    assert isinstance(value, dict)
                    copy_if_unset(value, dst_value)
                elif isinstance(dst_value, dict):
                    assert isinstance(value, dict)
                    copy_if_unset(value, dst_value)
    else:
        for key, value in src.items():
            if key not in dst.__pydantic_fields_set__:
                setattr(dst, key, value)
            elif deep:
                if isinstance(value, BaseModel):
                    value = value.model_dump()
                dst_value = getattr(dst, key)
                if isinstance(dst_value, BaseModel):
                    assert isinstance(value, dict)
                    copy_if_unset(value, dst_value)
                elif isinstance(dst_value, dict):
                    assert isinstance(value, dict)
                    copy_if_unset(value, dst_value)


def is_generator(func: Any) -> TypeGuard[Callable[..., Iterator]]:
    return (
            inspect.isgeneratorfunction(func)
            or (hasattr(func, "__call__") and inspect.isgeneratorfunction(func.__call__))
    )


def is_async_generator(func: Any) -> TypeGuard[Callable[..., AsyncIterator]]:
    return (
            inspect.isasyncgenfunction(func)
            or (hasattr(func, "__call__") and inspect.isasyncgenfunction(func.__call__))
    )


def accepts_input_var(func: Callable[..., Any], name: str) -> bool:
    try:
        return inspect.signature(func).parameters.get(name) is not None
    except ValueError:
        return False


def accepts_any_kwargs(func: Callable[..., Any]):
    arg_spec = inspect.getfullargspec(func)
    return arg_spec.varkw and arg_spec.annotations.get(arg_spec.varkw, Any) in (Any, "Any")


def accepts_config(func: Callable[..., Any]) -> bool:
    return accepts_input_var(func, "config") or accepts_any_kwargs(func)


def filter_config_by_method(kwargs: Dict[str, Any], func: Callable[..., Any]):
    if accepts_config(func):
        return kwargs

    return {k: v for k, v in kwargs.items() if k != "config"}


def env_is_set(env_var: str, default: bool | None = None) -> bool:
    if default is not None and env_var not in os.environ:
        return default

    return env_var in os.environ and os.environ[env_var] not in (
        "",
        "0",
        "false",
        "False",
    )


class NotGiven:
    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()
