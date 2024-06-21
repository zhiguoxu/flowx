import inspect
import os
from typing import Callable, List, Dict, Any, Type, TypeGuard, Iterator, AsyncIterator

from pydantic.main import Model


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


def filter_kwargs_by_pydantic(model_type: Type[Model],
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


def filter_kwargs_by_init_or_pydantic(model_type: Type[Model],
                                      kwargs: Dict[str, Any],
                                      exclude: set[str] | None = None,
                                      exclude_none: bool = False) -> Dict[str, Any]:
    kwargs = filter_kwargs_by_method(model_type.__init__, locals(), exclude=exclude, exclude_none=exclude_none)
    if len(kwargs) == 0:
        kwargs = filter_kwargs_by_pydantic(model_type, locals(), exclude=exclude, exclude_none=exclude_none)
    return kwargs


def get_model_field_type(model: Model | Type[Model], key: str):
    field_type = model.__annotations__.get(key)
    if not field_type:
        return None

    type_map = dict(List=list, Set=set, Tuple=tuple)
    if isinstance(field_type, str):
        # If caller use 'from __future__ import annotations', field_type's type will be str.
        for k, t in type_map.items():
            if field_type.startswith(k):
                return t

    elif hasattr(field_type, '__origin__'):
        return field_type.__origin__

    return field_type


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
