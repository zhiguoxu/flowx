import builtins
import inspect
from typing import Callable, List, Dict, Any, Type

from pydantic.main import Model


def get_method_parameters(obj: Callable,
                          exclude: set[str] | None = None,
                          exclude_self_cls: bool = True) -> List[str]:
    params = [param for param in inspect.signature(obj).parameters]
    exclude = exclude or set()
    if exclude_self_cls:
        exclude.update({"self", "cls"})
    return list(filter(lambda item: item not in exclude, params) if exclude else params)


def filter_kwargs_by_method(obj: Callable,
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
    type_map = dict(List=list, Set=set, Tuple=tuple)
    if isinstance(field_type, str):
        # If caller use 'from __future__ import annotations', field_type's type will be str.
        for k, t in type_map.items():
            if field_type.startswith(k):
                return t

    elif hasattr(field_type, '__origin__'):
        return field_type.__origin__

    return field_type
