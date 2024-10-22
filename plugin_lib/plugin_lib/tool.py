import inspect
import types
from functools import partial
from typing import Annotated, Any, Callable, _AnnotatedAlias  # type: ignore

from fastapi import APIRouter, Body, Header

from .utils import load_plugin_spec

router = APIRouter(prefix="/tools")


def get_header_params() -> list[str]:
    plugin_spec = load_plugin_spec()
    header_params = plugin_spec.tool_header
    return header_params


def transform_params(
    parameters: types.MappingProxyType[str, inspect.Parameter], header_params: list[str]
):
    new_params = []
    for name, param in parameters.items():
        if not isinstance(param.annotation, _AnnotatedAlias):
            raise TypeError(
                "All parameters must be Annotated, "
                f"Found '{name}' with type {param.annotation.__name__}"
            )
        param_type = Header if name in header_params else partial(Body, embed=True)
        new_params.append(
            inspect.Parameter(
                param.name,
                param.kind,
                default=param.default,
                annotation=Annotated[
                    param.annotation.__args__[0],
                    param_type(description=param.annotation.__metadata__[0]),
                ],
            )
        )
    return new_params


def tool(description: str):
    def decorator(func: Callable[..., Any]):
        sig = inspect.signature(func)
        parameters = sig.parameters
        header_params = get_header_params()
        new_params = transform_params(parameters, header_params)

        new_sig = sig.replace(parameters=new_params)

        def wrapper(*args: Any, **kwargs: Any):
            return func(*args, **kwargs)

        wrapper.__signature__ = new_sig

        router.post(
            f"/{func.__name__}",
            summary=func.__name__.replace("_", " ").title(),
            description=description,
        )(wrapper)
        return wrapper

    return decorator
