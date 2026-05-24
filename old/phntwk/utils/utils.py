import pandera as pa
from typing import Callable, TypedDict, Union, Any
from functools import wraps, reduce

class ModelConfigMetaUnits(TypedDict):
    units: dict[str, str]

def get_model_config_meta_units(*args:type[pa.DataFrameModel], **kwargs: str|None) -> ModelConfigMetaUnits:
    if len(args) > 1:
        raise ValueError("Only one positional argument is allowed.")
    if len(args) == 1 and not issubclass(args[0], pa.DataFrameModel):
        raise TypeError("The positional argument must be a DataFrameModel.")
    if (
         len(args) == 1 and 
         isinstance(metadata := args[0].Config.metadata, dict) and 
         "units" in metadata
    ):
        units = metadata["units"]
    else:
        units = {}
    return {
        "units": units | kwargs
    }

def chain_call(*funcs: Callable) -> Callable:
    @wraps(funcs[0])
    def inner(*args, **kwargs):
        return reduce(lambda x, f: f(x), funcs, *args, **kwargs)
    return inner

def chain_transform(init: Any, *funcs: Callable):
    return chain_call(*funcs)(init)

    