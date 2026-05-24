import numpy as np
import pandera as pa
import pandas as pd
from typing import Callable, Union, Literal
from functools import wraps
from sklearn import linear_model
from sklearn.metrics import r2_score
from numpy.typing import ArrayLike

AnyArray = Union[pd.Series, np.ndarray]

def _convert_any_array_2_numpy(_array: AnyArray|list) -> np.ndarray:
    if isinstance(_array, np.ndarray):
        return _array
    if isinstance(_array, pd.Series):
        return _array.to_numpy()
    if isinstance(_array, list):
        return np.array(_array)
    if isinstance(_array, (float, int)):
        return np.array([_array])
    return _array

def to_numpy(func: Callable) -> Callable:
    """convert func input args, kwargs to np.ndarray
    - support pd.Series, list, float, int

    Args:
        func (Callable): wrapped function

    Returns:
        Callable: function with np.ndarray input args, kwargs
    """
    @wraps(func)
    def inner(*args, **kwargs):
        _args = [_convert_any_array_2_numpy(arg) for arg in args]
        _kwargs = {k: _convert_any_array_2_numpy(v) for k, v in kwargs.items()}
        return func(*_args, **_kwargs)
    return inner

@to_numpy
def to_power(amplitude: AnyArray):
    return (np.conj(amplitude) * amplitude).astype(float)

@to_numpy
def normalize(value: AnyArray, ref: AnyArray, mode:Literal["linear", "log"] = "linear"):
    if mode == "linear":
        return value / ref
    if mode == "log":
        return value - ref

@to_numpy
def to_linear(value: AnyArray):
    return 10 ** (value / 10)

@to_numpy
def to_db(value: AnyArray):
    return 10 * np.log10(value)

def linear_fit(x: np.ndarray, y: np.ndarray):
    """perform linear fit on x, y
    - y = slope @ x + intercept

    Args:
        x (np.ndarray): shape (n, 1)
        y (np.ndarray): shape (n, m)

    Returns:
        tuple: slope, intercept, r2
    """
    reg = linear_model.LinearRegression().fit(x, y)
    slope = reg.coef_
    intercept = reg.intercept_
    r2 = r2_score(y, reg.predict(x), multioutput="raw_values")
    return slope, intercept, r2