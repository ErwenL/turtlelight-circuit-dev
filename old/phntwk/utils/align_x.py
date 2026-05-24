import numpy as np
import pandas as pd
from typing import Any, TypedDict, Callable, Optional
from loguru import logger
from functools import partial
import scipy.interpolate as itp

class AlignXInfo(TypedDict):
    x: str
    value: np.ndarray
    num_of_x: int
    num_of_param_sets: int
    params: list[str]
    props: list[str]

AlignHandler = Callable[[pd.DataFrame], pd.DataFrame]

def align_x(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    x: str, 
    params1: list[str],
    params2: list[str]|None=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """### align x col of df2 to df1
    - df1[x] that are not overlaped with df2[x] will be dropped
    - df2 will be interpolated to df1[x] using cubic spline
    

    Args:
        df1 (pd.DataFrame): reference dataframe
        df2 (pd.DataFrame): dataframe to align
        x (str): x column name
        params1 (list[str]): params list of df1 that won't interpolate
        params2 (list[str] | None, optional): params list of df2 that won't be interpolated. Defaults to None. if None, params2 will be set to params1.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: aligned df1 and df2
    """
    if params2 is None:
        params2 = params1

    _df1, info1 = validate_x_uniformity(df1, x, params1)
    _df2, info2 = validate_x_uniformity(df2, x, params2)

    if is_x_indentical(info1, info2):
        return _df1, _df2

    handler1, handler2 = get_align_handlers(info1, info2)

    return handler1(_df1), handler2(_df2)

def align_2_x(
    df: pd.DataFrame,
    x: str,
    value: np.ndarray,
    params: Optional[list[str]] = None,
) -> pd.DataFrame:
    """align df to x

    Args:
        df (pd.DataFrame): dataframe to align
        x (str): x column name
        value (np.ndarray): x value to align
        params (list[str]): params list of df that won't interpolate

    Returns:
        pd.DataFrame: aligned df
    """
    if params is None:
        params = []
    _df, info = validate_x_uniformity(df, x, params)
    if len(value) == info["num_of_x"] and np.all(info["value"] == value):
        return _df
    handler = partial(align_df_2_x, info=info, _to=value)
    return handler(_df)


def validate_x_uniformity(df: pd.DataFrame, x: str, params: list[str]) -> tuple[pd.DataFrame, AlignXInfo]:
    """validate x align and sort_value by x

    Args:
        df (pd.DataFrame): dataframe
        x (str): x column to align
        params (list[str]): other params column names
    """
    assert x in df.columns, f"{x} not in df.columns"
    num_of_x = len(df[x].unique())
    assert len(df) % num_of_x == 0, f"{x} length is not the same for each param set"

    _df = df.sort_values(by=params + [x])
    x_stack = _df[x].to_numpy().reshape(-1, num_of_x)
    assert np.all(x_stack == x_stack[0]), f"{x} is not the same for each param set"
    return _df, {
        "x": x,
        "value": x_stack[0],
        "num_of_x": num_of_x,
        "num_of_param_sets": len(x_stack),
        "params": params,
        "props": df.columns.difference(params + [x]).tolist()
    }

def is_x_indentical(info1: AlignXInfo, info2: AlignXInfo) -> bool:
    if info1["num_of_x"] != info2["num_of_x"]:
        return False
    return np.all(info1["value"] == info2["value"]) # type: ignore

def get_align_handlers(info1: AlignXInfo, info2: AlignXInfo) -> tuple[AlignHandler, AlignHandler]:

    x1 = info1["value"]
    x2 = info2["value"]
    x1_overlap = np.logical_and(x1 >= x2[0], x1 <= x2[-1])
    x1_trim = x1[np.logical_not(x1_overlap)]

    def trim_df1(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(df[df[info1["x"]].isin(x1_trim)].index)

    if len(x1_trim) == 0:
        handler1 = lambda df: df
    else:
        logger.warning(f"{info1['x']}: {x1_trim} of df1 are not overlaped with df2. Trim df1.")
        handler1 = trim_df1

    hander2 = partial(align_df_2_x, info=info2, _to=x1[x1_overlap])
    return handler1, hander2


def align_df_2_x(df: pd.DataFrame, info: AlignXInfo, _to:np.ndarray) -> pd.DataFrame:
    new_x = np.tile(_to, info["num_of_param_sets"]).reshape(-1, 1)
    new_length = len(new_x)

    props = df[info["props"]].to_numpy().T.reshape(-1, info["num_of_x"])
    interp = itp.CubicSpline(info["value"], props, axis=1)
    new_props = interp(_to).reshape(-1, new_length).T
    assert new_props.shape[1] == len(info["props"]), f"num of new_props cols is not equal to num of props cols"

    new_data = [new_x, new_props]

    if len(info["params"]) > 0:
        params = df[info["params"]].drop_duplicates().to_numpy()
        new_params = params.repeat(len(_to), axis=0)
        new_data.insert(0, new_params)

    return pd.DataFrame(
        np.hstack(new_data), 
        columns=info["params"] + [info["x"]] + info["props"]
    )




    

        

