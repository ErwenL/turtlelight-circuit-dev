import pandas as pd
import numpy as np
from numpy.typing import NDArray
import pandera as pa
from pandera.typing import DataFrame, Series

TransferMatrixSegma = pa.DataFrameSchema({
    "wavelength": pa.Column(
        float,
        coerce=True,
    ),
    r"t_[1-9]\d*_[1-9]\d*": pa.Column(
        complex,
        coerce=True,
        regex=True,
    )
})

TMatrix = NDArray[np.complexfloating]

def get_input_port(component: str) -> int:
    return int(component.split("_")[2])

def get_output_port(component: str) -> int:
    return int(component.split("_")[1])

def get_t_components(df: pd.DataFrame) -> list[str]:
    t_components = df.columns.to_list()
    t_components.remove("wavelength")
    return t_components

def get_num_of_input_port(df: pd.DataFrame) -> int:
    return max(get_input_port(component) for component in get_t_components(df))

def get_num_of_output_port(df: pd.DataFrame) -> int:
    return max(get_output_port(component) for component in get_t_components(df))

@pa.check_input(TransferMatrixSegma)
def to_matrix(df: pd.DataFrame) -> TMatrix:
    wavelength_num = len(df)
    num_of_input_port = get_num_of_input_port(df)
    num_of_output_port = get_num_of_output_port(df)
    matrix = np.zeros((wavelength_num, num_of_output_port, num_of_input_port), dtype=complex)
    for component in get_t_components(df):
        matrix[:, get_output_port(component) - 1, get_input_port(component) - 1] = df[component].to_numpy() 
    return matrix
