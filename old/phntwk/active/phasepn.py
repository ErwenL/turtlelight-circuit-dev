import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from typing import Optional, Union, Literal
import numpy as np
from siluxApi.phntwk.utils import get_model_config_meta_units, to_numpy, AnyArray, linear_fit, normalize, to_linear

class PhasepnParamsModel(pa.DataFrameModel):
    wavelength: Optional[Series[float]]
    group_index: Optional[Series[float]]
    bias: Series[float]
    vpil: Series[float]
    loss : Series[float]
    resistance: Series[float]
    capacitance: Series[float]

    class Config:
        metadata = get_model_config_meta_units(
            wavelength="um",
            group_index=None,
            bias="V",
            vpil="V*cm",
            loss="dB/cm",
            resistance="Ohm*um",
            capacitance="F/um"
        )