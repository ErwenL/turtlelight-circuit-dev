from ctypes.wintypes import WPARAM
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from typing import Optional, Union, Literal
import numpy as np
from siluxApi.phntwk.utils import get_model_config_meta_units, to_numpy, AnyArray, linear_fit, to_linear
from siluxApi.phntwk.core.transfer_matrix import TransferMatrixSegma
from ..waveguide import waveguide as wg

class DelayPairBaseModel(pa.DataFrameModel):
    wavelength: Series[float]
    delta_length: Optional[Series[float]]
    neff: Optional[Series[float]]
    ng: Optional[Series[float]]
    loss: Optional[Series[float]] 
    delta_insertion_loss: Optional[Series[float]] 
    delta_phase: Optional[Series[float]]
    delta_amplitude_coefficient: Optional[Series[complex]]
    delta_transmission: Optional[Series[float]]
    
    class Config:
        coerce = True
        metadata = get_model_config_meta_units(
            wavelength="um",
            neff=None,
            ng=None,
            loss="dB/um",
            delta_length="um",
            delta_insertion_loss="dB",
            delta_phase="rad",
            delta_amplitude_coefficient=None,
            delta_transmission=None,
        )

class DelayPairPropsModel(DelayPairBaseModel):
    delta_length: Series[float]
    delta_insertion_loss: Series[float]
    delta_phase: Series[float]
    delta_amplitude_coefficient: Series[complex]
    delta_transmission: Series[float]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(DelayPairBaseModel)


@pa.check_types(lazy=True)
def waveuguide_2_delaypair(df:DataFrame[wg.WgPropsModel]) -> DataFrame[DelayPairPropsModel]:
    _df = df.copy()
    _df.rename({
        "length": "delta_length",
        "insertion_loss": "delta_insertion_loss",
        "phase": "delta_phase",
        "amplitude_coefficient": "delta_amplitude_coefficient",
        "transmission": "delta_transmission"
    })
    return _df # type: ignore

@pa.check_types(lazy=True)
def swap_delayline(df: DataFrame[DelayPairPropsModel]) -> DataFrame[DelayPairPropsModel]:
    _df = df.copy()
    _df["delta_length"] = -_df["delta_length"]
    _df["delta_phase"] = -_df["delta_phase"]
    _df["delta_insertion_loss"] = -_df["delta_insertion_loss"]
    _df["delta_amplitude_coefficient"] = wg._props_2_amplitude(_df["delta_phase"], _df["delta_insertion_loss"])
    _df["delta_transmission"] = to_linear(_df["delta_insertion_loss"])
    return _df # type: ignore

@pa.check_types(lazy=True)
@pa.check_output(TransferMatrixSegma)
def props_2_trans_matrix(df: DataFrame[DelayPairPropsModel]):
    _df = pd.DataFrame()
    _df["wavelength"] = df["wavelength"]
    _df["t_1_1"] = df["amplitude_coefficient"]
    _df["t_2_2"] = complex(0)
    return _df

