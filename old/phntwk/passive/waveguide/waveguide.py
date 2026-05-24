import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from typing import Optional, Union, Literal
import numpy as np
from siluxApi.phntwk.utils import get_model_config_meta_units, to_numpy, AnyArray, linear_fit, normalize, to_linear
from siluxApi.phntwk.core.transfer_matrix import TransferMatrixSegma, TMatrix

class WgBaseModel(pa.DataFrameModel):
    wavelength: Series[float]
    neff: Optional[Series[float]]
    ng: Optional[Series[float]]
    loss: Optional[Series[float]] 
    length: Optional[Series[float]]
    insertion_loss: Optional[Series[float]] 
    phase: Optional[Series[float]]
    amplitude_coefficient: Optional[Series[complex]]
    transmission: Optional[Series[float]]
    
    class Config:
        coerce = True
        metadata = get_model_config_meta_units(
            wavelength="um",
            length="um",
            neff=None,
            ng=None,
            loss="dB/um",
            insertion_loss="dB",
            phase="rad",
            amplitude_coefficient=None,
            transmission=None,
        )

class WgParamsModel(WgBaseModel):
    length: Optional[Series[float]]
    neff: Series[float]
    ng: Optional[Series[float]]
    loss: Optional[Series[float]]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(WgBaseModel)

class WgPropsModel(WgBaseModel):
    length: Series[float]
    insertion_loss: Series[float]
    phase: Series[float]
    amplitude_coefficient: Series[complex]
    transmission: Series[float]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(WgBaseModel)

@pa.check_types(lazy=True)
def params_2_props(df: DataFrame[WgParamsModel], use_loss:bool = True) -> DataFrame[WgPropsModel]:
    _phase = _neff_2_phase(df["neff"], df["length"], df["wavelength"])
    if use_loss:
        _insertion_loss = _loss_2_insertion_loss(df["loss"], df["length"])
    else:
        _insertion_loss = np.zeros_like(_phase)
    return pd.DataFrame({
        **df,
        "insertion_loss": _insertion_loss,
        "phase": _phase,
        "amplitude_coefficient": _props_2_amplitude(_phase, _insertion_loss),
        "transmission": to_linear(_insertion_loss),
    }) # type: ignore


@pa.check_types(lazy=True)
@pa.check_output(TransferMatrixSegma)
def props_2_trans_matrix(df: DataFrame[WgPropsModel]):
    _df = pd.DataFrame({
        "wavelength": df["wavelength"],
        "t_1_1": df["amplitude_coefficient"],
    })
    return _df

def params_2_tmatrix(df: DataFrame[WgParamsModel], use_loss:bool = True) -> TMatrix:
    _phase = _neff_2_phase(df["neff"], df["length"], df["wavelength"])
    if use_loss:
        _insertion_loss = _loss_2_insertion_loss(df["loss"], df["length"])
    else:
        _insertion_loss = 0
    _amplitude = _props_2_amplitude(_phase, _insertion_loss)
    return np.moveaxis(np.array([[_amplitude]], dtype = complex), [0, 1, 2], [1, 2, 0])


@to_numpy
def _loss_2_insertion_loss(loss: AnyArray, length: AnyArray):
    return loss * length

@to_numpy
def _neff_2_phase(neff: AnyArray, length: AnyArray, wavelength: AnyArray):
    return 2 * np.pi * neff * length / wavelength

@to_numpy
def _props_2_amplitude(phase: AnyArray, insertion_loss: AnyArray):
    return np.exp(1j * phase) * np.exp(insertion_loss / 20)