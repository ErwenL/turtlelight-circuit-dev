import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from typing import Optional, Union, Literal
import numpy as np
from siluxApi.phntwk.utils import get_model_config_meta_units, to_numpy, AnyArray, linear_fit, normalize, to_linear
from siluxApi.phntwk.core.transfer_matrix import TransferMatrixSegma, TMatrix

class DcBaseModel(pa.DataFrameModel):
    wavelength: Series[float]
    couple_length: Optional[Series[float]]
    kappa: Optional[Series[float]]
    theta: Optional[Series[float]]
    phase: Optional[Series[float]]
    couple_loss: Optional[Series[float]] 
    fanout_loss: Optional[Series[float]] 
    insertion_loss: Optional[Series[float]] 
    through_coefficient: Optional[Series[complex]]
    cross_coefficient: Optional[Series[complex]]
    through_transmission: Optional[Series[float]]
    cross_transmission: Optional[Series[float]]
    total_transmission: Optional[Series[float]] = pa.Field(le=1)

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(
            wavelength="um",
            couple_length="um",
            kappa="rad/um",
            theta="rad",
            phase="rad",
            couple_loss="dB/um",
            fanout_loss="dB",
            insertion_loss="dB",
            through_transmission=None,
            cross_transmission=None,
            total_transmission=None,
            ref_measure="dB",
            through_measure="dB",
            cross_measure="dB",
            r2_phase=None,
            r2_loss=None
        )
        

class DcParamsModel(DcBaseModel):
    kappa: Series[float]
    theta: Series[float]
    couple_loss: Optional[Series[float]]
    fanout_loss: Optional[Series[float]]
    r2_phase: Optional[Series[float]]
    r2_loss: Optional[Series[float]]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(DcBaseModel)
    
class DcPropsModel(DcBaseModel):
    phase: Series[float]
    insertion_loss: Series[float]
    through_coefficient: Series[complex]
    cross_coefficient: Series[complex]
    through_transmission: Series[float]
    cross_transmission: Series[float]
    total_transmission: Series[float]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(DcBaseModel)


class DcMeasureModel(DcBaseModel):
    wavelength: Series[float]
    couple_length: Series[float]
    ref_measure: Series[float]
    through_measure: Series[float]
    cross_measure: Series[float]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(DcBaseModel)


@pa.check_types(lazy=True)
def params_2_props(df: DataFrame[DcParamsModel], use_loss:bool = False) -> DataFrame[DcPropsModel]:
    phase = _kappa_theta_2_phase(df["kappa"], df["theta"], df["couple_length"])
    if use_loss:
        insertion_loss = _couple_fanout_2_insertion_loss(df["couple_loss"], df["fanout_loss"], df["couple_length"])
    else:
        insertion_loss = 0
    r, k = _props_2_coefficients(phase, insertion_loss)
    t_throught = _coefficient_2_transmission(r)
    t_cross = _coefficient_2_transmission(k)
    return pd.DataFrame({
        **df,
        "through_coefficient": r,
        "cross_coefficient": k,
        "phase": phase,
        "insertion_loss": insertion_loss,
        "through_transmission": t_throught,
        "cross_transmission": t_cross,
        "total_transmission": t_throught + t_cross
    }) # type: ignore

@pa.check_types(lazy=True)
def measure_2_props(df: DataFrame[DcMeasureModel]) -> DataFrame[DcPropsModel]:
    _df = df.copy()
    t = to_linear(normalize(_df["through_measure"], _df["ref_measure"], mode="log"))
    c = to_linear(normalize(_df["cross_measure"], _df["ref_measure"], mode="log"))
    r = _transmission_2_coefficient(t)
    k = _transmission_2_coefficient(c, phase=90)
    _df["phase"] = _coefficient_2_phase(r, k)
    _df["insertion_loss"] = 10 * np.log10(t + c)
    _df["through_coefficient"] = r
    _df["cross_coefficient"] = k
    _df["through_transmission"] = t
    _df["cross_transmission"] = c
    _df["total_transmission"] = t + c
    return _df # type: ignore


@pa.check_types(lazy=True)
def props_2_params(df: DataFrame[DcPropsModel], params: list[str]|None=None) -> DataFrame[DcParamsModel]:
    if params is None:
        params = []
    _validate_kappa_theta_fit(df) # type: ignore
    _df = df.copy()
    _df.sort_values(by= params + ["wavelength", "couple_length"], inplace=True)
    _df = fit_kappa_theta(_df) # type: ignore
    _df = fit_loss(_df)
    return _df # type: ignore

def fit_kappa_theta(df:DataFrame[DcPropsModel]) -> DataFrame[DcPropsModel]:
    """linear fit kappa and theta
    - assume df are sorted by params + wavelength + couple_length

    Args:
        df (DataFrame[DcPropsModel]): props df

    Returns:
        DataFrame[DcPropsModel]: add cols kappa, theta, r2_phase
    """
    couple_lengths = df["couple_length"].unique().reshape(-1, 1)
    num_of_couple_lengths = len(couple_lengths)
    phase = df["phase"].to_numpy().reshape(-1, num_of_couple_lengths).T
    kappa, theta, r2 = linear_fit(couple_lengths, phase)
    df["kappa"] = np.repeat(kappa, num_of_couple_lengths)
    df["theta"] = np.repeat(theta, num_of_couple_lengths)
    df["r2_phase"] = np.repeat(r2, num_of_couple_lengths)
    return df

def fit_loss(df: DataFrame[DcPropsModel]) -> DataFrame[DcPropsModel]:
    """linear fit loss

    Args:
        df (DataFrame[DcPropsModel]): props df

    Returns:
        DataFrame[DcPropsModel]: add cols loss, r2_loss
    """
    couple_lengths = df["couple_length"].unique().reshape(-1, 1)
    num_of_couple_lengths = len(couple_lengths)
    insertion_loss= df["insertion_loss"].to_numpy().reshape(-1, num_of_couple_lengths).T
    couple_loss, fanout_loss, r2 = linear_fit(couple_lengths, insertion_loss)
    df["couple_loss"] = np.repeat(couple_loss, num_of_couple_lengths)
    df["fanout_loss"] = np.repeat(fanout_loss, num_of_couple_lengths)
    df["r2_loss"] = np.repeat(r2, num_of_couple_lengths)
    return df


@pa.check_types(lazy=True)
def _validate_kappa_theta_fit(df: DataFrame[DcBaseModel]) -> None:
    num_of_couple_length = len(df["couple_length"].unique())
    assert len(df) % num_of_couple_length == 0, "The number of rows must be a multiple of the number of couple lengths."

@pa.check_types(lazy=True)
@pa.check_output(TransferMatrixSegma)
def props_2_trans_matrix(df: DataFrame[DcPropsModel]):
    return pd.DataFrame({
        "wavelength": df["wavelength"],
        "t_1_1": df["through_coefficient"],
        "t_1_2": df["cross_coefficient"],
        "t_2_1": df["cross_coefficient"],
        "t_2_2": df["through_coefficient"],
    })

def params_2_tmatrix(df: DataFrame[DcParamsModel], use_loss: bool = False) -> TMatrix:
    phase = _kappa_theta_2_phase(df["kappa"], df["theta"], df["couple_length"])
    if use_loss:
        insertion_loss = _couple_fanout_2_insertion_loss(df["couple_loss"], df["fanout_loss"], df["couple_length"])
    else:
        insertion_loss = 0
    r, k = _props_2_coefficients(phase, insertion_loss)
    return np.moveaxis(np.array([
        [r, k],
        [k, r]
    ], dtype=complex), [0, 1, 2], [1, 2, 0])

def props_2_ntwk(df: DataFrame[DcPropsModel]):...

@to_numpy
def _kappa_theta_2_phase(kappa: AnyArray, theta: AnyArray, couple_length: AnyArray):
    return kappa * couple_length + theta

@to_numpy
def _couple_fanout_2_insertion_loss(couple_loss: AnyArray, fanout_loss: AnyArray, coupler_length: AnyArray):
    return couple_loss * coupler_length + fanout_loss 

@to_numpy
def _props_2_coefficients(phase: AnyArray, insertion_loss: AnyArray):
    """return coefficients in the order of (through, cross)

    Args:
        phase (AnyArray): phase    
        insertion_loss (AnyArray): insertion_loss   

    Returns:
        tuple(through, cross): _description_
    """
    _amp = np.exp(1j * phase) * 10 ** (insertion_loss / 20)
    return np.real(_amp), np.imag(_amp) * 1j

@to_numpy
def _coefficient_2_transmission(coefficient: AnyArray):
    return np.abs(coefficient)**2

@to_numpy
def _transmission_2_coefficient(transmission: AnyArray, phase: float = 0):
    """convert transmission to coefficient

    Args:
        transmission (AnyArray): float
        phase (float, optional): phase in degree. Defaults to 0.

    Returns:
        np.ndarray[complex]: coefficient
    """
    return np.sqrt(transmission) * np.exp(1j * phase * np.pi / 180)

@to_numpy
def _coefficient_2_phase(through_coefficient: AnyArray, cross_coefficient: AnyArray):
    return np.angle(through_coefficient + cross_coefficient)

@to_numpy
def _transmission_2_phase(through_transmission: AnyArray, cross_transmission: AnyArray):
    return _coefficient_2_phase(
        _transmission_2_coefficient(through_transmission),
        _transmission_2_coefficient(cross_transmission, phase=90)
    )



if __name__ == "__main__":
    help(_props_2_coefficients)
    print()

    
