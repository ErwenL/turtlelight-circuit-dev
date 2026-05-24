import numpy as np
from ..core.transfer_matrix import TransferMatrixSegma
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from numpy.typing import NDArray
from numbers import Number
from typing import Literal

@pa.check_output(TransferMatrixSegma)
def loss(wavelength: NDArray, loss: float|int|NDArray):
    """return transfer matrix dataframe of a abstract loss stage

    Args:
        wavelength (NDArray): wavelength, unit: um
        loss (Number|NDArray): unit: dB/um

    Returns:
        Dataframe[TransferMatrixSegma]: 1x1 transfer matrix dataframe
    """
    if isinstance(loss, Number):
        assert loss <= 0, "The loss must be negative."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": 10 ** ( loss / 20 ) * np.ones_like(wavelength, dtype=complex),
        })
    else:
        assert isinstance(loss, np.ndarray), "loss must be a number or a numpy array."
        assert np.all(loss <= 0), "The loss must be negative."
        assert loss.shape == wavelength.shape, "The shape of loss must be the same as wavelength."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": 10 ** ( loss / 20 ),
        })


@pa.check_output(TransferMatrixSegma)
def splitter(wavelength: NDArray, ratio: float|NDArray = 0.5, type=Literal["y", "dc", "ac"] = "y"):
    """return transfer matrix dataframe of a abstract splitter stage

    Args:
        wavelength (NDArray): wavelength, unit: um
        ratio (Number|NDArray): ratio of the splitter, 0 < ratio < 1
        type (Literal["y", "dc", "ac"]): type of the splitter, "y" or "dc" or "ac"

    Returns:
        Dataframe[TransferMatrixSegma]: 2x2 transfer matrix dataframe
    """
    match type:
        case "y":
            t11 = np.sqrt(ratio)
            t12 = np.sqrt(1 - ratio)
        case "dc":
            t11 = np.sqrt(ratio)
            t12 = 1j * np.sqrt(1 - ratio)
        case "ac":
            t11 = np.sqrt(ratio)
            t12 = -np.sqrt(1 - ratio)
        case _:
            raise ValueError("type must be 'y' or 'dc' or 'ac'.")
    if isinstance(ratio, Number):
        assert 0 < ratio < 1, "The ratio must be between 0 and 1."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": t11 * np.ones_like(wavelength, dtype=complex),
            "t_2_1": t12 * np.ones_like(wavelength, dtype=complex),
        })
    else:
        assert isinstance(ratio, np.ndarray), "ratio must be a number or a numpy array."
        assert np.all(0 < ratio) and np.all(ratio < 1), "The ratio must be between 0 and 1."
        assert ratio.shape == wavelength.shape, "The shape of ratio must be the same as wavelength."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": t11.astype(complex),
            "t_2_1": t12.astype(complex),
        })

@pa.check_output(TransferMatrixSegma)
def coupler(wavelength: NDArray, ratio: float|NDArray = 0.5, type: Literal["dc", "ac"] = "dc"):
    """return transfer matrix dataframe of a abstract coupler stage
    - if "ac" type
        - input port 1: odd mode like
        - input port 2: even mode like

    Args:
        wavelength (NDArray): wavelength, unit um.
        ratio (Number|NDArray, optional): ratio of the coupler, 0 < ratio < 1. Defaults to 0.5.
        type (Literal[&quot;dc&quot;, &quot;ac&quot;], optional): coupler type. Defaults to "dc".

    Returns:
        Dataframe[TransferMatrixSegma]: 2x2 transfer matrix dataframe
    """
    match type:
        case "dc":
            t12 = 1j * np.sqrt(1 - ratio)
            t21 = 1j * np.sqrt(1 - ratio)
        case "ac":
            t12 = np.sqrt(1 - ratio)
            t21 = -np.sqrt(1 - ratio)
        case _:
            raise ValueError("type must be 'dc' or 'ac'.")
    t11 = np.sqrt(ratio)
    if isinstance(ratio, Number):
        assert 0 < ratio < 1, "The ratio must be between 0 and 1."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": t11 * np.ones_like(wavelength, dtype=complex),
            "t_1_2": t12 * np.ones_like(wavelength, dtype=complex),
            "t_2_1": t21 * np.ones_like(wavelength, dtype=complex),
            "t_2_2": t11 * np.ones_like(wavelength, dtype=complex),
        })
    else:
        assert isinstance(ratio, np.ndarray), "ratio must be a number or a numpy array."
        assert np.all(0 < ratio) and np.all(ratio < 1), "The ratio must be between 0 and 1."
        assert ratio.shape == wavelength.shape, "The shape of ratio must be the same as wavelength."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": t11.astype(complex),
            "t_1_2": t12.astype(complex),
            "t_2_1": t21.astype(complex),
            "t_2_2": t11.astype(complex),
        })




@pa.check_output(TransferMatrixSegma)
def dc_like_coupler(wavelength: NDArray, ratio: float|NDArray = 0.5):
    """return transfer matrix dataframe of a abstract dc-like coupler stage
        
    Args:
        wavelength (NDArray): wavelength, unit: um
        ratio (Number|NDArray): ratio of the coupler, 0 < ratio < 1
    
    Returns:
        Dataframe[TransferMatrixSegma]: 2x2 transfer matrix dataframe
    """
    if isinstance(ratio, Number):
        assert 0 < ratio < 1, "The ratio must be between 0 and 1."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": np.sqrt(ratio) * np.ones_like(wavelength, dtype=complex),
            "t_1_2": 1j * np.sqrt(1 - ratio) * np.ones_like(wavelength, dtype=complex),
            "t_2_1": 1j * np.sqrt(1 - ratio) * np.ones_like(wavelength, dtype=complex),
            "t_2_2": np.sqrt(ratio) * np.ones_like(wavelength, dtype=complex),
        })
    else:
        assert isinstance(ratio, np.ndarray), "ratio must be a number or a numpy array."
        assert np.all(0 < ratio) and np.all(ratio < 1), "The ratio must be between 0 and 1."
        assert ratio.shape == wavelength.shape, "The shape of ratio must be the same as wavelength."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": np.sqrt(ratio).astype(complex),
            "t_1_2": 1j * np.sqrt(1 - ratio).astype(complex),
            "t_2_1": 1j * np.sqrt(1 - ratio).astype(complex),
            "t_2_2": np.sqrt(ratio).astype(complex),
        })

@pa.check_output(TransferMatrixSegma)
def ac_like_coupler(wavelength: NDArray, ratio: float|NDArray = 0.5):
    """return transfer matrix dataframe of a abstract ac-like coupler stage
    - input port 1: odd mode like
    - input port 2: even mode like
        
    Args:
        wavelength (NDArray): wavelength, unit: um
        ratio (Number|NDArray): ratio of the coupler, 0 < ratio < 1
    
    Returns:
        Dataframe[TransferMatrixSegma]: 2x2 transfer matrix dataframe
    """
    if isinstance(ratio, Number):
        assert 0 < ratio < 1, "The ratio must be between 0 and 1."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": np.sqrt(ratio) * np.ones_like(wavelength, dtype=complex),
            "t_1_2": np.sqrt(1 - ratio) * np.ones_like(wavelength, dtype=complex),
            "t_2_1": -np.sqrt(1 - ratio) * np.ones_like(wavelength, dtype=complex),
            "t_2_2": np.sqrt(ratio) * np.ones_like(wavelength, dtype=complex),
        })
    else:
        assert isinstance(ratio, np.ndarray), "ratio must be a number or a numpy array."
        assert np.all(0 < ratio) and np.all(ratio < 1), "The ratio must be between 0 and 1."
        assert ratio.shape == wavelength.shape, "The shape of ratio must be the same as wavelength."
        return pd.DataFrame({
            "wavelength": wavelength,
            "t_1_1": np.sqrt(ratio).astype(complex),
            "t_1_2": np.sqrt(1 - ratio).astype(complex),
            "t_2_1": -np.sqrt(1 - ratio).astype(complex),
            "t_2_2": np.sqrt(ratio).astype(complex),
        })



