from typing import Callable, Any, Optional, Literal
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from scipy import constants
from ....utils import (
    get_model_config_meta_units, to_db, to_linear
)
from ....utils.align_x import (
    validate_x_uniformity
)
from pydantic import BaseModel
from functools import cache, wraps

class Spectrum(pa.DataFrameModel):
    wavelength: Series[float]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(
            wavelength="um",
        )

class MziSpectrum(pa.DataFrameModel):
    wavelength: Series[float]
    # dc_component: Optional[Series[float]]
    # ac_component: Optional[Series[float]]
    # normalized_ac_component: Series[float]
    # top_envlope: Series[float]
    # bottom_envlope: Series[float]
    # residue: Series[float]
    transmission: Series[float]
    extinction_ratio: Series[float]

@pa.check_types(lazy=True)
def analyze_spectra_amplitude(
        df: DataFrame[Spectrum], 
        spectrum_name: str,
        params: list[str], 
        find_peaks_kwargs: dict[str, Any] = {},
        **kwargs
    ) -> DataFrame[MziSpectrum]:
    _df, x_info = validate_x_uniformity(df, "wavelength", params)
    _x = x_info["value"]
    _y = _df[spectrum_name].to_numpy().reshape(x_info["num_of_param_sets"], -1).T

    # _y_top_envelope, _y_bottom_envelope = _get_envelopes_by_signal_filter(_x, _y, **find_peaks_kwargs)
    _y_top_envelope, _y_bottom_envelope = _get_envelopes_by_log_curve_fit(_x, _y, **find_peaks_kwargs)

    _transmission = _y / _y_top_envelope
    _er = _y_bottom_envelope / _y_top_envelope

    _df["transmission"] = _transmission.T.reshape(-1)
    _df["extinction_ratio"] = _er.T.reshape(-1)

    return _df # type: ignore

def _get_envelopes_by_signal_filter(_x:NDArray, _y: NDArray, **kwargs):
    _filters = analyze_fft_spectra(
        *fft_spectra(_x, _y), 
        **kwargs
    )
    _y_dc = _filters.get_dc(_y)
    _y_1st_envelope = _filters.get_1st_envelope(_y)
    _y_top_envelope = _y_dc + _y_1st_envelope
    _y_bottom_envelope = _y_dc - _y_1st_envelope
    return _y_top_envelope, _y_bottom_envelope

def _get_envelopes_by_log_curve_fit(_x:NDArray, _y: NDArray, **kwargs):
    from scipy.signal import find_peaks
    from scipy.interpolate import CubicSpline
    _y_db = to_db(_y)
    _y_top_envelope, _y_bottom_envelope = fit_spectra_envelopes(_x, _y_db, **kwargs)
    return to_linear(_y_top_envelope), to_linear(_y_bottom_envelope)


def fft_spectra(x: NDArray, y: NDArray):
    """return fft of spectra

    Args:
        x (NDArray): shape (N, )
        y (NDArray): shape (N, M)

    Returns:
        tuple[NDArray, NDArray]: x_fft, y_fft
    """
    assert _validate_uniform_sampling(x), "x is not uniformly sampled"
    from scipy.fft import fft, fftfreq
    dx = x[1] - x[0]
    sample_num = len(y)
    fft_sample_num = sample_num //2
    y_fft = fft(y, axis=0)/ fft_sample_num # type: ignore
    x_fft = fftfreq(sample_num, dx)
    return x_fft, y_fft


def _validate_uniform_sampling(x: NDArray):
    """validate if x is uniformly sampled
    """
    dx = np.diff(x)
    return np.allclose(dx, dx[0])

class FftSpectrumAnalysis(BaseModel):
    get_dc: Callable[[NDArray], NDArray] # use object for unknown
    get_1st: Callable[[NDArray], NDArray]
    get_1st_envelope: Callable[[NDArray], NDArray]
    get_residue: Callable[[NDArray], NDArray]

def analyze_fft_spectra(xf: NDArray, yf: NDArray, **kwargs):
    assert len(xf) == yf.shape[0], "xf length should equal to first axis of yf"
    
    spectra_analysis = [
        analyze_fft_sprectrum(xf, _yf, **kwargs) for _yf in yf.T
    ]

    def analysis_helper(analysis: str):
        def get_analysis(ys: NDArray) -> NDArray:
            _ys = [
                getattr(spectrum_analysis, analysis)(_y) for _y, spectrum_analysis in zip(ys.T, spectra_analysis) 
            ]
            return np.vstack(_ys).T
        return get_analysis

    return FftSpectrumAnalysis(
        get_dc = analysis_helper("get_dc"),
        get_1st = analysis_helper("get_1st"),
        get_residue = analysis_helper("get_residue"),
        get_1st_envelope = analysis_helper("get_1st_envelope")
    )

def analyze_fft_sprectrum(xf: NDArray, yf: NDArray, **kwargs):
    assert len(yf.shape) == 1, "yf has to be 1D array"
    assert len(xf) == len(yf), "xf and yf has to be equal in length"
    from scipy.signal import find_peaks, butter, filtfilt, hilbert

    sample_num = len(xf)
    peaks, _ = find_peaks(np.abs(yf)[:sample_num//2], **kwargs)
    assert len(peaks) > 0, "no peak was found"
    first_peak = peaks[0]
    first_peak_xf = xf[first_peak]
    fs = xf[1] * sample_num

    def _get_dc(_y: NDArray) -> NDArray:
        b, a = butter(4, first_peak_xf/2, btype="lowpass", fs=fs)
        return filtfilt(b, a, _y)

    def _get_1st(_y: NDArray) -> NDArray:
        b, a = butter(2, [0.5*first_peak_xf, 1.5*first_peak_xf], btype="bandpass", fs=fs)
        return filtfilt(b, a, _y)

    def _get_residue(_y: NDArray) -> NDArray:
        return _y - _get_dc(_y) - _get_1st(_y)

    def _get_1st_envelope(_y: NDArray) -> NDArray:
        return np.abs(hilbert(_get_1st(_y))) # type: ignore

    return FftSpectrumAnalysis(
        get_dc = _get_dc,
        get_1st = _get_1st,
        get_residue = _get_residue,
        get_1st_envelope = _get_1st_envelope
    )

def fit_spectra_envelopes(x: NDArray, y: NDArray, **kwargs):
    _y_top_envelope = np.vstack([
        fit_spectrum_envelope(x, _y, mode="top", **kwargs) for _y in y.T
    ]).T
    _y_bottom_envelope = np.vstack([
        fit_spectrum_envelope(x, _y, mode="bottom", **kwargs) for _y in y.T
    ]).T
    return _y_top_envelope, _y_bottom_envelope


def fit_spectrum_envelope(x: NDArray, y: NDArray, mode: Literal["top", "bottom"], **kwargs):
    from scipy.signal import find_peaks
    from scipy.interpolate import CubicSpline
    match mode:
        case "top":
            _peaks, _ = find_peaks(y, **kwargs)
            interp = CubicSpline(x[_peaks], y[_peaks])
        case "bottom":
            _peaks, _ = find_peaks(-y, **kwargs)
            interp = CubicSpline(x[_peaks], y[_peaks])
    return interp(x)








