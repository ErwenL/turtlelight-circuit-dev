from __future__ import annotations
from ....utils.spectrum import Spectrum, XySpectrum
from typing import Callable, Any, Optional, Literal
import numpy as np
from numpy.typing import NDArray, ArrayLike
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from pydantic import BaseModel

@pa.check_types(lazy=True)
def analyze_ring_spectra(
    df: DataFrame[Spectrum],
    spectrum_name: str,
    params: list[str],
): 
    ...

class MRingValleySpectrum(XySpectrum):
    x_valley: float
    y_valley: float
    x0: float
    a: float
    gamma: float
    start_offset: Optional[float]
    end_offset: Optional[float]

    def get_lorentzian_params(self):
        return self.x0, self.a, self.gamma

class MRingSpectrumSummary(BaseModel):
    wavelength: NDArray
    fwhm: NDArray
    q_factor: NDArray
    fsr: NDArray
    a: NDArray
    er: NDArray

    class Config:
        arbitrary_types_allowed = True

class MRingSpectrum(XySpectrum):
    fit: Optional[NDArray]
    fit_residue: Optional[NDArray]
    top_envelope: Optional[NDArray]
    fit_envelope: Optional[NDArray]

GetSpectrum = Callable[[NDArray], NDArray]

class MRingSpectrumTools(BaseModel):
    get_top_envelope: GetSpectrum
    sample_at_peaks: GetSpectrum
    trim_spectrum: GetSpectrum
    chop_spectrum_by_peaks: Callable[[XySpectrum], list[XySpectrum]]
    trim_xy_spectrum: Callable[[XySpectrum], XySpectrum]
    
    class Config:
        arbitrary_types_allowed = True

NormMethod = Optional[Literal["top_envelop_first", "top_envelop_only"]]

@pa.check_types(lazy=True)
def analyze_ring_spectrum(
    df: DataFrame[Spectrum],
    spectrum_name: str,
    norm_method: NormMethod = None,
    find_peaks_kwargs: dict[str, Any] = {},
): 
    spectrum = XySpectrum(
        x=df["wavelength"].to_numpy(),
        y=df[spectrum_name].to_numpy().astype(float),
    )

    valley_spectra, _tools = chop_spectrum( spectrum, norm_method, **find_peaks_kwargs)
    valley_spectra = [ 
        analyze_microring_valley_spectrum(s, norm_method, **find_peaks_kwargs) 
        for s in valley_spectra
    ]
    get_top_envelope_lorentz_fit = _get_envelope_from_valley_spectra_offsets(valley_spectra)
    # valley summary
    _summary = summarize_valley_spectra(valley_spectra)
    _summary = pd.DataFrame(_summary.dict()) # type: ignore
    # spectrum
    _spectrum = MRingSpectrum.parse_obj(spectrum)
    _spectrum.fit = (
        _tools.get_top_envelope(_spectrum.x) + 
        get_top_envelope_lorentz_fit(_spectrum.x) +
        multi_lorentzian(
            _spectrum.x, 
            *_get_multi_lorentzian_params_from_valley_spectra(valley_spectra)
        )
    )
    _spectrum.fit_residue = _spectrum.y - _spectrum.fit
    _spectrum.top_envelope = _tools.get_top_envelope(_spectrum.x)
    _spectrum.fit_envelope = _spectrum.top_envelope + get_top_envelope_lorentz_fit(_spectrum.x)
    _spectrum = pd.DataFrame(_spectrum.dict()) # type: ignore
    _spectrum.rename(columns={"x": "wavelength", "y": spectrum_name}, inplace=True)

    return _spectrum, _summary, _tools


def chop_spectrum(
    spectrum: XySpectrum, 
    norm_method: NormMethod = None, 
    **kwargs
) -> tuple[ list[XySpectrum], MRingSpectrumTools ]:
    """split spectrum at peaks
    - return a list single fsr spectra

    Args:
        spectrum[XySpectrum]: transmission spectrum

    Returns:
        list[XySpectrum]: single fsr spectrum
        GetSpectrum: get top envelope if norm_method is "top_envelop"
    """
    _spectrum = XySpectrum(
        x=spectrum.x,
        y=spectrum.y.copy()
    )
    idx_peaks, _ = find_peaks(spectrum.y, **kwargs)

    def chop_spectrum_by_peaks(spectrum: XySpectrum) -> list[XySpectrum]:
        spectra = []
        for idx, start in enumerate(idx_peaks[:-1]):
            end = idx_peaks[idx+1]
            spectra.append(XySpectrum(
                x=spectrum.x[start:end],
                y=spectrum.y[start:end],
            ))
        return spectra

    def trim_spectrum(spectrum: NDArray) -> NDArray:
        return spectrum[idx_peaks[0]:idx_peaks[-1]]

    def trim_xy_spectrum(spectrum: XySpectrum) -> XySpectrum:
        return XySpectrum(
            x=trim_spectrum(spectrum.x),
            y=trim_spectrum(spectrum.y),
        )

    def sample_at_peaks(_: NDArray) -> NDArray:
        return _[idx_peaks]
    
    match norm_method:
        case "top_envelop_first" | "top_envelop_only":
            get_top_envelope = CubicSpline(spectrum.x[idx_peaks], spectrum.y[idx_peaks])
        case _:
            get_top_envelope = lambda x: np.ones(len(x))

    spectra = chop_spectrum_by_peaks(_spectrum)
    for s in spectra:
        s.y -= get_top_envelope(s.x)

    _tools = MRingSpectrumTools(
        get_top_envelope=get_top_envelope,
        sample_at_peaks=sample_at_peaks,
        trim_spectrum=trim_spectrum,
        chop_spectrum_by_peaks=chop_spectrum_by_peaks,
        trim_xy_spectrum=trim_xy_spectrum,
    )

    return spectra, _tools


def analyze_microring_valley_spectrum(spectrum: XySpectrum, norm_method: NormMethod = None, **kwargs) -> MRingValleySpectrum:
    idx_valley, _ = find_peaks(-spectrum.y, **kwargs)
    x_valley = spectrum.x[idx_valley][0]
    y_valley = spectrum.y[idx_valley][0]
    x_span = spectrum.x[-1] - spectrum.x[0]
    match norm_method:
        case "top_envelop_first" | None:
            params_guess = [
                x_valley,
                y_valley,
                x_span / 4,
                0,
                0,
            ]
            popt, _ = curve_fit(
                tilt_lorentzian, 
                spectrum.x, 
                spectrum.y, 
                p0=params_guess,
            )
            _x0, _a, _gamma, _start_offset, _end_offset = popt
        case "top_envelop_only":
            params_guess = [
                x_valley,
                y_valley,
                x_span / 4,
            ]
            popt, _ = curve_fit(
                lorentzian, 
                spectrum.x, 
                spectrum.y, 
                p0=params_guess,
            )
            _x0, _a, _gamma= popt
            _start_offset = 0
            _end_offset = 0

    return MRingValleySpectrum(
        x=spectrum.x,
        y=spectrum.y,
        x_valley=x_valley,
        y_valley=y_valley,
        x0=_x0,
        a=_a,
        gamma=_gamma,
        start_offset=_start_offset,
        end_offset=_end_offset,
    )

def lorentzian(x: NDArray, x0: float|NDArray, a: float|NDArray, gamma: float|NDArray) -> NDArray:
    return a * gamma**2 / (gamma**2 + (x-x0)**2)

def tilt_lorentzian(x: NDArray, x0: float, a: float, gamma: float, start_offset: float, end_offset: float) -> NDArray:
    tilt = np.interp(x, [x[0], x[-1]], [start_offset, end_offset])
    return lorentzian(x, x0, a, gamma) + tilt

def multi_lorentzian(x: NDArray, *params) -> NDArray:
    _params = np.array(params)
    _x0 = _params[::3].reshape(-1, 1) @ np.ones(len(x)).reshape(1, -1)
    _a = _params[1::3].reshape(-1, 1) @ np.ones(len(x)).reshape(1, -1)
    _gamma = _params[2::3].reshape(-1, 1) @ np.ones(len(x)).reshape(1, -1)
    _x = np.ones(len(_params)//3).reshape(-1, 1) @ x.reshape(1, -1)
    return lorentzian(_x, _x0, _a, _gamma).sum(axis=0)

def _get_multi_lorentzian_params_from_valley_spectra(spectra: list[MRingValleySpectrum]) -> NDArray:
    _params = [s.get_lorentzian_params() for s in spectra]
    return np.concatenate(_params)

def _get_envelope_from_valley_spectra_offsets(spectra: list[MRingValleySpectrum]) -> GetSpectrum:
    x = [s.x[0] for s in spectra] + [spectra[-1].x[-1]]
    _y = (np.array([s.end_offset for s in spectra[1:]]) + np.array([s.start_offset for s in spectra[:-1]])) / 2
    y = np.concatenate([[spectra[0].start_offset], _y, [spectra[-1].end_offset]]) # type: ignore
    return CubicSpline(x, y)

def summarize_valley_spectra(spectra: list[MRingValleySpectrum]) -> MRingSpectrumSummary:
    x0 = np.array([s.x0 for s in spectra])
    fwhm = np.abs(np.array([s.gamma * 2 for s in spectra]))
    q_factor = x0 / fwhm

    a = np.array([s.a for s in spectra])

    fsr = np.ones_like(x0)
    for idx, s in enumerate(spectra[:-1]):
        fsr[idx] = spectra[idx+1].x0 - s.x0
    fsr[-1] = spectra[-1].x[-1] - spectra[-1].x[0]

    return MRingSpectrumSummary(
        wavelength=x0,
        fwhm=fwhm,
        q_factor=q_factor,
        fsr=np.array(fsr),
        a=a,
        er=-np.log10(1 + a) * 10,
    )

from .phase import *

