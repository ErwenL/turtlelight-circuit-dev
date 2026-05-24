from __future__ import annotations
from tkinter import N
from ....utils.spectrum import Spectrum, XySpectrum
from typing import Callable, Any, Optional, Literal, Sequence, TypeVar
import numpy as np
from numpy.typing import NDArray, ArrayLike
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from .spectrum import GetSpectrum, MRingSpectrumTools
from .equation import all_pass_transmission
from pydantic import BaseModel, root_validator


class AllPassRingSpectrum(XySpectrum):
    x_sample: NDArray
    params: AllPassRingParams
    get_params: AllPassRingGetParams
    get_fit: Callable[[], NDArray]
    get_residue: Callable[[], NDArray]

    def swap_ra(self):
        self.params = _swap_ra(self.params)
        self.get_params = _swap_ra(self.get_params)

RaGuess = Optional[Sequence[float|NDArray]]
FitMode = Literal["ra", "rac"]
"""Fit mode: 'ra', 'rac'
- 'ra': fit only the ring loss and ring coupling
- 'rac': fit the ring loss, ring coupling and the constant offset
"""
FitRange = Literal["full", "valley"]
"""Fit range: 'full', 'valley'
- 'full': fit the full spectrum
- 'valley': fit the spectrum in the valley region
"""

@pa.check_types(lazy=True)
def fit_all_pass_transmission(
    spectrum: XySpectrum,
    tools: MRingSpectrumTools,
    guess: RaGuess = None,
    mode: FitMode = "rac",
    fit_range: FitRange = "full",
):

    phase_sample = tools.sample_at_peaks(spectrum.x)
    sample_num = len(phase_sample)
    param_guess = AllPassRingParams.validate_guess(guess, sample_num, mode)

    if fit_range == "full":
        return fit_all_pass_transmission_spectrum(
            spectrum, phase_sample, param_guess.to_fit_params(mode), mode,
            interp_method="cubic"
        )

    assert fit_range == "valley", "Unknown fit range"
    valley_spectra = tools.chop_spectrum_by_peaks(spectrum)

    fit_spectra: list[AllPassRingSpectrum] = []
    for idx, _spectrum in enumerate(valley_spectra):
        fit_spectra.append(
            fit_all_pass_transmission_spectrum(
                _spectrum, 
                phase_sample[[idx, idx + 1]], 
                param_guess[idx].to_fit_params(mode), 
                mode
            )
        )

    return join_ring_spectra(fit_spectra, mode)
    

class AllPassRingParams(BaseModel):
    a: NDArray
    r: NDArray
    c: Optional[NDArray] = None

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def sort_ar(cls, values):
        """assume a is always larger than r
        """
        a = values.pop("a")
        r = values.pop("r")
        ar_diff = a > r
        _a = a * ar_diff + r * ~ar_diff
        _r = r * ar_diff + a * ~ar_diff
        return {"a": _a, "r": _r, **values}

    @classmethod
    def from_fit_params(cls, params: ArrayLike, mode: FitMode):
        params = np.array(params)
        match mode:
            case "ra":
                return cls(a=params[::2], r=params[1::2])
            case "rac":
                return cls(a=params[::3], r=params[1::3], c=params[2::3])
            case _:
                raise ValueError(f"Unknown mode: {mode}")
    
    def to_fit_params(self, mode: FitMode):
        match mode:
            case "ra":
                return np.vstack([self.a, self.r]).T.reshape(-1)
            case "rac":
                assert self.c is not None, "c must be provided when mode is 'rac'"
                return np.vstack([self.a, self.r, self.c]).T.reshape(-1)
            case _:
                raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def validate_guess(cls, guess: RaGuess, sample_num: int, mode: FitMode):
        if guess is None:
            guess = (0.95, 0.9)

        assert len(guess) == 2, "guess must be a sequence of length 2"

        _guess : list[NDArray] = [] 

        for g in guess:
            assert isinstance(g, (float, np.ndarray)), "guess must be a tuple of float or ndarrays"
            if isinstance(g, np.ndarray):
                assert len(g) == sample_num, "guess ndarray must have the same length as the number of peaks"
            if isinstance(g, float):
                g = np.ones(sample_num) * g
            _guess.append(g)

        match mode:
            case "ra":
                return cls(a=_guess[0], r=_guess[1])
            case "rac":
                return cls(a=_guess[0], r=_guess[1], c=np.zeros(sample_num))
            case _:
                raise ValueError(f"Unknown mode: {mode}")

    def __getitem__(self, start:int, end:Optional[int] = None):
        if end is None:
            end = start + 2
        if self.c is None:
            return AllPassRingParams(a=self.a[start:end], r=self.r[start:end])
        return AllPassRingParams(a=self.a[start:end], r=self.r[start:end], c=self.c[start:end])

class AllPassRingGetParams(BaseModel):
    a: GetSpectrum
    r: GetSpectrum
    c: Optional[GetSpectrum] = None

    class Config:
        arbitrary_types_allowed = True


def fit_all_pass_transmission_spectrum(
    spectrum: XySpectrum,
    phase_sample: NDArray,
    param_guess: NDArray,
    mode: FitMode,
    interp_method: Literal["linear", "cubic"] = "linear",
):
    def _interp(y: NDArray) -> GetSpectrum: 
        match interp_method:
            case "linear":
                interp = linear_interp
            case "cubic":
                interp = CubicSpline
        dphase = phase_sample[1] - phase_sample[0]
        if dphase < 0:
            return interp(np.flip(phase_sample), np.flip(y))
        return interp(phase_sample, y)

    def _fit(phase: NDArray, *params):
        _ring_params = AllPassRingParams.from_fit_params(params, mode)
        _transmission = all_pass_transmission(
            phase,
            _interp(_ring_params.a)(phase),
            _interp(_ring_params.r)(phase),
        )
        match mode:
            case "ra":
                return _transmission
            case "rac":
                assert _ring_params.c is not None, "c must be provided when mode is 'rac'"
                return _transmission + _interp(_ring_params.c)(phase)
            case _:
                raise ValueError(f"Unknown mode: {mode}")
    
    popt, _ = curve_fit(_fit, spectrum.x, spectrum.y, p0=param_guess)
    ring_params = AllPassRingParams.from_fit_params(popt, mode)
    get_ring_params = AllPassRingGetParams(
        a=_interp(ring_params.a),
        r=_interp(ring_params.r),
        c=_interp(ring_params.c) if ring_params.c is not None else None,
    )
    return AllPassRingSpectrum(
        x=spectrum.x,
        y=spectrum.y,
        x_sample=phase_sample,
        params=ring_params,
        get_params=get_ring_params,
        get_fit=lambda : _fit(spectrum.x, *popt),
        get_residue=lambda : _fit(spectrum.x, *popt) - spectrum.y,
    )

def join_ring_spectra(spectra: list[AllPassRingSpectrum], mode: FitMode) -> AllPassRingSpectrum:
    _spectrum = _join_spectra(spectra)
    _x_sample = unsorted_unique(np.concatenate([spectrum.x_sample for spectrum in spectra]))

    def join_params(param: Literal["a", "r", "c"]):
        assert all(
            [
                hasattr(spectrum.params, param) and 
                getattr(spectrum.params, param) is not None 
                for spectrum in spectra]
        ), f"param {param} not found in spectra"
        param_spectra = [
            XySpectrum(
                x=spectrum.x_sample,
                y=getattr(spectrum.params, param),
            ) for spectrum in spectra
        ]
        return _join_spectra(param_spectra).y

    ring_params = AllPassRingParams(
        a=join_params("a"), # type: ignore
        r=join_params("r"), # type: ignore
    )
    get_ring_params = AllPassRingGetParams(
        a = linear_interp(_x_sample, ring_params.a),
        r = linear_interp(_x_sample, ring_params.r), 
    )
    if mode == "rac":
        ring_params.c = join_params("c")
        get_ring_params.c = linear_interp(_x_sample, ring_params.c)

    def join_fit():
        fit_spectra = [
            XySpectrum(
                x=spectrum.x,
                y=spectrum.get_fit(),
            ) for spectrum in spectra
        ]
        return _join_spectra(fit_spectra).y
    
    return AllPassRingSpectrum(
        x=_spectrum.x,
        y=_spectrum.y,
        x_sample=_x_sample,
        params=ring_params,
        get_params=get_ring_params,
        get_fit=join_fit,
        get_residue=lambda : join_fit() - _spectrum.y,
    )

AnySpectrum = TypeVar("AnySpectrum", bound=XySpectrum)
def _join_spectra(spectra: list[AnySpectrum]) -> XySpectrum:
    xs = np.concatenate([spectrum.x for spectrum in spectra])
    ys = np.concatenate([spectrum.y for spectrum in spectra])
    _x = unsorted_unique(xs)
    _y = np.zeros_like(_x)
    for i, x in enumerate(_x):
        idx = xs == x
        _y[i] = np.mean(ys[idx])
    return XySpectrum(x=_x, y=_y)

AnyParams = TypeVar("AnyParams", AllPassRingParams, AllPassRingGetParams)
def _swap_ra(params: AnyParams):
    params.a, params.r = params.r, params.a # type: ignore
    return params

def linear_interp(x: NDArray, y: NDArray) -> Callable[[NDArray], NDArray]:
    def interp(_x: NDArray) -> NDArray:
        return np.interp(_x, x, y)
    return interp

def unsorted_unique(_x: NDArray) -> NDArray:
    _x, _i = np.unique(_x, return_index=True)
    return _x[np.argsort(_i)]


AllPassRingSpectrum.update_forward_refs()
    
    
    