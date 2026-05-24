from __future__ import annotations
from typing import Callable, Any, Optional, Literal
from .spectrum import *

GetPhaseMode = Literal["harmonic", "interp_from_valleys"]

def get_ring_phase(
    df: DataFrame[Spectrum],
    spectrum_name: str,
    spectrum_summary: Optional[MRingSpectrumSummary|DataFrame[Spectrum]] = None,
    mode: GetPhaseMode = "interp_from_valleys",
    find_peaks_kwargs: dict[str, Any] = {},
):
    _df = df.copy()
    if isinstance(spectrum_summary, pd.DataFrame):
        spectrum_summary = MRingSpectrumSummary(
            **{str(key):value.to_numpy() for key, value in spectrum_summary.items()}
        )
        
    assert isinstance(spectrum_summary, MRingSpectrumSummary), "summary must be either a DataFrame or MRingSpectrumSummary"
    
    spectrum = XySpectrum(
        x=_df["wavelength"].to_numpy(),
        y=_df[spectrum_name].to_numpy().astype(float),
    )

    match mode:
        case "interp_from_valleys":
            phase = interp_phase_from_valleys(spectrum, spectrum_summary)
        case "harmonic":
            phase = calculate_phase_from_1st_harmonic(spectrum, spectrum_summary, **find_peaks_kwargs)
        case _:
            raise ValueError(f"invalid mode {mode}")
    
    _df["phase"] = phase
    return _df


def interp_phase_from_valleys(spectrum: XySpectrum, summary: MRingSpectrumSummary|DataFrame[Spectrum]):
    _x = summary.wavelength
    phase = np.flip(np.arange(len(_x))) * 2 * np.pi
    interp = CubicSpline(_x, phase, extrapolate=True)
    return interp(spectrum.x)

def calculate_phase_from_1st_harmonic(spectrum: XySpectrum, summary: MRingSpectrumSummary|DataFrame[Spectrum], **kwargs):
    _f_spectrum = spectrum.fft()
    sample_num = len(_f_spectrum.x)
    _f_peaks, _ = find_peaks(np.abs(_f_spectrum.y)[:sample_num//2], **kwargs)
    assert len(_f_peaks) > 0, "no peaks found"
    first_peak = _f_peaks[0]
    first_peak_x = _f_spectrum.x[first_peak]
    fs = _f_spectrum.x[1] * sample_num

    from scipy.signal import butter, filtfilt, hilbert

    def _get_dc(y):
        b, a = butter(4, first_peak_x/2, btype="lowpass", fs=fs)
        return filtfilt(b, a, y)

    def _get_1st(y):
        b, a = butter(2, [0.5*first_peak_x, 1.5*first_peak_x], btype="bandpass", fs=fs)
        return filtfilt(b, a, y)
    
    _y_1st_harmonic = _get_1st(spectrum.y)
    _y_analytic = hilbert(np.flip(_y_1st_harmonic))
    _phase = np.flip(np.unwrap(np.angle(_y_analytic))) # type: ignore
    return _get_dc(_phase)








