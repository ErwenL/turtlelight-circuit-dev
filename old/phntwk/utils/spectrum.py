import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from numpy.typing import NDArray
from .utils import get_model_config_meta_units
from pydantic import BaseModel

class Spectrum(pa.DataFrameModel):
    wavelength: Series[float]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(
            wavelength="um",
        )

class XySpectrum(BaseModel):
    x: NDArray
    y: NDArray

    class Config:
        arbitrary_types_allowed = True

    def fft(self):
        assert _validate_uniform_sampling(self.x), "x is not uniformly sampled"

        from scipy.fft import fft, fftfreq
        dx = self.x[1] - self.x[0]
        sample_num = len(self.y)
        fft_sample_num = sample_num //2
        y_fft = fft(self.y, axis=0)/ fft_sample_num # type: ignore
        x_fft = fftfreq(sample_num, dx)
        return XySpectrum(x=x_fft, y=y_fft)


def _validate_uniform_sampling(x: NDArray):
    """validate if x is uniformly sampled
    """
    dx = np.diff(x)
    return np.allclose(dx, dx[0])



