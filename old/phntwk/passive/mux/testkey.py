import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from siluxApi.phntwk.utils import get_model_config_meta_units

class BaseNtwkMeasurement(pa.DataFrameModel):
    wavelength: Series[float]
    ref_measure: Series[float]
    channel_1_measure: Series[float]
    channel_2_measure: Series[float]
    channel_3_measure: Series[float]
    channel_4_measure: Series[float]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(
            wavelength="um",
            ref_measure="dB",
            channel_1_measure="dB",
            channel_2_measure="dB",
            channel_3_measure="dB",
            channel_4_measure="dB",
        )