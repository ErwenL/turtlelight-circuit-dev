import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from siluxApi.phntwk.utils import get_model_config_meta_units
from typing import Optional

class MziDcTestkeyMeasurement(pa.DataFrameModel):
    wavelength: Series[float]
    through_measure: Series[float]
    couple_length: Optional[Series[float]]
    delta_length: Optional[Series[float]]

    class Config:
        coerce = True
        metadata = get_model_config_meta_units(
            wavelength="um",
            couple_length="um",
            delta_length="um",
            through_measure="dB",
        )

    