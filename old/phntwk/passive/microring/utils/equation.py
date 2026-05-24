from ....utils.math import to_numpy
import numpy as np
from numpy.typing import NDArray

@to_numpy
def all_pass_transmission(phase: NDArray, a: NDArray, r: NDArray) -> NDArray:
    num = a ** 2 + r ** 2 - 2 * a * r * np.cos(phase)
    den = 1 + (a * r) ** 2 - 2 * a * r * np.cos(phase)
    return num / den


