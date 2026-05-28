"""
eocircuit.core: Core data structures and base classes.

Provides fundamental abstractions for circuit elements, networks,
and simulation parameters used across the eocircuit framework.
"""

from eocircuit.core.types import (
    PortDomain,
    Port,
    FrequencyGrid,
    WavelengthGrid,
    ComplexArray,
    RealArray,
    SPEED_OF_LIGHT,
)

__all__ = [
    "PortDomain",
    "Port",
    "FrequencyGrid",
    "WavelengthGrid",
    "ComplexArray",
    "RealArray",
    "SPEED_OF_LIGHT",
]
