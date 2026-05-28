"""Core type definitions for eocircuit framework.

Provides fundamental types for electrical-optical circuit simulation:
- Port: N-port interface with domain (electrical/optical) and impedance
- FrequencyGrid: Frequency domain grid in Hz
- WavelengthGrid: Wavelength domain grid in meters
- Type aliases for numpy arrays
"""

from enum import Enum
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator, ConfigDict, Field


# Speed of light in m/s
SPEED_OF_LIGHT = 299792458.0

# Global port counter for auto-incrementing indices
_PORT_COUNTER = 0


class PortDomain(str, Enum):
    """Port domain enumeration."""

    ELECTRICAL = "electrical"
    OPTICAL = "optical"


def _get_next_port_index() -> int:
    """Get next port index and increment counter."""
    global _PORT_COUNTER
    current = _PORT_COUNTER
    _PORT_COUNTER += 1
    return current


class Port(BaseModel):
    """Port definition for circuit components.

    Represents a single port with domain (electrical/optical), impedance,
    and auto-incremented index. Immutable after creation.

    Attributes:
        name: Port identifier string
        domain: PortDomain (ELECTRICAL or OPTICAL)
        z0: Characteristic impedance in Ohms (default 50.0 for electrical)
        index: Auto-incremented port index
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    domain: PortDomain
    z0: float = 50.0
    index: int = Field(default_factory=_get_next_port_index)

    @field_validator("z0")
    @classmethod
    def validate_z0(cls, v: float) -> float:
        """Validate impedance is positive."""
        if v <= 0:
            raise ValueError("Impedance z0 must be positive")
        return v


class FrequencyGrid(BaseModel):
    """Frequency domain grid in Hz.

    Represents a set of frequency points for simulation.
    Validates that all frequencies are non-negative.

    Attributes:
        values: NDArray of frequency values in Hz
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: NDArray[np.floating]

    @field_validator("values")
    @classmethod
    def validate_non_negative(cls, v: NDArray) -> NDArray:
        """Validate all frequencies are non-negative."""
        if np.any(v < 0):
            raise ValueError("Frequency values must be non-negative")
        return v

    def to_wavelength(self) -> "WavelengthGrid":
        """Convert frequency grid to wavelength grid.

        Returns:
            WavelengthGrid: Wavelength values in meters (c / f)

        Raises:
            ValueError: If any frequency is zero (division by zero)
        """
        if np.any(self.values == 0):
            raise ValueError("Cannot convert zero frequency to wavelength")
        wavelengths = SPEED_OF_LIGHT / self.values
        return WavelengthGrid(values=wavelengths)


class WavelengthGrid(BaseModel):
    """Wavelength domain grid in meters.

    Represents a set of wavelength points for simulation.
    Validates that all wavelengths are non-negative.

    Attributes:
        values: NDArray of wavelength values in meters
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: NDArray[np.floating]

    @field_validator("values")
    @classmethod
    def validate_non_negative(cls, v: NDArray) -> NDArray:
        """Validate all wavelengths are non-negative."""
        if np.any(v < 0):
            raise ValueError("Wavelength values must be non-negative")
        return v

    def to_frequency(self) -> FrequencyGrid:
        """Convert wavelength grid to frequency grid.

        Returns:
            FrequencyGrid: Frequency values in Hz (c / λ)

        Raises:
            ValueError: If any wavelength is zero (division by zero)
        """
        if np.any(self.values == 0):
            raise ValueError("Cannot convert zero wavelength to frequency")
        frequencies = SPEED_OF_LIGHT / self.values
        return FrequencyGrid(values=frequencies)


# Type aliases for numpy arrays
ComplexArray = Annotated[NDArray[np.complexfloating], "Complex-valued array"]
RealArray = Annotated[NDArray[np.floating], "Real-valued array"]
