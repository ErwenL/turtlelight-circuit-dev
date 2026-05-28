"""Microring resonator photonic component.

v1 implements the all-pass (2-port, through-only) microring resonator model.
Add-drop (4-port) support is intentionally left for a later revision.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from eocircuit.core.component import Component, ComponentParams, s_to_z
from eocircuit.core.types import FrequencyGrid, Port, PortDomain, SPEED_OF_LIGHT


class MicroringResonatorParams(ComponentParams):
    """Parameters for an all-pass microring resonator.

    Attributes:
        radius: Ring radius in meters
        neff: Effective index for phase accumulation
        ng: Group index used for FSR estimation
        coupling_kappa: Coupling coefficient (0 <= kappa <= 1)
        loss: Propagation loss in dB/cm
    """

    radius: float
    neff: float
    ng: float
    coupling_kappa: float
    loss: float = 0.0

    @field_validator("radius")
    @classmethod
    def validate_radius(cls, v: float) -> float:
        """Validate ring radius is positive."""
        if v <= 0:
            raise ValueError("radius must be positive")
        return v

    @field_validator("neff", "ng")
    @classmethod
    def validate_indices(cls, v: float) -> float:
        """Validate refractive/group index is positive."""
        if v <= 0:
            raise ValueError("neff and ng must be positive")
        return v

    @field_validator("coupling_kappa")
    @classmethod
    def validate_coupling_kappa(cls, v: float) -> float:
        """Validate coupling coefficient kappa in [0, 1]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("coupling_kappa must be in [0, 1]")
        return v

    @field_validator("loss")
    @classmethod
    def validate_loss(cls, v: float) -> float:
        """Validate propagation loss is non-negative."""
        if v < 0:
            raise ValueError("loss must be non-negative")
        return v


class MicroringResonator(Component):
    """All-pass microring resonator (2-port through-only).

    Port order: [in, thru]

    Through transfer function:
        T_thru = (t - alpha * t * exp(-j * phi_rt)) / (1 - alpha * t^2 * exp(-j * phi_rt))

    where:
        phi_rt = 2*pi*neff*(2*pi*R)/lambda
        alpha  = 10^(-loss * 2*pi*R / 20)  (per project v1 spec)
        t      = sqrt(1-kappa^2)
        c      = j*kappa  (cross coefficient, not directly used in all-pass through equation)
    """

    def __init__(self, **data):
        """Initialize all-pass MRR with default 2 optical ports."""
        if "ports" not in data:
            data["ports"] = [
                Port(name="in", domain=PortDomain.OPTICAL, z0=1.0),
                Port(name="thru", domain=PortDomain.OPTICAL, z0=1.0),
            ]
        super().__init__(**data)

    @property
    def circumference_m(self) -> float:
        """Ring circumference in meters."""
        params = cast(MicroringResonatorParams, self.params)
        return 2 * np.pi * params.radius

    def round_trip_phase(self, freq: FrequencyGrid) -> NDArray[np.floating]:
        """Compute round-trip phase for each frequency point.

        Uses:
            lambda = c / f
            phi_rt = 2*pi*neff*(2*pi*R)/lambda
        """
        wavelengths = SPEED_OF_LIGHT / freq.values
        params = cast(MicroringResonatorParams, self.params)
        return 2 * np.pi * params.neff * self.circumference_m / wavelengths

    def round_trip_loss_amplitude(self) -> float:
        """Compute round-trip field attenuation alpha.

        Uses project v1 analytical specification directly:
            alpha = 10^(-loss * 2*pi*R / 20)
        """
        params = cast(MicroringResonatorParams, self.params)
        return float(10 ** (-params.loss * self.circumference_m / 20.0))

    def fsr_at_wavelength(self, wavelength_m: float) -> float:
        """Approximate FSR in wavelength domain: lambda^2 / (ng * 2*pi*R)."""
        if wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        params = cast(MicroringResonatorParams, self.params)
        return wavelength_m**2 / (params.ng * self.circumference_m)

    def get_s(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get 2x2 S-parameters for all-pass ring."""
        n_freq = len(freq.values)
        s_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)

        params = cast(MicroringResonatorParams, self.params)
        kappa = params.coupling_kappa
        t = np.sqrt(1.0 - kappa**2)
        # Coupler coefficients:
        #   t = sqrt(1-kappa^2), c = j*kappa
        # c is not explicitly needed in all-pass through-only transfer evaluation.
        alpha = self.round_trip_loss_amplitude()

        phi_rt = self.round_trip_phase(freq)
        exp_term = np.exp(-1j * phi_rt)

        # Through transfer (all-pass):
        # T_thru = (t - alpha*t*exp(-j*phi_rt)) / (1 - alpha*t^2*exp(-j*phi_rt))
        t_thru = (t - alpha * t * exp_term) / (1.0 - alpha * (t**2) * exp_term)

        for i in range(n_freq):
            s_matrix[i, 0, 1] = t_thru[i]
            s_matrix[i, 1, 0] = t_thru[i]

        return s_matrix

    def get_z(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Convert S-parameters to Z-parameters."""
        s_matrix = self.get_s(freq)
        z0_values = [port.z0 for port in self.ports]
        return s_to_z(s_matrix, z0_values)

    def get_y(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Compute Y-parameters as matrix inverse of Z per frequency."""
        n_freq = len(freq.values)
        z_matrix = self.get_z(freq)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        for i in range(n_freq):
            y_matrix[i] = np.linalg.inv(z_matrix[i])
        return y_matrix


def create_mrr(
    name: str,
    radius: float,
    neff: float,
    ng: float,
    coupling_kappa: float,
    loss: float = 0.0,
) -> MicroringResonator:
    """Factory helper for creating an all-pass MicroringResonator."""
    return MicroringResonator(
        name=name,
        params=MicroringResonatorParams(
            radius=radius,
            neff=neff,
            ng=ng,
            coupling_kappa=coupling_kappa,
            loss=loss,
        ),
    )
