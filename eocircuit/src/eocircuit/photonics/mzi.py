"""Mach-Zehnder Interferometer (MZI) photonic component.

Implements a 2-port MZI using an analytical transfer model:

T = t1*t2*exp(-j*phi_top) - c1*c2*exp(-j*phi_bottom)

with
phi = 2*pi*neff*L/lambda and lambda = c/f.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from eocircuit.core.component import Component, ComponentParams, s_to_z
from eocircuit.core.types import FrequencyGrid, Port, PortDomain, SPEED_OF_LIGHT
from eocircuit.photonics.dc import DirectionalCoupler
from eocircuit.photonics.waveguide import Waveguide
from eocircuit.solvers.s_param import SParamSolver


class MZIParams(ComponentParams):
    """Parameters for a Mach-Zehnder Interferometer.

    Attributes:
        dc_kappa: Directional coupler amplitude coupling coefficient.
        arm_length_top: Top arm length in meters.
        arm_length_bottom: Bottom arm length in meters.
        neff: Effective refractive index.
        loss: Propagation loss in dB/m.
    """

    dc_kappa: float
    arm_length_top: float
    arm_length_bottom: float
    neff: float
    loss: float = 0.0


class MZI(Component):
    """2-port Mach-Zehnder Interferometer.

    Port order: ["1in", "1out"].

    Internally modeled as:
    DC1 -> WG_top || WG_bottom -> DC2

    Uses analytical transfer for stable and direct 2-port response.
    """

    def __init__(self, **data):
        if "ports" not in data:
            data["ports"] = [
                Port(name="1in", domain=PortDomain.OPTICAL, z0=1.0),
                Port(name="1out", domain=PortDomain.OPTICAL, z0=1.0),
            ]
        super().__init__(**data)

    @staticmethod
    def _arm_transfer(
        freq_values: NDArray[np.floating], length: float, neff: float, loss_db_per_m: float
    ) -> NDArray[np.complexfloating]:
        """Return arm complex transfer a*exp(-j*phi) over frequency."""
        wavelengths = SPEED_OF_LIGHT / freq_values
        phi = 2 * np.pi * neff * length / wavelengths
        amplitude = 10 ** (-loss_db_per_m * length / 20)
        return amplitude * np.exp(-1j * phi)

    def transfer_analytical(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Compute analytical MZI complex transfer T(f)."""
        kappa = self.params.dc_kappa
        t1 = np.sqrt(1.0 - kappa**2)
        t2 = np.sqrt(1.0 - kappa**2)
        c1 = 1j * kappa
        c2 = 1j * kappa

        h_top = self._arm_transfer(
            freq.values,
            self.params.arm_length_top,
            self.params.neff,
            self.params.loss,
        )
        h_bottom = self._arm_transfer(
            freq.values,
            self.params.arm_length_bottom,
            self.params.neff,
            self.params.loss,
        )

        return t1 * t2 * h_top - c1 * c2 * h_bottom

    def get_s(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get 2x2 S-parameters for MZI.

        The MZI is modeled as reciprocal and reflectionless:
        S = [[0, T], [T, 0]]
        where T is the analytical transfer from transfer_analytical().
        """
        t = self.transfer_analytical(freq)
        n_freq = len(freq.values)
        s_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        s_matrix[:, 0, 1] = t
        s_matrix[:, 1, 0] = t
        return s_matrix

    def get_z(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Convert S-parameters to Z-parameters."""
        s_matrix = self.get_s(freq)
        z0_values = [port.z0 for port in self.ports]
        return s_to_z(s_matrix, z0_values)

    def get_y(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Compute Y-parameters as inverse of Z for each frequency."""
        z_matrix = self.get_z(freq)
        y_matrix = np.zeros_like(z_matrix)
        for i in range(len(freq.values)):
            y_matrix[i] = np.linalg.inv(z_matrix[i])
        return y_matrix


def create_mzi(
    name: str,
    dc_kappa: float,
    arm_length_top: float,
    arm_length_bottom: float,
    neff: float,
    loss: float = 0.0,
) -> MZI:
    """Factory for MZI component."""
    params = MZIParams(
        dc_kappa=dc_kappa,
        arm_length_top=arm_length_top,
        arm_length_bottom=arm_length_bottom,
        neff=neff,
        loss=loss,
    )
    return MZI(name=name, params=params)


# Keep explicit imports required by design notes for internal MZI realization.
__all__ = [
    "MZI",
    "MZIParams",
    "create_mzi",
    "Waveguide",
    "DirectionalCoupler",
    "SParamSolver",
]
