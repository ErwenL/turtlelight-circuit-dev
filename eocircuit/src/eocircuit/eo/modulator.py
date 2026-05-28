"""Electro-optic modulators based on phenomenological v1 models.

Implements:
- MZModulator: electrical bias maps to MZI phase modulation
- MRRModulator: electrical bias maps to resonance wavelength shift
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from eocircuit.core.component import Component, ComponentParams, s_to_z
from eocircuit.core.types import FrequencyGrid, Port, PortDomain, SPEED_OF_LIGHT
from eocircuit.photonics.mzi import MZI, MZIParams
from eocircuit.photonics.mrr import MicroringResonator as MRR
from eocircuit.photonics.mrr import MicroringResonatorParams


class MZModulatorMZIParams(ComponentParams):
    """Embedded MZI parameters for MZModulator.

    Attributes:
        dc_kappa: Directional coupler amplitude coupling coefficient.
        arm_lengths: Tuple of (top, bottom) arm lengths in meters.
        neff: Effective refractive index.
        loss: Waveguide loss in dB/m.
    """

    dc_kappa: float
    arm_lengths: tuple[float, float]
    neff: float
    loss: float = 0.0


class MZModulatorParams(ComponentParams):
    """Parameters for Mach-Zehnder modulator.

    Attributes:
        vpi: Half-wave voltage in volts.
        v_bias: Electrical bias voltage in volts.
        mzi_params: Internal MZI geometry/optical parameters.
        phase_bias_rad: Static phase bias (radians). Default pi gives
            max transmission at V=vpi for this v1 model.
    """

    vpi: float
    v_bias: float = 0.0
    mzi_params: MZModulatorMZIParams
    phase_bias_rad: float = float(np.pi)


class MZModulator(Component):
    """3-port MZI-based EO modulator.

    Ports: [elec (electrical), opt_in (optical), opt_out (optical)]

    v1 phenomenological model:
        dphi = pi * V_bias / Vpi
        T = cos^2((phase_bias_rad + dphi) / 2)
    """

    params: MZModulatorParams

    def __init__(self, **data):
        if "ports" not in data:
            data["ports"] = [
                Port(name="elec", domain=PortDomain.ELECTRICAL, z0=50.0),
                Port(name="opt_in", domain=PortDomain.OPTICAL, z0=1.0),
                Port(name="opt_out", domain=PortDomain.OPTICAL, z0=1.0),
            ]
        super().__init__(**data)

    def _build_internal_mzi(self) -> MZI:
        p = self.params.mzi_params
        return MZI(
            name=f"{self.name}_internal_mzi",
            params=MZIParams(
                dc_kappa=p.dc_kappa,
                arm_length_top=p.arm_lengths[0],
                arm_length_bottom=p.arm_lengths[1],
                neff=p.neff,
                loss=p.loss,
            ),
        )

    def phase_shift(self) -> float:
        """Return electrical phase shift dphi = pi*V/Vpi."""
        return float(np.pi * self.params.v_bias / self.params.vpi)

    def transmission(self) -> float:
        """Return intensity transmission based on phase bias + electrical shift."""
        phase_total = self.params.phase_bias_rad + self.phase_shift()
        return float(np.cos(phase_total / 2.0) ** 2)

    def get_s(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get 3x3 S-parameters with EO-controlled optical transmission."""
        n_freq = len(freq.values)
        s_matrix = np.zeros((n_freq, 3, 3), dtype=np.complex128)

        # Wrap internal MZI and apply EO transmission envelope.
        internal_mzi = self._build_internal_mzi()
        s_internal = internal_mzi.get_s(freq)
        base_t = s_internal[:, 1, 0]

        amp = np.sqrt(self.transmission())
        modulated_t = amp * np.exp(1j * np.angle(base_t))
        s_matrix[:, 1, 2] = modulated_t
        s_matrix[:, 2, 1] = modulated_t

        return s_matrix

    def get_z(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Convert S-parameters to Z-parameters."""
        s_matrix = self.get_s(freq)
        z0_values = [port.z0 for port in self.ports]
        return s_to_z(s_matrix, z0_values)

    def get_y(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Compute Y-parameters from Z by per-frequency inversion."""
        z_matrix = self.get_z(freq)
        y_matrix = np.zeros_like(z_matrix)
        for i in range(len(freq.values)):
            y_matrix[i] = np.linalg.inv(z_matrix[i])
        return y_matrix


class MRRModulatorParams(ComponentParams):
    """Parameters for MRR EO modulator.

    Attributes:
        tuning_efficiency_nm_per_v: Resonance tuning efficiency in nm/V.
        v_bias: Electrical bias voltage in volts.
        mrr_params: Internal MRR parameters.
    """

    tuning_efficiency_nm_per_v: float
    v_bias: float = 0.0
    mrr_params: MicroringResonatorParams


class MRRModulator(Component):
    """3-port MRR-based EO modulator.

    Ports: [elec (electrical), opt_in (optical), opt_out (optical)]

    v1 tuning model:
        lambda_res(V) = lambda_res(0) + tuning_efficiency * V_bias
    """

    params: MRRModulatorParams

    def __init__(self, **data):
        if "ports" not in data:
            data["ports"] = [
                Port(name="elec", domain=PortDomain.ELECTRICAL, z0=50.0),
                Port(name="opt_in", domain=PortDomain.OPTICAL, z0=1.0),
                Port(name="opt_out", domain=PortDomain.OPTICAL, z0=1.0),
            ]
        super().__init__(**data)

    def _build_internal_mrr(self) -> MRR:
        return MRR(name=f"{self.name}_internal_mrr", params=self.params.mrr_params)

    def resonance_shift_nm(self) -> float:
        """Return resonance shift in nm for the current bias."""
        return float(self.params.tuning_efficiency_nm_per_v * self.params.v_bias)

    def get_s(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get 3x3 S-parameters with electrically shifted optical response."""
        n_freq = len(freq.values)
        s_matrix = np.zeros((n_freq, 3, 3), dtype=np.complex128)

        # Apply lambda-domain shift: T(lambda; V) = T0(lambda - d_lambda)
        d_lambda_m = self.resonance_shift_nm() * 1e-9
        wavelengths_m = SPEED_OF_LIGHT / freq.values
        shifted_wavelengths_m = wavelengths_m - d_lambda_m
        if np.any(shifted_wavelengths_m <= 0):
            raise ValueError("Bias-induced wavelength shift produced non-positive wavelength")

        shifted_freq = FrequencyGrid(values=SPEED_OF_LIGHT / shifted_wavelengths_m)

        internal_mrr = self._build_internal_mrr()
        s_internal = internal_mrr.get_s(shifted_freq)
        t = s_internal[:, 1, 0]
        s_matrix[:, 1, 2] = t
        s_matrix[:, 2, 1] = t

        return s_matrix

    def get_z(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Convert S-parameters to Z-parameters."""
        s_matrix = self.get_s(freq)
        z0_values = [port.z0 for port in self.ports]
        return s_to_z(s_matrix, z0_values)

    def get_y(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Compute Y-parameters from Z by per-frequency inversion."""
        z_matrix = self.get_z(freq)
        y_matrix = np.zeros_like(z_matrix)
        for i in range(len(freq.values)):
            y_matrix[i] = np.linalg.inv(z_matrix[i])
        return y_matrix


def create_mz_modulator(
    name: str,
    vpi: float,
    mzi_params: MZModulatorMZIParams,
    v_bias: float = 0.0,
    phase_bias_rad: float = float(np.pi),
) -> MZModulator:
    """Factory helper for MZModulator."""
    return MZModulator(
        name=name,
        params=MZModulatorParams(
            vpi=vpi,
            v_bias=v_bias,
            mzi_params=mzi_params,
            phase_bias_rad=phase_bias_rad,
        ),
    )


def create_mrr_modulator(
    name: str,
    tuning_efficiency_nm_per_v: float,
    mrr_params: MicroringResonatorParams,
    v_bias: float = 0.0,
) -> MRRModulator:
    """Factory helper for MRRModulator."""
    return MRRModulator(
        name=name,
        params=MRRModulatorParams(
            tuning_efficiency_nm_per_v=tuning_efficiency_nm_per_v,
            v_bias=v_bias,
            mrr_params=cast(MicroringResonatorParams, mrr_params),
        ),
    )


__all__ = [
    "MZModulatorMZIParams",
    "MZModulatorParams",
    "MZModulator",
    "MRRModulatorParams",
    "MRRModulator",
    "create_mz_modulator",
    "create_mrr_modulator",
]
