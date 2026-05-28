"""Tests for electro-optic modulator components."""

from __future__ import annotations

import numpy as np

from eocircuit.core.types import FrequencyGrid, SPEED_OF_LIGHT
from eocircuit.eo.modulator import (
    MZModulatorMZIParams,
    create_mz_modulator,
    create_mrr_modulator,
)
from eocircuit.photonics.mrr import MicroringResonatorParams


def _freq_from_wavelength(wavelength_m: float) -> float:
    return SPEED_OF_LIGHT / wavelength_m


def _default_mzi_params() -> MZModulatorMZIParams:
    return MZModulatorMZIParams(
        dc_kappa=1 / np.sqrt(2),
        arm_lengths=(200e-6, 200e-6),
        neff=2.4,
        loss=0.0,
    )


def test_mzm_at_vpi() -> None:
    """V=Vpi should produce max transmission for default phase bias."""
    wavelength = 1550e-9
    freq = FrequencyGrid(values=np.array([_freq_from_wavelength(wavelength)]))

    vpi = 2.0
    mzm = create_mz_modulator(
        name="mzm_vpi",
        vpi=vpi,
        v_bias=vpi,
        mzi_params=_default_mzi_params(),
    )

    s = mzm.get_s(freq)
    power = np.abs(s[0, 2, 1]) ** 2
    np.testing.assert_allclose(power, 1.0, atol=1e-12)


def test_mzm_at_vpi_half() -> None:
    """V=Vpi/2 should produce ~3 dB (0.5) transmission."""
    wavelength = 1550e-9
    freq = FrequencyGrid(values=np.array([_freq_from_wavelength(wavelength)]))

    vpi = 2.0
    mzm = create_mz_modulator(
        name="mzm_vpi_half",
        vpi=vpi,
        v_bias=vpi / 2.0,
        mzi_params=_default_mzi_params(),
    )

    s = mzm.get_s(freq)
    power = np.abs(s[0, 2, 1]) ** 2
    np.testing.assert_allclose(power, 0.5, atol=1e-12)


def test_mzm_at_zero() -> None:
    """At V=0, transmission should depend on static phase bias."""
    wavelength = 1550e-9
    freq = FrequencyGrid(values=np.array([_freq_from_wavelength(wavelength)]))

    mzm_constructive = create_mz_modulator(
        name="mzm_zero_constructive",
        vpi=2.0,
        v_bias=0.0,
        mzi_params=_default_mzi_params(),
        phase_bias_rad=0.0,
    )
    mzm_destructive = create_mz_modulator(
        name="mzm_zero_destructive",
        vpi=2.0,
        v_bias=0.0,
        mzi_params=_default_mzi_params(),
        phase_bias_rad=np.pi,
    )

    p_constructive = np.abs(mzm_constructive.get_s(freq)[0, 2, 1]) ** 2
    p_destructive = np.abs(mzm_destructive.get_s(freq)[0, 2, 1]) ** 2

    np.testing.assert_allclose(p_constructive, 1.0, atol=1e-12)
    np.testing.assert_allclose(p_destructive, 0.0, atol=1e-12)


def test_mrr_tuning() -> None:
    """Resonance shift should scale linearly with voltage change."""
    mrr_params = MicroringResonatorParams(
        radius=10e-6,
        neff=2.4,
        ng=4.0,
        coupling_kappa=0.3,
        loss=0.0,
    )

    tuning_eff = 0.08  # nm/V
    mod_v1 = create_mrr_modulator(
        name="mrr_mod_v1",
        tuning_efficiency_nm_per_v=tuning_eff,
        mrr_params=mrr_params,
        v_bias=1.0,
    )
    mod_v2 = create_mrr_modulator(
        name="mrr_mod_v2",
        tuning_efficiency_nm_per_v=tuning_eff,
        mrr_params=mrr_params,
        v_bias=3.5,
    )

    delta_lambda_nm = mod_v2.resonance_shift_nm() - mod_v1.resonance_shift_nm()
    expected_nm = tuning_eff * (3.5 - 1.0)
    np.testing.assert_allclose(delta_lambda_nm, expected_nm, atol=1e-15)
