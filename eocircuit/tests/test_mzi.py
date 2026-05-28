"""Tests for Mach-Zehnder Interferometer (MZI) photonic component."""

from __future__ import annotations

# pyright: reportMissingImports=false

import numpy as np

from eocircuit.core.types import FrequencyGrid, SPEED_OF_LIGHT
from eocircuit.photonics.mzi import MZI, MZIParams, create_mzi


def _freq_from_wavelength(wavelength_m: float) -> float:
    return SPEED_OF_LIGHT / wavelength_m


def _analytical_mzi_transfer(
    freq_hz: np.ndarray,
    dc_kappa: float,
    arm_length_top: float,
    arm_length_bottom: float,
    neff: float,
    loss: float,
) -> np.ndarray:
    wavelengths = SPEED_OF_LIGHT / freq_hz
    phi_top = 2 * np.pi * neff * arm_length_top / wavelengths
    phi_bottom = 2 * np.pi * neff * arm_length_bottom / wavelengths
    a_top = 10 ** (-loss * arm_length_top / 20)
    a_bottom = 10 ** (-loss * arm_length_bottom / 20)

    t1 = np.sqrt(1.0 - dc_kappa**2)
    t2 = np.sqrt(1.0 - dc_kappa**2)
    c1 = 1j * dc_kappa
    c2 = 1j * dc_kappa

    return t1 * t2 * a_top * np.exp(-1j * phi_top) - c1 * c2 * a_bottom * np.exp(-1j * phi_bottom)


def test_mzi_constructive() -> None:
    """Δφ=0 should yield maximal transmission."""
    wavelength = 1550e-9
    freq = FrequencyGrid(values=np.array([_freq_from_wavelength(wavelength)]))

    mzi = create_mzi(
        name="mzi_constructive",
        dc_kappa=1 / np.sqrt(2),
        arm_length_top=200e-6,
        arm_length_bottom=200e-6,
        neff=2.4,
        loss=0.0,
    )

    s = mzi.get_s(freq)
    t = s[0, 1, 0]
    assert np.isclose(np.abs(t) ** 2, 1.0, atol=1e-12)


def test_mzi_destructive() -> None:
    """Δφ=π should yield near-zero transmission."""
    wavelength = 1550e-9
    neff = 2.4
    base_length = 200e-6
    delta_l = wavelength / (2 * neff)  # gives Δφ = π
    freq = FrequencyGrid(values=np.array([_freq_from_wavelength(wavelength)]))

    mzi = create_mzi(
        name="mzi_destructive",
        dc_kappa=1 / np.sqrt(2),
        arm_length_top=base_length,
        arm_length_bottom=base_length + delta_l,
        neff=neff,
        loss=0.0,
    )

    s = mzi.get_s(freq)
    t = s[0, 1, 0]
    assert np.abs(t) < 1e-9


def test_mzi_analytical() -> None:
    """Component transfer should match analytical model tightly."""
    wavelengths = np.linspace(1530e-9, 1570e-9, 101)
    freq_values = SPEED_OF_LIGHT / wavelengths
    freq = FrequencyGrid(values=freq_values)

    params = dict(
        dc_kappa=0.6,
        arm_length_top=320e-6,
        arm_length_bottom=350e-6,
        neff=2.35,
        loss=0.0,
    )
    mzi = create_mzi(name="mzi_analytical", **params)

    s = mzi.get_s(freq)
    model_t = s[:, 1, 0]
    expected_t = _analytical_mzi_transfer(freq_values, **params)

    max_err = np.max(np.abs(model_t - expected_t))
    assert max_err < 1e-6


def test_mzi_fsr() -> None:
    """FSR from sweep should match ΔL-based theoretical FSR."""
    c = SPEED_OF_LIGHT
    neff = 2.4
    l_top = 1.0e-3
    l_bottom = 1.1e-3
    delta_l = abs(l_bottom - l_top)

    f0 = _freq_from_wavelength(1550e-9)
    fsr_theory = c / (neff * delta_l)

    span = 1.3 * fsr_theory
    freq_values = np.linspace(f0 - span, f0 + span, 20001)
    freq = FrequencyGrid(values=freq_values)

    mzi = MZI(
        name="mzi_fsr",
        params=MZIParams(
            dc_kappa=1 / np.sqrt(2),
            arm_length_top=l_top,
            arm_length_bottom=l_bottom,
            neff=neff,
            loss=0.0,
        ),
    )

    power = np.abs(mzi.get_s(freq)[:, 1, 0]) ** 2

    # Find local maxima, then use adjacent peak spacing near sweep center.
    peak_indices = np.where((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:]))[0] + 1
    assert len(peak_indices) >= 3

    center_idx = len(freq_values) // 2
    nearest_peak_pos = int(np.argmin(np.abs(peak_indices - center_idx)))
    if nearest_peak_pos == 0:
        left_peak = peak_indices[0]
        right_peak = peak_indices[1]
    elif nearest_peak_pos == len(peak_indices) - 1:
        left_peak = peak_indices[-2]
        right_peak = peak_indices[-1]
    else:
        left_peak = peak_indices[nearest_peak_pos - 1]
        right_peak = peak_indices[nearest_peak_pos]

    fsr_measured = freq_values[right_peak] - freq_values[left_peak]

    assert np.isclose(fsr_measured, fsr_theory, rtol=0.02)
