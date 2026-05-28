"""Tests for MicroringResonator all-pass component."""

# pyright: reportMissingImports=false

import numpy as np

from eocircuit.core.types import FrequencyGrid, SPEED_OF_LIGHT
from eocircuit.photonics.mrr import create_mrr


def _resonant_frequency(neff: float, radius: float, mode_number: int) -> float:
    """Return resonant frequency for mode m from phi_rt = 2*pi*m."""
    circumference = 2 * np.pi * radius
    wavelength = neff * circumference / mode_number
    return SPEED_OF_LIGHT / wavelength


def test_mrr_resonance():
    """At resonance, through power should be lower than nearby detuned points."""
    radius = 10e-6
    neff = 2.4
    ng = 4.0
    mrr = create_mrr(
        name="mrr_res",
        radius=radius,
        neff=neff,
        ng=ng,
        coupling_kappa=0.35,
        loss=5_000.0,
    )

    f_res = _resonant_frequency(neff=neff, radius=radius, mode_number=97)
    freq_grid = FrequencyGrid(values=np.array([f_res - 1e11, f_res, f_res + 1e11]))

    s = mrr.get_s(freq_grid)
    power = np.abs(s[:, 1, 0]) ** 2

    assert power[1] < power[0]
    assert power[1] < power[2]


def test_mrr_fsr():
    """FSR follows lambda^2 / (ng * 2*pi*R)."""
    radius = 12e-6
    ng = 4.2
    wavelength = 1550e-9

    mrr = create_mrr(
        name="mrr_fsr",
        radius=radius,
        neff=2.5,
        ng=ng,
        coupling_kappa=0.2,
        loss=0.0,
    )

    expected_fsr = wavelength**2 / (ng * 2 * np.pi * radius)
    actual_fsr = mrr.fsr_at_wavelength(wavelength)

    np.testing.assert_allclose(actual_fsr, expected_fsr, rtol=1e-12)


def test_mrr_critical_coupling():
    """With kappa equal to round-trip loss amplitude (=1 here), through is zero."""
    mrr = create_mrr(
        name="mrr_cc",
        radius=10e-6,
        neff=2.4,
        ng=4.0,
        coupling_kappa=1.0,
        loss=0.0,
    )

    wavelength = 1550e-9
    freq_grid = FrequencyGrid(values=np.array([SPEED_OF_LIGHT / wavelength]))
    s = mrr.get_s(freq_grid)

    np.testing.assert_allclose(s[0, 1, 0], 0.0 + 0.0j, atol=1e-14)


def test_mrr_all_pass():
    """All-pass ring works as through-only reciprocal 2-port response."""
    mrr = create_mrr(
        name="mrr_all_pass",
        radius=8e-6,
        neff=2.35,
        ng=4.1,
        coupling_kappa=0.4,
        loss=0.0,
    )

    freqs = np.linspace(190e12, 196e12, 41)
    s = mrr.get_s(FrequencyGrid(values=freqs))
    mag = np.abs(s[:, 1, 0])

    assert np.all(mag <= 1.0 + 1e-12)
    assert np.all(mag >= 0.0)
    np.testing.assert_allclose(s[:, 1, 0], s[:, 0, 1], atol=1e-14)
    np.testing.assert_allclose(s[:, 0, 0], 0.0 + 0.0j, atol=1e-14)
    np.testing.assert_allclose(s[:, 1, 1], 0.0 + 0.0j, atol=1e-14)
