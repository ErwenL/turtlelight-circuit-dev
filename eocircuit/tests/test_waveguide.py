"""Tests for Waveguide photonic component.

Tests cover:
- Phase shift calculation
- Loss calculation
- Reciprocity property
- Zero-length edge case
"""

import numpy as np
import pytest

from eocircuit.core.types import FrequencyGrid, SPEED_OF_LIGHT
from eocircuit.photonics.waveguide import Waveguide, WaveguideParams, create_waveguide
from eocircuit.core.types import Port, PortDomain


class TestWaveguidePhase:
    """Test phase shift calculation."""

    def test_waveguide_phase(self):
        """Test phase calculation: L=100μm, neff=2.5, λ=1550nm.
        
        Expected phase ≈ 2π * neff * L / λ
        """
        # Create waveguide
        length = 100e-6  # 100 μm
        neff = 2.5
        waveguide = create_waveguide("wg", length=length, neff=neff, loss=0.0)

        # Frequency corresponding to λ=1550nm
        wavelength = 1550e-9  # 1550 nm
        freq_hz = SPEED_OF_LIGHT / wavelength
        freq_grid = FrequencyGrid(values=np.array([freq_hz]))

        # Get S-parameters
        s_matrix = waveguide.get_s(freq_grid)

        # Extract transmission coefficient
        t_forward = s_matrix[0, 1, 0]  # S21

        # Expected phase
        expected_phase = 2 * np.pi * neff * length / wavelength

        # Extract phase from transmission coefficient
        actual_phase = -np.angle(t_forward)  # negative because exp(-j*phase)

        # Normalize both phases to [-π, π] for comparison
        expected_phase_normalized = np.angle(np.exp(1j * expected_phase))
        actual_phase_normalized = np.angle(np.exp(1j * actual_phase))

        # Verify phase (within numerical precision)
        np.testing.assert_allclose(actual_phase_normalized, expected_phase_normalized, rtol=1e-10)

    def test_waveguide_loss(self):
        """Test loss calculation: loss=3dB/cm, L=2cm.
        
        Expected |S21|² = 10^(-6/10) ≈ 0.251
        """
        # Create waveguide
        length = 2e-2  # 2 cm
        loss = 3.0  # 3 dB/cm
        waveguide = create_waveguide("wg", length=length, neff=2.0, loss=loss)

        # Arbitrary frequency (loss is frequency-independent)
        freq_grid = FrequencyGrid(values=np.array([1e14]))

        # Get S-parameters
        s_matrix = waveguide.get_s(freq_grid)

        # Extract transmission coefficient
        t_forward = s_matrix[0, 1, 0]  # S21

        # Expected amplitude: 10^(-loss * length / 20)
        expected_amplitude = 10 ** (-loss * length / 20)

        # Expected |S21|²
        expected_power = expected_amplitude ** 2

        # Actual |S21|²
        actual_power = np.abs(t_forward) ** 2

        # Verify power (within numerical precision)
        np.testing.assert_allclose(actual_power, expected_power, rtol=1e-10)

    def test_waveguide_reciprocity(self):
        """Test reciprocity: S21 = S12."""
        waveguide = create_waveguide("wg", length=100e-6, neff=2.5, loss=1.0)

        freq_grid = FrequencyGrid(values=np.array([1e14, 2e14, 3e14]))

        s_matrix = waveguide.get_s(freq_grid)

        # S21 = s_matrix[:, 1, 0]
        # S12 = s_matrix[:, 0, 1]
        s21 = s_matrix[:, 1, 0]
        s12 = s_matrix[:, 0, 1]

        np.testing.assert_array_almost_equal(s21, s12)

    def test_waveguide_zero_length(self):
        """Test zero-length waveguide: L=0 → S=[0,1;1,0]."""
        waveguide = create_waveguide("wg", length=0.0, neff=2.5, loss=0.0)

        freq_grid = FrequencyGrid(values=np.array([1e14]))

        s_matrix = waveguide.get_s(freq_grid)

        # Expected S-matrix for zero-length: [[0, 1], [1, 0]]
        expected_s = np.array([[[0, 1], [1, 0]]], dtype=np.complex128)

        np.testing.assert_array_almost_equal(s_matrix, expected_s)
