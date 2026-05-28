"""Tests for eocircuit.core.types module.

TDD tests for Port, FrequencyGrid, WavelengthGrid, and type aliases.
"""

import pytest
import numpy as np
from pydantic import ValidationError

from eocircuit.core.types import (
    Port,
    PortDomain,
    FrequencyGrid,
    WavelengthGrid,
    ComplexArray,
    RealArray,
    SPEED_OF_LIGHT,
)


class TestPortCreation:
    """Test Port creation and default values."""

    def test_port_creation_with_defaults(self):
        """Verify default z0=50, name set correctly."""
        port = Port(name="P1", domain=PortDomain.ELECTRICAL)
        assert port.name == "P1"
        assert port.domain == PortDomain.ELECTRICAL
        assert port.z0 == 50.0
        assert isinstance(port.index, int)

    def test_port_creation_with_custom_z0(self):
        """Verify custom impedance is set."""
        port = Port(name="P2", domain=PortDomain.OPTICAL, z0=75.0)
        assert port.z0 == 75.0

    def test_port_auto_increment_index(self):
        """Verify port indices auto-increment."""
        # Reset counter for test
        import eocircuit.core.types as types_module
        types_module._PORT_COUNTER = 0
        port1 = Port(name="P1", domain=PortDomain.ELECTRICAL)
        port2 = Port(name="P2", domain=PortDomain.ELECTRICAL)
        port3 = Port(name="P3", domain=PortDomain.ELECTRICAL)
        
        assert port1.index == 0
        assert port2.index == 1
        assert port3.index == 2


class TestPortDomainValidation:
    """Test PortDomain enum validation."""

    def test_electrical_domain(self):
        """Verify ELECTRICAL domain."""
        port = Port(name="P1", domain=PortDomain.ELECTRICAL)
        assert port.domain == PortDomain.ELECTRICAL
        assert port.domain.value == "electrical"

    def test_optical_domain(self):
        """Verify OPTICAL domain."""
        port = Port(name="P1", domain=PortDomain.OPTICAL)
        assert port.domain == PortDomain.OPTICAL
        assert port.domain.value == "optical"

    def test_invalid_domain_raises_error(self):
        """Verify invalid domain raises ValidationError."""
        with pytest.raises(ValidationError):
            Port(name="P1", domain="invalid")


class TestPortImmutability:
    """Test Port immutability after creation."""

    def test_port_immutability_name(self):
        """Verify port.name cannot be modified."""
        port = Port(name="P1", domain=PortDomain.ELECTRICAL)
        with pytest.raises(Exception):  # ValidationError or AttributeError
            port.name = "P2"

    def test_port_immutability_z0(self):
        """Verify port.z0 cannot be modified."""
        port = Port(name="P1", domain=PortDomain.ELECTRICAL)
        with pytest.raises(Exception):
            port.z0 = 75.0

    def test_port_immutability_domain(self):
        """Verify port.domain cannot be modified."""
        port = Port(name="P1", domain=PortDomain.ELECTRICAL)
        with pytest.raises(Exception):
            port.domain = PortDomain.OPTICAL


class TestPortValidation:
    """Test Port field validation."""

    def test_negative_z0_raises_error(self):
        """Verify negative impedance raises ValidationError."""
        with pytest.raises(ValidationError):
            Port(name="P1", domain=PortDomain.ELECTRICAL, z0=-50.0)

    def test_zero_z0_raises_error(self):
        """Verify zero impedance raises ValidationError."""
        with pytest.raises(ValidationError):
            Port(name="P1", domain=PortDomain.ELECTRICAL, z0=0.0)


class TestFrequencyGridValidation:
    """Test FrequencyGrid validation."""

    def test_frequency_grid_creation(self):
        """Verify FrequencyGrid creation with valid frequencies."""
        freqs = np.array([1e9, 10e9, 100e9])
        grid = FrequencyGrid(values=freqs)
        np.testing.assert_array_equal(grid.values, freqs)

    def test_negative_frequency_raises_error(self):
        """Verify negative frequency raises ValidationError."""
        freqs = np.array([1e9, -10e9, 100e9])
        with pytest.raises(ValidationError):
            FrequencyGrid(values=freqs)

    def test_zero_frequency_allowed(self):
        """Verify zero frequency is allowed in grid."""
        freqs = np.array([0.0, 1e9, 10e9])
        grid = FrequencyGrid(values=freqs)
        np.testing.assert_array_equal(grid.values, freqs)


class TestWavelengthGridValidation:
    """Test WavelengthGrid validation."""

    def test_wavelength_grid_creation(self):
        """Verify WavelengthGrid creation with valid wavelengths."""
        wavelengths = np.array([1550e-9, 1310e-9, 1064e-9])
        grid = WavelengthGrid(values=wavelengths)
        np.testing.assert_array_equal(grid.values, wavelengths)

    def test_negative_wavelength_raises_error(self):
        """Verify negative wavelength raises ValidationError."""
        wavelengths = np.array([1550e-9, -1310e-9, 1064e-9])
        with pytest.raises(ValidationError):
            WavelengthGrid(values=wavelengths)

    def test_zero_wavelength_allowed(self):
        """Verify zero wavelength is allowed in grid."""
        wavelengths = np.array([0.0, 1550e-9, 1310e-9])
        grid = WavelengthGrid(values=wavelengths)
        np.testing.assert_array_equal(grid.values, wavelengths)


class TestFrequencyToWavelengthConversion:
    """Test frequency to wavelength conversion."""

    def test_wavelength_to_frequency_conversion(self):
        """Verify 1550nm → ~193.4 THz conversion."""
        wavelength_m = 1550e-9  # 1550 nm in meters
        expected_freq_hz = SPEED_OF_LIGHT / wavelength_m
        expected_freq_thz = expected_freq_hz / 1e12
        
        # Should be approximately 193.4 THz
        assert 193.3 < expected_freq_thz < 193.5

    def test_frequency_grid_to_wavelength(self):
        """Verify FrequencyGrid.to_wavelength() conversion."""
        # 193.4 THz corresponds to ~1550 nm
        freq_hz = np.array([193.4e12])
        freq_grid = FrequencyGrid(values=freq_hz)
        wl_grid = freq_grid.to_wavelength()
        
        expected_wl = SPEED_OF_LIGHT / freq_hz[0]
        np.testing.assert_allclose(wl_grid.values, [expected_wl], rtol=1e-6)

    def test_wavelength_grid_to_frequency(self):
        """Verify WavelengthGrid.to_frequency() conversion."""
        wl_m = np.array([1550e-9])
        wl_grid = WavelengthGrid(values=wl_m)
        freq_grid = wl_grid.to_frequency()
        
        expected_freq = SPEED_OF_LIGHT / wl_m[0]
        np.testing.assert_allclose(freq_grid.values, [expected_freq], rtol=1e-6)

    def test_frequency_to_wavelength_zero_raises_error(self):
        """Verify zero frequency raises error in to_wavelength()."""
        freqs = np.array([0.0, 1e9])
        freq_grid = FrequencyGrid(values=freqs)
        with pytest.raises(ValueError, match="Cannot convert zero frequency"):
            freq_grid.to_wavelength()

    def test_wavelength_to_frequency_zero_raises_error(self):
        """Verify zero wavelength raises error in to_frequency()."""
        wavelengths = np.array([0.0, 1550e-9])
        wl_grid = WavelengthGrid(values=wavelengths)
        with pytest.raises(ValueError, match="Cannot convert zero wavelength"):
            wl_grid.to_frequency()


class TestTypeAliases:
    """Test type aliases for numpy arrays."""

    def test_complex_array_type(self):
        """Verify ComplexArray type alias."""
        arr = np.array([1 + 2j, 3 + 4j])
        assert arr.dtype == np.complex128

    def test_real_array_type(self):
        """Verify RealArray type alias."""
        arr = np.array([1.0, 2.0, 3.0])
        assert np.issubdtype(arr.dtype, np.floating)
