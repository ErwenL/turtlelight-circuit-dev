"""Tests for Component abstract base class and utilities.

Tests cover:
- Component ABC instantiation rules
- Concrete component implementation
- Z↔S parameter conversion
- Component immutability
- Matrix shape validation
"""

import numpy as np
import pytest
from typing import List
from numpy.typing import NDArray

from eocircuit.core.component import (
    Component,
    ComponentParams,
    z_to_s,
    s_to_z,
)
from eocircuit.core.types import Port, PortDomain, FrequencyGrid, ComplexArray


class ResistorParams(ComponentParams):
    """Parameters for a simple resistor component."""

    resistance: float


class ConcreteResistor(Component):
    """Concrete 2-port resistor for testing.
    
    Simple component that returns Z = diag(R, R) for all frequencies.
    """

    def get_s(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get S-parameters for resistor."""
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        
        # Get Z-parameters first
        z_matrix = self.get_z(freq)
        
        # Convert to S-parameters using z0 from ports
        z0_values = [port.z0 for port in self.ports]
        s_matrix = z_to_s(z_matrix, z0_values)
        
        return s_matrix

    def get_z(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get Z-parameters for resistor.
        
        Returns diagonal matrix with resistance on diagonal.
        """
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        
        z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        
        for i in range(n_freq):
            z_matrix[i] = np.diag([self.params.resistance] * n_ports)
        
        return z_matrix

    def get_y(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get Y-parameters for resistor.
        
        Y = inv(Z) for each frequency.
        """
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        
        y_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        z_matrix = self.get_z(freq)
        
        for i in range(n_freq):
            y_matrix[i] = np.linalg.inv(z_matrix[i])
        
        return y_matrix


class TestComponentABC:
    """Test Component abstract base class."""

    def test_component_abc_cannot_instantiate(self):
        """Direct Component() instantiation should raise TypeError."""
        port1 = Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        port2 = Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        
        with pytest.raises(TypeError):
            Component(ports=[port1, port2], params=ComponentParams())

    def test_concrete_component_instantiation(self):
        """ConcreteResistor should instantiate successfully."""
        port1 = Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        port2 = Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        params = ResistorParams(resistance=50.0)
        
        resistor = ConcreteResistor(name="R1", ports=[port1, port2], params=params)
        
        assert resistor.num_of_ports == 2
        assert resistor.params.resistance == 50.0


class TestZToSConversion:
    """Test Z-to-S parameter conversion."""

    def test_z_to_s_matched(self):
        """Z=50Ω diagonal should convert to S≈0 matrix."""
        # Create a 2-port Z-matrix with Z=50Ω on diagonal
        n_freq = 3
        n_ports = 2
        z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        
        for i in range(n_freq):
            z_matrix[i] = np.diag([50.0, 50.0])
        
        # Convert to S-parameters with z0=50Ω
        s_matrix = z_to_s(z_matrix, 50.0)
        
        # S should be approximately zero (matched impedance)
        np.testing.assert_allclose(s_matrix, 0.0, atol=1e-10)

    def test_z_to_s_open_circuit(self):
        """Z=∞ (open circuit) should convert to S≈1 on diagonal, 0 off-diagonal."""
        n_freq = 2
        n_ports = 2
        z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        
        # Large impedance to approximate open circuit
        for i in range(n_freq):
            z_matrix[i] = np.diag([1e10, 1e10])
        
        s_matrix = z_to_s(z_matrix, 50.0)
        
        # S_ii should be approximately 1 (open circuit reflection)
        # S_ij (i≠j) should be 0 (no coupling between independent ports)
        for i in range(n_freq):
            np.testing.assert_allclose(np.diag(s_matrix[i]), [1.0, 1.0], atol=1e-3)
            # Check off-diagonal are zero
            s_copy = s_matrix[i].copy()
            np.fill_diagonal(s_copy, 0)
            np.testing.assert_allclose(s_copy, 0.0, atol=1e-10)

    def test_z_to_s_short_circuit(self):
        """Z=0 (short circuit) should convert to S≈-1 on diagonal, 0 off-diagonal."""
        n_freq = 2
        n_ports = 2
        z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        
        # Small impedance to approximate short circuit
        for i in range(n_freq):
            z_matrix[i] = np.diag([1e-10, 1e-10])
        
        s_matrix = z_to_s(z_matrix, 50.0)
        
        # S_ii should be approximately -1 (short circuit reflection)
        # S_ij (i≠j) should be 0 (no coupling between independent ports)
        for i in range(n_freq):
            np.testing.assert_allclose(np.diag(s_matrix[i]), [-1.0, -1.0], atol=1e-3)
            # Check off-diagonal are zero
            s_copy = s_matrix[i].copy()
            np.fill_diagonal(s_copy, 0)
            np.testing.assert_allclose(s_copy, 0.0, atol=1e-10)

    def test_z_to_s_with_list_z0(self):
        """z_to_s should accept list of z0 values."""
        n_freq = 1
        n_ports = 2
        z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        z_matrix[0] = np.diag([50.0, 75.0])
        
        # Use different z0 for each port
        s_matrix = z_to_s(z_matrix, [50.0, 75.0])
        
        # Should be approximately zero (matched)
        np.testing.assert_allclose(s_matrix, 0.0, atol=1e-10)

    def test_z_to_s_invalid_z0_length(self):
        """z_to_s should raise ValueError if z0 list length mismatches."""
        n_freq = 1
        n_ports = 2
        z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        z_matrix[0] = np.diag([50.0, 50.0])
        
        with pytest.raises(ValueError, match="z0 length"):
            z_to_s(z_matrix, [50.0])  # Wrong length


class TestSToZConversion:
    """Test S-to-Z parameter conversion."""

    def test_s_to_z_identity(self):
        """S=0 should convert to Z≈50Ω diagonal."""
        n_freq = 3
        n_ports = 2
        s_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        
        # S=0 (matched)
        z_matrix = s_to_z(s_matrix, 50.0)
        
        # Z should be approximately 50Ω on diagonal
        for i in range(n_freq):
            np.testing.assert_allclose(
                np.diag(z_matrix[i]),
                [50.0, 50.0],
                atol=1e-10,
            )

    def test_s_to_z_open_circuit(self):
        """S≈1 should convert to Z≈∞."""
        n_freq = 1
        n_ports = 2
        s_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        s_matrix[0] = np.diag([0.99999, 0.99999])  # Close to 1
        
        z_matrix = s_to_z(s_matrix, 50.0)
        
        # Z should be very large (open circuit)
        assert np.abs(z_matrix[0, 0, 0]) > 1e6
        assert np.abs(z_matrix[0, 1, 1]) > 1e6

    def test_s_to_z_with_list_z0(self):
        """s_to_z should accept list of z0 values."""
        n_freq = 1
        n_ports = 2
        s_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        
        z_matrix = s_to_z(s_matrix, [50.0, 75.0])
        
        # Z should match z0 values on diagonal
        np.testing.assert_allclose(
            np.diag(z_matrix[0]),
            [50.0, 75.0],
            atol=1e-10,
        )

    def test_s_to_z_invalid_z0_length(self):
        """s_to_z should raise ValueError if z0 list length mismatches."""
        n_freq = 1
        n_ports = 2
        s_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        
        with pytest.raises(ValueError, match="z0 length"):
            s_to_z(s_matrix, [50.0])  # Wrong length
