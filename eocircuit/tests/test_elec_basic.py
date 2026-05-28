"""Tests for basic electrical components."""

import numpy as np
import pytest
from pydantic import ValidationError

from eocircuit.core.types import FrequencyGrid
from eocircuit.elec.basic import Resistor, Capacitor, Inductor, Conductor


class TestResistor:
    """Tests for Resistor component."""
    
    def test_resistor_impedance(self):
        """Test resistor impedance is constant across frequency.
        
        For R=100Ω, Z should be [[100, 0], [0, 100]] at any frequency.
        """
        r = Resistor.new("R1", 100.0)
        
        # Test at multiple frequencies
        freq = FrequencyGrid(values=np.array([1e6, 1e9, 10e9]))
        z = r.get_z(freq)
        
        # Check shape
        assert z.shape == (3, 2, 2)
        
        # Check values at each frequency
        for i in range(3):
            assert np.isclose(z[i, 0, 0], 100.0)
            assert np.isclose(z[i, 1, 1], 100.0)
            assert np.isclose(z[i, 0, 1], 0.0)
            assert np.isclose(z[i, 1, 0], 0.0)
    
    def test_resistor_admittance(self):
        """Test resistor admittance is 1/R across frequency."""
        r = Resistor.new("R1", 100.0)
        
        freq = FrequencyGrid(values=np.array([1e6, 1e9]))
        y = r.get_y(freq)
        
        # Check shape
        assert y.shape == (2, 2, 2)
        
        # Check values: Y = 1/R = 0.01 S
        for i in range(2):
            assert np.isclose(y[i, 0, 0], 0.01)
            assert np.isclose(y[i, 1, 1], 0.01)
            assert np.isclose(y[i, 0, 1], 0.0)
            assert np.isclose(y[i, 1, 0], 0.0)
    
    def test_resistor_negative_resistance(self):
        """Test that negative resistance raises ValidationError."""
        with pytest.raises(ValidationError):
            Resistor.new("R1", -50.0)


class TestCapacitor:
    """Tests for Capacitor component."""
    
    def test_capacitor_impedance(self):
        """Test capacitor impedance at 1GHz.
        
        For C=1pF at f=1GHz:
        Z = 1/(jωC) = 1/(j*2π*1e9*1e-12) ≈ -j159.15Ω
        |Z| ≈ 159Ω, phase ≈ -90°
        """
        c = Capacitor.new("C1", 1e-12)  # 1 pF
        
        freq = FrequencyGrid(values=np.array([1e9]))  # 1 GHz
        z = c.get_z(freq)
        
        # Check shape
        assert z.shape == (1, 2, 2)
        
        # Calculate expected impedance
        # Z = 1/(j*2π*f*C)
        jw = 2j * np.pi * 1e9
        z_expected = 1.0 / (jw * 1e-12)
        
        # Check magnitude and phase
        z_actual = z[0, 0, 0]
        assert np.isclose(np.abs(z_actual), np.abs(z_expected), rtol=1e-6)
        assert np.isclose(np.angle(z_actual), np.angle(z_expected), atol=1e-6)
        
        # Verify phase is approximately -90°
        phase_deg = np.angle(z_actual) * 180 / np.pi
        assert np.isclose(phase_deg, -90.0, atol=1.0)
    
    def test_capacitor_admittance(self):
        """Test capacitor admittance Y = jωC."""
        c = Capacitor.new("C1", 1e-12)  # 1 pF
        
        freq = FrequencyGrid(values=np.array([1e9]))  # 1 GHz
        y = c.get_y(freq)
        
        # Check shape
        assert y.shape == (1, 2, 2)
        
        # Calculate expected admittance
        # Y = j*2π*f*C
        jw = 2j * np.pi * 1e9
        y_expected = jw * 1e-12
        
        # Check values
        y_actual = y[0, 0, 0]
        assert np.isclose(y_actual, y_expected, rtol=1e-6)
        
        # Verify phase is approximately +90°
        phase_deg = np.angle(y_actual) * 180 / np.pi
        assert np.isclose(phase_deg, 90.0, atol=1.0)


class TestInductor:
    """Tests for Inductor component."""
    
    def test_inductor_impedance(self):
        """Test inductor impedance at 1GHz.
        
        For L=1nH at f=1GHz:
        Z = jωL = j*2π*1e9*1e-9 ≈ j6.28Ω
        |Z| ≈ 6.28Ω, phase ≈ +90°
        """
        l = Inductor.new("L1", 1e-9)  # 1 nH
        
        freq = FrequencyGrid(values=np.array([1e9]))  # 1 GHz
        z = l.get_z(freq)
        
        # Check shape
        assert z.shape == (1, 2, 2)
        
        # Calculate expected impedance
        # Z = j*2π*f*L
        jw = 2j * np.pi * 1e9
        z_expected = jw * 1e-9
        
        # Check magnitude and phase
        z_actual = z[0, 0, 0]
        assert np.isclose(np.abs(z_actual), np.abs(z_expected), rtol=1e-6)
        assert np.isclose(np.angle(z_actual), np.angle(z_expected), atol=1e-6)
        
        # Verify phase is approximately +90°
        phase_deg = np.angle(z_actual) * 180 / np.pi
        assert np.isclose(phase_deg, 90.0, atol=1.0)
    
    def test_inductor_admittance(self):
        """Test inductor admittance Y = 1/(jωL)."""
        l = Inductor.new("L1", 1e-9)  # 1 nH
        
        freq = FrequencyGrid(values=np.array([1e9]))  # 1 GHz
        y = l.get_y(freq)
        
        # Check shape
        assert y.shape == (1, 2, 2)
        
        # Calculate expected admittance
        # Y = 1/(j*2π*f*L)
        jw = 2j * np.pi * 1e9
        y_expected = 1.0 / (jw * 1e-9)
        
        # Check values
        y_actual = y[0, 0, 0]
        assert np.isclose(y_actual, y_expected, rtol=1e-6)
        
        # Verify phase is approximately -90°
        phase_deg = np.angle(y_actual) * 180 / np.pi
        assert np.isclose(phase_deg, -90.0, atol=1.0)


class TestConductor:
    """Tests for Conductor component."""
    
    def test_conductor_admittance(self):
        """Test conductor admittance is constant across frequency.
        
        For G=0.01S, Y should be [[0.01, 0], [0, 0.01]] at any frequency.
        """
        g = Conductor.new("G1", 0.01)
        
        # Test at multiple frequencies
        freq = FrequencyGrid(values=np.array([1e6, 1e9, 10e9]))
        y = g.get_y(freq)
        
        # Check shape
        assert y.shape == (3, 2, 2)
        
        # Check values at each frequency
        for i in range(3):
            assert np.isclose(y[i, 0, 0], 0.01)
            assert np.isclose(y[i, 1, 1], 0.01)
            assert np.isclose(y[i, 0, 1], 0.0)
            assert np.isclose(y[i, 1, 0], 0.0)
    
    def test_conductor_impedance(self):
        """Test conductor impedance is 1/G across frequency."""
        g = Conductor.new("G1", 0.01)
        
        freq = FrequencyGrid(values=np.array([1e6, 1e9]))
        z = g.get_z(freq)
        
        # Check shape
        assert z.shape == (2, 2, 2)
        
        # Check values: Z = 1/G = 100Ω
        for i in range(2):
            assert np.isclose(z[i, 0, 0], 100.0)
            assert np.isclose(z[i, 1, 1], 100.0)
            assert np.isclose(z[i, 0, 1], 0.0)
            assert np.isclose(z[i, 1, 0], 0.0)
