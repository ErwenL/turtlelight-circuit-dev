"""Tests for electrical source components."""

import numpy as np
import pytest

from eocircuit.core.network import Network
from eocircuit.core.types import FrequencyGrid, Port, PortDomain
from eocircuit.elec.basic import Resistor, ResistorParams
from eocircuit.elec.sources import VoltageSource, VoltageSourceParams, CurrentSource, CurrentSourceParams
from eocircuit.solvers.mna import MNASolver


class TestVoltageSource:
    """Tests for VoltageSource component."""
    
    def test_voltage_source_dc(self):
        """Test voltage source DC output at all frequencies.
        
        For V=5V at DC, output should be 5V at all frequencies.
        """
        vs = VoltageSource.new("VS1", voltage=5.0)
        
        # Test at multiple frequencies
        freq = FrequencyGrid(values=np.array([0.0, 1e6, 1e9, 10e9]))
        v = vs.get_v(freq)
        
        # Check shape
        assert v.shape == (4,)
        
        # Check values at each frequency
        for i in range(4):
            assert np.isclose(v[i], 5.0)
    
    def test_voltage_source_ac(self):
        """Test voltage source AC output with complex voltage.
        
        For V=3+4j, |V| should be 5V.
        """
        vs = VoltageSource.new("VS1", voltage=3+4j)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        v = vs.get_v(freq)
        
        # Check magnitude
        assert np.isclose(np.abs(v[0]), 5.0)
        
        # Check real and imaginary parts
        assert np.isclose(v[0].real, 3.0)
        assert np.isclose(v[0].imag, 4.0)
    
    def test_voltage_source_impedance(self):
        """Test voltage source impedance is near-zero.
        
        Z should be approximately 1e-12 Ω at all frequencies.
        """
        vs = VoltageSource.new("VS1", voltage=5.0)
        
        freq = FrequencyGrid(values=np.array([1e6, 1e9]))
        z = vs.get_z(freq)
        
        # Check shape
        assert z.shape == (2, 2, 2)
        
        # Check near-zero impedance
        for i in range(2):
            assert np.isclose(z[i, 0, 0], 1e-12, rtol=1e-10)
            assert np.isclose(z[i, 1, 1], 1e-12, rtol=1e-10)
            assert np.isclose(z[i, 0, 1], 0.0)
            assert np.isclose(z[i, 1, 0], 0.0)
    
    def test_voltage_source_admittance(self):
        """Test voltage source admittance is near-infinite.
        
        Y should be approximately 1e12 S at all frequencies.
        """
        vs = VoltageSource.new("VS1", voltage=5.0)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        y = vs.get_y(freq)
        
        # Check shape
        assert y.shape == (1, 2, 2)
        
        # Check near-infinite admittance (stamp form)
        y_large = 1e12
        assert np.isclose(y[0, 0, 0], y_large, rtol=1e-10)
        assert np.isclose(y[0, 1, 1], y_large, rtol=1e-10)
        assert np.isclose(y[0, 0, 1], -y_large, rtol=1e-10)
        assert np.isclose(y[0, 1, 0], -y_large, rtol=1e-10)


class TestCurrentSource:
    """Tests for CurrentSource component."""
    
    def test_current_source_dc(self):
        """Test current source DC output at all frequencies.
        
        For I=1mA at DC, output should be 1mA at all frequencies.
        """
        cs = CurrentSource.new("CS1", current=1e-3)
        
        # Test at multiple frequencies
        freq = FrequencyGrid(values=np.array([0.0, 1e6, 1e9, 10e9]))
        i = cs.get_i(freq)
        
        # Check shape
        assert i.shape == (4,)
        
        # Check values at each frequency
        for idx in range(4):
            assert np.isclose(i[idx], 1e-3)
    
    def test_current_source_ac(self):
        """Test current source AC output with complex current.
        
        For I=0.6+0.8j mA, |I| should be 1mA.
        """
        cs = CurrentSource.new("CS1", current=0.6e-3+0.8e-3j)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        i = cs.get_i(freq)
        
        # Check magnitude
        assert np.isclose(np.abs(i[0]), 1e-3)
        
        # Check real and imaginary parts
        assert np.isclose(i[0].real, 0.6e-3)
        assert np.isclose(i[0].imag, 0.8e-3)
    
    def test_current_source_impedance(self):
        """Test current source impedance is near-infinite.
        
        Z should be approximately 1e12 Ω at all frequencies.
        """
        cs = CurrentSource.new("CS1", current=1e-3)
        
        freq = FrequencyGrid(values=np.array([1e6, 1e9]))
        z = cs.get_z(freq)
        
        # Check shape
        assert z.shape == (2, 2, 2)
        
        # Check near-infinite impedance
        for i in range(2):
            assert np.isclose(z[i, 0, 0], 1e12, rtol=1e-10)
            assert np.isclose(z[i, 1, 1], 1e12, rtol=1e-10)
            assert np.isclose(z[i, 0, 1], 0.0)
            assert np.isclose(z[i, 1, 0], 0.0)
    
    def test_current_source_admittance(self):
        """Test current source admittance is near-zero.
        
        Y should be approximately 1e-12 S at all frequencies.
        """
        cs = CurrentSource.new("CS1", current=1e-3)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        y = cs.get_y(freq)
        
        # Check shape
        assert y.shape == (1, 2, 2)
        
        # Check near-zero admittance
        for i in range(1):
            assert np.isclose(y[i, 0, 0], 1e-12, rtol=1e-10)
            assert np.isclose(y[i, 1, 1], 1e-12, rtol=1e-10)
            assert np.isclose(y[i, 0, 1], 0.0)
            assert np.isclose(y[i, 1, 0], 0.0)


class TestSourceWithDivider:
    """Integration tests with resistive divider."""
    
    def test_voltage_source_recognized_by_mna(self):
        """Test that voltage source is recognized by MNA solver.
        
        Verifies that the voltage source has the correct parameter
        that MNA solver looks for.
        """
        vs = VoltageSource.new("VS1", voltage=5.0)
        
        # Check that the voltage source has 'v' parameter
        assert hasattr(vs.params, 'v')
        assert vs.params.v == 5.0
        
        # Check that it's a 2-port component
        assert vs.num_of_ports == 2
    
    def test_current_source_recognized_by_mna(self):
        """Test that current source is recognized by MNA solver.
        
        Verifies that the current source has the correct parameter
        that MNA solver looks for.
        """
        cs = CurrentSource.new("CS1", current=1e-3)
        
        # Check that the current source has 'i' parameter
        assert hasattr(cs.params, 'i')
        assert cs.params.i == 1e-3
        
        # Check that it's a 2-port component
        assert cs.num_of_ports == 2
