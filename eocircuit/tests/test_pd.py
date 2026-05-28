"""Tests for Photodetector (PD) component."""

import numpy as np
import pytest

from eocircuit.core.types import FrequencyGrid, Port, PortDomain
from eocircuit.eo.pd import Photodetector, PhotodetectorParams, create_pd


class TestPhotodetector:
    """Tests for Photodetector component."""
    
    def test_pd_creation(self):
        """Test photodetector creation with factory method."""
        pd = Photodetector.new("PD1", responsivity=0.8, dark_current=1e-9)
        
        assert pd.name == "PD1"
        assert pd.num_of_ports == 3
        assert pd.params.responsivity == 0.8
        assert pd.params.dark_current == 1e-9
    
    def test_pd_ports(self):
        """Test photodetector has correct ports."""
        pd = Photodetector.new("PD1", responsivity=0.8)
        
        # Check port count
        assert len(pd.ports) == 3
        
        # Check port names and domains
        port_names = {p.name for p in pd.ports}
        assert "opt_in" in port_names
        assert "elec_p" in port_names
        assert "elec_n" in port_names
        
        # Check port domains
        opt_port = next(p for p in pd.ports if p.name == "opt_in")
        assert opt_port.domain == PortDomain.OPTICAL
        
        elec_p_port = next(p for p in pd.ports if p.name == "elec_p")
        assert elec_p_port.domain == PortDomain.ELECTRICAL
        
        elec_n_port = next(p for p in pd.ports if p.name == "elec_n")
        assert elec_n_port.domain == PortDomain.ELECTRICAL
    
    def test_pd_responsivity(self):
        """Test photodetector responsivity: 1mW optical, R=0.8 A/W → I=0.8mA.
        
        For 1mW (0.001W) optical power and responsivity 0.8 A/W:
        I_photo = 0.8 × 0.001 = 0.0008 A = 0.8 mA
        """
        pd = Photodetector.new("PD1", responsivity=0.8, dark_current=0.0)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        p_opt = 0.001  # 1 mW in Watts
        
        i = pd.get_i(freq, p_opt)
        
        # Check shape
        assert i.shape == (1,)
        
        # Check photocurrent value: 0.8 A/W × 0.001 W = 0.0008 A
        assert np.isclose(i[0], 0.0008, rtol=1e-6)
    
    def test_pd_dark_current(self):
        """Test photodetector dark current: no optical input → I=I_dark=10nA."""
        dark_current = 1e-8  # 10 nA
        pd = Photodetector.new("PD1", responsivity=0.8, dark_current=dark_current)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        p_opt = 0.0  # No optical power
        
        i = pd.get_i(freq, p_opt)
        
        # Check shape
        assert i.shape == (1,)
        
        # Check dark current: I = 0.8 × 0 + 1e-8 = 1e-8 A
        assert np.isclose(i[0], dark_current, rtol=1e-6)
    
    def test_pd_with_load(self):
        """Test photodetector with load resistor: PD + R_load=1kΩ → V=0.8V.
        
        For I_photo = 0.8 mA and R_load = 1 kΩ:
        V = I × R = 0.0008 × 1000 = 0.8 V
        """
        pd = Photodetector.new("PD1", responsivity=0.8, dark_current=0.0)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        p_opt = 0.001  # 1 mW
        
        i = pd.get_i(freq, p_opt)
        
        # Simulate load resistor
        r_load = 1000.0  # 1 kΩ
        v_out = i[0] * r_load
        
        # Check voltage: 0.0008 A × 1000 Ω = 0.8 V
        assert np.isclose(v_out, 0.8, rtol=1e-6)
    
    def test_pd_s_parameters(self):
        """Test photodetector S-parameters are near-zero (absorbing + current source)."""
        pd = Photodetector.new("PD1", responsivity=0.8)
        
        freq = FrequencyGrid(values=np.array([1e9, 10e9]))
        s = pd.get_s(freq)
        
        # Check shape
        assert s.shape == (2, 3, 3)
        
        # All S-parameters should be near-zero (ideal absorption + current source)
        assert np.allclose(s, 0.0, atol=1e-10)
    
    def test_pd_z_parameters(self):
        """Test photodetector Z-parameters.
        
        Optical port: near-zero impedance (absorbs light)
        Electrical ports: near-infinite impedance (current source)
        """
        pd = Photodetector.new("PD1", responsivity=0.8)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        z = pd.get_z(freq)
        
        # Check shape
        assert z.shape == (1, 3, 3)
        
        # Optical port (index 0): near-zero impedance
        assert np.isclose(z[0, 0, 0], 1e-12, rtol=1e-10)
        
        # Electrical ports (indices 1, 2): near-infinite impedance
        assert np.isclose(z[0, 1, 1], 1e12, rtol=1e-10)
        assert np.isclose(z[0, 2, 2], 1e12, rtol=1e-10)
    
    def test_pd_y_parameters(self):
        """Test photodetector Y-parameters.
        
        Optical port: near-infinite admittance (absorbs light)
        Electrical ports: near-zero admittance (current source)
        """
        pd = Photodetector.new("PD1", responsivity=0.8)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        y = pd.get_y(freq)
        
        # Check shape
        assert y.shape == (1, 3, 3)
        
        # Optical port (index 0): near-infinite admittance
        assert np.isclose(y[0, 0, 0], 1e12, rtol=1e-10)
        
        # Electrical ports (indices 1, 2): near-zero admittance
        assert np.isclose(y[0, 1, 1], 1e-12, rtol=1e-10)
        assert np.isclose(y[0, 2, 2], 1e-12, rtol=1e-10)
    
    def test_pd_factory_function(self):
        """Test create_pd factory function."""
        pd = create_pd("PD2", responsivity=0.5, dark_current=5e-9)
        
        assert pd.name == "PD2"
        assert pd.params.responsivity == 0.5
        assert pd.params.dark_current == 5e-9
    
    def test_pd_multiple_frequencies(self):
        """Test photodetector current across multiple frequencies."""
        pd = Photodetector.new("PD1", responsivity=0.8, dark_current=1e-9)
        
        freq = FrequencyGrid(values=np.array([1e6, 1e9, 10e9, 100e9]))
        p_opt = 0.001  # 1 mW
        
        i = pd.get_i(freq, p_opt)
        
        # Check shape
        assert i.shape == (4,)
        
        # Current should be constant across all frequencies (DC model)
        expected_i = 0.8 * 0.001 + 1e-9
        for freq_idx in range(4):
            assert np.isclose(i[freq_idx], expected_i, rtol=1e-6)
    
    def test_pd_zero_responsivity(self):
        """Test photodetector with zero responsivity."""
        pd = Photodetector.new("PD1", responsivity=0.0, dark_current=1e-9)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        p_opt = 0.001  # 1 mW
        
        i = pd.get_i(freq, p_opt)
        
        # Current should be only dark current
        assert np.isclose(i[0], 1e-9, rtol=1e-6)
    
    def test_pd_high_power(self):
        """Test photodetector with high optical power."""
        pd = Photodetector.new("PD1", responsivity=0.8, dark_current=0.0)
        
        freq = FrequencyGrid(values=np.array([1e9]))
        p_opt = 0.1  # 100 mW
        
        i = pd.get_i(freq, p_opt)
        
        # Current should be 0.8 × 0.1 = 0.08 A = 80 mA
        assert np.isclose(i[0], 0.08, rtol=1e-6)
