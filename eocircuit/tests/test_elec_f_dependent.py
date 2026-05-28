"""Tests for frequency-dependent electrical components."""

import numpy as np
import pytest
from loguru import logger

from eocircuit.core.types import FrequencyGrid
from eocircuit.elec.f_dependent import (
    FreqDependentResistor,
    FreqDependentCapacitor,
    FreqDependentInductor,
)


class TestFreqDependentResistor:
    """Tests for FreqDependentResistor component."""
    
    def test_freq_dep_resistor(self):
        """Test frequency-dependent resistor with interpolation.
        
        At f=[1,2,3]GHz, r=[50,60,70]Ω
        Z should vary with frequency.
        """
        freq_points = [1e9, 2e9, 3e9]  # 1, 2, 3 GHz
        r_values = [50.0, 60.0, 70.0]  # 50, 60, 70 Ohms
        
        r = FreqDependentResistor.new("R_freq", freq_points, r_values)
        
        # Test at the defined frequencies
        freq = FrequencyGrid(values=np.array([1e9, 2e9, 3e9]))
        z = r.get_z(freq)
        
        # Check shape
        assert z.shape == (3, 2, 2)
        
        # Check values at each frequency
        assert np.isclose(z[0, 0, 0], 50.0)
        assert np.isclose(z[1, 0, 0], 60.0)
        assert np.isclose(z[2, 0, 0], 70.0)
        
        # Check diagonal elements
        assert np.isclose(z[0, 1, 1], 50.0)
        assert np.isclose(z[1, 1, 1], 60.0)
        assert np.isclose(z[2, 1, 1], 70.0)
        
        # Check off-diagonal elements are zero
        assert np.isclose(z[0, 0, 1], 0.0)
        assert np.isclose(z[0, 1, 0], 0.0)
    
    def test_freq_dep_resistor_interpolation(self):
        """Test interpolation between defined frequency points."""
        freq_points = [1e9, 3e9]  # 1, 3 GHz
        r_values = [50.0, 70.0]   # 50, 70 Ohms
        
        r = FreqDependentResistor.new("R_freq", freq_points, r_values)
        
        # Test at midpoint (2 GHz) - should interpolate to 60 Ohms
        freq = FrequencyGrid(values=np.array([2e9]))
        z = r.get_z(freq)
        
        assert np.isclose(z[0, 0, 0], 60.0)
    
    def test_extrapolation_warning(self, caplog):
        """Test that warning is issued when evaluating outside parameter range.
        
        When f > f_max, loguru should issue a WARNING.
        """
        freq_points = [1e9, 2e9, 3e9]
        r_values = [50.0, 60.0, 70.0]
        
        r = FreqDependentResistor.new("R_freq", freq_points, r_values)
        
        # Evaluate at frequency beyond the range
        freq = FrequencyGrid(values=np.array([5e9]))  # 5 GHz > 3 GHz max
        
        # Capture loguru output via handler
        messages = []
        def capture_handler(message):
            messages.append(message.record["message"])
        
        logger.add(capture_handler, level="WARNING")
        z = r.get_z(freq)
        logger.remove()
        
        # Check that warning was logged
        assert any("outside parameter range" in msg for msg in messages)
    
    def test_inside_range_no_warning(self, caplog):
        """Test that no warning is issued when evaluating inside parameter range."""
        freq_points = [1e9, 2e9, 3e9]
        r_values = [50.0, 60.0, 70.0]
        
        r = FreqDependentResistor.new("R_freq", freq_points, r_values)
        
        # Evaluate at frequency within the range
        freq = FrequencyGrid(values=np.array([1.5e9]))  # 1.5 GHz is within [1, 3] GHz
        
        # Capture loguru output via handler
        messages = []
        def capture_handler(message):
            messages.append(message.record["message"])
        
        logger.add(capture_handler, level="WARNING")
        z = r.get_z(freq)
        logger.remove()
        
        # Check that no warning was logged
        assert not any("outside parameter range" in msg for msg in messages)


class TestFreqDependentCapacitor:
    """Tests for FreqDependentCapacitor component."""
    
    def test_freq_dep_capacitor(self):
        """Test frequency-dependent capacitor with interpolation."""
        freq_points = [1e9, 2e9, 3e9]
        c_values = [1e-12, 2e-12, 3e-12]  # 1, 2, 3 pF
        
        c = FreqDependentCapacitor.new("C_freq", freq_points, c_values)
        
        # Test at the defined frequencies
        freq = FrequencyGrid(values=np.array([1e9, 2e9, 3e9]))
        z = c.get_z(freq)
        
        # Check shape
        assert z.shape == (3, 2, 2)
        
        # Check that impedance varies with frequency
        # Z = 1/(jωC), so higher C means lower |Z|
        z_mag_1 = np.abs(z[0, 0, 0])
        z_mag_2 = np.abs(z[1, 0, 0])
        z_mag_3 = np.abs(z[2, 0, 0])
        
        # Impedance should decrease as capacitance increases
        assert z_mag_1 > z_mag_2 > z_mag_3
    
    def test_freq_dep_capacitor_extrapolation_warning(self, caplog):
        """Test warning for capacitor outside range."""
        freq_points = [1e9, 2e9]
        c_values = [1e-12, 2e-12]
        
        c = FreqDependentCapacitor.new("C_freq", freq_points, c_values)
        
        # Evaluate below minimum frequency
        freq = FrequencyGrid(values=np.array([0.5e9]))
        
        # Capture loguru output via handler
        messages = []
        def capture_handler(message):
            messages.append(message.record["message"])
        
        logger.add(capture_handler, level="WARNING")
        z = c.get_z(freq)
        logger.remove()
        
        assert any("outside parameter range" in msg for msg in messages)


class TestFreqDependentInductor:
    """Tests for FreqDependentInductor component."""
    
    def test_freq_dep_inductor(self):
        """Test frequency-dependent inductor with interpolation."""
        freq_points = [1e9, 2e9, 3e9]
        l_values = [1e-9, 2e-9, 3e-9]  # 1, 2, 3 nH
        
        l = FreqDependentInductor.new("L_freq", freq_points, l_values)
        
        # Test at the defined frequencies
        freq = FrequencyGrid(values=np.array([1e9, 2e9, 3e9]))
        z = l.get_z(freq)
        
        # Check shape
        assert z.shape == (3, 2, 2)
        
        # Check that impedance varies with frequency
        # Z = jωL, so higher L means higher |Z|
        z_mag_1 = np.abs(z[0, 0, 0])
        z_mag_2 = np.abs(z[1, 0, 0])
        z_mag_3 = np.abs(z[2, 0, 0])
        
        # Impedance should increase as inductance increases
        assert z_mag_1 < z_mag_2 < z_mag_3
    
    def test_freq_dep_inductor_extrapolation_warning(self, caplog):
        """Test warning for inductor outside range."""
        freq_points = [1e9, 2e9]
        l_values = [1e-9, 2e-9]
        
        l = FreqDependentInductor.new("L_freq", freq_points, l_values)
        
        # Evaluate above maximum frequency
        freq = FrequencyGrid(values=np.array([5e9]))
        
        # Capture loguru output via handler
        messages = []
        def capture_handler(message):
            messages.append(message.record["message"])
        
        logger.add(capture_handler, level="WARNING")
        z = l.get_z(freq)
        logger.remove()
        
        assert any("outside parameter range" in msg for msg in messages)
