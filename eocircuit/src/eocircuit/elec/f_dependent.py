"""Frequency-dependent electrical components with interpolation.

Provides frequency-dependent versions of Resistor, Capacitor, and Inductor
that interpolate their values across a frequency range using numpy.interp.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from loguru import logger
from pydantic import Field, field_validator

from eocircuit.core.component import Component, ComponentParams, z_to_s
from eocircuit.core.types import Port, PortDomain, FrequencyGrid, ComplexArray
from eocircuit.elec.basic import Resistor, Capacitor, Inductor


class FreqDependentResistorParams(ComponentParams):
    """Parameters for FreqDependentResistor component.
    
    Attributes:
        freq_points: Frequency points in Hz where resistance is defined
        r_values: Resistance values in Ohms at corresponding frequencies
    """
    
    freq_points: list = Field(description="Frequency points in Hz")
    r_values: list = Field(description="Resistance values in Ohms")
    
    @field_validator("freq_points", "r_values")
    @classmethod
    def validate_positive(cls, v):
        """Ensure all values are positive."""
        if not all(x > 0 for x in v):
            raise ValueError("All frequency and resistance values must be positive")
        return v
    
    @field_validator("freq_points")
    @classmethod
    def validate_freq_sorted(cls, v):
        """Ensure frequency points are sorted."""
        if v != sorted(v):
            raise ValueError("Frequency points must be sorted in ascending order")
        return v


class FreqDependentResistor(Component):
    """2-port frequency-dependent resistor component.
    
    Impedance: Z = R(f) where R is interpolated from frequency-dependent data
    """
    
    params: FreqDependentResistorParams
    
    @classmethod
    def new(cls, name: str, freq_points: list[float], r_values: list[float], **kwargs) -> Self:
        """Factory method to create a FreqDependentResistor.
        
        Args:
            name: Component name
            freq_points: Frequency points in Hz
            r_values: Resistance values in Ohms
            **kwargs: Additional component arguments
            
        Returns:
            New FreqDependentResistor instance
        """
        ports = [
            Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = FreqDependentResistorParams(freq_points=freq_points, r_values=r_values)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def _get_r_at_freq(self, freq_array: np.ndarray) -> np.ndarray:
        """Interpolate resistance at given frequencies.
        
        Args:
            freq_array: Array of frequency values
            
        Returns:
            Interpolated resistance values
        """
        # Check for out-of-range frequencies
        f_min = min(self.params.freq_points)
        f_max = max(self.params.freq_points)
        
        out_of_range = (freq_array < f_min) | (freq_array > f_max)
        if np.any(out_of_range):
            logger.warning(
                f"Evaluation frequency outside parameter range [{f_min}, {f_max}]. "
                f"Extrapolating may produce inaccurate results."
            )
        
        # Interpolate using numpy.interp
        r_interp = np.interp(freq_array, self.params.freq_points, self.params.r_values)
        return r_interp
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for frequency-dependent resistor.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            S-parameter matrix with shape (n_freq, 2, 2)
        """
        z_matrix = self.get_z(freq)
        z0 = 50.0  # Characteristic impedance
        return z_to_s(z_matrix, z0)
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for frequency-dependent resistor.
        
        Z = [[R(f), 0], [0, R(f)]] where R is interpolated
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Z-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Get interpolated resistance values
        r_interp = self._get_r_at_freq(freq.values)
        
        # Diagonal impedance matrix
        z_matrix[:, 0, 0] = r_interp
        z_matrix[:, 1, 1] = r_interp
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for frequency-dependent resistor.
        
        Y = [[1/R(f), 0], [0, 1/R(f)]]
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Y-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Get interpolated resistance values
        r_interp = self._get_r_at_freq(freq.values)
        
        # Diagonal admittance matrix
        y_matrix[:, 0, 0] = 1.0 / r_interp
        y_matrix[:, 1, 1] = 1.0 / r_interp
        
        return y_matrix


class FreqDependentCapacitorParams(ComponentParams):
    """Parameters for FreqDependentCapacitor component.
    
    Attributes:
        freq_points: Frequency points in Hz where capacitance is defined
        c_values: Capacitance values in Farads at corresponding frequencies
    """
    
    freq_points: list = Field(description="Frequency points in Hz")
    c_values: list = Field(description="Capacitance values in Farads")
    
    @field_validator("freq_points", "c_values")
    @classmethod
    def validate_positive(cls, v):
        """Ensure all values are positive."""
        if not all(x > 0 for x in v):
            raise ValueError("All frequency and capacitance values must be positive")
        return v
    
    @field_validator("freq_points")
    @classmethod
    def validate_freq_sorted(cls, v):
        """Ensure frequency points are sorted."""
        if v != sorted(v):
            raise ValueError("Frequency points must be sorted in ascending order")
        return v


class FreqDependentCapacitor(Component):
    """2-port frequency-dependent capacitor component.
    
    Impedance: Z = 1/(jωC(f)) where C is interpolated from frequency-dependent data
    """
    
    params: FreqDependentCapacitorParams
    
    @classmethod
    def new(cls, name: str, freq_points: list[float], c_values: list[float], **kwargs) -> Self:
        """Factory method to create a FreqDependentCapacitor.
        
        Args:
            name: Component name
            freq_points: Frequency points in Hz
            c_values: Capacitance values in Farads
            **kwargs: Additional component arguments
            
        Returns:
            New FreqDependentCapacitor instance
        """
        ports = [
            Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = FreqDependentCapacitorParams(freq_points=freq_points, c_values=c_values)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def _get_c_at_freq(self, freq_array: np.ndarray) -> np.ndarray:
        """Interpolate capacitance at given frequencies.
        
        Args:
            freq_array: Array of frequency values
            
        Returns:
            Interpolated capacitance values
        """
        # Check for out-of-range frequencies
        f_min = min(self.params.freq_points)
        f_max = max(self.params.freq_points)
        
        out_of_range = (freq_array < f_min) | (freq_array > f_max)
        if np.any(out_of_range):
            logger.warning(
                f"Evaluation frequency outside parameter range [{f_min}, {f_max}]. "
                f"Extrapolating may produce inaccurate results."
            )
        
        # Interpolate using numpy.interp
        c_interp = np.interp(freq_array, self.params.freq_points, self.params.c_values)
        return c_interp
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for frequency-dependent capacitor.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            S-parameter matrix with shape (n_freq, 2, 2)
        """
        z_matrix = self.get_z(freq)
        z0 = 50.0  # Characteristic impedance
        return z_to_s(z_matrix, z0)
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for frequency-dependent capacitor.
        
        Z = [[1/(jωC(f)), 0], [0, 1/(jωC(f))]]
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Z-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # jω = 2πj × freq
        jw = 2j * np.pi * freq.values
        
        # Get interpolated capacitance values
        c_interp = self._get_c_at_freq(freq.values)
        
        # Z = 1/(jωC)
        z_c = 1.0 / (jw * c_interp)
        
        z_matrix[:, 0, 0] = z_c
        z_matrix[:, 1, 1] = z_c
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for frequency-dependent capacitor.
        
        Y = [[jωC(f), 0], [0, jωC(f)]]
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Y-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # jω = 2πj × freq
        jw = 2j * np.pi * freq.values
        
        # Get interpolated capacitance values
        c_interp = self._get_c_at_freq(freq.values)
        
        # Y = jωC
        y_c = jw * c_interp
        
        y_matrix[:, 0, 0] = y_c
        y_matrix[:, 1, 1] = y_c
        
        return y_matrix


class FreqDependentInductorParams(ComponentParams):
    """Parameters for FreqDependentInductor component.
    
    Attributes:
        freq_points: Frequency points in Hz where inductance is defined
        l_values: Inductance values in Henries at corresponding frequencies
    """
    
    freq_points: list = Field(description="Frequency points in Hz")
    l_values: list = Field(description="Inductance values in Henries")
    
    @field_validator("freq_points", "l_values")
    @classmethod
    def validate_positive(cls, v):
        """Ensure all values are positive."""
        if not all(x > 0 for x in v):
            raise ValueError("All frequency and inductance values must be positive")
        return v
    
    @field_validator("freq_points")
    @classmethod
    def validate_freq_sorted(cls, v):
        """Ensure frequency points are sorted."""
        if v != sorted(v):
            raise ValueError("Frequency points must be sorted in ascending order")
        return v


class FreqDependentInductor(Component):
    """2-port frequency-dependent inductor component.
    
    Impedance: Z = jωL(f) where L is interpolated from frequency-dependent data
    """
    
    params: FreqDependentInductorParams
    
    @classmethod
    def new(cls, name: str, freq_points: list[float], l_values: list[float], **kwargs) -> Self:
        """Factory method to create a FreqDependentInductor.
        
        Args:
            name: Component name
            freq_points: Frequency points in Hz
            l_values: Inductance values in Henries
            **kwargs: Additional component arguments
            
        Returns:
            New FreqDependentInductor instance
        """
        ports = [
            Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = FreqDependentInductorParams(freq_points=freq_points, l_values=l_values)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def _get_l_at_freq(self, freq_array: np.ndarray) -> np.ndarray:
        """Interpolate inductance at given frequencies.
        
        Args:
            freq_array: Array of frequency values
            
        Returns:
            Interpolated inductance values
        """
        # Check for out-of-range frequencies
        f_min = min(self.params.freq_points)
        f_max = max(self.params.freq_points)
        
        out_of_range = (freq_array < f_min) | (freq_array > f_max)
        if np.any(out_of_range):
            logger.warning(
                f"Evaluation frequency outside parameter range [{f_min}, {f_max}]. "
                f"Extrapolating may produce inaccurate results."
            )
        
        # Interpolate using numpy.interp
        l_interp = np.interp(freq_array, self.params.freq_points, self.params.l_values)
        return l_interp
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for frequency-dependent inductor.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            S-parameter matrix with shape (n_freq, 2, 2)
        """
        z_matrix = self.get_z(freq)
        z0 = 50.0  # Characteristic impedance
        return z_to_s(z_matrix, z0)
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for frequency-dependent inductor.
        
        Z = [[jωL(f), 0], [0, jωL(f)]]
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Z-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # jω = 2πj × freq
        jw = 2j * np.pi * freq.values
        
        # Get interpolated inductance values
        l_interp = self._get_l_at_freq(freq.values)
        
        # Z = jωL
        z_l = jw * l_interp
        
        z_matrix[:, 0, 0] = z_l
        z_matrix[:, 1, 1] = z_l
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for frequency-dependent inductor.
        
        Y = [[1/(jωL(f)), 0], [0, 1/(jωL(f))]]
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Y-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # jω = 2πj × freq
        jw = 2j * np.pi * freq.values
        
        # Get interpolated inductance values
        l_interp = self._get_l_at_freq(freq.values)
        
        # Y = 1/(jωL)
        y_l = 1.0 / (jw * l_interp)
        
        y_matrix[:, 0, 0] = y_l
        y_matrix[:, 1, 1] = y_l
        
        return y_matrix
