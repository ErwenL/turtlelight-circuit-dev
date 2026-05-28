"""Electrical source components: VoltageSource, CurrentSource.

Provides 2-port active components for voltage and current sources with
frequency-dependent behavior and phase control.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field

from eocircuit.core.component import Component, ComponentParams, z_to_s
from eocircuit.core.types import Port, PortDomain, FrequencyGrid, ComplexArray


class VoltageSourceParams(ComponentParams):
    """Parameters for VoltageSource component.
    
    Attributes:
        v: Voltage magnitude (complex) in Volts
        phase: Phase shift in radians
    """
    
    v: complex = Field(description="Voltage magnitude (complex) in Volts")
    phase: float = Field(default=0.0, description="Phase shift in radians")


class VoltageSource(Component):
    """2-port voltage source component.
    
    Represents an ideal voltage source with near-zero impedance.
    Impedance: Z ≈ 1e-12 Ω (near-zero to avoid singularity)
    """
    
    params: VoltageSourceParams
    
    @classmethod
    def new(cls, name: str, voltage: complex, phase: float = 0.0, **kwargs) -> Self:
        """Factory method to create a VoltageSource.
        
        Args:
            name: Component name
            voltage: Voltage magnitude (complex) in Volts
            phase: Phase shift in radians (default 0.0)
            **kwargs: Additional component arguments
            
        Returns:
            New VoltageSource instance
        """
        ports = [
            Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = VoltageSourceParams(v=voltage, phase=phase)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def get_v(self, freq: FrequencyGrid) -> ComplexArray:
        """Get voltage output across frequency.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Voltage array with shape (n_freq,)
        """
        n_freq = len(freq.values)
        # Apply phase shift to voltage
        v_output = self.params.v * np.exp(1j * self.params.phase)
        return np.full(n_freq, v_output, dtype=np.complex128)
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for voltage source.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            S-parameter matrix with shape (n_freq, 2, 2)
        """
        z_matrix = self.get_z(freq)
        z0 = 50.0  # Characteristic impedance
        return z_to_s(z_matrix, z0)
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for voltage source.
        
        Z ≈ 1e-12 Ω (near-zero impedance) for all frequencies
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Z-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Near-zero impedance (1e-12 to avoid singularity)
        z_small = 1e-12
        z_matrix[:, 0, 0] = z_small
        z_matrix[:, 1, 1] = z_small
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for voltage source.
        
        Y ≈ 1e12 S (near-infinite admittance) for all frequencies
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Y-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Near-infinite admittance (1e12 to avoid singularity)
        y_large = 1e12
        y_stamp = np.array([[y_large, -y_large], [-y_large, y_large]], dtype=np.complex128)
        for i in range(n_freq):
            y_matrix[i] = y_stamp
        
        return y_matrix


class CurrentSourceParams(ComponentParams):
    """Parameters for CurrentSource component.
    
    Attributes:
        i: Current magnitude (complex) in Amperes
        phase: Phase shift in radians
    """
    
    i: complex = Field(description="Current magnitude (complex) in Amperes")
    phase: float = Field(default=0.0, description="Phase shift in radians")


class CurrentSource(Component):
    """2-port current source component.
    
    Represents an ideal current source with near-infinite impedance.
    Admittance: Y ≈ 1e-12 S (near-zero to avoid singularity)
    """
    
    params: CurrentSourceParams
    
    @classmethod
    def new(cls, name: str, current: complex, phase: float = 0.0, **kwargs) -> Self:
        """Factory method to create a CurrentSource.
        
        Args:
            name: Component name
            current: Current magnitude (complex) in Amperes
            phase: Phase shift in radians (default 0.0)
            **kwargs: Additional component arguments
            
        Returns:
            New CurrentSource instance
        """
        ports = [
            Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = CurrentSourceParams(i=current, phase=phase)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def get_i(self, freq: FrequencyGrid) -> ComplexArray:
        """Get current output across frequency.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Current array with shape (n_freq,)
        """
        n_freq = len(freq.values)
        # Apply phase shift to current
        i_output = self.params.i * np.exp(1j * self.params.phase)
        return np.full(n_freq, i_output, dtype=np.complex128)
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for current source.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            S-parameter matrix with shape (n_freq, 2, 2)
        """
        z_matrix = self.get_z(freq)
        z0 = 50.0  # Characteristic impedance
        return z_to_s(z_matrix, z0)
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for current source.
        
        Z ≈ 1e12 Ω (near-infinite impedance) for all frequencies
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Z-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Near-infinite impedance (1e12 to avoid singularity)
        z_large = 1e12
        z_matrix[:, 0, 0] = z_large
        z_matrix[:, 1, 1] = z_large
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for current source.
        
        Y ≈ 1e-12 S (near-zero admittance) for all frequencies
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Y-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Near-zero admittance (1e-12 to avoid singularity)
        y_small = 1e-12
        y_matrix[:, 0, 0] = y_small
        y_matrix[:, 1, 1] = y_small
        
        return y_matrix
