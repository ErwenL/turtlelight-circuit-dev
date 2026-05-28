"""Basic electrical components: Resistor, Capacitor, Inductor, Conductor.

Provides fundamental 2-port passive components with frequency-dependent
Z-parameters and Y-parameters.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field, field_validator

from eocircuit.core.component import Component, ComponentParams, z_to_s
from eocircuit.core.types import Port, PortDomain, FrequencyGrid, ComplexArray


class ResistorParams(ComponentParams):
    """Parameters for Resistor component.
    
    Attributes:
        r: Resistance in Ohms (must be positive)
    """
    
    r: float = Field(gt=0, description="Resistance in Ohms")


class Resistor(Component):
    """2-port resistor component.
    
    Impedance: Z = R (constant across frequency)
    Admittance: Y = 1/R (constant across frequency)
    """
    
    params: ResistorParams
    
    @classmethod
    def new(cls, name: str, value: float, **kwargs) -> Self:
        """Factory method to create a Resistor.
        
        Args:
            name: Component name
            value: Resistance in Ohms
            **kwargs: Additional component arguments
            
        Returns:
            New Resistor instance
        """
        ports = [
            Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = ResistorParams(r=value)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for resistor.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            S-parameter matrix with shape (n_freq, 2, 2)
        """
        z_matrix = self.get_z(freq)
        z0 = 50.0  # Characteristic impedance
        return z_to_s(z_matrix, z0)
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for resistor.
        
        Z = [[R, 0], [0, R]] for all frequencies
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Z-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Diagonal impedance matrix
        z_matrix[:, 0, 0] = self.params.r
        z_matrix[:, 1, 1] = self.params.r
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for resistor.
        
        Y = [[1/R, 0], [0, 1/R]] for all frequencies
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Y-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Diagonal admittance matrix
        y_matrix[:, 0, 0] = 1.0 / self.params.r
        y_matrix[:, 1, 1] = 1.0 / self.params.r
        
        return y_matrix


class CapacitorParams(ComponentParams):
    """Parameters for Capacitor component.
    
    Attributes:
        c: Capacitance in Farads (must be positive)
    """
    
    c: float = Field(gt=0, description="Capacitance in Farads")


class Capacitor(Component):
    """2-port capacitor component.
    
    Impedance: Z = 1/(jωC)
    Admittance: Y = jωC
    """
    
    params: CapacitorParams
    
    @classmethod
    def new(cls, name: str, value: float, **kwargs) -> Self:
        """Factory method to create a Capacitor.
        
        Args:
            name: Component name
            value: Capacitance in Farads
            **kwargs: Additional component arguments
            
        Returns:
            New Capacitor instance
        """
        ports = [
            Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = CapacitorParams(c=value)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for capacitor.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            S-parameter matrix with shape (n_freq, 2, 2)
        """
        z_matrix = self.get_z(freq)
        z0 = 50.0  # Characteristic impedance
        return z_to_s(z_matrix, z0)
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for capacitor.
        
        Z = [[1/(jωC), 0], [0, 1/(jωC)]]
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Z-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # jω = 2πj × freq
        jw = 2j * np.pi * freq.values
        
        # Z = 1/(jωC)
        z_c = 1.0 / (jw * self.params.c)
        
        z_matrix[:, 0, 0] = z_c
        z_matrix[:, 1, 1] = z_c
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for capacitor.
        
        Y = [[jωC, 0], [0, jωC]]
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Y-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # jω = 2πj × freq
        jw = 2j * np.pi * freq.values
        
        # Y = jωC
        y_c = jw * self.params.c
        
        y_matrix[:, 0, 0] = y_c
        y_matrix[:, 1, 1] = y_c
        
        return y_matrix


class InductorParams(ComponentParams):
    """Parameters for Inductor component.
    
    Attributes:
        l: Inductance in Henries (must be positive)
    """
    
    l: float = Field(gt=0, description="Inductance in Henries")


class Inductor(Component):
    """2-port inductor component.
    
    Impedance: Z = jωL
    Admittance: Y = 1/(jωL)
    """
    
    params: InductorParams
    
    @classmethod
    def new(cls, name: str, value: float, **kwargs) -> Self:
        """Factory method to create an Inductor.
        
        Args:
            name: Component name
            value: Inductance in Henries
            **kwargs: Additional component arguments
            
        Returns:
            New Inductor instance
        """
        ports = [
            Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = InductorParams(l=value)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for inductor.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            S-parameter matrix with shape (n_freq, 2, 2)
        """
        z_matrix = self.get_z(freq)
        z0 = 50.0  # Characteristic impedance
        return z_to_s(z_matrix, z0)
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for inductor.
        
        Z = [[jωL, 0], [0, jωL]]
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Z-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # jω = 2πj × freq
        jw = 2j * np.pi * freq.values
        
        # Z = jωL
        z_l = jw * self.params.l
        
        z_matrix[:, 0, 0] = z_l
        z_matrix[:, 1, 1] = z_l
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for inductor.
        
        Y = [[1/(jωL), 0], [0, 1/(jωL)]]
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Y-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # jω = 2πj × freq
        jw = 2j * np.pi * freq.values
        
        # Y = 1/(jωL)
        y_l = 1.0 / (jw * self.params.l)
        
        y_matrix[:, 0, 0] = y_l
        y_matrix[:, 1, 1] = y_l
        
        return y_matrix


class ConductorParams(ComponentParams):
    """Parameters for Conductor component.
    
    Attributes:
        g: Conductance in Siemens (must be positive)
    """
    
    g: float = Field(gt=0, description="Conductance in Siemens")


class Conductor(Component):
    """2-port conductor component.
    
    Admittance: Y = G (constant across frequency)
    Impedance: Z = 1/G (constant across frequency)
    """
    
    params: ConductorParams
    
    @classmethod
    def new(cls, name: str, value: float, **kwargs) -> Self:
        """Factory method to create a Conductor.
        
        Args:
            name: Component name
            value: Conductance in Siemens
            **kwargs: Additional component arguments
            
        Returns:
            New Conductor instance
        """
        ports = [
            Port(name="p1", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="p2", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = ConductorParams(g=value)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for conductor.
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            S-parameter matrix with shape (n_freq, 2, 2)
        """
        z_matrix = self.get_z(freq)
        z0 = 50.0  # Characteristic impedance
        return z_to_s(z_matrix, z0)
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for conductor.
        
        Z = [[1/G, 0], [0, 1/G]] for all frequencies
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Z-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Diagonal impedance matrix
        z_matrix[:, 0, 0] = 1.0 / self.params.g
        z_matrix[:, 1, 1] = 1.0 / self.params.g
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for conductor.
        
        Y = [[G, 0], [0, G]] for all frequencies
        
        Args:
            freq: FrequencyGrid with frequency points
            
        Returns:
            Y-parameter matrix with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        
        # Diagonal admittance matrix
        y_matrix[:, 0, 0] = self.params.g
        y_matrix[:, 1, 1] = self.params.g
        
        return y_matrix
