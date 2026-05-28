"""Photodetector (PD) component for electro-optic circuits.

Provides a 3-port photodetector component that converts optical power
to electrical current. Models a simple responsivity-based photodetector
with dark current.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field

from eocircuit.core.component import Component, ComponentParams
from eocircuit.core.types import Port, PortDomain, FrequencyGrid, ComplexArray


class PhotodetectorParams(ComponentParams):
    """Parameters for Photodetector component.
    
    Attributes:
        responsivity: Photodetector responsivity in A/W
        dark_current: Dark current in Amperes (default 0)
    """
    
    responsivity: float = Field(description="Responsivity in A/W")
    dark_current: float = Field(default=0.0, description="Dark current in Amperes")


class Photodetector(Component):
    """3-port photodetector component.
    
    Converts optical power at the optical input port to electrical current
    at the electrical output ports. Acts as a current source in the electrical
    domain.
    
    Ports:
        - opt_in (OPTICAL): Optical input port (absorbs all light)
        - elec_p (ELECTRICAL): Positive electrical output
        - elec_n (ELECTRICAL): Negative electrical output (reference)
    
    Attributes:
        name: Component identifier
        ports: List of 3 ports (1 optical, 2 electrical)
        params: PhotodetectorParams with responsivity and dark_current
    """
    
    params: PhotodetectorParams
    
    @classmethod
    def new(cls, name: str, responsivity: float, dark_current: float = 0.0, **kwargs) -> Self:
        """Factory method to create a Photodetector.
        
        Args:
            name: Component name
            responsivity: Responsivity in A/W
            dark_current: Dark current in Amperes (default 0.0)
            **kwargs: Additional component arguments
            
        Returns:
            New Photodetector instance
        """
        ports = [
            Port(name="opt_in", domain=PortDomain.OPTICAL, z0=50.0),
            Port(name="elec_p", domain=PortDomain.ELECTRICAL, z0=50.0),
            Port(name="elec_n", domain=PortDomain.ELECTRICAL, z0=50.0),
        ]
        params = PhotodetectorParams(responsivity=responsivity, dark_current=dark_current)
        return cls(name=name, ports=ports, params=params, **kwargs)
    
    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters for photodetector.
        
        The optical input port absorbs all light (S21=0 at opt_in).
        The electrical ports have near-infinite impedance (current source behavior).
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, 3, 3)
            S[i, j, k] = S-parameter at frequency i, from port k to port j
        """
        n_freq = len(freq.values)
        s_matrix = np.zeros((n_freq, 3, 3), dtype=np.complex128)
        
        # Optical port (index 0) absorbs all light: S21 = 0 (no reflection/transmission)
        # Electrical ports (indices 1, 2) have near-infinite impedance (current source)
        # For a current source, S-parameters are near-zero (high impedance)
        
        # All S-parameters are approximately zero for ideal photodetector
        # (optical absorption + current source behavior)
        
        return s_matrix
    
    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters for photodetector.
        
        Optical port: absorbs light (Z ≈ 0 for optical domain)
        Electrical ports: near-infinite impedance (current source behavior)
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, 3, 3)
        """
        n_freq = len(freq.values)
        z_matrix = np.zeros((n_freq, 3, 3), dtype=np.complex128)
        
        # Optical port (index 0): near-zero impedance (absorbs light)
        z_small = 1e-12
        z_matrix[:, 0, 0] = z_small
        
        # Electrical ports (indices 1, 2): near-infinite impedance (current source)
        z_large = 1e12
        z_matrix[:, 1, 1] = z_large
        z_matrix[:, 2, 2] = z_large
        
        return z_matrix
    
    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters for photodetector.
        
        Optical port: near-infinite admittance (absorbs light)
        Electrical ports: near-zero admittance (current source behavior)
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, 3, 3)
        """
        n_freq = len(freq.values)
        y_matrix = np.zeros((n_freq, 3, 3), dtype=np.complex128)
        
        # Optical port (index 0): near-infinite admittance (absorbs light)
        y_large = 1e12
        y_matrix[:, 0, 0] = y_large
        
        # Electrical ports (indices 1, 2): near-zero admittance (current source)
        y_small = 1e-12
        y_matrix[:, 1, 1] = y_small
        y_matrix[:, 2, 2] = y_small
        
        return y_matrix
    
    def get_i(self, freq: FrequencyGrid, p_opt: float) -> ComplexArray:
        """Get photocurrent output based on optical input power.
        
        Computes the photocurrent from optical power using the responsivity
        and adds the dark current.
        
        I_photo = responsivity × P_opt + dark_current
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            p_opt: Optical input power in Watts
            
        Returns:
            ComplexArray with shape (n_freq,) representing photocurrent in Amperes
        """
        n_freq = len(freq.values)
        
        # Compute photocurrent: I = R × P_opt + I_dark
        i_photo = self.params.responsivity * p_opt + self.params.dark_current
        
        # Return constant current across all frequencies
        return np.full(n_freq, i_photo, dtype=np.complex128)


def create_pd(name: str, responsivity: float, dark_current: float = 0.0) -> Photodetector:
    """Factory function to create a Photodetector component.
    
    Args:
        name: Component name
        responsivity: Responsivity in A/W
        dark_current: Dark current in Amperes (default 0.0)
        
    Returns:
        New Photodetector instance
    """
    return Photodetector.new(name, responsivity, dark_current)
