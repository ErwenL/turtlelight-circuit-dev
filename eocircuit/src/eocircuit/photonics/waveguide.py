"""Waveguide photonic component for eocircuit framework.

Provides a 2-port optical waveguide component with frequency-dependent
S-parameters including phase shift and loss effects.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from eocircuit.core.component import Component, ComponentParams
from eocircuit.core.types import (
    Port,
    PortDomain,
    FrequencyGrid,
    SPEED_OF_LIGHT,
)


class WaveguideParams(ComponentParams):
    """Parameters for waveguide component.
    
    Attributes:
        length: Waveguide length in meters
        neff: Effective refractive index (float or array)
        loss: Loss in dB/m (default 0)
    """

    length: float
    neff: float
    loss: float = 0.0


class Waveguide(Component):
    """2-port optical waveguide component.
    
    Models a straight waveguide with frequency-dependent phase shift
    and loss. Implements reciprocal transmission with no reflection.
    
    Attributes:
        name: Component identifier
        ports: List of 2 optical ports
        params: WaveguideParams with length, neff, and loss
    """

    params: WaveguideParams

    def get_s(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get S-parameters for waveguide.
        
        Computes S-matrix with phase shift and loss:
        - wavelength = c / freq
        - phase = 2π * neff * length / wavelength
        - amplitude = 10^(-loss * length / 20)
        - t_forward = amplitude * exp(-j * phase)
        - S = [[0, t_reverse], [t_forward, 0]] (reciprocal)
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, 2, 2)
            S[i, j, k] = S-parameter at frequency i, from port k to port j
        """
        n_freq = len(freq.values)
        s_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)

        # Compute wavelength for each frequency
        wavelengths = SPEED_OF_LIGHT / freq.values

        # Compute phase shift: 2π * neff * length / wavelength
        phases = 2 * np.pi * self.params.neff * self.params.length / wavelengths

        # Compute amplitude from loss: 10^(-loss * length / 20)
        amplitudes = 10 ** (-self.params.loss * self.params.length / 20)

        # Compute transmission coefficient: amplitude * exp(-j * phase)
        t_forward = amplitudes * np.exp(-1j * phases)

        # S-matrix for reciprocal 2-port with no reflection:
        # S = [[0, t_reverse], [t_forward, 0]]
        # where t_reverse = t_forward (reciprocity)
        for i in range(n_freq):
            s_matrix[i, 0, 1] = t_forward[i]  # S12: port 1 to port 0
            s_matrix[i, 1, 0] = t_forward[i]  # S21: port 0 to port 1

        return s_matrix

    def get_z(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get Z-parameters for waveguide.
        
        Converts S-parameters to Z-parameters using characteristic impedance.
        For optical ports, z0 is typically 1.0 (normalized).
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, 2, 2)
        """
        from eocircuit.core.component import s_to_z

        s_matrix = self.get_s(freq)
        z0_values = [port.z0 for port in self.ports]
        z_matrix = s_to_z(s_matrix, z0_values)

        return z_matrix

    def get_y(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get Y-parameters for waveguide.
        
        Computes Y = inv(Z) for each frequency.
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, 2, 2)
        """
        n_freq = len(freq.values)
        z_matrix = self.get_z(freq)
        y_matrix = np.zeros((n_freq, 2, 2), dtype=np.complex128)

        for i in range(n_freq):
            y_matrix[i] = np.linalg.inv(z_matrix[i])

        return y_matrix


def create_waveguide(
    name: str,
    length: float,
    neff: float,
    loss: float = 0.0,
) -> Waveguide:
    """Factory function to create a Waveguide component.
    
    Args:
        name: Component identifier
        length: Waveguide length in meters
        neff: Effective refractive index
        loss: Loss in dB/m (default 0)
        
    Returns:
        Waveguide component with 2 optical ports
    """
    ports = [
        Port(name="in", domain=PortDomain.OPTICAL, z0=1.0),
        Port(name="out", domain=PortDomain.OPTICAL, z0=1.0),
    ]

    params = WaveguideParams(length=length, neff=neff, loss=loss)

    return Waveguide(name=name, ports=ports, params=params)
