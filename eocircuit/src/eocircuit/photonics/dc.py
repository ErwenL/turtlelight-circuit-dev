"""Directional Coupler photonic component.

Implements a 4-port optical directional coupler with frequency-independent
S-parameters defined by coupling coefficient and excess loss.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator

from eocircuit.core.component import Component, ComponentParams
from eocircuit.core.types import Port, PortDomain, FrequencyGrid


class DirectionalCouplerParams(ComponentParams):
    """Parameters for DirectionalCoupler component.
    
    Attributes:
        coupling_coefficient: Coupling coefficient kappa (0 ≤ kappa ≤ 1)
        excess_loss: Excess loss in dB (default 0)
    """

    coupling_coefficient: float
    excess_loss: float = 0.0

    @field_validator("coupling_coefficient")
    @classmethod
    def validate_coupling_coefficient(cls, v: float) -> float:
        """Validate coupling coefficient is in [0, 1]."""
        if not (0 <= v <= 1):
            raise ValueError("coupling_coefficient must be in [0, 1]")
        return v

    @field_validator("excess_loss")
    @classmethod
    def validate_excess_loss(cls, v: float) -> float:
        """Validate excess loss is non-negative."""
        if v < 0:
            raise ValueError("excess_loss must be non-negative")
        return v


class DirectionalCoupler(Component):
    """4-port optical directional coupler component.
    
    A directional coupler splits optical power between two paths with
    a specified coupling coefficient and excess loss.
    
    Port order: [in1, in2, out1, out2]
    - in1, in2: Input ports (indices 0, 1)
    - out1, out2: Output ports (indices 2, 3)
    
    S-matrix structure (lossless, kappa=0.5):
    - S[out1][in1] = t (through)
    - S[out2][in1] = c (cross, 90° phase)
    - S[out1][in2] = c (cross, 90° phase)
    - S[out2][in2] = t (through)
    - All other elements = 0 (no reflection, reciprocal)
    
    Where:
    - t = sqrt(1 - kappa²) * 10^(-loss/20)
    - c = 1j * kappa * 10^(-loss/20)
    """

    params: DirectionalCouplerParams

    def __init__(self, **data):
        """Initialize DirectionalCoupler with 4 optical ports."""
        if "ports" not in data:
            data["ports"] = [
                Port(name="in1", domain=PortDomain.OPTICAL),
                Port(name="in2", domain=PortDomain.OPTICAL),
                Port(name="out1", domain=PortDomain.OPTICAL),
                Port(name="out2", domain=PortDomain.OPTICAL),
            ]
        super().__init__(**data)

    def get_s(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get S-parameters (scattering parameters).
        
        Returns 4×4 S-matrix for each frequency point.
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, 4, 4)
        """
        n_freq = len(freq.values)
        s_matrix = np.zeros((n_freq, 4, 4), dtype=np.complex128)

        # Extract parameters
        kappa = self.params.coupling_coefficient
        loss_db = self.params.excess_loss

        # Calculate coefficients
        # Through coefficient: t = sqrt(1 - kappa²) * 10^(-loss/20)
        t = np.sqrt(1 - kappa**2) * 10 ** (-loss_db / 20)

        # Cross coefficient: c = 1j * kappa * 10^(-loss/20)
        c = 1j * kappa * 10 ** (-loss_db / 20)

        # Fill S-matrix for all frequencies (frequency-independent)
        # Port mapping: 0=in1, 1=in2, 2=out1, 3=out2
        # For reciprocal device: S[i,j] = S[j,i]
        for i in range(n_freq):
            # Through paths (diagonal-like)
            s_matrix[i, 2, 0] = t  # out1 <- in1 (through)
            s_matrix[i, 0, 2] = t  # in1 <- out1 (reciprocal)
            s_matrix[i, 3, 1] = t  # out2 <- in2 (through)
            s_matrix[i, 1, 3] = t  # in2 <- out2 (reciprocal)
            
            # Cross paths
            s_matrix[i, 3, 0] = c  # out2 <- in1 (cross)
            s_matrix[i, 0, 3] = c  # in1 <- out2 (reciprocal)
            s_matrix[i, 2, 1] = c  # out1 <- in2 (cross)
            s_matrix[i, 1, 2] = c  # in2 <- out1 (reciprocal)

        return s_matrix

    def get_z(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get Z-parameters (impedance parameters).
        
        For optical components, Z-parameters are not typically used.
        Returns zeros for compatibility with Component interface.
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, 4, 4) of zeros
        """
        n_freq = len(freq.values)
        return np.zeros((n_freq, 4, 4), dtype=np.complex128)

    def get_y(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get Y-parameters (admittance parameters).
        
        For optical components, Y-parameters are not typically used.
        Returns zeros for compatibility with Component interface.
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, 4, 4) of zeros
        """
        n_freq = len(freq.values)
        return np.zeros((n_freq, 4, 4), dtype=np.complex128)
