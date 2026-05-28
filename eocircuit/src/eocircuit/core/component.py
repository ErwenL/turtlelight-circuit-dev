"""Component abstract base class for eocircuit framework.

Provides the core N-port component abstraction with frequency-dependent
S-parameters, Z-parameters, and Y-parameters. All components are immutable
and use Pydantic v2 for validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, Union, List

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from eocircuit.core.types import Port, FrequencyGrid, ComplexArray, RealArray


class ComponentParams(BaseModel):
    """Base class for component parameters.
    
    All component parameter classes should inherit from this.
    Provides a common base for type checking and validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Component(BaseModel, ABC):
    """Abstract base class for circuit components.
    
    Represents an N-port component with frequency-dependent response.
    All components are immutable (use set_params to create modified copies).
    
    Attributes:
        name: Component identifier string
        ports: List of Port objects defining the component's ports
        params: ComponentParams instance with component-specific parameters
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    ports: List[Port]
    params: ComponentParams

    @property
    def num_of_ports(self) -> int:
        """Return the number of ports."""
        return len(self.ports)

    @abstractmethod
    def get_s(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get S-parameters (scattering parameters).
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, n_ports, n_ports)
            S[i, j, k] = S-parameter at frequency i, from port k to port j
        """
        pass

    @abstractmethod
    def get_z(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get Z-parameters (impedance parameters).
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, n_ports, n_ports)
            Z[i, j, k] = Z-parameter at frequency i, element (j, k)
        """
        pass

    @abstractmethod
    def get_y(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        """Get Y-parameters (admittance parameters).
        
        Args:
            freq: FrequencyGrid with frequency points in Hz
            
        Returns:
            ComplexArray with shape (n_freq, n_ports, n_ports)
            Y[i, j, k] = Y-parameter at frequency i, element (j, k)
        """
        pass

    def set_params(self, **kwargs) -> Self:
        """Create a new component with updated parameters.
        
        Implements immutable pattern: returns a new Component instance
        with updated params, leaving the original unchanged.
        
        Args:
            **kwargs: Parameter updates to apply to self.params
            
        Returns:
            New Component instance with updated params
        """
        updated_params = self.params.model_copy(update=kwargs)
        return self.model_copy(update={"params": updated_params})


def z_to_s(z_matrix: NDArray[np.complexfloating], z0: Union[float, List[float]]) -> NDArray[np.complexfloating]:
    """Convert Z-parameters to S-parameters.
    
    Implements the conversion: S = (Z - Z0*I) @ inv(Z + Z0*I)
    
    Args:
        z_matrix: Z-parameter matrix with shape (n_freq, n_ports, n_ports)
        z0: Characteristic impedance(s). If scalar, applied to all ports.
            If list, must have length n_ports.
            
    Returns:
        S-parameter matrix with same shape as z_matrix
        
    Raises:
        ValueError: If z0 list length doesn't match number of ports
        LinAlgError: If (Z + Z0*I) is singular
    """
    n_freq, n_ports, _ = z_matrix.shape
    
    # Convert z0 to array
    if isinstance(z0, (int, float)):
        z0_array = np.full(n_ports, z0, dtype=np.float64)
    else:
        z0_array = np.array(z0, dtype=np.float64)
        if len(z0_array) != n_ports:
            raise ValueError(
                f"z0 length {len(z0_array)} doesn't match n_ports {n_ports}"
            )
    
    # Create Z0 diagonal matrix for each frequency
    s_matrix = np.zeros_like(z_matrix)
    
    for i in range(n_freq):
        z0_i = np.diag(z0_array)
        numerator = z_matrix[i] - z0_i
        denominator = z_matrix[i] + z0_i
        s_matrix[i] = numerator @ np.linalg.inv(denominator)
    
    return s_matrix


def s_to_z(s_matrix: NDArray[np.complexfloating], z0: Union[float, List[float]]) -> NDArray[np.complexfloating]:
    """Convert S-parameters to Z-parameters.
    
    Implements the inverse conversion: Z = Z0 @ inv(I - S) @ (I + S)
    
    Args:
        s_matrix: S-parameter matrix with shape (n_freq, n_ports, n_ports)
        z0: Characteristic impedance(s). If scalar, applied to all ports.
            If list, must have length n_ports.
            
    Returns:
        Z-parameter matrix with same shape as s_matrix
        
    Raises:
        ValueError: If z0 list length doesn't match number of ports
        LinAlgError: If (I - S) is singular
    """
    n_freq, n_ports, _ = s_matrix.shape
    
    # Convert z0 to array
    if isinstance(z0, (int, float)):
        z0_array = np.full(n_ports, z0, dtype=np.float64)
    else:
        z0_array = np.array(z0, dtype=np.float64)
        if len(z0_array) != n_ports:
            raise ValueError(
                f"z0 length {len(z0_array)} doesn't match n_ports {n_ports}"
            )
    
    # Create Z0 diagonal matrix for each frequency
    z_matrix = np.zeros_like(s_matrix)
    
    for i in range(n_freq):
        z0_i = np.diag(z0_array)
        identity = np.eye(n_ports, dtype=s_matrix.dtype)
        numerator = identity + s_matrix[i]
        denominator = identity - s_matrix[i]
        z_matrix[i] = z0_i @ numerator @ np.linalg.inv(denominator)
    
    return z_matrix
