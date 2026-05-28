"""Pytest configuration and shared fixtures for eocircuit tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_freq():
    """
    Fixture providing a sample frequency grid.
    
    Returns:
        np.ndarray: Frequency grid from 1 MHz to 1 THz (1e6 to 1e12 Hz)
                   with 1000 logarithmically-spaced points.
    """
    return np.logspace(6, 12, 1000)


@pytest.fixture
def sample_wavelength():
    """
    Fixture providing a sample wavelength grid.
    
    Returns:
        np.ndarray: Wavelength grid from 1.5 µm to 1.6 µm (1.5e-6 to 1.6e-6 m)
                   with 1000 linearly-spaced points.
    """
    return np.linspace(1.5e-6, 1.6e-6, 1000)
