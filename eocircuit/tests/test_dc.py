"""Tests for DirectionalCoupler photonic component.

Tests cover:
- 50/50 coupler (kappa=√0.5)
- Power conservation (lossless)
- Bar state (kappa=0, all through)
- Cross state (kappa=1, all cross)
- Lossy coupler (3dB excess loss)
"""

import numpy as np
import pytest

from eocircuit.photonics.dc import DirectionalCoupler, DirectionalCouplerParams
from eocircuit.core.types import FrequencyGrid


class TestDirectionalCoupler:
    """Test DirectionalCoupler component."""

    def test_dc_50_50(self):
        """Test 50/50 coupler: kappa=√0.5 → |t|²=|c|²=0.5."""
        # For 50/50 coupler: kappa = sqrt(0.5)
        kappa = np.sqrt(0.5)
        dc = DirectionalCoupler(
            name="dc_50_50",
            params=DirectionalCouplerParams(coupling_coefficient=kappa, excess_loss=0),
        )

        # Create frequency grid
        freq = FrequencyGrid(values=np.array([1e12, 2e12, 3e12]))

        # Get S-parameters
        s_matrix = dc.get_s(freq)

        # Check shape
        assert s_matrix.shape == (3, 4, 4)

        # For 50/50 coupler (lossless):
        # t = sqrt(1 - 0.5) = sqrt(0.5)
        # c = 1j * sqrt(0.5)
        # |t|² = 0.5, |c|² = 0.5
        t_expected = np.sqrt(0.5)
        c_expected = 1j * np.sqrt(0.5)

        for i in range(3):
            # Check through coefficients
            assert np.isclose(s_matrix[i, 2, 0], t_expected)
            assert np.isclose(s_matrix[i, 3, 1], t_expected)

            # Check cross coefficients
            assert np.isclose(s_matrix[i, 3, 0], c_expected)
            assert np.isclose(s_matrix[i, 2, 1], c_expected)

            # Check power conservation: |t|² + |c|² = 1
            power = np.abs(t_expected) ** 2 + np.abs(c_expected) ** 2
            assert np.isclose(power, 1.0)

    def test_dc_power_conservation(self):
        """Test lossless coupler: |t|² + |c|² = 1.0."""
        kappa = 0.3
        dc = DirectionalCoupler(
            name="dc_lossless",
            params=DirectionalCouplerParams(coupling_coefficient=kappa, excess_loss=0),
        )

        freq = FrequencyGrid(values=np.array([1e12]))
        s_matrix = dc.get_s(freq)

        # Extract coefficients
        t = s_matrix[0, 2, 0]
        c = s_matrix[0, 3, 0]

        # Power conservation: |t|² + |c|² = 1
        power = np.abs(t) ** 2 + np.abs(c) ** 2
        assert np.isclose(power, 1.0)

    def test_dc_bar_state(self):
        """Test bar state: kappa=0 → t=1, c=0."""
        dc = DirectionalCoupler(
            name="dc_bar",
            params=DirectionalCouplerParams(coupling_coefficient=0, excess_loss=0),
        )

        freq = FrequencyGrid(values=np.array([1e12]))
        s_matrix = dc.get_s(freq)

        # In bar state (kappa=0):
        # t = sqrt(1 - 0) = 1
        # c = 1j * 0 = 0
        assert np.isclose(s_matrix[0, 2, 0], 1.0)  # through
        assert np.isclose(s_matrix[0, 3, 0], 0.0)  # cross
        assert np.isclose(s_matrix[0, 2, 1], 0.0)  # cross
        assert np.isclose(s_matrix[0, 3, 1], 1.0)  # through

    def test_dc_cross_state(self):
        """Test cross state: kappa=1 → t=0, c=j."""
        dc = DirectionalCoupler(
            name="dc_cross",
            params=DirectionalCouplerParams(coupling_coefficient=1, excess_loss=0),
        )

        freq = FrequencyGrid(values=np.array([1e12]))
        s_matrix = dc.get_s(freq)

        # In cross state (kappa=1):
        # t = sqrt(1 - 1) = 0
        # c = 1j * 1 = 1j
        assert np.isclose(s_matrix[0, 2, 0], 0.0)  # through
        assert np.isclose(s_matrix[0, 3, 0], 1j)  # cross
        assert np.isclose(s_matrix[0, 2, 1], 1j)  # cross
        assert np.isclose(s_matrix[0, 3, 1], 0.0)  # through

    def test_dc_lossy(self):
        """Test lossy coupler: loss=3dB → |t|²+|c|² = 10^(-3/10)."""
        kappa = np.sqrt(0.5)  # 50/50 coupler
        loss_db = 3.0
        dc = DirectionalCoupler(
            name="dc_lossy",
            params=DirectionalCouplerParams(
                coupling_coefficient=kappa, excess_loss=loss_db
            ),
        )

        freq = FrequencyGrid(values=np.array([1e12]))
        s_matrix = dc.get_s(freq)

        # Extract coefficients
        t = s_matrix[0, 2, 0]
        c = s_matrix[0, 3, 0]

        # Power conservation with loss: |t|² + |c|² = 10^(-loss/10)
        power = np.abs(t) ** 2 + np.abs(c) ** 2
        expected_power = 10 ** (-loss_db / 10)
        assert np.isclose(power, expected_power)

    def test_dc_port_count(self):
        """Test that DirectionalCoupler has 4 ports."""
        dc = DirectionalCoupler(
            name="dc_test",
            params=DirectionalCouplerParams(coupling_coefficient=0.5),
        )
        assert dc.num_of_ports == 4

    def test_dc_port_names(self):
        """Test that ports have correct names."""
        dc = DirectionalCoupler(
            name="dc_test",
            params=DirectionalCouplerParams(coupling_coefficient=0.5),
        )
        port_names = [port.name for port in dc.ports]
        assert port_names == ["in1", "in2", "out1", "out2"]

    def test_dc_port_domain(self):
        """Test that all ports are optical domain."""
        from eocircuit.core.types import PortDomain

        dc = DirectionalCoupler(
            name="dc_test",
            params=DirectionalCouplerParams(coupling_coefficient=0.5),
        )
        for port in dc.ports:
            assert port.domain == PortDomain.OPTICAL

    def test_dc_invalid_kappa_negative(self):
        """Test that negative kappa raises ValueError."""
        with pytest.raises(ValueError, match="coupling_coefficient must be in"):
            DirectionalCouplerParams(coupling_coefficient=-0.1)

    def test_dc_invalid_kappa_greater_than_one(self):
        """Test that kappa > 1 raises ValueError."""
        with pytest.raises(ValueError, match="coupling_coefficient must be in"):
            DirectionalCouplerParams(coupling_coefficient=1.1)

    def test_dc_invalid_loss_negative(self):
        """Test that negative loss raises ValueError."""
        with pytest.raises(ValueError, match="excess_loss must be non-negative"):
            DirectionalCouplerParams(coupling_coefficient=0.5, excess_loss=-1.0)

    def test_dc_reciprocal_symmetry(self):
        """Test that S-matrix is reciprocal (symmetric for passive devices)."""
        dc = DirectionalCoupler(
            name="dc_test",
            params=DirectionalCouplerParams(coupling_coefficient=0.3, excess_loss=0.5),
        )

        freq = FrequencyGrid(values=np.array([1e12]))
        s_matrix = dc.get_s(freq)

        # For a reciprocal passive device: S[i,j] = S[j,i]
        # Check the non-zero elements
        # S[2,0] = t should equal S[0,2] = t
        assert np.isclose(s_matrix[0, 2, 0], s_matrix[0, 0, 2])
        # S[3,0] = c should equal S[0,3] = c
        assert np.isclose(s_matrix[0, 3, 0], s_matrix[0, 0, 3])
        # S[2,1] = c should equal S[1,2] = c
        assert np.isclose(s_matrix[0, 2, 1], s_matrix[0, 1, 2])
        # S[3,1] = t should equal S[1,3] = t
        assert np.isclose(s_matrix[0, 3, 1], s_matrix[0, 1, 3])

    def test_dc_no_reflection(self):
        """Test that there is no reflection (S[in][in] = 0)."""
        dc = DirectionalCoupler(
            name="dc_test",
            params=DirectionalCouplerParams(coupling_coefficient=0.5),
        )

        freq = FrequencyGrid(values=np.array([1e12]))
        s_matrix = dc.get_s(freq)

        # No reflection on input ports
        assert np.isclose(s_matrix[0, 0, 0], 0.0)  # in1 <- in1
        assert np.isclose(s_matrix[0, 1, 1], 0.0)  # in2 <- in2
        # No reflection on output ports
        assert np.isclose(s_matrix[0, 2, 2], 0.0)  # out1 <- out1
        assert np.isclose(s_matrix[0, 3, 3], 0.0)  # out2 <- out2
