"""Tests for Network class.

Tests cover:
- Adding components to network
- Connecting ports between components
- Getting external and internal ports
- Duplicate component name validation
- Floating port detection
- Duplicate connection rejection
"""

import pytest
import numpy as np

from eocircuit.core.network import Network
from eocircuit.core.component import Component, ComponentParams
from eocircuit.core.types import Port, PortDomain, FrequencyGrid, ComplexArray


class SimpleParams(ComponentParams):
    """Parameters for a simple test component."""
    pass


class SimpleComponent(Component):
    """Simple 2-port component for testing."""

    params: SimpleParams

    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Get S-parameters (identity for testing)."""
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        s_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        for i in range(n_freq):
            s_matrix[i] = np.eye(n_ports)
        return s_matrix

    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Z-parameters (50 ohm for testing)."""
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        for i in range(n_freq):
            z_matrix[i] = np.eye(n_ports) * 50.0
        return z_matrix

    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Get Y-parameters (1/50 for testing)."""
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        y_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        for i in range(n_freq):
            y_matrix[i] = np.eye(n_ports) / 50.0
        return y_matrix


class TestNetwork:
    """Test Network class functionality."""

    def test_add_component(self):
        """Test adding components to network."""
        network = Network(name="test_network")
        
        # Create two components
        p1 = Port(name="c1_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p2 = Port(name="c1_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c1 = SimpleComponent(name="comp1", ports=[p1, p2], params=SimpleParams())
        
        p3 = Port(name="c2_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p4 = Port(name="c2_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c2 = SimpleComponent(name="comp2", ports=[p3, p4], params=SimpleParams())
        
        # Add components
        network.add_component(c1)
        network.add_component(c2)
        
        # Verify components list
        assert len(network.components) == 2
        assert network.components[0].name == "comp1"
        assert network.components[1].name == "comp2"

    def test_connect_two_components(self):
        """Test connecting ports between two components."""
        network = Network(name="test_network")
        
        # Create two components
        p1 = Port(name="c1_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p2 = Port(name="c1_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c1 = SimpleComponent(name="comp1", ports=[p1, p2], params=SimpleParams())
        
        p3 = Port(name="c2_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p4 = Port(name="c2_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c2 = SimpleComponent(name="comp2", ports=[p3, p4], params=SimpleParams())
        
        network.add_component(c1)
        network.add_component(c2)
        
        # Connect p2 of c1 to p1 of c2
        network.connect(p2, p3)
        
        # Verify connection exists
        assert network.connections.is_connected(p2)
        assert network.connections.is_connected(p3)

    def test_get_external_ports(self):
        """Test getting external ports from network."""
        network = Network(name="test_network")
        
        # Create two components with 2 ports each
        p1 = Port(name="c1_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p2 = Port(name="c1_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c1 = SimpleComponent(name="comp1", ports=[p1, p2], params=SimpleParams())
        
        p3 = Port(name="c2_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p4 = Port(name="c2_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c2 = SimpleComponent(name="comp2", ports=[p3, p4], params=SimpleParams())
        
        network.add_component(c1)
        network.add_component(c2)
        
        # Connect p2 of c1 to p1 of c2 (1 connection)
        network.connect(p2, p3)
        
        # Get external ports: should be p1 and p4 (2 external)
        external = network.get_external_ports()
        assert len(external) == 2
        assert p1 in external
        assert p4 in external

    def test_duplicate_component_name(self):
        """Test that duplicate component names raise ValueError."""
        network = Network(name="test_network")
        
        # Create two components with same name
        p1 = Port(name="c1_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p2 = Port(name="c1_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c1 = SimpleComponent(name="comp1", ports=[p1, p2], params=SimpleParams())
        
        p3 = Port(name="c2_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p4 = Port(name="c2_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c2 = SimpleComponent(name="comp1", ports=[p3, p4], params=SimpleParams())
        
        # Add first component
        network.add_component(c1)
        
        # Adding second component with same name should raise ValueError
        with pytest.raises(ValueError, match="already exists"):
            network.add_component(c2)

    def test_floating_port_detected(self):
        """Test that floating ports are detected during validation."""
        network = Network(name="test_network")
        
        # Create a component with 2 ports
        p1 = Port(name="c1_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p2 = Port(name="c1_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c1 = SimpleComponent(name="comp1", ports=[p1, p2], params=SimpleParams())
        
        # Create another component with 2 ports
        p3 = Port(name="c2_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p4 = Port(name="c2_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c2 = SimpleComponent(name="comp2", ports=[p3, p4], params=SimpleParams())
        
        network.add_component(c1)
        network.add_component(c2)
        
        # Connect only p2 of c1 to p1 of c2
        network.connect(p2, p3)
        
        # Validation should fail because p1 is floating (c1 has connected port p2 but p1 is not connected)
        with pytest.raises(ValueError, match="Floating port"):
            network.validate()

    def test_duplicate_connection_rejected(self):
        """Test that connecting the same port twice raises ValueError."""
        network = Network(name="test_network")
        
        # Create two components
        p1 = Port(name="c1_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p2 = Port(name="c1_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c1 = SimpleComponent(name="comp1", ports=[p1, p2], params=SimpleParams())
        
        p3 = Port(name="c2_p1", domain=PortDomain.ELECTRICAL, z0=50.0)
        p4 = Port(name="c2_p2", domain=PortDomain.ELECTRICAL, z0=50.0)
        c2 = SimpleComponent(name="comp2", ports=[p3, p4], params=SimpleParams())
        
        network.add_component(c1)
        network.add_component(c2)
        
        # Connect p2 to p3
        network.connect(p2, p3)
        
        # Try to connect p2 again (should fail)
        with pytest.raises(ValueError, match="already connected"):
            network.connect(p2, p4)
