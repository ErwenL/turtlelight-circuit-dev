"""Tests for port-to-port connection model.

TDD tests for ConnectionMap class covering:
- Single link creation and retrieval
- External/internal port detection
- Multiple links
- Duplicate connection prevention
- Bidirectional lookup
"""

import pytest
from eocircuit.core.types import Port, PortDomain
from eocircuit.core.connection import ConnectionMap, PortLink
from eocircuit.core.component import Component, ComponentParams


class DummyParams(ComponentParams):
    """Dummy component parameters for testing."""
    pass


class DummyComponent(Component):
    """Dummy component for testing."""
    
    def get_s(self, freq):
        """Dummy S-parameter implementation."""
        return None
    
    def get_z(self, freq):
        """Dummy Z-parameter implementation."""
        return None
    
    def get_y(self, freq):
        """Dummy Y-parameter implementation."""
        return None


def test_add_single_link():
    """Test adding a single link and verifying connected port retrieval."""
    conn_map = ConnectionMap()
    p1 = Port(name="p1", domain=PortDomain.ELECTRICAL)
    p2 = Port(name="p2", domain=PortDomain.ELECTRICAL)
    
    conn_map.add_link(p1, p2)
    
    # Verify both ports are connected
    assert conn_map.is_connected(p1)
    assert conn_map.is_connected(p2)
    
    # Verify bidirectional lookup
    assert conn_map._links[p1.name] == p2.name
    assert conn_map._links[p2.name] == p1.name


def test_get_external_ports():
    """Test identifying external (unconnected) ports."""
    conn_map = ConnectionMap()
    p1 = Port(name="p1", domain=PortDomain.ELECTRICAL)
    p2 = Port(name="p2", domain=PortDomain.ELECTRICAL)
    p3 = Port(name="p3", domain=PortDomain.ELECTRICAL)
    p4 = Port(name="p4", domain=PortDomain.ELECTRICAL)
    
    all_ports = [p1, p2, p3, p4]
    
    # Connect p1 to p2
    conn_map.add_link(p1, p2)
    
    # Get external ports (should be p3 and p4)
    external = conn_map.get_external_ports(all_ports)
    external_names = {p.name for p in external}
    
    assert len(external) == 2
    assert external_names == {"p3", "p4"}


def test_get_internal_ports():
    """Test identifying internal (connected) ports."""
    conn_map = ConnectionMap()
    p1 = Port(name="p1", domain=PortDomain.ELECTRICAL)
    p2 = Port(name="p2", domain=PortDomain.ELECTRICAL)
    p3 = Port(name="p3", domain=PortDomain.ELECTRICAL)
    p4 = Port(name="p4", domain=PortDomain.ELECTRICAL)
    
    all_ports = [p1, p2, p3, p4]
    
    # Connect p1 to p2
    conn_map.add_link(p1, p2)
    
    # Get internal ports (should be p1 and p2)
    internal = conn_map.get_internal_ports(all_ports)
    internal_names = {p.name for p in internal}
    
    assert len(internal) == 2
    assert internal_names == {"p1", "p2"}


def test_multiple_links():
    """Test adding multiple links and verifying all are correct."""
    conn_map = ConnectionMap()
    p1 = Port(name="p1", domain=PortDomain.ELECTRICAL)
    p2 = Port(name="p2", domain=PortDomain.ELECTRICAL)
    p3 = Port(name="p3", domain=PortDomain.ELECTRICAL)
    p4 = Port(name="p4", domain=PortDomain.ELECTRICAL)
    p5 = Port(name="p5", domain=PortDomain.ELECTRICAL)
    p6 = Port(name="p6", domain=PortDomain.ELECTRICAL)
    
    # Add 3 links
    conn_map.add_link(p1, p2)
    conn_map.add_link(p3, p4)
    conn_map.add_link(p5, p6)
    
    # Verify all connections
    assert conn_map._links[p1.name] == p2.name
    assert conn_map._links[p2.name] == p1.name
    assert conn_map._links[p3.name] == p4.name
    assert conn_map._links[p4.name] == p3.name
    assert conn_map._links[p5.name] == p6.name
    assert conn_map._links[p6.name] == p5.name


def test_duplicate_connection():
    """Test that connecting an already-connected port raises ValueError."""
    conn_map = ConnectionMap()
    p1 = Port(name="p1", domain=PortDomain.ELECTRICAL)
    p2 = Port(name="p2", domain=PortDomain.ELECTRICAL)
    p3 = Port(name="p3", domain=PortDomain.ELECTRICAL)
    
    # Connect p1 to p2
    conn_map.add_link(p1, p2)
    
    # Try to connect p1 to p3 (should fail)
    with pytest.raises(ValueError, match="already connected"):
        conn_map.add_link(p1, p3)
    
    # Try to connect p3 to p2 (should fail)
    with pytest.raises(ValueError, match="already connected"):
        conn_map.add_link(p3, p2)


def test_bidirectional_lookup():
    """Test that bidirectional lookup works correctly."""
    conn_map = ConnectionMap()
    p1 = Port(name="p1", domain=PortDomain.ELECTRICAL)
    p2 = Port(name="p2", domain=PortDomain.ELECTRICAL)
    
    conn_map.add_link(p1, p2)
    
    # Verify bidirectional lookup
    assert conn_map._links[p1.name] == p2.name
    assert conn_map._links[p2.name] == p1.name
    
    # Verify is_connected works both ways
    assert conn_map.is_connected(p1)
    assert conn_map.is_connected(p2)


def test_validate_no_short_circuit():
    """Test that short circuit detection works."""
    conn_map = ConnectionMap()
    
    # Create a component with 2 ports
    p1 = Port(name="p1", domain=PortDomain.ELECTRICAL)
    p2 = Port(name="p2", domain=PortDomain.ELECTRICAL)
    
    component = DummyComponent(
        name="test_comp",
        ports=[p1, p2],
        params=DummyParams()
    )
    
    # Connect the two ports of the same component (short circuit)
    conn_map.add_link(p1, p2)
    
    # Validation should raise ValueError
    with pytest.raises(ValueError, match="Short circuit detected"):
        conn_map.validate_no_short_circuit([component])


def test_validate_no_short_circuit_multiple_components():
    """Test short circuit detection with multiple components."""
    conn_map = ConnectionMap()
    
    # Create two components
    p1 = Port(name="p1", domain=PortDomain.ELECTRICAL)
    p2 = Port(name="p2", domain=PortDomain.ELECTRICAL)
    p3 = Port(name="p3", domain=PortDomain.ELECTRICAL)
    p4 = Port(name="p4", domain=PortDomain.ELECTRICAL)
    
    comp1 = DummyComponent(
        name="comp1",
        ports=[p1, p2],
        params=DummyParams()
    )
    
    comp2 = DummyComponent(
        name="comp2",
        ports=[p3, p4],
        params=DummyParams()
    )
    
    # Connect p1 to p3 (different components - OK)
    conn_map.add_link(p1, p3)
    
    # This should pass without error
    conn_map.validate_no_short_circuit([comp1, comp2])
    
    # Now connect p2 to p4 (still different components - OK)
    conn_map.add_link(p2, p4)
    
    # This should still pass
    conn_map.validate_no_short_circuit([comp1, comp2])
