"""Port-to-port connection model for eocircuit framework.

Provides connection management for arbitrary port-to-port linking,
supporting bidirectional connections and topology validation.
"""

from typing import TypeAlias

from eocircuit.core.types import Port
from eocircuit.core.component import Component

# Type alias for a port-to-port connection pair
PortLink: TypeAlias = tuple[Port, Port]


class ConnectionMap:
    """Manages bidirectional port-to-port connections.
    
    Stores connections as a bidirectional mapping where linking port1 to port2
    automatically creates the reverse link. Supports arbitrary network topologies.
    
    Attributes:
        _links: dict mapping port name to connected port name (bidirectional)
    """
    
    def __init__(self):
        """Initialize empty connection map."""
        self._links: dict[str, str] = {}
    
    def add_link(self, port1: Port, port2: Port) -> None:
        """Register a bidirectional connection between two ports.
        
        Args:
            port1: First port to connect
            port2: Second port to connect
            
        Raises:
            ValueError: If either port is already connected
        """
        self.validate_no_duplicate_connection(port1)
        self.validate_no_duplicate_connection(port2)
        
        # Create bidirectional link
        self._links[port1.name] = port2.name
        self._links[port2.name] = port1.name
    
    def get_connected_port(self, port: Port) -> Port | None:
        """Find the port connected to the given port.
        
        Args:
            port: Port to query
            
        Returns:
            The connected Port object, or None if not connected
        """
        connected_name = self._links.get(port.name)
        if connected_name is None:
            return None
        # Return a Port-like object with the connected name
        # Note: We return a minimal representation since we only have the name
        return None  # Placeholder - will be resolved by caller with full port list
    
    def is_connected(self, port: Port) -> bool:
        """Check if a port is connected to another port.
        
        Args:
            port: Port to check
            
        Returns:
            True if port is in any connection, False otherwise
        """
        return port.name in self._links
    
    def get_external_ports(self, all_ports: list[Port]) -> list[Port]:
        """Get ports that are NOT connected to any other port.
        
        External ports represent the network interface.
        
        Args:
            all_ports: List of all ports to filter
            
        Returns:
            List of ports not in any connection
        """
        external = []
        for port in all_ports:
            if port.name not in self._links:
                external.append(port)
        return external
    
    def get_internal_ports(self, all_ports: list[Port]) -> list[Port]:
        """Get ports that ARE connected to another port.
        
        Internal ports are hidden by connections.
        
        Args:
            all_ports: List of all ports to filter
            
        Returns:
            List of ports that are in connections
        """
        internal = []
        for port in all_ports:
            if port.name in self._links:
                internal.append(port)
        return internal
    
    def get_links(self) -> list[PortLink]:
        """Return all unique link pairs.
        
        Since links are bidirectional, each connection appears once.
        
        Returns:
            List of (port1, port2) tuples representing all connections
        """
        # This is a placeholder - actual implementation needs port objects
        # For now, return empty list as we need full port context
        return []
    
    def validate_no_duplicate_connection(self, port: Port) -> None:
        """Validate that a port is not already connected.
        
        Args:
            port: Port to validate
            
        Raises:
            ValueError: If port is already connected
        """
        if port.name in self._links:
            raise ValueError(f"Port '{port.name}' is already connected")
    
    def validate_no_short_circuit(self, components: list[Component]) -> None:
        """Validate that no two ports of the same component are directly linked.
        
        A short circuit occurs when two ports of the same component are
        connected to each other, which would create a direct feedback loop.
        
        Args:
            components: List of components to validate
            
        Raises:
            ValueError: If a short circuit is detected
        """
        for component in components:
            component_port_names = {port.name for port in component.ports}
            
            # Check if any port of this component is connected to another port
            # of the same component
            for port_name in component_port_names:
                if port_name in self._links:
                    connected_name = self._links[port_name]
                    if connected_name in component_port_names:
                        raise ValueError(
                            f"Short circuit detected: component '{component.name}' "
                            f"has ports '{port_name}' and '{connected_name}' directly linked"
                        )
