"""Network class for managing circuit topology and component connections.

Provides a container for components and their port-to-port connections,
with validation for network correctness.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from eocircuit.core.component import Component
from eocircuit.core.connection import ConnectionMap
from eocircuit.core.types import Port


class Network(BaseModel):
    """Container for circuit components and their connections.

    Manages a collection of components and the connections between their ports.
    Provides validation to ensure network correctness (no duplicate names,
    no floating ports, etc.).

    Attributes:
        name: Network identifier string
        components: List of Component objects in the network
        connections: ConnectionMap managing port-to-port connections
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    components: list[Component] = Field(default_factory=list)
    connections: ConnectionMap = Field(default_factory=ConnectionMap)
    
    def __init__(self, **data):
        """Initialize Network with fresh instances."""
        super().__init__(**data)

    def add_component(self, component: Component) -> None:
        """Add a component to the network.

        Args:
            component: Component to add

        Raises:
            ValueError: If component name already exists in network
        """
        # Check for duplicate names
        for existing in self.components:
            if existing.name == component.name:
                raise ValueError(
                    f"Component with name '{component.name}' already exists in network"
                )
        self.components.append(component)

    def connect(self, port1: Port, port2: Port) -> None:
        """Connect two ports together.

        Args:
            port1: First port to connect
            port2: Second port to connect

        Raises:
            ValueError: If ports are already connected or invalid
        """
        # Delegate to ConnectionMap with duplicate validation
        self.connections.add_link(port1, port2)

    def validate(self) -> None:
        """Validate network correctness.

        Runs all validation checks:
        - No duplicate component names
        - No floating ports (all ports connected or tagged external)

        Raises:
            ValueError: If any validation fails
        """
        self.validate_no_duplicate_names()
        self.validate_no_floating_ports()

    def validate_no_duplicate_names(self) -> None:
        """Ensure all component names are unique.

        Raises:
            ValueError: If duplicate names found
        """
        names = set()
        for component in self.components:
            if component.name in names:
                raise ValueError(
                    f"Duplicate component name: '{component.name}'"
                )
            names.add(component.name)

    def validate_no_floating_ports(self) -> None:
        """Ensure every port is connected or tagged external.

        A port is floating if:
        - It has no connection AND
        - The component has at least one other port that IS connected

        Raises:
            ValueError: If floating ports found
        """
        all_ports = self.all_ports

        for component in self.components:
            component_ports = component.ports
            if not component_ports:
                continue

            # Check if any port in this component is connected
            has_connected_port = any(
                self.connections.is_connected(port) for port in component_ports
            )

            # If component has connected ports, all ports must be connected
            if has_connected_port:
                for port in component_ports:
                    if not self.connections.is_connected(port):
                        raise ValueError(
                            f"Floating port '{port.name}' in component '{component.name}': "
                            f"component has connected ports but this port is not connected"
                        )

    def get_external_ports(self) -> list[Port]:
        """Get ports that are not connected to any other port.

        Returns:
            List of external ports
        """
        return self.connections.get_external_ports(self.all_ports)

    def get_internal_ports(self) -> list[Port]:
        """Get ports that are connected to another port.

        Returns:
            List of internal ports
        """
        return self.connections.get_internal_ports(self.all_ports)

    def get_component_by_name(self, name: str) -> Optional[Component]:
        """Find a component by name.

        Args:
            name: Component name to search for

        Returns:
            Component with matching name, or None if not found
        """
        for component in self.components:
            if component.name == name:
                return component
        return None

    @property
    def all_ports(self) -> list[Port]:
        """Get a flat list of all component ports.

        Returns:
            List of all Port objects from all components
        """
        all_ports = []
        for component in self.components:
            all_ports.extend(component.ports)
        return all_ports
