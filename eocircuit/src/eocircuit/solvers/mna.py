"""Modified Nodal Analysis (MNA) solver for electrical networks.

This solver builds a nodal system directly from `Network` topology:
- Node extraction from port-to-port connections
- Component Y-matrix stamping into nodal admittance
- Independent source handling (current + voltage)
- Linear solve per frequency point
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eocircuit.core.network import Network
from eocircuit.core.types import FrequencyGrid, Port


@dataclass(frozen=True)
class MNASolution:
    """MNA solve result over a frequency grid."""

    node_voltages: dict[str, NDArray[np.complexfloating]]
    port_voltages: dict[str, NDArray[np.complexfloating]]
    branch_currents: dict[str, NDArray[np.complexfloating]]
    frequency: FrequencyGrid

    def get_node_voltage(self, node_name: str) -> NDArray[np.complexfloating]:
        """Return voltage array for a node."""
        return self.node_voltages[node_name]

    def get_port_voltage(self, port_name: str) -> NDArray[np.complexfloating]:
        """Return voltage array for a port."""
        return self.port_voltages[port_name]

    def get_branch_currents(self, component_name: str) -> NDArray[np.complexfloating]:
        """Return port-current array (n_freq, n_ports) for a component."""
        return self.branch_currents[component_name]


@dataclass(frozen=True)
class _SourceSpec:
    component_name: str
    kind: str  # "voltage" | "current"
    p_name: str
    n_name: str
    value: NDArray[np.complexfloating]


class MNASolver:
    """Modified nodal solver for electrical-only networks."""

    _VOLTAGE_NAMES = ("voltage", "v", "dc_voltage", "value")
    _CURRENT_NAMES = ("current", "i", "dc_current")

    def solve(self, network: Network, freq: FrequencyGrid) -> MNASolution:
        """Solve node voltages and branch currents for a network."""
        n_freq = len(freq.values)
        if n_freq == 0:
            raise ValueError("Frequency grid must contain at least one point")

        port_by_name = {port.name: port for port in network.all_ports}
        component_by_port = {
            port.name: component
            for component in network.components
            for port in component.ports
        }

        port_to_node, node_to_ports, ground_node = self._build_nodes(network, port_by_name)
        non_ground_nodes = [node for node in node_to_ports if node != ground_node]
        node_index = {node: idx for idx, node in enumerate(non_ground_nodes)}

        sources = self._extract_sources(network, n_freq)
        voltage_sources = [s for s in sources if s.kind == "voltage"]
        current_sources = [s for s in sources if s.kind == "current"]

        n_nodes = len(non_ground_nodes)
        n_vsrc = len(voltage_sources)
        g = np.zeros((n_freq, n_nodes, n_nodes), dtype=np.complex128)
        i_rhs = np.zeros((n_freq, n_nodes), dtype=np.complex128)

        voltage_component_names = {src.component_name for src in voltage_sources}

        for component in network.components:
            if component.name in voltage_component_names:
                continue

            y_comp = component.get_y(freq)
            if y_comp.shape != (n_freq, component.num_of_ports, component.num_of_ports):
                raise ValueError(
                    f"Component '{component.name}' get_y shape {y_comp.shape} "
                    f"does not match expected {(n_freq, component.num_of_ports, component.num_of_ports)}"
                )

            for f_idx in range(n_freq):
                for a_idx, port_a in enumerate(component.ports):
                    node_a = port_to_node[port_a.name]
                    if node_a == ground_node:
                        continue
                    row = node_index[node_a]

                    for b_idx, port_b in enumerate(component.ports):
                        node_b = port_to_node[port_b.name]
                        if node_b == ground_node:
                            continue
                        col = node_index[node_b]
                        g[f_idx, row, col] += y_comp[f_idx, a_idx, b_idx]

        for src in current_sources:
            node_p = port_to_node[src.p_name]
            node_n = port_to_node[src.n_name]
            if node_p != ground_node:
                i_rhs[:, node_index[node_p]] -= src.value
            if node_n != ground_node:
                i_rhs[:, node_index[node_n]] += src.value

        solution_nodes = np.zeros((n_freq, n_nodes), dtype=np.complex128)
        source_currents = np.zeros((n_freq, n_vsrc), dtype=np.complex128)

        b = np.zeros((n_nodes, n_vsrc), dtype=np.complex128)
        e_rhs = np.zeros((n_freq, n_vsrc), dtype=np.complex128)
        for k, src in enumerate(voltage_sources):
            node_p = port_to_node[src.p_name]
            node_n = port_to_node[src.n_name]
            if node_p != ground_node:
                b[node_index[node_p], k] += 1.0
            if node_n != ground_node:
                b[node_index[node_n], k] -= 1.0
            e_rhs[:, k] = src.value

        for f_idx in range(n_freq):
            if n_vsrc == 0:
                if n_nodes == 0:
                    continue
                solution_nodes[f_idx] = np.linalg.solve(g[f_idx], i_rhs[f_idx])
                continue

            a = np.zeros((n_nodes + n_vsrc, n_nodes + n_vsrc), dtype=np.complex128)
            rhs = np.zeros(n_nodes + n_vsrc, dtype=np.complex128)

            if n_nodes > 0:
                a[:n_nodes, :n_nodes] = g[f_idx]
                a[:n_nodes, n_nodes:] = b
                a[n_nodes:, :n_nodes] = b.T
                rhs[:n_nodes] = i_rhs[f_idx]
            rhs[n_nodes:] = e_rhs[f_idx]

            solved = np.linalg.solve(a, rhs)
            if n_nodes > 0:
                solution_nodes[f_idx] = solved[:n_nodes]
            source_currents[f_idx] = solved[n_nodes:]

        node_voltages: dict[str, NDArray[np.complexfloating]] = {}
        for node in non_ground_nodes:
            node_voltages[node] = solution_nodes[:, node_index[node]].copy()
        node_voltages[ground_node] = np.zeros(n_freq, dtype=np.complex128)

        port_voltages: dict[str, NDArray[np.complexfloating]] = {}
        for port_name, node_name in port_to_node.items():
            port_voltages[port_name] = node_voltages[node_name]

        branch_currents: dict[str, NDArray[np.complexfloating]] = {}
        vsrc_index = {src.component_name: k for k, src in enumerate(voltage_sources)}
        current_src_names = {src.component_name: src for src in current_sources}

        for component in network.components:
            currents = np.zeros((n_freq, component.num_of_ports), dtype=np.complex128)

            if component.name in vsrc_index:
                k = vsrc_index[component.name]
                currents[:, 0] = source_currents[:, k]
                currents[:, 1] = -source_currents[:, k]
                branch_currents[component.name] = currents
                continue

            if component.name in current_src_names:
                src = current_src_names[component.name]
                currents[:, 0] = src.value
                currents[:, 1] = -src.value
                branch_currents[component.name] = currents
                continue

            y_comp = component.get_y(freq)
            for f_idx in range(n_freq):
                v_ports = np.array(
                    [port_voltages[p.name][f_idx] for p in component.ports],
                    dtype=np.complex128,
                )
                currents[f_idx] = y_comp[f_idx] @ v_ports
            branch_currents[component.name] = currents

        return MNASolution(
            node_voltages=node_voltages,
            port_voltages=port_voltages,
            branch_currents=branch_currents,
            frequency=freq,
        )

    def _build_nodes(
        self,
        network: Network,
        port_by_name: dict[str, Port],
    ) -> tuple[dict[str, str], dict[str, list[str]], str]:
        """Build connectivity nodes from pairwise port links."""
        parents = {name: name for name in port_by_name}

        def find(x: str) -> str:
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parents[rb] = ra

        links = network.connections._links
        for a, b in links.items():
            if a in parents and b in parents:
                union(a, b)

        groups: dict[str, list[str]] = {}
        for p_name in port_by_name:
            root = find(p_name)
            groups.setdefault(root, []).append(p_name)

        nodes_sorted = sorted(
            (sorted(members) for members in groups.values()),
            key=lambda members: members[0],
        )
        node_to_ports: dict[str, list[str]] = {
            f"n{idx}": members for idx, members in enumerate(nodes_sorted)
        }

        port_to_node: dict[str, str] = {}
        for node_name, members in node_to_ports.items():
            for p_name in members:
                port_to_node[p_name] = node_name

        ground_node = self._find_ground_node(node_to_ports, port_by_name)
        return port_to_node, node_to_ports, ground_node

    def _find_ground_node(
        self,
        node_to_ports: dict[str, list[str]],
        port_by_name: dict[str, Port],
    ) -> str:
        """Find designated ground node.

        Priority:
        1) Any node containing a port with index == 0
        2) Any node containing a port named gnd/ground/0
        3) Lexicographically first node
        """
        for node_name, members in node_to_ports.items():
            if any(port_by_name[p_name].index == 0 for p_name in members):
                return node_name

        for node_name, members in node_to_ports.items():
            lowered = {p_name.lower() for p_name in members}
            if any(name in {"gnd", "ground", "0"} for name in lowered):
                return node_name

        return sorted(node_to_ports.keys())[0]

    def _extract_sources(self, network: Network, n_freq: int) -> list[_SourceSpec]:
        """Extract independent voltage/current sources from component params."""
        sources: list[_SourceSpec] = []
        for component in network.components:
            if component.num_of_ports != 2:
                continue

            voltage_value = self._find_param_value(component.params, self._VOLTAGE_NAMES)
            current_value = self._find_param_value(component.params, self._CURRENT_NAMES)

            if voltage_value is not None:
                sources.append(
                    _SourceSpec(
                        component_name=component.name,
                        kind="voltage",
                        p_name=component.ports[0].name,
                        n_name=component.ports[1].name,
                        value=self._to_frequency_vector(voltage_value, n_freq),
                    )
                )
                continue

            if current_value is not None:
                sources.append(
                    _SourceSpec(
                        component_name=component.name,
                        kind="current",
                        p_name=component.ports[0].name,
                        n_name=component.ports[1].name,
                        value=self._to_frequency_vector(current_value, n_freq),
                    )
                )

        return sources

    @staticmethod
    def _find_param_value(params: Any, names: tuple[str, ...]) -> complex | float | NDArray[np.complexfloating] | None:
        for name in names:
            if hasattr(params, name):
                return getattr(params, name)
        return None

    @staticmethod
    def _to_frequency_vector(value: Any, n_freq: int) -> NDArray[np.complexfloating]:
        if np.isscalar(value):
            return np.full(n_freq, complex(value), dtype=np.complex128)

        arr = np.asarray(value, dtype=np.complex128)
        if arr.ndim != 1 or arr.shape[0] != n_freq:
            raise ValueError(
                f"Source value must be scalar or shape ({n_freq},), got {arr.shape}"
            )
        return arr
