"""Hierarchical electrical subcircuit component.

`SubCircuit` wraps an internal :class:`~eocircuit.core.network.Network` and
exposes selected internal ports as external component ports.
"""

# pyright: reportMissingImports=false, reportInvalidTypeForm=false, reportIncompatibleVariableOverride=false

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field, PrivateAttr, field_validator

from eocircuit.core.component import Component, ComponentParams, z_to_s, s_to_z
from eocircuit.core.network import Network
from eocircuit.core.types import ComplexArray, FrequencyGrid, Port, PortDomain
from eocircuit.elec.sources import CurrentSource, CurrentSourceParams
from eocircuit.solvers.mna import MNASolver


class SubCircuitParams(ComponentParams):
    """Parameter overrides for internal components.

    Mapping format:
        {
            "component_name": {"param_name": value, ...},
            ...
        }
    """

    component_params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SubCircuit(Component):
    """Composite component backed by an internal electrical network."""

    params: SubCircuitParams = Field(default_factory=SubCircuitParams)
    internal_network: Network
    circuit_ports: list[Port]
    max_depth: int = 10

    _inherited_component_params: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _active_depth: int = PrivateAttr(default=0)

    @field_validator("max_depth")
    @classmethod
    def _validate_max_depth(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_depth must be non-negative")
        return v

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if len(self.circuit_ports) != self.num_of_ports:
            raise ValueError(
                "Number of circuit_ports must match number of external ports: "
                f"{len(self.circuit_ports)} != {self.num_of_ports}"
            )

        internal_names = {p.name for p in self.internal_network.all_ports}
        for port in self.circuit_ports:
            if port.name not in internal_names:
                raise ValueError(
                    f"circuit_port '{port.name}' is not part of internal network '{self.internal_network.name}'"
                )

    def _validate_nesting_depth(self) -> None:
        """Pre-check nested SubCircuit depth to fail fast."""

        visited: set[int] = set()

        def _walk(subckt: "SubCircuit", depth: int) -> None:
            if depth > subckt.max_depth:
                raise RecursionError(
                    f"SubCircuit recursion depth exceeded: depth={depth} > max_depth={subckt.max_depth}"
                )

            marker = id(subckt)
            if marker in visited:
                return
            visited.add(marker)

            for component in subckt.internal_network.components:
                if isinstance(component, SubCircuit):
                    _walk(component, depth + 1)

        _walk(self, self._active_depth)

    def _combined_component_params(self) -> dict[str, dict[str, Any]]:
        combined = {name: values.copy() for name, values in self._inherited_component_params.items()}
        for name, values in self.params.component_params.items():
            if name in combined:
                merged = combined[name]
                merged.update(values)
                combined[name] = merged
            else:
                combined[name] = values.copy()
        return combined

    def _clone_with_inherited_params(self, inherited: dict[str, dict[str, Any]]) -> "SubCircuit":
        clone = self.model_copy(deep=True)
        clone._inherited_component_params = {k: v.copy() for k, v in inherited.items()}
        clone._active_depth = self._active_depth + 1
        if clone._active_depth > clone.max_depth:
            raise RecursionError(
                f"SubCircuit recursion depth exceeded: depth={clone._active_depth} > max_depth={clone.max_depth}"
            )
        return clone

    def _prepare_internal_network(
        self,
    ) -> tuple[Network, list[Port]]:
        overrides = self._combined_component_params()

        prepared = Network(name=f"{self.internal_network.name}__prepared")
        port_lookup: dict[str, Port] = {}

        for component in self.internal_network.components:
            updated_component: Component = component

            if component.name in overrides:
                updated_component = updated_component.set_params(**overrides[component.name])

            if isinstance(updated_component, SubCircuit):
                updated_component = updated_component._clone_with_inherited_params(overrides)

            prepared.add_component(updated_component)
            for port in updated_component.ports:
                port_lookup[port.name] = port

        seen: set[frozenset[str]] = set()
        for a, b in self.internal_network.connections._links.items():
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            prepared.connect(port_lookup[a], port_lookup[b])

        mapped_circuit_ports = [port_lookup[p.name] for p in self.circuit_ports]
        return prepared, mapped_circuit_ports

    @staticmethod
    def _inject_probe(network: Network, target_port: Port, probe_name: str) -> None:
        probe_p = Port(
            name=f"{probe_name}_p",
            domain=PortDomain.ELECTRICAL,
            z0=target_port.z0,
            index=1,
        )
        probe_n = Port(
            name=f"{probe_name}_n",
            domain=PortDomain.ELECTRICAL,
            z0=target_port.z0,
            index=0,
        )
        probe = CurrentSource(
            name=probe_name,
            ports=[probe_p, probe_n],
            params=CurrentSourceParams(i=1.0 + 0.0j),
        )
        network.add_component(probe)
        network.connect(probe_p, target_port)

    def _solve_internal_z(self, freq: FrequencyGrid, depth: int) -> ComplexArray:
        if depth > self.max_depth:
            raise RecursionError(
                f"SubCircuit recursion depth exceeded: depth={depth} > max_depth={self.max_depth}"
            )

        prepared_network, mapped_ports = self._prepare_internal_network()
        n_freq = len(freq.values)
        n_ports = len(mapped_ports)
        z = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        solver = MNASolver()

        for excite_idx, excite_port in enumerate(mapped_ports):
            solve_network = prepared_network.model_copy(deep=True)
            target_port = next(p for p in solve_network.all_ports if p.name == excite_port.name)
            self._inject_probe(solve_network, target_port, f"probe_{self.name}_{excite_idx}")

            solution = solver.solve(solve_network, freq)
            for meas_idx, meas_port in enumerate(mapped_ports):
                z[:, meas_idx, excite_idx] = solution.get_port_voltage(meas_port.name)

        return z

    def get_s(self, freq: FrequencyGrid) -> ComplexArray:
        """Return S-parameters from internal-network Z extraction."""
        self._validate_nesting_depth()
        z_matrix = self._solve_internal_z(freq, depth=self._active_depth)
        return z_to_s(z_matrix, [port.z0 for port in self.ports])

    def get_z(self, freq: FrequencyGrid) -> ComplexArray:
        """Return Z-parameters from S-parameters via S→Z conversion."""
        s_matrix = self.get_s(freq)
        return s_to_z(s_matrix, [port.z0 for port in self.ports])

    def get_y(self, freq: FrequencyGrid) -> ComplexArray:
        """Return Y-parameters from Z inversion."""
        z_from_s = self.get_z(freq)

        n_freq = z_from_s.shape[0]
        y_matrix = np.zeros_like(z_from_s)
        for i in range(n_freq):
            y_matrix[i] = np.linalg.pinv(z_from_s[i])
        return y_matrix
