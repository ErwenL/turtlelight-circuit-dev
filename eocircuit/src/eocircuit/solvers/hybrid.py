"""Hybrid electrical-photonic solver.

Implements a v1 linear electro-optic flow:
1) Partition mixed-domain network into electrical and photonic sub-networks
2) Solve electrical network with MNA
3) Push electrical bias into EO devices
4) Solve photonic network with updated EO S-parameters
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eocircuit.core.component import Component
from eocircuit.core.network import Network
from eocircuit.core.types import FrequencyGrid, Port, PortDomain
from eocircuit.solvers.mna import MNASolution, MNASolver
from eocircuit.solvers.s_param import SParamSolver


@dataclass(frozen=True)
class HybridSolution:
    """Hybrid solve result for a mixed-domain network."""

    elec_solution: MNASolution
    photonic_result: NDArray[np.complexfloating]
    eo_state: dict[str, dict[str, NDArray[np.complexfloating]]]


class _ProjectedComponent(Component):
    """Domain-projected view of a mixed-domain component."""

    source_component: Component
    source_port_indices: list[int]

    def get_s(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        s_full = self.source_component.get_s(freq)
        idx = self.source_port_indices
        return s_full[:, idx][:, :, idx]

    def get_z(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        z_full = self.source_component.get_z(freq)
        idx = self.source_port_indices
        return z_full[:, idx][:, :, idx]

    def get_y(self, freq: FrequencyGrid) -> NDArray[np.complexfloating]:
        y_full = self.source_component.get_y(freq)
        idx = self.source_port_indices
        return y_full[:, idx][:, :, idx]


class HybridSolver:
    """v1 linear hybrid solver for mixed electrical-optical networks."""

    def __init__(
        self,
        mna_solver: MNASolver | None = None,
        s_param_solver: SParamSolver | None = None,
    ) -> None:
        self.mna_solver = mna_solver or MNASolver()
        self.s_param_solver = s_param_solver or SParamSolver()

    def partition_network(self, network: Network) -> tuple[Network, Network, list[Component]]:
        """Partition a mixed network into electrical subnet, photonic subnet, EO devices."""
        eo_devices = [
            component
            for component in network.components
            if self._is_eo_device(component)
        ]
        elec_subnet = self._build_domain_subnetwork(network, PortDomain.ELECTRICAL)
        photonic_subnet = self._build_domain_subnetwork(network, PortDomain.OPTICAL)
        return elec_subnet, photonic_subnet, eo_devices

    def solve(self, network: Network, freq: FrequencyGrid) -> HybridSolution:
        """Run v1 linear EO flow: electrical solve -> EO update -> photonic solve."""
        elec_subnet, _, eo_devices = self.partition_network(network)
        elec_solution = (
            self.mna_solver.solve(elec_subnet, freq)
            if elec_subnet.components
            else self._empty_mna_solution(freq)
        )

        eo_state: dict[str, dict[str, NDArray[np.complexfloating]]] = {}
        updated_components: dict[str, Component] = {}
        for eo_device in eo_devices:
            bias_by_port = self._extract_electrical_bias(eo_device, elec_solution)
            eo_state[eo_device.name] = bias_by_port
            updated_components[eo_device.name] = self._apply_electrical_bias(eo_device, bias_by_port)

        photonic_subnet = self._build_domain_subnetwork(
            network,
            PortDomain.OPTICAL,
            component_overrides=updated_components,
        )
        photonic_result = self.s_param_solver.solve(photonic_subnet, freq)
        return HybridSolution(
            elec_solution=elec_solution,
            photonic_result=photonic_result,
            eo_state=eo_state,
        )

    @staticmethod
    def _component_domains(component: Component) -> set[PortDomain]:
        return {port.domain for port in component.ports}

    @classmethod
    def _is_eo_device(cls, component: Component) -> bool:
        domains = cls._component_domains(component)
        return PortDomain.ELECTRICAL in domains and PortDomain.OPTICAL in domains

    def _build_domain_subnetwork(
        self,
        network: Network,
        domain: PortDomain,
        component_overrides: dict[str, Component] | None = None,
    ) -> Network:
        subnet = Network(name=f"{network.name}_{domain.value}")
        component_overrides = component_overrides or {}

        for original in network.components:
            component = component_overrides.get(original.name, original)
            projected = self._project_component_for_domain(component, domain)
            if projected is not None:
                subnet.add_component(projected)

        subnet_ports = {port.name: port for port in subnet.all_ports}
        seen_pairs: set[tuple[str, str]] = set()
        for p1_name, p2_name in network.connections._links.items():
            pair_names = sorted((p1_name, p2_name))
            pair = (pair_names[0], pair_names[1])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            p1 = subnet_ports.get(p1_name)
            p2 = subnet_ports.get(p2_name)
            if p1 is not None and p2 is not None:
                subnet.connect(p1, p2)

        return subnet

    def _project_component_for_domain(
        self,
        component: Component,
        domain: PortDomain,
    ) -> Component | None:
        indices = [idx for idx, port in enumerate(component.ports) if port.domain == domain]
        if not indices:
            return None

        if len(indices) == component.num_of_ports:
            return component

        projected_ports: list[Port] = [component.ports[idx] for idx in indices]
        return _ProjectedComponent(
            name=component.name,
            ports=projected_ports,
            params=component.params,
            source_component=component,
            source_port_indices=indices,
        )

    @staticmethod
    def _extract_electrical_bias(
        component: Component,
        elec_solution: MNASolution,
    ) -> dict[str, NDArray[np.complexfloating]]:
        bias_by_port: dict[str, NDArray[np.complexfloating]] = {}
        for port in component.ports:
            if port.domain != PortDomain.ELECTRICAL:
                continue
            if port.name in elec_solution.port_voltages:
                bias_by_port[port.name] = elec_solution.port_voltages[port.name]
        return bias_by_port

    @staticmethod
    def _apply_electrical_bias(
        component: Component,
        bias_by_port: dict[str, NDArray[np.complexfloating]],
    ) -> Component:
        if hasattr(component, "set_electrical_bias"):
            maybe_updated = component.set_electrical_bias(bias_by_port)
            if isinstance(maybe_updated, Component):
                return maybe_updated
            return component
        return component

    @staticmethod
    def _empty_mna_solution(freq: FrequencyGrid) -> MNASolution:
        return MNASolution(
            node_voltages={},
            port_voltages={},
            branch_currents={},
            frequency=freq,
        )
