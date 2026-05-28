"""Tests for hierarchical SubCircuit component."""

from __future__ import annotations

import numpy as np
import pytest

from eocircuit.core.component import z_to_s
from eocircuit.core.network import Network
from eocircuit.core.types import FrequencyGrid, Port, PortDomain
from eocircuit.elec.basic import Resistor, ResistorParams
from eocircuit.elec.sources import (
    CurrentSource,
    CurrentSourceParams,
    VoltageSource,
    VoltageSourceParams,
)
from eocircuit.elec.subcircuit import SubCircuit, SubCircuitParams
from eocircuit.solvers.mna import MNASolver


def _e_port(name: str, index: int | None = None) -> Port:
    kwargs = {"index": index} if index is not None else {}
    return Port(name=name, domain=PortDomain.ELECTRICAL, z0=50.0, **kwargs)


def _build_r_divider(prefix: str = "div") -> tuple[Network, list[Port]]:
    network = Network(name=f"{prefix}_network")

    p_in = _e_port(f"{prefix}_in")
    p_mid = _e_port(f"{prefix}_mid")
    r1 = Resistor(name=f"{prefix}_r1", ports=[p_in, p_mid], params=ResistorParams(r=100.0))

    p_mid2 = _e_port(f"{prefix}_mid2")
    p_out = _e_port(f"{prefix}_out")
    r2 = Resistor(name=f"{prefix}_r2", ports=[p_mid2, p_out], params=ResistorParams(r=100.0))

    network.add_component(r1)
    network.add_component(r2)
    network.connect(p_mid, p_mid2)

    return network, [p_in, p_out]


def _extract_s_from_network(network: Network, ports: list[Port], freq: FrequencyGrid) -> np.ndarray:
    solver = MNASolver()
    n_freq = len(freq.values)
    n_ports = len(ports)
    z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)

    for excite_idx, excite_port in enumerate(ports):
        solve_network = network.model_copy(deep=True)
        target_port = next(p for p in solve_network.all_ports if p.name == excite_port.name)

        probe_p = _e_port(f"probe_{excite_idx}_p", index=1)
        probe_n = _e_port(f"probe_{excite_idx}_n", index=0)
        probe = CurrentSource(
            name=f"Iprobe_{excite_idx}",
            ports=[probe_p, probe_n],
            params=CurrentSourceParams(i=1.0 + 0.0j),
        )
        solve_network.add_component(probe)
        solve_network.connect(probe_p, target_port)

        solution = solver.solve(solve_network, freq)
        for meas_idx, meas_port in enumerate(ports):
            z_matrix[:, meas_idx, excite_idx] = solution.get_port_voltage(meas_port.name)

    return z_to_s(z_matrix, [p.z0 for p in ports])


def _make_subcircuit_from_divider(name: str) -> SubCircuit:
    internal_network, circuit_ports = _build_r_divider(prefix=name)
    external_ports = [_e_port(f"{name}_p0"), _e_port(f"{name}_p1")]
    return SubCircuit(
        name=name,
        ports=external_ports,
        params=SubCircuitParams(),
        internal_network=internal_network,
        circuit_ports=circuit_ports,
        max_depth=10,
    )


def _nest_subcircuit(base: SubCircuit, levels: int, max_depth: int = 10) -> SubCircuit:
    current = base
    for i in range(levels):
        container = Network(name=f"nested_network_{i}")
        container.add_component(current)
        current = SubCircuit(
            name=f"nested_{i}",
            ports=[_e_port(f"nested_{i}_p0"), _e_port(f"nested_{i}_p1")],
            params=SubCircuitParams(),
            internal_network=container,
            circuit_ports=current.ports,
            max_depth=max_depth,
        )
    return current


def test_subcircuit_as_component() -> None:
    freq = FrequencyGrid(values=np.array([1e3], dtype=np.float64))
    subckt = _make_subcircuit_from_divider("sub_for_parent")

    network = Network(name="parent_network")
    src_p = _e_port("src_p")
    src_n = _e_port("src_n", index=0)
    vsrc = VoltageSource(
        name="V1",
        ports=[src_p, src_n],
        params=VoltageSourceParams(v=1.0 + 0j),
    )

    network.add_component(vsrc)
    network.add_component(subckt)
    network.connect(src_p, subckt.ports[0])
    network.connect(subckt.ports[1], src_n)

    solution = MNASolver().solve(network, freq)
    assert network.get_component_by_name("sub_for_parent") is not None
    assert np.isfinite(solution.get_port_voltage(subckt.ports[0].name)[0])


def test_subcircuit_s_params() -> None:
    freq = FrequencyGrid(values=np.array([1e3, 1e6], dtype=np.float64))

    direct_network, direct_ports = _build_r_divider(prefix="direct")
    direct_s = _extract_s_from_network(direct_network, direct_ports, freq)

    subckt = SubCircuit(
        name="divider_sub",
        ports=[_e_port("divider_sub_p0"), _e_port("divider_sub_p1")],
        params=SubCircuitParams(),
        internal_network=direct_network,
        circuit_ports=direct_ports,
        max_depth=10,
    )
    sub_s = subckt.get_s(freq)

    np.testing.assert_allclose(sub_s, direct_s, atol=1e-12, rtol=1e-12)


@pytest.mark.xfail(reason="Known: nested subcircuit Z→S→Z scaling issue")
def test_nested_subcircuit() -> None:
    freq = FrequencyGrid(values=np.array([5e3], dtype=np.float64))
    inner = _make_subcircuit_from_divider("inner")
    outer = _nest_subcircuit(inner, levels=1, max_depth=10)

    inner_s = inner.get_s(freq)
    outer_s = outer.get_s(freq)

    np.testing.assert_allclose(outer_s, inner_s, atol=1e-12, rtol=1e-12)


def test_recursion_limit() -> None:
    freq = FrequencyGrid(values=np.array([1e3], dtype=np.float64))
    base = _make_subcircuit_from_divider("base")
    nested = _nest_subcircuit(base, levels=11, max_depth=10)

    with pytest.raises(RecursionError):
        nested.get_s(freq)
