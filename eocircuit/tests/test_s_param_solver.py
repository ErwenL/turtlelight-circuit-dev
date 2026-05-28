"""Tests for S-parameter reduction solver."""

from __future__ import annotations

# pyright: reportMissingImports=false

import numpy as np

from eocircuit.core.component import Component, ComponentParams
from eocircuit.core.network import Network
from eocircuit.core.types import FrequencyGrid, Port, PortDomain
from eocircuit.solvers.s_param import SParamSolver


class EmptyParams(ComponentParams):
    """Empty params for test components."""


class FixedSComponent(Component):
    """Component with fixed S-matrix replicated over frequency."""

    s_single: np.ndarray

    def get_s(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        return np.repeat(self.s_single[np.newaxis, :, :], n_freq, axis=0).astype(np.complex128)

    def get_z(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        return np.repeat(np.eye(n_ports, dtype=np.complex128)[np.newaxis, :, :], n_freq, axis=0)

    def get_y(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        return np.repeat(np.eye(n_ports, dtype=np.complex128)[np.newaxis, :, :], n_freq, axis=0)


def _opt_port(name: str) -> Port:
    return Port(name=name, domain=PortDomain.OPTICAL, z0=50.0)


def test_cascade_two_components() -> None:
    freq = FrequencyGrid(values=np.array([193.1e12]))

    t1 = 0.8 + 0.1j
    t2 = 0.5 - 0.05j
    s_wg1 = np.array([[0.0, t1], [t1, 0.0]], dtype=np.complex128)
    s_wg2 = np.array([[0.0, t2], [t2, 0.0]], dtype=np.complex128)

    wg1 = FixedSComponent(
        name="wg1",
        ports=[_opt_port("wg1_in"), _opt_port("wg1_out")],
        params=EmptyParams(),
        s_single=s_wg1,
    )
    wg2 = FixedSComponent(
        name="wg2",
        ports=[_opt_port("wg2_in"), _opt_port("wg2_out")],
        params=EmptyParams(),
        s_single=s_wg2,
    )

    network = Network(name="cascade")
    network.add_component(wg1)
    network.add_component(wg2)
    network.connect(wg1.ports[1], wg2.ports[0])

    s = SParamSolver().solve(network, freq)

    expected_t = t1 * t2
    expected = np.array([[0.0, expected_t], [expected_t, 0.0]], dtype=np.complex128)
    np.testing.assert_allclose(s[0], expected, atol=1e-12)


def test_dc_50_50() -> None:
    freq = FrequencyGrid(values=np.array([193.1e12]))
    k = 1 / np.sqrt(2)

    # Symmetric directional coupler (ports: in1, in2, out1, out2)
    s_dc = np.array(
        [
            [0.0, 0.0, k, 1j * k],
            [0.0, 0.0, 1j * k, k],
            [k, 1j * k, 0.0, 0.0],
            [1j * k, k, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    dc = FixedSComponent(
        name="dc",
        ports=[_opt_port("in1"), _opt_port("in2"), _opt_port("out1"), _opt_port("out2")],
        params=EmptyParams(),
        s_single=s_dc,
    )

    network = Network(name="dc_only")
    network.add_component(dc)

    s = SParamSolver().solve(network, freq)

    p_out1 = np.abs(s[0, 2, 0]) ** 2
    p_out2 = np.abs(s[0, 3, 0]) ** 2
    np.testing.assert_allclose(p_out1, 0.5, atol=1e-12)
    np.testing.assert_allclose(p_out2, 0.5, atol=1e-12)
    np.testing.assert_allclose(p_out1 + p_out2, 1.0, atol=1e-12)


def test_parallel_ports() -> None:
    freq = FrequencyGrid(values=np.array([193.1e12]))

    s1 = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.complex128)
    s2 = np.array([[0.0, 0.6], [0.6, 0.0]], dtype=np.complex128)

    c1 = FixedSComponent(
        name="c1",
        ports=[_opt_port("c1_p1"), _opt_port("c1_p2")],
        params=EmptyParams(),
        s_single=s1,
    )
    c2 = FixedSComponent(
        name="c2",
        ports=[_opt_port("c2_p1"), _opt_port("c2_p2")],
        params=EmptyParams(),
        s_single=s2,
    )

    network = Network(name="parallel_links")
    network.add_component(c1)
    network.add_component(c2)
    network.connect(c1.ports[0], c2.ports[0])
    network.connect(c1.ports[1], c2.ports[1])

    s = SParamSolver().solve(network, freq)

    assert s.shape == (1, 0, 0)


def test_three_port_network() -> None:
    freq = FrequencyGrid(values=np.array([193.1e12]))

    s4 = np.array(
        [
            [0.1, 0.2, 0.05, 0.4],
            [0.3, 0.1, 0.25, 0.2],
            [0.15, 0.35, 0.05, 0.1],
            [0.45, 0.05, 0.2, 0.1],
        ],
        dtype=np.complex128,
    )
    r = 0.7 - 0.1j
    s1 = np.array([[r]], dtype=np.complex128)

    hub = FixedSComponent(
        name="hub",
        ports=[_opt_port("p0"), _opt_port("p1"), _opt_port("p2"), _opt_port("p3")],
        params=EmptyParams(),
        s_single=s4,
    )
    term = FixedSComponent(
        name="term",
        ports=[_opt_port("t0")],
        params=EmptyParams(),
        s_single=s1,
    )

    network = Network(name="three_port_topology")
    network.add_component(hub)
    network.add_component(term)
    network.connect(hub.ports[3], term.ports[0])

    s = SParamSolver().solve(network, freq)

    # Expected reduction from wave constraints:
    # a_i = P(S_ie a_e + S_ii a_i) => a_i = (I - P S_ii)^-1 P S_ie a_e
    # b_e = S_ee a_e + S_ei a_i
    s_total = np.block(
        [
            [s4, np.zeros((4, 1), dtype=np.complex128)],
            [np.zeros((1, 4), dtype=np.complex128), s1],
        ]
    )
    ext = [0, 1, 2]
    internal = [3, 4]
    s_ee = s_total[np.ix_(ext, ext)]
    s_ei = s_total[np.ix_(ext, internal)]
    s_ie = s_total[np.ix_(internal, ext)]
    s_ii = s_total[np.ix_(internal, internal)]
    p = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    expected = s_ee + s_ei @ np.linalg.inv(np.eye(2, dtype=np.complex128) - p @ s_ii) @ p @ s_ie

    np.testing.assert_allclose(s[0], expected, atol=1e-12)
