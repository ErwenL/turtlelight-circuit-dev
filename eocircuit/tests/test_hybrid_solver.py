"""Tests for hybrid electrical-photonic solver."""

from __future__ import annotations

# pyright: reportMissingImports=false

import numpy as np
from typing import cast

from eocircuit.core.component import Component, ComponentParams
from eocircuit.core.network import Network
from eocircuit.core.types import FrequencyGrid, Port, PortDomain
from eocircuit.solvers.hybrid import HybridSolver


class EmptyParams(ComponentParams):
    """Empty params for fixed-domain components."""


class VoltageSourceParams(ComponentParams):
    voltage: complex


class ResistorParams(ComponentParams):
    resistance: float


class MockEOParams(ComponentParams):
    base_t: complex = 0.4 + 0.0j
    gain: float = 0.1
    bias_voltage: float = 0.0
    elec_g: float = 1.0e-3


class MockPDParams(ComponentParams):
    optical_reflection: complex = 0.0 + 0.0j
    elec_g: float = 1.0e-3


class FixedDomainComponent(Component):
    """Simple fixed-domain component with identity Y/Z and zero S."""

    def get_s(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        return np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)

    def get_z(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        out = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        for i in range(n_freq):
            out[i] = np.eye(n_ports, dtype=np.complex128)
        return out

    def get_y(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        n_ports = self.num_of_ports
        out = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)
        for i in range(n_freq):
            out[i] = np.eye(n_ports, dtype=np.complex128)
        return out


class VoltageSource(Component):
    def get_s(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        return np.zeros((n_freq, 2, 2), dtype=np.complex128)

    def get_z(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        return np.zeros((n_freq, 2, 2), dtype=np.complex128)

    def get_y(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        return np.zeros((n_freq, 2, 2), dtype=np.complex128)


class Resistor(Component):
    def get_s(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        return np.zeros((n_freq, 2, 2), dtype=np.complex128)

    def get_z(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        z = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        params = cast(ResistorParams, self.params)
        for i in range(n_freq):
            z[i] = np.array(
                [[params.resistance, -params.resistance], [-params.resistance, params.resistance]],
                dtype=np.complex128,
            )
        return z

    def get_y(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        params = cast(ResistorParams, self.params)
        g = 1.0 / params.resistance
        y = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        stamp = np.array([[g, -g], [-g, g]], dtype=np.complex128)
        for i in range(n_freq):
            y[i] = stamp
        return y


class MockEO(Component):
    """EO modulator with 1 electrical + 2 optical ports."""

    def get_s(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        s = np.zeros((n_freq, 3, 3), dtype=np.complex128)
        params = cast(MockEOParams, self.params)
        t = params.base_t + params.gain * params.bias_voltage
        s[:, 1, 2] = t
        s[:, 2, 1] = t
        return s

    def get_z(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        z = np.zeros((n_freq, 3, 3), dtype=np.complex128)
        params = cast(MockEOParams, self.params)
        z[:, 0, 0] = 1.0 / params.elec_g
        z[:, 1, 1] = 1.0
        z[:, 2, 2] = 1.0
        return z

    def get_y(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        y = np.zeros((n_freq, 3, 3), dtype=np.complex128)
        params = cast(MockEOParams, self.params)
        y[:, 0, 0] = params.elec_g
        return y

    def set_electrical_bias(self, bias_by_port: dict[str, np.ndarray]) -> "MockEO":
        elec_port_name = self.ports[0].name
        bias = bias_by_port.get(elec_port_name)
        if bias is None:
            return self
        return self.set_params(bias_voltage=float(np.real(bias[0])))


class MockPD(Component):
    """Simple PD placeholder with 1 optical + 2 electrical ports."""

    def get_s(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        s = np.zeros((n_freq, 3, 3), dtype=np.complex128)
        params = cast(MockPDParams, self.params)
        s[:, 0, 0] = params.optical_reflection
        return s

    def get_z(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        z = np.zeros((n_freq, 3, 3), dtype=np.complex128)
        z[:, 1, 1] = 1.0
        z[:, 2, 2] = 1.0
        return z

    def get_y(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        params = cast(MockPDParams, self.params)
        y = np.zeros((n_freq, 3, 3), dtype=np.complex128)
        stamp = np.array(
            [[params.elec_g, -params.elec_g], [-params.elec_g, params.elec_g]],
            dtype=np.complex128,
        )
        for i in range(n_freq):
            y[i, 1:, 1:] = stamp
        return y


class Waveguide2Port(Component):
    """Simple 2-port optical through element."""

    def get_s(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        s = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        s[:, 0, 1] = 0.9 + 0.0j
        s[:, 1, 0] = 0.9 + 0.0j
        return s

    def get_z(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        z = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        for i in range(n_freq):
            z[i] = np.eye(2, dtype=np.complex128)
        return z

    def get_y(self, freq: FrequencyGrid) -> np.ndarray:
        n_freq = len(freq.values)
        y = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        for i in range(n_freq):
            y[i] = np.eye(2, dtype=np.complex128)
        return y


def _ep(name: str, index: int | None = None) -> Port:
    kwargs = {"index": index} if index is not None else {}
    return Port(name=name, domain=PortDomain.ELECTRICAL, z0=50.0, **kwargs)


def _op(name: str) -> Port:
    return Port(name=name, domain=PortDomain.OPTICAL, z0=50.0)


def test_partition() -> None:
    network = Network(name="mixed_partition")

    e1 = FixedDomainComponent(name="e1", ports=[_ep("e1_p1"), _ep("e1_p2")], params=EmptyParams())
    e2 = FixedDomainComponent(name="e2", ports=[_ep("e2_p1"), _ep("e2_p2")], params=EmptyParams())
    p1 = FixedDomainComponent(name="p1", ports=[_op("p1_in"), _op("p1_out")], params=EmptyParams())
    p2 = FixedDomainComponent(name="p2", ports=[_op("p2_in"), _op("p2_out")], params=EmptyParams())
    mzm = MockEO(
        name="mzm",
        ports=[_ep("mzm_bias"), _op("mzm_in"), _op("mzm_out")],
        params=MockEOParams(),
    )

    for comp in (e1, e2, p1, p2, mzm):
        network.add_component(comp)

    network.connect(e1.ports[1], mzm.ports[0])
    network.connect(mzm.ports[2], p1.ports[0])
    network.connect(p1.ports[1], p2.ports[0])

    elec_subnet, photonic_subnet, eo_devices = HybridSolver().partition_network(network)

    assert {comp.name for comp in eo_devices} == {"mzm"}
    assert {comp.name for comp in elec_subnet.components} == {"e1", "e2", "mzm"}
    assert {comp.name for comp in photonic_subnet.components} == {"p1", "p2", "mzm"}


def test_eo_link_basic() -> None:
    freq = FrequencyGrid(values=np.array([193.1e12], dtype=np.float64))
    network = Network(name="eo_link")

    vsrc = VoltageSource(
        name="vsrc",
        ports=[_ep("vsrc_p"), _ep("gnd", index=0)],
        params=VoltageSourceParams(voltage=1.5 + 0.0j),
    )
    mzm = MockEO(
        name="mzm",
        ports=[_ep("mzm_bias"), _op("mzm_in"), _op("mzm_out")],
        params=MockEOParams(base_t=0.2 + 0j, gain=0.5, bias_voltage=0.0, elec_g=2.0e-3),
    )
    wg = Waveguide2Port(name="wg", ports=[_op("wg_in"), _op("wg_out")], params=EmptyParams())
    pd = MockPD(
        name="pd",
        ports=[_op("pd_opt"), _ep("pd_out"), _ep("pd_gnd")],
        params=MockPDParams(),
    )
    r_load = Resistor(
        name="r_load",
        ports=[_ep("r_load_p"), _ep("r_load_n")],
        params=ResistorParams(resistance=50.0),
    )

    for comp in (vsrc, mzm, wg, pd, r_load):
        network.add_component(comp)

    # electrical chain
    network.connect(vsrc.ports[0], mzm.ports[0])
    network.connect(pd.ports[1], r_load.ports[0])
    network.connect(r_load.ports[1], vsrc.ports[1])

    # optical chain
    network.connect(mzm.ports[2], wg.ports[0])
    network.connect(wg.ports[1], pd.ports[0])

    result = HybridSolver().solve(network, freq)

    assert "mzm" in result.eo_state
    assert "mzm_bias" in result.eo_state["mzm"]
    np.testing.assert_allclose(result.eo_state["mzm"]["mzm_bias"], np.array([1.5 + 0j]))
    assert "pd" in result.eo_state
    assert result.photonic_result.shape == (1, 1, 1)
    assert np.isfinite(np.abs(result.photonic_result[0, 0, 0]))
