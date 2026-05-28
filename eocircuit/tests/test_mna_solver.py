"""Tests for eocircuit MNA solver."""

# pyright: reportMissingImports=false

import numpy as np

from eocircuit.core.component import Component, ComponentParams
from eocircuit.core.network import Network
from eocircuit.core.types import FrequencyGrid, Port, PortDomain
from eocircuit.solvers.mna import MNASolver


class ResistorParams(ComponentParams):
    resistance: float


class CapacitorParams(ComponentParams):
    capacitance: float


class VoltageSourceParams(ComponentParams):
    voltage: float


class CurrentSourceParams(ComponentParams):
    current: float


class MnaResistor(Component):
    def get_s(self, freq: FrequencyGrid):
        raise NotImplementedError

    def get_z(self, freq: FrequencyGrid):
        raise NotImplementedError

    def get_y(self, freq: FrequencyGrid):
        n_freq = len(freq.values)
        y = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        g = 1.0 / self.params.resistance
        stamp = np.array([[g, -g], [-g, g]], dtype=np.complex128)
        for i in range(n_freq):
            y[i] = stamp
        return y


class MnaCapacitor(Component):
    def get_s(self, freq: FrequencyGrid):
        raise NotImplementedError

    def get_z(self, freq: FrequencyGrid):
        raise NotImplementedError

    def get_y(self, freq: FrequencyGrid):
        n_freq = len(freq.values)
        y = np.zeros((n_freq, 2, 2), dtype=np.complex128)
        for i, f in enumerate(freq.values):
            yc = 1j * 2.0 * np.pi * f * self.params.capacitance
            y[i] = np.array([[yc, -yc], [-yc, yc]], dtype=np.complex128)
        return y


class MnaVoltageSource(Component):
    def get_s(self, freq: FrequencyGrid):
        raise NotImplementedError

    def get_z(self, freq: FrequencyGrid):
        raise NotImplementedError

    def get_y(self, freq: FrequencyGrid):
        n_freq = len(freq.values)
        return np.zeros((n_freq, 2, 2), dtype=np.complex128)


class MnaCurrentSource(Component):
    def get_s(self, freq: FrequencyGrid):
        raise NotImplementedError

    def get_z(self, freq: FrequencyGrid):
        raise NotImplementedError

    def get_y(self, freq: FrequencyGrid):
        n_freq = len(freq.values)
        return np.zeros((n_freq, 2, 2), dtype=np.complex128)


def _ef_port(name: str) -> Port:
    return Port(name=name, domain=PortDomain.ELECTRICAL, z0=50.0, index=1)


def test_resistive_divider() -> None:
    freq = FrequencyGrid(values=np.array([1.0e3], dtype=np.float64))
    solver = MNASolver()
    network = Network(name="divider")

    p_src_p = _ef_port("src_p")
    p_src_n = Port(name="src_n", domain=PortDomain.ELECTRICAL, z0=50.0, index=0)
    vsrc = MnaVoltageSource(name="V1", ports=[p_src_p, p_src_n], params=VoltageSourceParams(voltage=5.0))

    p_r1_a = _ef_port("r1_a")
    p_r1_b = _ef_port("r1_b")
    r1 = MnaResistor(name="R1", ports=[p_r1_a, p_r1_b], params=ResistorParams(resistance=100.0))

    p_r2_a = _ef_port("r2_a")
    p_r2_b = _ef_port("r2_b")
    r2 = MnaResistor(name="R2", ports=[p_r2_a, p_r2_b], params=ResistorParams(resistance=100.0))

    for comp in (vsrc, r1, r2):
        network.add_component(comp)

    network.connect(p_src_p, p_r1_a)
    network.connect(p_r1_b, p_r2_a)
    network.connect(p_r2_b, p_src_n)

    sol = solver.solve(network, freq)
    v_mid = sol.get_port_voltage("r1_b")[0]
    assert np.isclose(v_mid.real, 2.5, atol=1e-10)
    assert np.isclose(v_mid.imag, 0.0, atol=1e-12)


def test_rc_lowpass() -> None:
    f0 = 159_000.0
    freq = FrequencyGrid(values=np.array([f0], dtype=np.float64))
    solver = MNASolver()
    network = Network(name="rc")

    p_src_p = _ef_port("src_p")
    p_src_n = Port(name="src_n", domain=PortDomain.ELECTRICAL, z0=50.0, index=0)
    vsrc = MnaVoltageSource(name="V1", ports=[p_src_p, p_src_n], params=VoltageSourceParams(voltage=1.0))

    p_r_a = _ef_port("r_a")
    p_r_b = _ef_port("r_b")
    r1 = MnaResistor(name="R1", ports=[p_r_a, p_r_b], params=ResistorParams(resistance=1_000.0))

    p_c_a = _ef_port("c_a")
    p_c_b = _ef_port("c_b")
    c1 = MnaCapacitor(name="C1", ports=[p_c_a, p_c_b], params=CapacitorParams(capacitance=1e-9))

    for comp in (vsrc, r1, c1):
        network.add_component(comp)

    network.connect(p_src_p, p_r_a)
    network.connect(p_r_b, p_c_a)
    network.connect(p_c_b, p_src_n)

    sol = solver.solve(network, freq)
    vin = sol.get_port_voltage("src_p")[0]
    vout = sol.get_port_voltage("r_b")[0]
    gain = np.abs(vout / vin)
    assert np.isclose(gain, 1 / np.sqrt(2), rtol=2e-2)


def test_kcl_at_node() -> None:
    freq = FrequencyGrid(values=np.array([10_000.0], dtype=np.float64))
    solver = MNASolver()
    network = Network(name="kcl")

    p_src_p = _ef_port("src_p")
    p_src_n = Port(name="src_n", domain=PortDomain.ELECTRICAL, z0=50.0, index=0)
    vsrc = MnaVoltageSource(name="V1", ports=[p_src_p, p_src_n], params=VoltageSourceParams(voltage=3.0))

    p_r1_a = _ef_port("r1_a")
    p_r1_b = _ef_port("mid")
    r1 = MnaResistor(name="R1", ports=[p_r1_a, p_r1_b], params=ResistorParams(resistance=200.0))

    p_r2_a = _ef_port("r2_a")
    p_r2_b = _ef_port("r2_b")
    r2 = MnaResistor(name="R2", ports=[p_r2_a, p_r2_b], params=ResistorParams(resistance=300.0))

    for comp in (vsrc, r1, r2):
        network.add_component(comp)

    network.connect(p_src_p, p_r1_a)
    network.connect(p_r1_b, p_r2_a)
    network.connect(p_r2_b, p_src_n)
    sol = solver.solve(network, freq)
    i_r1 = sol.get_branch_currents("R1")[0, 1]
    i_r2 = sol.get_branch_currents("R2")[0, 0]
    assert np.isclose((i_r1 + i_r2).real, 0.0, atol=1e-10)
    assert np.isclose((i_r1 + i_r2).imag, 0.0, atol=1e-10)


def test_voltage_source() -> None:
    freq = FrequencyGrid(values=np.array([1_000.0], dtype=np.float64))
    solver = MNASolver()
    network = Network(name="vsrc")

    p_p = _ef_port("vp")
    p_n = Port(name="vn", domain=PortDomain.ELECTRICAL, z0=50.0, index=0)
    vsrc = MnaVoltageSource(name="V1", ports=[p_p, p_n], params=VoltageSourceParams(voltage=3.0))

    network.add_component(vsrc)
    sol = solver.solve(network, freq)
    assert np.isclose(sol.get_port_voltage("vp")[0].real, 3.0, atol=1e-12)
    assert np.isclose(sol.get_port_voltage("vn")[0].real, 0.0, atol=1e-12)


def test_two_resistors_series_total_impedance() -> None:
    freq = FrequencyGrid(values=np.array([2_000.0], dtype=np.float64))
    solver = MNASolver()
    network = Network(name="series_r")

    p_v_p = _ef_port("v_p")
    p_v_n = Port(name="v_n", domain=PortDomain.ELECTRICAL, z0=50.0, index=0)
    vsrc = MnaVoltageSource(name="V1", ports=[p_v_p, p_v_n], params=VoltageSourceParams(voltage=1.0))

    p_r1_a = _ef_port("r1_a")
    p_r1_b = _ef_port("r1_b")
    r1 = MnaResistor(name="R1", ports=[p_r1_a, p_r1_b], params=ResistorParams(resistance=150.0))

    p_r2_a = _ef_port("r2_a")
    p_r2_b = _ef_port("r2_b")
    r2 = MnaResistor(name="R2", ports=[p_r2_a, p_r2_b], params=ResistorParams(resistance=350.0))

    for comp in (vsrc, r1, r2):
        network.add_component(comp)

    network.connect(p_v_p, p_r1_a)
    network.connect(p_r1_b, p_r2_a)
    network.connect(p_r2_b, p_v_n)

    sol = solver.solve(network, freq)
    i_source = sol.get_branch_currents("V1")[0, 0]
    z_total = 1.0 / np.abs(i_source)
    assert np.isclose(z_total.real, 500.0, atol=1e-10)
    assert np.isclose(z_total.imag, 0.0, atol=1e-10)
