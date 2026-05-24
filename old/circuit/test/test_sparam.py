import skrf as rf
from siluxApi.circuit.core import Circuit, Element
from siluxApi.circuit.core.lib import Capacitor, Inductor, Composite
from siluxApi.circuit.core.sparam import (
    create_testbench_circuit,
    solve_circuit_sparams,
    SParamTestBench
)
import numpy as np
from time import time

freq = rf.Frequency(0.1, 10, 1001, "GHz")

def get_reference_ntwk():
    """example from skrf.circuit lc filter example"""
    media = rf.DefinedGammaZ0(freq, z0=50, gamma=1j * freq.w / rf.c)
    C1 = media.capacitor(3.222e-12, name='C1')
    C2 = media.capacitor(82.25e-15, name='C2')
    C3 = media.capacitor(3.222e-12, name='C3')
    L2 = media.inductor(8.893e-9, name='L2')
    RL = media.resistor(50, name='RL')
    gnd = rf.Circuit.Ground(freq, name='gnd')
    port1 = rf.Circuit.Port(freq, name='port1', z0=50)
    port2 = rf.Circuit.Port(freq, name='port2', z0=50)

    cnx = [
        [(port1, 0), (C1, 0), (L2, 0), (C2, 0)],
        [(L2, 1), (C2, 1), (C3, 0), (port2, 0)],
        [(gnd, 0), (C1, 1), (C3, 1)],
    ]
    cir = rf.Circuit(cnx)
    ntwk = cir.network
    return ntwk

def get_test_circuit():
    circuit = Circuit(
        elements = [
            Capacitor.new(name="C1", c=3.222e-12),
            Capacitor.new(name="C2", c=82.25e-15),
            Capacitor.new(name="C3", c=3.222e-12),
            Inductor.new(name="L2", l=8.893e-9),
        ],
        links = [
            "C1 p1 gnd",
            "C2 p1 p2",
            "L2 p1 p2",
            "C3 p2 gnd"
        ]
    )
    circuit.connect()
    circuit.set_f(freq.f)
    return circuit

def test_testbench_circuit():
    test_circuit = create_testbench_circuit(
        get_test_circuit(), 
        ports=["p1", "p2"], 
        z0=[50, 50]
    )
    return

def test_solve_sparams():
    circuit = get_test_circuit()
    s = solve_circuit_sparams(circuit, ports=["p1", "p2"], z0=[50, 50])
    return s

def test_solve_sparams_ntwk():
    s = test_solve_sparams()
    ntwk = rf.Network(frequency=freq, s=s)
    assert ntwk == get_reference_ntwk()
    return ntwk

def test_inner_circuit_sparam():
    lc_filter = Element(
        name = "lc_filter",
        num_of_ports= 2,
        circuit = get_test_circuit(),
        circuit_ports= ["p1", "p2"],
        z0 = [50, 50],
    )
    lc_filter.connect()
    lc_filter.set_f(freq.f)
    s = lc_filter.get_s()
    assert np.allclose(s,  get_reference_ntwk().s)
    return

def test_parallel_composite():
    l = 8.893e-9
    c = 82.25e-15
    parallel = Composite(
        name = "parallel",
        num_of_ports= 2,
        params = {
            "l": l,
            "c": c
        },
        params_listeners= {
            "l": "L2::l",
            "c": "C2::c"
        },
        circuit = Circuit(
            elements=[
                Capacitor.new(name="C2", c=c),
                Inductor.new(name="L2", l=l)
            ],
            links=[
                "C2 p1 p2",
                "L2 p1 p2"
            ]
        ),
        circuit_ports= ["p1", "p2"],
    )
    parallel.connect()
    parallel.set_f(freq.f)
    z = parallel.get_impedance("0", "1")
    z_expected = 1 / (1j * freq.w * c + 1 / (1j * freq.w * l))
    assert np.allclose(z, z_expected)
    return parallel

def test_nested_composite():
    lc_filter = Composite(
        name = "lc_filter",
        num_of_ports= 2,
        params = {
            "l": 8.893e-9,
            "c1": 3.222e-12,
            "c2": 82.25e-15,
            "c3": 3.222e-12
        },
        params_listeners= {
            "l": "parallel::l",
            "c1": "C1::c",
            "c2": "parallel::c",
            "c3": "C3::c"
        },
        circuit = Circuit(
            elements=[
                test_parallel_composite(),
                Capacitor.new(name="C1", c=3.222e-12),
                Capacitor.new(name="C3", c=3.222e-12),
            ],
            links = [
                "parallel p1 p2",
                "C1 p1 gnd",
                "C3 p2 gnd",
            ]
        ),
        circuit_ports= ["p1", "p2"],
        z0 = [50, 50],
    )
    lc_filter.connect()
    lc_filter.set_f(freq.f)
    s = lc_filter.get_s()
    assert np.allclose(s,  get_reference_ntwk().s)
    return lc_filter

def test_load_lib():
    lc_filter = Element.parse(test_nested_composite().json())
    lc_filter.connect()
    lc_filter.set_f(freq.f)
    s = lc_filter.get_s()
    assert isinstance(lc_filter, Composite)
    assert np.allclose(s,  get_reference_ntwk().s)
    return lc_filter

def test_set_params():
    lc_filter = test_nested_composite()

    fig, ax = plt.subplots()
    for l in np.arange(5e-9, 10e-9, 1e-9):
        lc_filter.set_params(l=l)
        ax.plot(freq.f, lc_filter.get_s_component("0", "1"))
    save_png(fig)
    return
    

def test_performance():
    # skrf
    t0 = time()
    for _ in range(100):
        get_reference_ntwk()
    print(f"skrf Time: {time() - t0} s")

    # siluxApi func
    t0 = time()
    circuit = get_test_circuit()
    for _ in range(100):
        s = solve_circuit_sparams(circuit, ports=["p1", "p2"], z0=[50, 50])
    print(f"siluxApi func Time: {time() - t0} s")

    # siluxApi class
    t0 = time()
    for _ in range(100):
        testbench = SParamTestBench(circuit, ["p1", "p2"], [50, 50])
        s = testbench.get_s()
    print(f"siluxApi class creation and solve Time: {time() - t0} s")

    t0 = time()
    for _ in range(100):
        s = testbench.get_s()
    print(f"siluxApi class solve Time: {time() - t0} s")

    t0 = time()
    lc = test_nested_composite()
    for _ in range(100):
        lc.get_s()
    print(f"siluxApi nested solve Time: {time() - t0} s")

    return

def test_plot_ntwk():
    ref = get_reference_ntwk()
    test = test_solve_sparams_ntwk()
    fig, ax = plt.subplots()
    ref.plot_s_db(0, 0, logx=True, label='Reference S11', ax=ax)
    ref.plot_s_db(1, 0, logx=True, label='Reference S21', ax=ax)
    test.plot_s_db(0, 0, logx=True, ls="--", label='siluxApi S11', ax=ax)
    test.plot_s_db(1, 0, logx=True, ls="--", label='siluxApi S21', ax=ax)
    save_png(fig)
    return

def get_shunt_c_ntwk():
    """example from skrf.circuit lc filter example"""
    media = rf.DefinedGammaZ0(freq, z0=50, gamma=1j * freq.w / rf.c)
    C1 = media.capacitor(10e-15, name='C1')
    gnd = rf.Circuit.Ground(freq, name='gnd')
    port1 = rf.Circuit.Port(freq, name='port1', z0=100)
    port2 = rf.Circuit.Port(freq, name='port2', z0=100)

    cnx = [
        [(port1, 0), (C1, 0), (port2, 0)],
        [(gnd, 0), (C1, 1)],
    ]
    cir = rf.Circuit(cnx)
    ntwk = cir.network
    return ntwk

def test_shunt_c_circuit():
    shunt = Composite(
        name = "shunt_c",
        num_of_ports= 2,
        params = {
            "c": 10e-15,
        },
        params_listeners= {
            "c": "C::c"
        },
        circuit = Circuit(
            elements=[
                Capacitor.new(name="C", c=10e-15),
            ],
            links = [
                "C p0 gnd",
            ]
        ),
        circuit_ports= ["p0", "p0"],
        z0 = [100, 100],
    )
    shunt.connect()
    shunt.set_f(freq.f)
    s = shunt.get_s()
    target_s = get_shunt_c_ntwk().s
    assert np.allclose(s,  target_s)
    return s


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from erwen_env import load_erwen_style, save_png
    load_erwen_style()
    
    # test_testbench_circuit()
    # test_solve_sparams()
    test_solve_sparams_ntwk()
    # test_inner_circuit_sparam()
    # test_parallel_composite()
    test_nested_composite()
    # test_set_params()
    # test_load_lib()
    # test_performance()
    # test_plot_ntwk()
    test_shunt_c_circuit()
    