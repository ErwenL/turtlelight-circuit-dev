from siluxApi.circuit.core import (
    Circuit,
    Element,
)
from siluxApi.circuit.core.lib import (
    Resistor,
    VoltageSource,
    CurrentSource,
)
from siluxApi.circuit.core.graph import (
    circuit_2_graph,
    get_cotree,
    get_incidence_matrix,
    get_cut_matrix,
    get_circuit_matrix,
    sort_branches
)
from siluxApi.circuit.core.solver import (
    solve_iv_loop,
    solve_circuit_loop
)
import numpy as np

def get_sample_circuit():
    circuit = Circuit(
        elements = [
            CurrentSource.new(name="e1", i=1, z0=50),
            Resistor.new(name="e2", r=3),
            Resistor.new(name="e3", r=1),
            Resistor.new(name="e4", r=1),
            Resistor.new(name="e5", r=1),
            VoltageSource.new(name="e6", v=2),
        ],
        links = [
            "e1 gnd V2",
            "e2 V1 gnd",
            "e3 V0 V2",
            "e4 V0 V1",
            "e5 V1 V2",
            "e6 V0 gnd",
        ]
    )
    circuit.connect()
    return circuit

def test_circuit_serialization():
    circuit = get_sample_circuit()
    reconstruct_circuit = Circuit.parse_obj(circuit.dict())
    assert circuit.json() == reconstruct_circuit.json()
    return

def test_circuit_matrix_2_iv():
    b = {
        "12":np.array([[0,0,-1,-1]]), 
        "13":np.array([[1]]),
        "22":np.array([
            [1,0,1,0], 
            [0,1,-1,-1]
        ]), 
        "23":np.array([[-1], [0]])
    }
    z = np.diag([3, 1, 1, 1])
    i_1 = np.array([[1]])
    v_3 = np.array([[2]])
    i, v = solve_iv_loop(b, z, i_1, v_3)
    assert np.allclose((i * 11).flatten(), np.array([11, 7, -5, 1, -6, 4]))
    assert np.allclose((v * 11).flatten(), np.array([-27, 21, -5, 1, -6, 22]))
    i_1 = np.array([np.array([[1]]) for _ in range(11)])
    v_3 = np.array([np.array([[2]]) for _ in range(11)])
    i, v = solve_iv_loop(b, z, i_1, v_3)
    return

def test_solver():
    circuit = get_sample_circuit()
    g = circuit_2_graph(circuit)
    g.branches.sort(key=lambda x: x.element.name)
    tree = g.sort_tree([branch for branch in g.branches if branch.element.name in ["e4", "e5", "e6"]])
    cotree = g.sort_tree(get_cotree(g.branches, tree))
    q = get_cut_matrix(g, tree, cotree)
    b = get_circuit_matrix(g, cotree, tree)
    assert np.all(
        b == np.array([
            [1,0,0,-1,-1,1],
            [0,1,0,1,0,-1],
            [0,0,1,-1,-1,0]
        ])
    )
    assert np.all(
        q == np.array([
            [1,-1,1,1,0,0],
            [1,0,1,0,1,0],
            [-1,1,0,0,0,1]
        ])
    )
    g = solve_circuit_loop(circuit, "gnd", 0.)
    g.branches.sort(key=lambda x: x.element.name)
    i = [branch.get_current().flatten() for branch in g.branches]
    v = [branch.get_voltage().flatten() for branch in g.branches]
    assert np.allclose(np.real(np.stack(i) * 11).flatten(), np.array([11, 7, -5, 1, -6, 4]))
    assert np.allclose(np.real(np.stack(v) * 11).flatten(), np.array([-27, 21, -5, 1, -6, 22]))
    return

def test_solver_with_f():
    circuit = get_sample_circuit()
    circuit.set_f(np.linspace(1e9, 40e9, 40))
    g = solve_circuit_loop(circuit, "gnd", 0.)
    i = [branch.get_current() for branch in g.branches]
    v = [branch.get_voltage() for branch in g.branches]
    return

def test_element_copy():
    circuit = get_sample_circuit()
    circuit.set_f(np.linspace(1e9, 40e9, 40))
    r1 = circuit.elements[1]
    r2 = Resistor.parse_obj(r1.copy())
    return




if __name__ == "__main__":
    test_circuit_serialization()
    test_circuit_matrix_2_iv()
    test_solver()
    test_solver_with_f()
    test_element_copy()