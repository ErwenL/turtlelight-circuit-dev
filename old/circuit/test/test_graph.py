from siluxApi.circuit.core import Circuit, Element
from siluxApi.circuit.core.graph import (
    circuit_2_graph,
    get_cotree,
    get_incidence_matrix,
    get_cut_matrix,
    get_circuit_matrix
)
import numpy as np

def test_graph():
    circuit = Circuit(
        elements = [
            Element(name="e1", num_of_ports=2),
            Element(name="e2", num_of_ports=2),
            Element(name="e3", num_of_ports=2),
            Element(name="e4", num_of_ports=2),
            Element(name="e5", num_of_ports=2),
            Element(name="e6", num_of_ports=2),
            Element(name="e7", num_of_ports=2),
        ],
        links = [
            "e1 V1 V2",
            "e2 V2 V3",
            "e3 V3 V4",
            "e4 V5 V4",
            "e5 V3 V5",
            "e6 V1 V4",
            "e7 V5 V1"
        ]
    )
    circuit.connect()
    g = circuit_2_graph(circuit)
    g.branches.sort(key=lambda x: x.element.name)
    tree = [_ for i, _ in enumerate(g.branches) if i in [0, 1, 4, 5]]
    cotree = get_cotree(g.branches, tree)

    a = get_incidence_matrix(g)
    a_target = np.array([
        [1, 0, 0, 0, 0, 1, -1],
        [-1, 1, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 1, 0, 0],
        [0, 0, -1, -1, 0, -1, 0],
        [0, 0, 0, 1, -1, 0, 1]
    ])
    assert np.all(a == a_target), "Incidence matrix is wrong"

    q = get_cut_matrix(g, tree, cotree)
    q_target = np.array([
        [1, 0, 0, 0, -1, -1, -1],
        [0, 1, 0, 0, -1, -1, -1],
        [0, 0, 1, 0, 0, -1, -1],
        [0, 0, 0, 1, 1, 1, 0]
    ])
    assert np.all(q[:, np.array([1,2,5,6,3,4,7])-1] == q_target), "Cut matrix is wrong"

    b = get_circuit_matrix(g, cotree, tree)
    b_target = np.array([
        [1, 0, 0, 1, 1, 0, -1],
        [0,1,0,1,1,1,-1],
        [0,0,1,1,1,1,0]
    ])
    assert np.all(b[:, np.array([3,4,7,1,2,5,6])-1] == b_target), "Circuit matrix is wrong"

    for i in range(len(tree)):
        for j in range(len(cotree)):
            assert np.sum(b[j,:]*q[i,:]) == 0
    return

if __name__ == "__main__":
    test_graph()