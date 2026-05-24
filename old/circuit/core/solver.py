"""
## Solve Circuit based on graph theory

Dimensional symbol:
    - F: frequency length
    - N_i: number of current sources
    - N_v: number of voltage sources
    - N_rlc: number of rlc elements
        - N_rlc_s: number of rlc elements in cutset or loop
    - N: number of branches
        - N = N_i + N_v + N_rlc

### Circuit Equations

Rerange branch sequence:
    $$
    I = \\begin{bmatrix}
        I_1 \\\\
        I_2 \\\\
        I_3 \\\\
    \\end{bmatrix}, V = \\begin{bmatrix}
        V_1 \\\\
        V_2 \\\\
        V_3 \\\\
    \\end{bmatrix}
    $$

    1. Current Source Branches
    2. RLC Branches
    3. Voltage Source Branches
    
Fundamental Circuit Matrix:
    $$
    B = \\begin{bmatrix}
        U & B_{12} & B_{13} \\\\
        0 & B_{22} & B_{23} \\
    \\end{bmatrix}
    $$

KVL:
    $$
    BV = 
    \\begin{bmatrix}
        U & B_{12} & B_{13} \\\\
        0 & B_{22} & B_{23} \\
    \\end{bmatrix} \\begin{bmatrix}
        V_1 \\\\
        V_2 \\\\
        V_3 \\\\
    \\end{bmatrix} = 0
    $$

Fundamental Cut Matrix:

KCL:

"""
from .circuit import Circuit, Node, Branch, AnyValueArray, ComplexArray
from .graph import (
    CircuitGraph,
    circuit_2_graph,
    graph_2_tree,
    graph_spanning_tree_iterator,
    get_cut_matrix,
    get_circuit_matrix
)
import numpy as np
from numpy.typing import NDArray
from loguru import logger
from functools import cache


def get_num_of_sources(g:CircuitGraph):
    current_sources = [branch for branch in g.branches if branch.element.type == "current_source"]
    voltage_sources = [branch for branch in g.branches if branch.element.type == "voltage_source"]
    return len(current_sources), len(voltage_sources)

def _get_rlc_branch_params(g:CircuitGraph, get_param: str) -> ComplexArray:
    n_i, n_v = get_num_of_sources(g)
    num_of_rlc = len(g.branches) - n_i - n_v
    param_by_branch = [getattr(branch, get_param)() for branch in g.branches[n_i:n_i+num_of_rlc]]
    assert isinstance(param_by_branch[0], np.ndarray), "branch params need to be np.ndarray"
    param = np.zeros((param_by_branch[0].size, num_of_rlc, num_of_rlc), dtype=complex)
    for i, _p in enumerate(param_by_branch):
        param[:, i, i] = _p
    return param

def get_rlc_impedances(g:CircuitGraph) -> ComplexArray:
    """shape: (F, N_rlc, N_rlc)"""
    return _get_rlc_branch_params(g, "get_impedance")

def get_rlc_admittances(g:CircuitGraph) -> ComplexArray:
    """shape: (F, N_rlc, N_rlc)"""
    return _get_rlc_branch_params(g, "get_admittance")

def get_current_source_currents(g:CircuitGraph) -> ComplexArray:
    """shape: (F, N_i, 1)"""
    n_i, _ = get_num_of_sources(g)
    if n_i == 0:
        return np.array([])
    i_by_source = [branch.get_current() for branch in g.branches[:n_i]]
    i = np.zeros((i_by_source[0].size, n_i, 1), dtype=complex)
    for idx, _i in enumerate(i_by_source):
        i[:, idx, 0] = _i
    return i

def get_voltage_source_voltages(g:CircuitGraph) -> ComplexArray:
    """shape: (F, N_v, 1)"""
    _, n_v = get_num_of_sources(g)
    if n_v == 0:
        return np.array([])
    v_by_source = [branch.get_voltage() for branch in g.branches[-n_v:]]
    v = np.zeros((v_by_source[0].size, n_v, 1), dtype=complex)
    for idx, _v in enumerate(v_by_source):
        v[:, idx, 0] = _v
    return v

def stack_branches_iv(*ivs: ComplexArray) -> ComplexArray:
    """
    Stack branch currents and voltages vectors

    len(iv.shape):
        - (n, 1): vstack, first axis
        - (F, n, 1): hstack, second axis

    Args:
        ivs (ComplexArray): branch currents and voltages
    """
    if len(ivs[0].shape) == 2:
        return np.vstack(ivs)
    if len(ivs[0].shape) == 3:
        return np.hstack(ivs)
    raise ValueError("Invalid branch currents and voltages shape")

def update_branches_iv(g: CircuitGraph, i: ComplexArray, v: ComplexArray):
    """broadcast iv vectors to branches iv cache
    
    - branches order should match the iv vectors order

    Args:
        g (CircuitGraph): circuit graph
        i (ComplexArray): current vector, shape: (F, n, 1)
        v (ComplexArray): voltage vector, shape: (F, n, 1)
    """
    for idx, branch in enumerate(g.branches):
        if branch.element_type != "current_source":
            branch.set_current(i[:, idx, 0].flatten())
        if branch.element_type != "voltage_source":
            branch.set_voltage(v[:, idx, 0].flatten())
    return 

def update_node_voltages(g: CircuitGraph, root: str, reference_voltage: float):
    """calculate and update node voltages in a circuit graph based on branch voltages

    - search all node based on DFS algorithm
    
    Args:
        g (CircuitGraph): circuit graph
        root (str): root node name
        reference_voltage (float): reference voltage
    """
    _root = g.nodes_dict[root]
    _root.v = _root.elements[0].as_f(reference_voltage).astype(complex)

    def dfs(node: Node, visited: set[str]):
        visited.add(node.name)
        for branch in node.branches:
            next_node = branch.get_the_other_node(node)
            if next_node.name not in visited:
                assert node.v is not None, "Node voltage has not been set"
                next_node.v = node.v - branch.get_voltage(node.name, next_node.name)
                dfs(next_node, visited)
        return

    dfs(g.nodes_dict[root], set())
    return

###############################################################################
#		MARK: Loop Analysis
###############################################################################

CircuitMatrixBreakdown = dict[str, NDArray[np.integer]]
"""
$$
\\begin{bmatrix}
    B_{11} & B_{12} & B_{13} \\\\
    B_{21} & B_{22} & B_{23} \\
\\end{bmatrix}
$$

Shape:
    - $B_{11}$: (N_i, N_i)
    - $B_{12}$: (N_i, N_rlc)
    - $B_{13}$: (N_i, N_v)
    - $B_{21}$: (N_rlc_s, N_i)
    - $B_{22}$: (N_rlc_s, N_rlc)
    - $B_{23}$: (N_rlc_s, N_v)
"""

def get_broken_circuit_matrix(g:CircuitGraph, cotree: list[Branch], tree: list[Branch]) -> CircuitMatrixBreakdown:    
    """
    Returns:
        circuit matrix breakdown
    """
    b = get_circuit_matrix(g, cotree, tree)
    n_i, n_v = get_num_of_sources(g)
    # col num between B_i2 and B_i3
    _n_v = b.shape[1] - n_v 
    return {
        "11": b[:n_i, :n_i],
        "12": b[:n_i, n_i:_n_v],
        "13": b[:n_i, _n_v:],
        "21": b[n_i:, :n_i],
        "22": b[n_i:, n_i:_n_v],
        "23": b[n_i:, _n_v:],
    }

def _is_good_circuit_matrix_breakdown(b: CircuitMatrixBreakdown) -> bool:
    if not np.all(b["21"]==0):
        return False
    return np.all(b["11"] == np.eye(b["11"].shape[0], dtype=int)) # type: ignore

def find_best_circuit_set(g: CircuitGraph, start: str = "gnd") -> CircuitMatrixBreakdown:
    # nodes = list(g.nodes_dict.keys())
    # if start in nodes:
    #     nodes.remove(start)
    #     nodes.insert(0, start)
    # for node in nodes:
    #     for method in ["dfs", "bfs"]:
    #         tree, cotree = graph_2_tree(g, node, method) # type: ignore
    #         print(tree)
    #         b = get_broken_circuit_matrix(g, cotree, tree)
    #         if _is_good_circuit_matrix_breakdown(b):
    #             return b
    for tree, cotree in graph_spanning_tree_iterator(g, start):
        b = get_broken_circuit_matrix(g, cotree, tree)
        if _is_good_circuit_matrix_breakdown(b):
            return b
    raise ValueError("No good circuit matrix breakdown found")

def get_known_args_loop(g: CircuitGraph) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """get known arguments for loop analysis solver

    Known Arguments:
        - z: rlc impedance
        - i_1: current source current
        - v_3: voltage source voltage

    Args:
        g (CircuitGraph): circuit graph

    Returns:
        rlc impedance, current source current, voltage source voltage
    """
    z = get_rlc_impedances(g)
    i_1 = get_current_source_currents(g)
    v_3 = get_voltage_source_voltages(g)
    return z, i_1, v_3

def solve_iv_loop(
    b: CircuitMatrixBreakdown,
    z: ComplexArray,
    i_1: ComplexArray,
    v_3: ComplexArray
) -> tuple[ComplexArray, ComplexArray]:
    """calculate the current and voltage of branches based on loop method

    Args:
        b (CircuitMatrixBreakdown): cicuit matrix breakdown
        z (ComplexArray): rlc impedance
        i_1 (ComplexArray): current source current
        v_3 (ComplexArray): voltge source voltage

    Returns:
        tuple[ComplexArray, ComplexArray]: branch currents and voltages
    """
    calculate_voltage = v_3.shape[0] > 0

    @cache
    def get_loop_current():
        """shape: (F, N_rlc, 1)"""
        loop_impedance = b["22"] @ z @ b["22"].T
        v_loop = - b["22"] @ z @ b["12"].T @ i_1
        if calculate_voltage > 0:
            v_loop += -b["23"] @ v_3
        return np.linalg.inv(loop_impedance) @ v_loop
    
    def get_rlc_iv():
        i = b["22"].T @ get_loop_current() + b["12"].T @ i_1
        v = z @ i
        return i, v

    def calculate_branches_iv():
        """shape: (F, N, 1)"""
        i_2, v_2 = get_rlc_iv()
        v_1 = -b["12"] @ v_2
        if calculate_voltage > 0:
            v_1 += -b["13"] @ v_3
            i_3 = b["13"].T @ i_1 + b["23"].T @ get_loop_current()
            return stack_branches_iv(i_1, i_2, i_3), stack_branches_iv(v_1, v_2, v_3)
        return stack_branches_iv(i_1, i_2), stack_branches_iv(v_1, v_2)
    
    return calculate_branches_iv()

def solve_circuit_loop(circuit: Circuit, root:str = "gnd", reference_voltage:float = 0.) -> CircuitGraph:
    """solve circuit based on loop analysis

    Args:
        circuit (Circuit): Circuit object
        root (str, optional): root node with reference voltage set. Defaults to "gnd".
        reference_voltage (float, optional): voltage of root node. Defaults to 0..

    Returns:
        CircuitGraph: corresponding CircuitGraph object

    Fundamental Circuit Matrix:
        $$
        B = \\begin{bmatrix}
            U & B_{12} & B_{13} \\\\
            0 & B_{22} & B_{23} \\
        \\end{bmatrix}
        $$

    KVL:
        $$
        BV = 
        \\begin{bmatrix}
            U & B_{12} & B_{13} \\\\
            0 & B_{22} & B_{23} \\
        \\end{bmatrix} \\begin{bmatrix}
            V_1 \\\\
            V_2 \\\\
            V_3 \\\\
        \\end{bmatrix} = 0
        $$

    """
    g = circuit_2_graph(circuit)
    n_i, n_v = get_num_of_sources(g)
    if n_i == 0:
        logger.warning("No current sources in the circuit, switch to cutset analysis")
        return solve_circuit_cutset(circuit, root, reference_voltage)

    b = find_best_circuit_set(g, root)
    i, v = solve_iv_loop(b, *get_known_args_loop(g))

    update_branches_iv(g, i, v)
    update_node_voltages(g, root, reference_voltage)
    return g

###############################################################################
#		MARK: Cutset Analysis
###############################################################################

def solve_circuit_cutset(circuit: Circuit, root:str = "gnd", reference_voltage:float = 0.) -> CircuitGraph:
    raise NotImplementedError("Cutset analysis is not implemented yet")
