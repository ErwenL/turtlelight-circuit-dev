"""
## Calculate S-parameters from circuit solution

Steps:
    - create a testbench circuit from target circuit
    - solve the testbench circuit with different source setting
    - calculate S-parameters from the solution

Reference:
    - [Matlab: Extract S-Parameters from Circuit](https://www.mathworks.com/help/rf/ug/extract-s-parameters-from-circuit.html)
"""
from .circuit import Circuit, Node, Element, ComplexArray
from .solver import *
from .lib.basic import Resistor, CurrentSource
import numpy as np
from typing import Callable, Optional

###############################################################################
#		MARK: function approach
###############################################################################

def solve_circuit_sparams(circuit: Circuit, ports: list[str], z0: list[float]) -> ComplexArray:
    testbench = create_testbench_circuit(circuit, ports, z0)
    g = circuit_2_graph(testbench)
    b = find_best_circuit_set(g, "gnd")

    def _solve_input(input_port_num: int):
        setup_testbench_current_source(testbench, input_port_num, ports, z0)
        i, v = solve_iv_loop(b, *get_known_args_loop(g))
        update_branches_iv(g, i, v)
        update_node_voltages(g, "gnd", 0.)
        return

    return extract_s(g, ports, _solve_input)

def create_testbench_circuit(circuit: Circuit, ports: list[str], z0: list[float]) -> Circuit:

    def copy_element(element: Element) -> Element:
        _element =  element.parse_obj(element.copy())
        # sync s cache during element copy
        try:
            _element.s = element.get_s()
        except NotImplementedError:
            pass
        return _element
    
    _elements = [copy_element(element) for element in circuit.elements]
    _nodes = [Node(name=node.name) for node in circuit.nodes]
    _links = circuit.links.copy()

    def gnd_in_nodes():
        for node in _nodes:
            if node.name == "gnd":
                return True
        return False

    if not gnd_in_nodes():
        _nodes.append(Node(name="gnd"))

    for idx, (port, _z0) in enumerate(zip(ports, z0)):
        _elements.extend([
            CurrentSource.new(name=f"i_source_{idx}", i=0.),
            Resistor.new(name=f"terminal_{idx}", r=_z0)
        ])
        _links.extend([
            f"i_source_{idx} gnd {port}",
            f"terminal_{idx} {port} gnd"
        ])

    # skip Circuit validation for better performance
    testbench = Circuit.construct(
        {"elements", "nodes", "links"},
        elements = _elements,
        nodes = _nodes,
        links = _links
    )
    testbench.connect()
    testbench.set_f(circuit.elements[0].frequency)
    return testbench

def setup_testbench_current_source(
    testbench: Circuit, input_port_num: int,
    ports: list[str], z0: list[float]
):
    for idx, _port in enumerate(ports):
        if input_port_num == idx:
            testbench.get_element(f"i_source_{idx}").set_params(i=1. / z0[idx])
        else:
            testbench.get_element(f"i_source_{idx}").set_params(i=0)
    return

SolveInput = Callable[[int], None]

def extract_s_component(
    g: CircuitGraph, input_port_num: int, output_port_num: int, 
    ports: list[str], solve_input: Optional[SolveInput] = None
) -> ComplexArray:
    # solve circuit if needed
    if solve_input is not None:
        solve_input(input_port_num)
    # calculate s-parameter
    _v = g.nodes_dict[output_port := ports[output_port_num]].v
    assert _v is not None, f"Node voltage of {output_port} is not set"
    if input_port_num == output_port_num:
        return 2 * _v - 1
    return 2 * _v

def extract_s_column(
    g: CircuitGraph, input_port_num: int, ports: list[str], 
    solve_input: SolveInput,
) -> ComplexArray:
    solve_input(input_port_num)
    return np.vstack([
        extract_s_component(g, input_port_num, output_port_num, ports) 
        for output_port_num, _ in enumerate(ports)
    ]).T

def extract_s(
    g: CircuitGraph, ports: list[str], solve_input: SolveInput,
) -> ComplexArray:
    return np.dstack([
        extract_s_column(g, input_port_num, ports, solve_input) 
        for input_port_num, _ in enumerate(ports)
    ])

###############################################################################
#		MARK: class approach
###############################################################################

class SParamTestBench:
    circuit: Circuit
    g: CircuitGraph
    b: CircuitMatrixBreakdown
    ports: list[str]
    z0: list[float]
    current_solved: Optional[int]

    def __init__(self, circuit: Circuit, ports: list[str], z0: list[float]) -> None:
        self.circuit = create_testbench_circuit(circuit, ports, z0)
        self.g = circuit_2_graph(self.circuit)
        self.b = find_best_circuit_set(self.g, "gnd")
        self.ports = ports
        self.z0 = z0
        self.current_solved = None

    def _setup_current_source(self, input_port_num: int):
        setup_testbench_current_source(self.circuit, input_port_num, self.ports, self.z0)
        return

    def _solve(self):
        i, v = solve_iv_loop(self.b, *get_known_args_loop(self.g))
        update_branches_iv(self.g, i, v)
        update_node_voltages(self.g, "gnd", 0.)
        return
    
    def solve_input(self, input_port_num: int):
        if self.current_solved == input_port_num:
            return
        self._setup_current_source(input_port_num)
        self._solve()
        self.current_solved = input_port_num
        return

    def get_s_component(self, input_port_num: int, output_port_num:int) -> ComplexArray:
        return extract_s_component(self.g, input_port_num, output_port_num, self.ports, self.solve_input)

    def get_s(self) -> ComplexArray:
        return extract_s(self.g, self.ports, self.solve_input)




