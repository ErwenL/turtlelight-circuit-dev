# Circuit (siluxApi.circuit)

## Circuit

The [Circuit class](../../reference/circuit/core.md#circuit.core.circuit.Circuit) describes a circuit that consists of a list of elements (`Element`) and a list of nodes (`Node`). How elments and nodes are connected is decribed by a list of link strings (`LinkStr`).

### Concepts

#### [Element](../../reference/circuit/core.md#circuit.core.circuit.Element) 

> A circuit component with arbitrary number of ports(pins).

- *Each element in a circuit should have its own unique name*.
- Each port of an element also has a unique name, default to be the string representation of the port index.
    - The sequence of ports matters, which defines the direction of current flow (voltage drop).
- For a connected circuit, every port of a element should be connected to a node.

!!! Note

    - `i_source_{num}` and `terminal_{num}` are reserved element names for the [testbench circuit](../../reference/circuit/solver.md#circuit.core.sparam.create_testbench_circuit) during solving s-parameters.

#### [Node](../../reference/circuit/core.md#circuit.core.circuit.Node) 

> An abstract point in the circuit where two or more elements are connected.

- A node can be easily understood as connected wires in a circuit that have the same voltage.
- *Each node in a circuit should have its own unique name.*
- Currently, each node is allowed to connect only one port of an element.
    - Multiple ports of an element connecting to the same node is to be supported in the future.

!!! Note

    - `gnd` is a reserved node name for the ground node in the circuit.

#### [LinkStr](../../reference/circuit/core.md#circuit.core.circuit.LinkStr) 

> A link string describing how ports of an element and nodes are connected.

- supports two formats:
    - `element_name::port_name node_name` (intuitive and readable)
    - `element_name *node_names` (inspired by SPICE netlist)
        - node_name sequence should match the port sequence of the element

```py title="LinkStr format"
# for a "mos" element which has the ports in the order of "s", "g", "d"
# the following two link strings are equivalent

links_1 = [
    "mos::s n1",
    "mos::g n2",
    "mos::d n3",
]

links_2 = [
    "mos n1 n2 n3",
]
```

!!! warning "Attention"

    - The order of element ports (connected nodes) defines the direction of current flow or voltage drop.
        - For 2-port passive elements, such as R,L,C, the order of ports may be trivial.
        - For voltage source and current source, the order of ports matters.
        - For N-port elements (N > 2), recommend to use `element_name::port_name node_name` format to avoid confusion.

#### [CircuitGraph](../../reference/circuit/solver.md#circuit.core.graph.CircuitGraph)

    to be added

#### [Branch](../../reference/circuit/core.md#circuit.core.circuit.Branch)

    to be added

### Define a Circuit

#### Example

![lc_filter_circuit](https://scikit-rf.readthedocs.io/en/latest/_images/circuit_filter1.svg)

The above LC filter circuit can be defined as following python code:

```python
from siluxApi.circuit.core import Circuit
from siluxApi.circuit.core.lib import Capacitor, Inductor, Resistor, VoltageSource

# LC filter only
filter_circuit = Circuit(
    elements = [
        Capacitor.new(name="C1", c=3.222e-12),
        Capacitor.new(name="C2", c=82.25e-15),
        Capacitor.new(name="C3", c=3.222e-12),
        Inductor.new(name="L2", l=8.893e-9),
    ],
    # nodes: p1, p2, gnd
    links = [
        "C1 p1 gnd",
        "C2 p1 p2",
        "L2 p1 p2",
        "C3 p2 gnd"
    ]
)

# full circuit
circuit = Circuit(
    elements = [
        Capacitor.new(name="C1", c=3.222e-12),
        Capacitor.new(name="C2", c=82.25e-15),
        Capacitor.new(name="C3", c=3.222e-12),
        Inductor.new(name="L2", l=8.893e-9),
        Resistor.new(name="Rs", r=50),
        Resistor.new(name="Rl", r=50),
        VoltageSource.new(name="V", v=1.0),
    ],
    # nodes: p1, p2, gnd, s,
    links = [
        "C1 p1 gnd",
        "C2 p1 p2",
        "L2 p1 p2",
        "C3 p2 gnd",
        "V gnd s",
        "Rs s p1",
        "Rl p2 gnd"
    ]
)
```

Defining a circuit starts with claiming all the `Element`s used in the circuit.

- A `Element` can always be defined by the default init method `Element(*, **kwargs)` (Pydantic built-in initialization).
- `Element.new()` is the customized constructor for all `Element` subclasses, which provides a more intuitive interface to define an element.
  - It is recommended to use `Element.new()` for all basic elements.

```py title="create an Element"
# The following snippets are equivalent

# built-in init
c = Capacitor(name="C1", params={"c": 3.222e-12})
# customized constructor
c = Capacitor.new(name="C1", c=3.222e-12)
c = Capacitor.new("C1", 3.222e-12)
```

By principle, we should also claiming all the `Node`s used in the circuit.

- However, the initialization of a `Node` object only requires the node name, which is trivial.
- All the `Node` name informations have been included in the `LinkStr` list. Therefore, as long as we providing a valid `LinkStr` list, we don't need to claim all the nodes explicitly.
- `Node`s are automatically created during the validation process of the `Circuit` object.

### Connect a circuit

After the creation of a `Circuit`, we just claim the `Element`s and `Node`s. Each `Element` and `Node` still doesn't know each other.

We need run `circuit.connect()`, which add pointers to all connected `Node`s to the `Elment` object and vice versa based on the `LinkStr` list. 

```py title="connect elements and nodes in a circuit"
circuit.connect()
```

Then, we have successfully built a graph structure for the circuit. We can then go to solve the circuit.

### Solve a circuit

#### Set Frequency

Currrently, `siluxApi.circuit` focuses on solving linear circuit in frequency domain. Therefore, unless we are solving pure resistive circuits, we always need to set the frequency before solving the circuit.

Any values calculated during the circuit solving process will be aligned to the freqeuency shape.

- branch impedance, admittance, voltage, current
- s-parameters

We can set frequency to either the `Circuit` object or the `Element` object.

- `circuit.set_f(freqs: np.ndarray)` syncs the `frequency` attribute of all the elements in the circuit.
- `element.set_f(freqs: np.ndarray)` sets the `frequency` attribute of the element and broadcasts to all the elements in the element tree.

Therefore, we only need set frequency once either to a `Circuit` object or the any one `Element` object in the element tree. 

```py title="set frequency to circuit and element"
freqs = np.linspace(0.1e9, 10e9, 1001)

# set frequency to circuit
circuit.set_f(freqs)

# set frequency to element. 
element.set_f(freqs)
```

#### Solve IV

[`solve_circuit_loop`](../../reference/circuit/solver.md#circuit.core.sparam.solve_circuit_loop) solves the circuit based on [loop method](../../reference/circuit/solver.md#circuit.core.solver.solve_circuit_loop) and return a `CircuitGraph` object which stores current flow and voltage drop in `Branch` and absolute voltage in `Node`.

```py title="solve a circuit"
from siluxApi.circuit.core.solver import solve_circuit_loop

# solve the circuit and get the CircuitGraph object
g = solve_circuit_loop(circuit, "gnd", 0.)

# access branches IV
branch_currents = [branch.get_current() for branch in g.branches]
branch_voltages = [branch.get_voltage() for branch in g.branches]

# access nodes voltage
node_voltages = {node.name: node.v for node in g.nodes}
```
!!! warning "Attention"

    - For a circuit with only voltage sources and no current sources, it need to be solved by [cutset method](../../reference/circuit/solver.md#circuit.core.solver.solve_circuit_cutset), which has not been implemented yet.

#### Solve S-parameters

Having a passive circuit, we can consider it as a network by clarifying the nodes we want external ports to connect and the correspond port impedance. Then we can solve the s-parameters of the circuit.

[`solve_circuit_sparams`](../../reference/circuit/solver.md#circuit.core.sparam.solve_circuit_sparams) solve the s-parameters of the circuit. 

- Under the hood, it creates a testbench circuit by adding current sources and terminal resistors to port nodes of the original circuit.
- By solving the IV of the testbench circuit, we can get the s-parameters of the original circuit. Details of the method can be found [here](https://www.mathworks.com/help/rf/ug/extract-s-parameters-from-circuit.html).


![s_param_testbench_circuit](https://www.mathworks.com/help/examples/rf/win64/ExtractingSParametersExample_02.png){ width="500" }

```py title="solve circuit s-parameters"
from siluxApi.circuit.core import solve_circuit_sparams

# solve the s-parameters of the above LC filter circuit example
# port node: p1, p2
# port impedance: 50 ohm
s = solve_circuit_sparams(filter_circuit, ports=["p1", "p2"], z0=[50, 50])
```

