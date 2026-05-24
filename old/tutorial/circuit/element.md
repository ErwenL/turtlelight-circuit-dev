# Circuit (siluxApi.circuit)

## Element

The [Element class](../../reference/circuit/core.md#circuit.core.Element) is the base class for all circuit elements, which is the center of `siluxApi.circuit` package.

### Element Class breakdown

#### Ports

When we think about a circuit `Element`, the first thing that comes to mind is how many ports (pins, nets, etc.) it has. Anyway we choose the word "port" to represent the connection point of an `Element`. 

[`ports`](../../reference/circuit/core.md#circuit.core.circuit.Element.port_order) attributes is a list of portnames (`str`).

Default to be the string representation of the index of the port. For example, a 2-port `Element` will have `["0", "1"]` as its default `ports` attribute.

- Portname may be trival for passive circuit elements, like R, L, C. But for some elements, you may want to assign meaningful portnames. 
- For example, you may want to assign `"p", "n"` as portnames for a diode, and `"s", "d", "g"` for a MOS transistor.

The sequence of ports matters, which defines the direction of current flow (voltage drop). 

- Current flows from port with lower index to port with higher index.
- The direction of the current flow definition can be overridden by the [`port_order`](../../reference/circuit/core.md#circuit.core.circuit.Element.port_order) attribute.

!!!warning "Attention"

    You need to pay attention to the current flow direction when you define the voltage source or current source elements.

#### Portname interfaced properties

There are some important `Element` methods that are interfaced with portnames, which are:

- [`get_impedance`](../../reference/circuit/core.md#circuit.core.circuit.Element.get_impedance)
- [`get_admittance`](../../reference/circuit/core.md#circuit.core.circuit.Element.get_admittance)
- [`get_voltage`](../../reference/circuit/core.md#circuit.core.circuit.Element.get_voltage)
- [`get_current`](../../reference/circuit/core.md#circuit.core.circuit.Element.get_current)
- [`get_s_component`](../../reference/circuit/core.md#circuit.core.circuit.Element.get_s_component)

They all share similar interface as `Element.get_some_property(_from_port_name, _to_port_name)`.

When defining a new basic `Element` subclass, the key is to implement one or some of these methods.

[`element_type`](../../reference/circuit/core.md#circuit.core.circuit.Element.element_type) attribute labels the type of the `Element` subclass.

- For `rlc` type `Element`, you need to implement `get_impedance` or `get_admittance` method, defining how the impedance or admittance is related with its `params` attribute.
- For `voltage_source` or `current_source` type `Element`, you need to implement `get_voltage` or `get_current` method, respectively.

!!!note

    - For `Composite` element, `get_impedance` is calculated based on its s-parameters.

#### Inner Circuit

- [`circuit`](../../reference/circuit/core.md#circuit.core.circuit.Element.circuit) attribute stores the inner `Circuit` of the `element`
- [`circuit_ports`](../../reference/circuit/core.md#circuit.core.circuit.Element.circuit_ports) attribute defines the `*ode` names of the inner `Circuit` that is connected to `Element` port. Therefore, the order of the `circuit_ports` should match the order of the `ports`.
- [`z0`](../../reference/circuit/core.md#circuit.core.circuit.Element.z0) attribute stores the port impedance of the `Element`, so as the inner `Circuit`.

- [`Element.get_s()`](../../reference/circuit/core.md#circuit.core.circuit.Element.get_s) method calculates the s-parameters of the `Element` based on its inner `Circuit`.
    - `circuit`, `circuit_ports`, `z0` together define a complete `Circuit` based network, of which we can [solve the s-parameters](circuit.md#solve-s-parameters).

```py title="calculate the s-parameters of an element's inner circuit"
# the following two lines are equivalent
s = element.get_s()
s = solve_circuit_sparams(element.circuit, element.circuit_ports, element.z0)
```

#### Element Tree

- [`children`](../../reference/circuit/core.md#circuit.core.circuit.Element.children) attribute stores pointers to all the `elements` of a `Element`'s inner `Circuit`.
    - Therefore, the `Element` and `elments` in its inner `Circuit` form a tree structure.
- [`parent`](../../reference/circuit/core.md#circuit.core.circuit.Element.parent) attribute stores pointer to its parent `Element` in the `Element` tree.

Typically, upon creating an `Element` instance, the `Element` and its `children` don't know the existence of each other. We need call [`Element.connect()`](../../reference/circuit/core.md#circuit.core.circuit.Element.connect) to creating the pointers.


```py title="connect element tree"
element.connect()
```

#### Params management

[`params`](../../reference/circuit/core.md#circuit.core.circuit.Element.params) attribute stores the parameters of the `Element`.

- When creating of an `Element` instance, the `params` can be defined as a `dict`.
- After creation of an `Element` instance, the `params` attribute will be converted to a `Element.Params(pydantic.BaseModel)` object.
- All params values are supposed to be `float` or `np.ndarray[float]`. No complex number is allowed. This is majorly due to the data storage and serialization issue. 

```py title="define params of an element"

# define params as a dict
lc_filter = Element(
    name = "lc_filter,
    num_of_ports = 2,
    params = {
        "l": 8.893e-9,
        "c1": 3.222e-12,
        "c2": 82.25e-15,
        "c3": 3.222e-12
    }
) 
# this also works
lc_filter = Element(
    name = "lc_filter,
    num_of_ports = 2,
    params = dict(
        l= 8.893e-9,
        c1= 3.222e-12,
        c2= 82.25e-15,
        c3= 3.222e-12
    )
) 

# we can access the params as a object
>>> print(lc_filter.params.l)
>>> 8.893e-9

```

[`params_listeners`](../../reference/circuit/core.md#circuit.core.circuit.Element.params_listeners) attribute defines how the `Element` params is related to child `Element`'s params or other `Element`'s params in the element tree.

- `params_listeners` is a `dict` with key as the `params` name and values as [`ParamListener`](../../reference/circuit/core.md#circuit.core.circuit.ParamListener) string.
    - format: 
        - `child_element_name::param_name`
        - `".".join(node_names_in_path)::param_name`

```py title="define params listeners"

# suppose we have an element tree like this
# lc_filter {
#       Parallel: { L, C },
#       C1,
#       C3
# }

parallel = Element(
    name = "Parallel",
    num_of_ports = 2,
    params = dict(
        l= 8.893e-9,
        c= 3.222e-12
    )
    params_listeners = {
        "l": "L::l",
        "c": "C::c",
    },
    ...
)

# we can define the params_listeners of lc_filter like this
lc_filter = Element(
    name = "lc_filter,
    num_of_ports = 2,
    params = dict(
        l= 8.893e-9,
        c1= 3.222e-12,
        c2= 82.25e-15,
        c3= 3.222e-12
    ),
    params_listeners = {
        # link params to child element's params
        "l": "Parallel::l"
        "c1": "C1::c",
        "c2": "Parallel::c",
        "c3": "C3::c"
    }
) 
# this also works
lc_filter = Element(
    name = "LC_filter,
    num_of_ports = 2,
    params = dict(
        l= 8.893e-9,
        c1= 3.222e-12,
        c2= 82.25e-15,
    ),
    params_listeners = {
        # link params to grand child (element in the element tree) element's params
        "l": "Parallel.L::l"
        "c2": "Parallel.C::c",
        # link params to multiple listeners
        "c1": ["C1::c", "C3::c"],
    }
) 

```

Any change in the `Element.params` will trigger the `Element.params_listeners` to update the related `Element.params`.

- Directly modifying `Element.params` is not allowed
- Always update the `Element.params` through the [`Element.set_params(**kwargs)`](../../reference/circuit/core.md#circuit.core.circuit.Element.set_params) method.

```py title="update params"
# DON'T do this
lc_filter.params.l = 8e-9

# do this
lc_filter.set_params(l=8e-9) 
```

### Core Element Library

#### Basic Elements

- [Resistor](../../reference/circuit/library.md#circuit.core.lib.basic.Resistor)
- [Capacitor](../../reference/circuit/library.md#circuit.core.lib.basic.Capacitor)
- [Inductor](../../reference/circuit/library.md#circuit.core.lib.basic.Inductor)
- [VoltageSource](../../reference/circuit/library.md#circuit.core.lib.basic.VoltageSource)
- [CurrentSource](../../reference/circuit/library.md#circuit.core.lib.basic.CurrentSource)

#### Composite Element

Despite the basic elements, all the other elements should be instance or subclass instance of the [`Composite`](../../reference/circuit/library.md#circuit.core.lib.composite.Composite) class.

- `get_impedance` of a `Composite` element is calculated based on its s-parameters.

```py title="Composite Element example"

# the LC filter in the previous circuit example can be defined as a Composite element

parallel = Composite(
    name = "Parallel",
    num_of_ports= 2,
    params = {
        "l": 8.893e-9,
        "c": 82.25e-15
    },
    params_listeners= {
        "l": "L2::l",
        "c": "C2::c"
    },
    circuit = Circuit(
        elements=[
            Capacitor.new(name="C2", c=82.25e-15),
            Inductor.new(name="L2", l=8.893e-9)
        ],
        links=[
            "C2 p1 p2",
            "L2 p1 p2"
        ]
    ),
    circuit_ports= ["p1", "p2"],
)

lc_filter = Composite(
    name = "LC_filter",
    num_of_ports= 2,
    params = {
        "l": 8.893e-9,
        "c1": 3.222e-12,
        "c2": 82.25e-15,
        "c3": 3.222e-12
    },
    params_listeners= {
        "l": "Parallel::l"
        "c1": "C1::c",
        "c2": "Parallel::c",
        "c3": "C3::c"
    },
    circuit = Circuit(
        elements=[
            Capacitor.new(name="C1", c=3.222e-12),
            parallel,
            Capacitor.new(name="C3", c=3.222e-12)
        ],
        links=[
            "C1 p1 gnd",
            "Parallel p1 p2",
            "C3 p2 gnd"
        ]
    ),
    circuit_ports= ["p1", "p2"],
)

```

