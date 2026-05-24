"""
"""
from __future__ import annotations
from pydantic import BaseModel, Field, validator, root_validator
from dataclasses import dataclass
from typing import Optional, Literal, Any, Sequence, SupportsFloat, Self
import numpy as np
from numpy.typing import ArrayLike, NDArray
from numbers import Number
import json

AnyValueArray = NDArray[np.floating|np.complexfloating]
ComplexArray = NDArray[np.complexfloating]
AnyParamValue = float|NDArray[np.floating]|Sequence[float]|None

class Node(BaseModel):
    name: str
    elements: list[Element] = Field(default_factory=lambda:[], exclude=True)
    v: Optional[ComplexArray] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"{self.name}:{[element.name for element in self.elements]}"

    @property
    def neighbours(self) -> list[Node]:
        """Get all connected nodes"""
        nodes = {}
        for element in self.elements:
            for node in element.nodes:
                if node is not None and node.name != self.name:
                    nodes[node.name] = node
        return list(nodes.values())

    @property
    def branches(self) -> list[Branch]:
        """Get all branches connected to the node"""
        branches = []
        for element in self.elements:
            # TODO: current version doesn't consider the case that 
            # TODO: multiple ports of a element connecting to same node
            for i, node in enumerate(element.nodes):
                assert node is not None, f"{element}::{element.ports[i]} is not connected to any node"
                if node.name != self.name:

                    def get_order(node_name:str) -> int:
                        node_names = [node.name for node in element.nodes] # type: ignore
                        return element.port_order.index(
                            element.ports[node_names.index(node_name)]
                        )

                    self_order = get_order(self.name)
                    node_order = get_order(node.name)
                    _from = min(self_order, node_order)
                    _to = max(self_order, node_order)
                    _from_node = element.nodes[_from]
                    _to_node = element.nodes[_to]
                    assert isinstance(_from_node, Node), "Node must be a Node object"
                    assert isinstance(_to_node, Node), "Node must be a Node object"
                    branches.append(Branch(
                        element=element,
                        from_port=element.ports[_from],
                        to_port=element.ports[_to],
                        from_node=_from_node,
                        to_node=_to_node
                    ))
        return branches

BrachKey = tuple[set[str], str, set[str]]
"""BranchKey format:
(node_names, element_name, port_names)
"""

@dataclass
class Branch:
    """Branch serves as Edge in the circuit graph.

    - It also acts as another layer between node and element,
    which translates the *portname-interfaced* element methods to *nodename-interfaced*.
    """
    element: Element
    from_port: str
    to_port: str
    from_node: Node
    to_node: Node

    def __str__(self):
        """
        Example:
            vdd-0:R:1->gnd
        """
        return f"{self.from_node.name}-{self.from_port}:{self.element.name}:{self.to_port}->{self.to_node.name}"

    def __repr__(self):
        return f"Branch({self.__str__()})"

    @property
    def link_strs(self) -> list[LinkStr]:
        return [
            f"{self.element.name}::{self.from_port} {self.from_node.name}",
            f"{self.element.name}::{self.to_port} {self.to_node.name}",
        ]

    @property
    def name(self):
        return str(self)

    @property
    def key(self) -> BrachKey:
        return (
            {self.from_node.name, self.to_node.name}, 
            self.element.name, 
            {self.from_port, self.to_port}
        )

    @property
    def element_type(self) -> ElementType:
        return self.element.type

    def get_impedance(self) -> ComplexArray:
        return np.array(
            self.element.get_impedance(self.from_port, self.to_port),
            dtype=complex
        )

    def get_admittance(self) -> ComplexArray:
        return np.array(
            self.element.get_admittance(self.from_port, self.to_port),
            dtype=complex
        )
    
    def nodename_pair_2_port_pair(self, _from: Optional[str] = None, _to: Optional[str] = None) -> tuple[str, str]:
        """validate and convert nodename pair to port name pair

        Args:
            _from (Optional[str], optional): from node name. Defaults to None.
            _to (Optional[str], optional): to node name. Defaults to None.

        Returns:
            from port name, to port name
        """
        if _from is None:
            _from = self.from_node.name
        if _to is None:
            _to = self.to_node.name
        assert {_from, _to} == {self.from_node.name, self.to_node.name}, f"Invalid port from {_from}, to {_to}"
        nodename_2_port = {
            self.from_node.name: self.from_port,
            self.to_node.name: self.to_port
        }
        return nodename_2_port[_from], nodename_2_port[_to]

    def get_current(self, _from: Optional[str] = None, _to: Optional[str] = None) -> ComplexArray:
        return np.array(
            self.element.get_current(*self.nodename_pair_2_port_pair(_from, _to)),
            dtype=complex
        )

    def get_voltage(self, _from: Optional[str] = None, _to: Optional[str] = None) -> ComplexArray:
        return np.array(
            self.element.get_voltage(*self.nodename_pair_2_port_pair(_from, _to)),
            dtype=complex
        )

    def set_current(self, value: ComplexArray, _from: Optional[str] = None, _to: Optional[str] = None):
        self.element.set_current(*self.nodename_pair_2_port_pair(_from, _to), value)

    def set_voltage(self, value: ComplexArray, _from: Optional[str] = None, _to: Optional[str] = None):
        self.element.set_voltage(*self.nodename_pair_2_port_pair(_from, _to), value)

    def get_the_other_node(self, node: Node) -> Node:
        if node.name == self.from_node.name:
            return self.to_node
        if node.name == self.to_node.name:
            return self.from_node
        raise ValueError("Node not in the branch")
    

LibName = str
"""for `libname` attr of Element class

Format:
    - `{library_name}.{element_class_name}`
    - `{library_name}.{'.'.join(module_names)}.{element_class_name}`
    - default library name is `core`, which can be omitted

"""
ElementType = Literal["current_source", "rlc", "voltage_source"]
IvCache = dict[tuple[str, str], ComplexArray|None]
ParamListener = str
"""str to define the param listener (linked param) of the element

Format:
    - child listener: 
        - `{child element name}::{child element param name}`
    - any listener in the element tree: 
        - `{".".join(node_names_in_path)}::{child element param name}`
"""

class Element(BaseModel):
    """Base class for all elements in the circuit.
    """
    name: str
    libname: LibName = "Element"
    num_of_ports: int
    ports: list[str] = Field(default_factory=lambda:[])
    port_order: list[str] = Field(default_factory=lambda:[])
    """Current flow direction is defined based on port order:

    - flow from lower index to higher index
    """
    nodes: list[Node|None] = Field(default_factory=lambda:[], exclude=True)
    """List of nodes connected to the element."""
    parent: Optional[Element] = Field(default=None, exclude=True)
    children: list[Element] = Field(default_factory=lambda:[], exclude=True)
    circuit: Circuit = Field(default=None)
    """internal circuit of the element"""
    circuit_ports: Optional[list[str]] = None
    """nodes of the internal circuit corresponding to the element ports"""
    type: ElementType = "rlc"
    """type of element for arranging the branches during the network analysis"""

    class Config:
        arbitrary_types_allowed = True

    @validator("ports", always=True)
    def validate_ports(cls, v:list, values:dict) -> list[str]:
        """validate and set_default port names

        - Defaults to str(index) of port number if not provided

        Args:
            v (list): input port names. If input is empty. Default port names will be set.
            values (dict): default port names will be generated based the set values.
                - fields: num_of_ports

        Raises:
            AssertionError: Number of nodes must be equal to num_of_ports

        Returns:
            list[str]: port names
        """
        assert isinstance(v, list), "Ports must be a list"
        if len(v) == 0:
            return [f"{i}" for i in range(values["num_of_ports"])]
        if len(v) == values["num_of_ports"]:
            return v
        raise ValueError(f"Number of ports must be {values['num_of_ports']}")

    @validator("port_order", always=True)
    def validate_port_order(cls, v:list, values:dict) -> list[str]:
        """validate and set_default port order which defines the current flow direction

        - If not provided, default port order will be set to ports.
        - If provided, it must be a permutation of ports.

        Args:
            v (list): input port order. If input is empty. Default port order will be set.
            values (dict): default port order will be generated based the set values.
                - fields: ports

        Returns:
            list[str]: port order
        """
        assert isinstance(v, list), "Port order must be a list"
        if len(v) == 0:
            return values["ports"]
        else:
            assert len(v) == len(values["ports"]), "Port order must be a permutation of ports"
            assert set(v) == set(values["ports"]), "Port order must be a permutation of ports"
        return v

    @validator("nodes", always=True)
    def validate_nodes(cls, v:list, values:dict) -> list[Node|None]:
        """validate and set_default nodes

        Args:
            v (list): input nodes. If input is empty. Default nodes will be set to list of None.
            values (dict): default nodes will be set based on "num_of_ports"

        Returns:
            list[Node]: nodes
        """
        assert isinstance(v, list), "Nodes must be a list"
        if len(v) == 0:
            return [None for _ in range(values["num_of_ports"])]
        assert len(v) == values["num_of_ports"], "Number of nodes must be equal to num_of_ports"
        return v

    ###############################################################################
    #		MARK: element tree
    ###############################################################################

    @classmethod
    def new(cls, **kwargs) -> Self:
        """custom construction method creating a new element instance

        - To be overriden by subclasses
            - for the ease of creating new element instances with self-defined interfaces
            - avoid overriding pydantic BaseModel __init__ method
        """
        return cls(**kwargs)

    @staticmethod
    def parse(input:Any) -> Element:
        """Customized parse method

        - parse input to corresponding Element subclass object based on `libname` attr.
        - valid input types:
            - json str
            - dict from `element.dict()` method (pydantic)
            - Element object

        Args:
            input (Any): input to parse

        Raises:
            ValueError: input must be a string, dict or Element object

        Returns:
            Element: correspond Element subclass obj
        """
        if isinstance(input, str):
            return load_element_json(input)
        if isinstance(input, dict|Element):
            return parse_element(input)
        raise ValueError("input must be a string, dict or Element object")

    def connect(self):
        """
        - connect internal circuit elements to the element
        - connect elements and ports of the internal circuit
        """
        if not isinstance(self.circuit, Circuit):
            return self
        self.children = self.circuit.elements
        for element in self.children:
            element.parent = self
        self.circuit.connect()
        return self
    
    @property
    def children_dict(self) -> dict[str, Element]:
        return {element.name: element for element in self.children}

    @property
    def root_element(self):
        if self.parent is None:
            return self
        return self.parent.root_element

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}:{[node.name for node in self.nodes if node is not None]})"

    def get_impedance(self, _from:Optional[str]=None, _to:Optional[str]=None, recursion_depth:int=0) -> AnyValueArray:
        """Get impedance matrix components of the element

        Args:
            _from (str): from port name
            _to (str): to port name
            recursion_depth (int): recursion depth

        Returns:
            impedance matrix element
        """
        if recursion_depth > 1:
            raise NotImplementedError("either get_impedance or get_admittance method need to be implemented")
        try:
            return 1 / np.array(self.get_admittance(_from, _to, recursion_depth+1))
        except NotImplementedError as e:
            raise e

    def get_admittance(self, _from:Optional[str]=None, _to:Optional[str]=None, recursion_depth:int=0) -> AnyValueArray:
        """Get admittance matrix components of the element

        Args:
            _from (str): from port name
            _to (str): to port name
            recursion_depth (int): recursion depth

        Returns:
            admittance matrix element
        """
        if recursion_depth > 1:
            raise NotImplementedError("either get_impedance or get_admittance method need to be implemented")
        try:
            return 1 / np.array(self.get_impedance(_from, _to, recursion_depth+1))
        except NotImplementedError as e:
            raise e
    
    ###############################################################################
    #		MARK: iv management
    ###############################################################################

    i: IvCache = Field(default=None, exclude=True)
    """current cache of the element, default None"""
    v: IvCache = Field(default=None, exclude=True)
    """voltage cache of the element, default None"""

    @validator("i", "v", always=True)
    def initialize_iv(
        cls, 
        v:Optional[IvCache],
        values:dict
    ) -> IvCache:
        if v is None:
            _v = {}
            for i, _from in enumerate(values["port_order"]):
                for j, _to in enumerate(values["port_order"]):
                    if i < j:
                        _v[(_from, _to)] = None
            return _v
        return v
    
    @property
    def port_vectors(self) -> list[tuple[str, str]]:
        vectors = []
        for i, _from in enumerate(self.port_order):
            for j, _to in enumerate(self.port_order):
                if i < j:
                    vectors.append((_from, _to))
        return vectors
    
    def get_current_direction(self, _from:str, _to:str) -> Literal[1, -1]:
        """get corresponding current flow direction 

        Args:
            _from (str): from port name
            _to (str): to port name

        Raises:
            ValueError: invalid port from _from, to _to

        Returns:
            sign of current direction
        """
        if (_from, _to) in self.port_vectors:
            return 1
        if (_to, _from) in self.port_vectors:
            return -1
        raise ValueError(f"invalid port from {_from}, to {_to} ")
    
    def get_voltage_direction(self, _from:str, _to:str) -> Literal[1, -1]:
        """corresponding voltage drop direction
        Args:
            _from (str): from port name
            _to (str): to port name
        """
        return self.get_current_direction(_from, _to)

    def _get_iv_cache(self, _from:str, _to:str, attr:Literal["i", "v"]) -> NDArray[np.complexfloating]:
        """Get iv cache from _from to _to

        Args:
            _from (str): from port name
            _to (str): to port name
            attr (Literal[&quot;i&quot;, &quot;v&quot;]): attr name, i or v

        Returns:
            current/voltage from _from port to _to port
        """
        _attr = getattr(self, attr)
        _attr_name = {
            "i": "current",
            "v": "voltage"
        }
        if (_from, _to) in _attr:
            _p = _attr[(_from, _to)]
        elif (_to, _from) in _attr:
            _p = _attr[(_to, _from)]
        else:
            raise ValueError(f"invalid port from {_from}, to {_to} ")
        assert _p is not None, f"{_attr_name[attr]}({_from}->{_to}) is None"
        return _p

    def _set_iv_cache(self, _from:str, _to:str, value:ComplexArray, attr:Literal["i", "v"]):
        """Set iv cache from _from to _to

        Args:
            _from (str): from port name
            _to (str): to port name
            value (ComplexArray): values to be set
            attr (Literal[&quot;i&quot;, &quot;v&quot;]): attr name, i or v
        """
        if (_from, _to) in self.port_vectors:
            getattr(self, attr)[(_from, _to)] = value
        elif (_to, _from) in self.port_vectors:
            getattr(self, attr)[(_to, _from)] = -value
        else:
            raise ValueError(f"invalid port from {_from}, to {_to} ")

    def get_current(self, _from:str, _to:str) -> ComplexArray:
        """Get current flowing from _from to _to

        Args:
            _from (str): from port name
            _to (str): to port name

        Returns:
            current flowing from _from port to _to port
        """
        if self.type == "current_source":
            raise NotImplementedError("get_current method of current_source need to be implemented separately")
        return self._get_iv_cache(_from, _to, "i") * self.get_current_direction(_from, _to)

    def get_voltage(self, _from:str, _to:str) -> ComplexArray:
        """Get voltage drop from _from to _to

        Args:
            _from (str): from port name
            _to (str): to port name

        Returns:
            voltage drop from _from port to _to port
        """
        if self.type == "voltage_source":
            raise NotImplementedError("get_voltage method of voltage_source need to be implemented separately")
        return self._get_iv_cache(_from, _to, "v") * self.get_voltage_direction(_from, _to)

    def set_current(self, _from:str, _to:str, value: ComplexArray):
        """
        Args:
            _from (str): from port name
            _to (str): to port name
        """
        assert self.type != "current_source", "current current_source can't been set via set_current"
        self._set_iv_cache(_from, _to, value, "i")


    def set_voltage(self, _from:str, _to:str, value: ComplexArray):
        """
        Args:
            _from (str): from port name
            _to (str): to port name
        """
        assert self.type != "voltage_source", "current voltage_source can't been set via set_voltage"
        self._set_iv_cache(_from, _to, value, "v")

    ###############################################################################
    #		MARK: params management
    ###############################################################################

    params: Element.Params|dict[str, AnyParamValue] = Field(default_factory=lambda:{})
    """element params

    - Param values are  designed to be all real for the ease of json serialization
    """
    params_listeners: dict[str, ParamListener|list[ParamListener]] = Field(default_factory=lambda:{})
    """internal circuit params that listens to the element params changes

    - key: element param name
    - value: `ParamListener`
        - child listener: 
            - `{child element name}::{child element param name}`
        - any listener in the element tree: 
            - `{".".join(node_names_in_path)}::{child element param name}`
    """

    class Params(BaseModel):
        class Config:
            arbitrary_types_allowed = True
            extra = "allow"
            allow_mutation = False
            json_encoders = {
                np.ndarray: lambda v: v.tolist()
            }
            json_loads = lambda x: {
                key:np.array(value) 
                for key, value in json.loads(x).items() 
                if isinstance(value, list)
            }

    @validator("params", always=True)
    def validate_params(cls: Element, v:dict) -> Element.Params:
        return cls.Params.parse_obj(v)

    def get_child(self, name:str) -> Element:
        """get child element or grandchild element down the element tree by name

        Args:
            name (str): child element name
                - child name
                - ".".join(element names in the path to the grandchild)
        
        Returns:
            child or grandchild element
        """
        if "." not in name:
            return self.children_dict[name]
        path_to_child = name.split(".")
        try:
            _next = self
            for node_name in path_to_child:
                _next = _next.children_dict[node_name]
            return _next
        except KeyError:
            raise KeyError(f"Element '{name}' not found in the children")

    def notifiy_params_listeners(self, **kwargs: AnyParamValue):
        kwargs_by_child = {}
        for key, value in kwargs.items():
            if key in self.params_listeners:
                _listensers = self.params_listeners[key]
                if isinstance(_listensers, str):
                    _listensers = [_listensers]
                for _listener in _listensers:
                    _child, _key = _listener.split("::")
                    if _child in kwargs_by_child:
                        kwargs_by_child[_child][_key] = value
                    else:
                        kwargs_by_child[_child] = {_key: value}
        for _child, _kwargs in kwargs_by_child.items():
            self.get_child(_child).set_params(**_kwargs)
        return

    def set_params(self, **kwargs: AnyParamValue):
        """update element params and notify listeners

        - always update the params object as whole
        """
        assert isinstance(self.params, self.Params), "params need to be of type cls.Params"
        # convetion to self.Params obj has been done in __setattr__
        self.params = {**self.params.dict(), **kwargs}
        self.notifiy_params_listeners(**kwargs)
        return

    ###############################################################################
    #		MARK: ac analysis
    ###############################################################################

    z0: list[float] = Field(default=None)
    """port impedance list corresponding to the port, default to be 50 ohm

    Attenions:
        - z0 definition is different from skrf which is a fxn complex array
        - It is very uncommon that we need define complex frequency dependent port impedance
        - we will convert z0 to frequency shape during the ac analysis
    """

    @validator("z0", always=True, pre=True)
    def validate_z0(cls, v: Any, values: dict) -> list[float]:
        return cls._validate_z0(v, values["num_of_ports"])

    @staticmethod
    def _validate_z0(z0:Any, num_of_ports:int) -> list[float]:
        if z0 is None:
            return [50. for _ in range(num_of_ports)]
        if isinstance(z0, SupportsFloat):
            return [float(z0) for _ in range(num_of_ports)]
        if isinstance(z0, Sequence):
            z0 = [float(_v) for _v in z0]
            assert len(z0) == num_of_ports, "Number of z0 must be equal to num_of_ports"
            return z0
        raise ValueError("z0 must be a number or a list of numbers")

    def set_z0(self, z0: AnyParamValue):
        """always update z0 list as a whole, instead of individual port impedance"""
        super().__setattr__("z0", self._validate_z0(z0, self.num_of_ports))
        return

    def port_z0(self, port:str) -> float:
        """get port impedance

        Args:
            port (str): port name

        Returns:
            float: port impedance
        """
        return self.z0[self.ports.index(port)]

    frequency: NDArray[np.floating] = Field(default=None, exclude=True)
    """frequencies of the ac analysis"""

    @property
    def f(self):
        """alias of frequency"""
        return self.frequency

    @property
    def jw(self) -> AnyValueArray:
        """angular frequency: $j \\omega$"""
        assert self.frequency is not None, "frequency must be set"
        return 1j * 2 * np.pi * self.frequency

    def _frequency_shape_adaptor(self, param: AnyParamValue|AnyValueArray) -> AnyValueArray:
        """convert param to array of frequencies shape

        - param can be port_z0, any param value

        Args:
            param (AnyParamValue): param value

        Returns:
            AnyValueArray: param value as array
        """
        if self.frequency is None:
            return np.array(param)
        if np.array(param).size == 1:
            return np.ones_like(self.frequency) * param
        assert np.array(param).size == self.frequency.size, "param size must be 1 or equal to frequencies size"
        return np.array(param)

    def as_f(self, param: AnyParamValue|AnyValueArray) -> AnyValueArray:
        """alias of _frequency_shape_adaptor
        """
        return self._frequency_shape_adaptor(param)

    def set_f(self, frequency: AnyParamValue):
        """set and broadcast frequency to the element tree"""
        if self.root_element is self:
            if not isinstance(frequency, np.ndarray):
                frequency = np.array(frequency)
            self.frequency = frequency
        else:
            assert self.parent is not None, "None-root element must have parent element"
            if frequency is not self.parent.frequency:
                # always update frequency from the root element
                self.root_element.set_f(frequency)
            elif self.frequency is not self.parent.frequency:
                # sync frequency with parent element
                self.frequency = self.parent.frequency
            else:
                return
        # broadcast frequency to the children
        for element in self.children:
            element.set_f(frequency)
        return

    s: ComplexArray|None = Field(default=None, exclude=True)
    """s param cache of the element, default None

    Reset mechanism:
        - s cache watches changes of `params`, `frequency`, `z0`
        - changes of watched fields and s itself will trigger reset_s
        - reset_s will reset s cache of all the elements above in the element tree
    """

    def reset_s(self, include_self:bool = True):
        """reset s param cache and all the element above in the tree"""
        if include_self:
            self.s = None
        if self.parent is not None:
            self.parent.reset_s()
        return

    def __setattr__(self, name, value):
        """
        Customization:
            - watch changes of `params`, `frequency`, `z0`, `s` and trigger `reset_s`
        """
        match name:
            case "frequency":
                super().__setattr__(name, value)
                self.reset_s()
            case "z0":
                self.set_z0(value)
                self.reset_s()
            case "params":
                super().__setattr__(name, self.Params.parse_obj(value))
                self.reset_s()
            case "s":
                super().__setattr__(name, value)
                self.reset_s(include_self=False)
            case _:
                super().__setattr__(name, value)
    
    def get_s(self, z0: Optional[list[float]] = None) -> ComplexArray:
        if self.circuit is None:
            raise NotImplementedError("basic circuit element doesn't support s parameter analysis")
        if self.s is not None:
            return self.s
        # TODO: implement s params solver options
        from .sparam import solve_circuit_sparams
        assert self.circuit_ports is not None, "circuit_ports must be set"
        if z0 is None:
            z0 = self.z0
        s = solve_circuit_sparams(self.circuit, self.circuit_ports, z0)
        self.s = s
        return s
    
    def get_s_component(self, _from:str, _to:str) -> ComplexArray:
        """Get s param component from _from to _to

        Args:
            _from (str): from port name
            _to (str): to port name

        Returns:
            s param component from _from to _to
        """
        return self.get_s()[:, self.ports.index(_to), self.ports.index(_from)].flatten()

    @property
    def ntwk(self):
        """return skrf network object of the element"""
        import skrf as rf
        return rf.Network(
            name = self.name,
            frequency = rf.Frequency.from_f(self.frequency, unit="Hz"),
            s = self.get_s(),
            z0 = self.z0
        )


LinkStr = str
"""
LinkStr format:
    - `element_name::portname node_name`
    - `element_name *node_name`
"""

@dataclass
class Link:
    element: Element
    port: str
    node: Node

    @property
    def element_port_num(self) -> int:
        assert isinstance(self.element.ports, list), "Ports must be a list"
        return self.element.ports.index(self.port)

    def __str__(self):
        return f"{self.element.name}::{self.port} {self.node.name}"

    def __repr__(self):
        return self.__str__()

    def connect(self):
        """Connect element and node, update fields respectively"""
        self.element.nodes[self.element_port_num] = self.node
        node_elements = [element.name for element in self.node.elements]
        if self.element.name not in node_elements:
            self.node.elements.append(self.element)

def _parse_link_str_1(link:LinkStr, elements: dict[str, Element], nodes: dict[str, Node]) -> list[Link]:
    try:
        element_port, node_name = link.split(" ")
    except ValueError as e:
        raise ValueError(
            "Invalid LinkStr format. \
            It should be either be `element_name::portname node_name` or `element_name *node_name`"
        ) from e
    element, port = element_port.split("::")
    return [Link(
        element=elements[element],
        port=port,
        node=nodes[node_name]
    )]

def _parse_link_str_2(link:LinkStr, elements: dict[str, Element], nodes: dict[str, Node]) -> list[Link]:
    element_name, *node_names = link.split(" ")
    element = elements[element_name]
    assert isinstance(element.ports, list), "Ports must be a list"
    assert len(node_names) == element.num_of_ports, "Number of nodes must be equal to num_of_ports"
    return [
        Link(
            element=element,
            port=element.ports[i],
            node=nodes[node_name]
        )
        for i, node_name in enumerate(node_names)
    ]

def parse_link_str(link:LinkStr, elements: dict[str, Element], nodes: dict[str, Node]) -> list[Link]:
    """parse link string to list of Link objects
    
    - link string format:
        - `element_name::portname node_name`
        - `element_name *node_name`

    Args:
        link (LinkStr): link string
        elements (dict[str, Element]): Elements by name
        nodes (dict[str, Node]): Nodes by name

    Returns:
        list[Link]: List of Link objects
    """
    if "::" in link:
        return _parse_link_str_1(link, elements, nodes)
    return _parse_link_str_2(link, elements, nodes)

def validate_link(link:Link, elements: dict[str, Element], nodes: dict[str, Node]) -> Link:
    """check link.element and link.node are in elements and nodes respectively"""
    assert link.element in elements.values(), "Element not found in elements"
    assert link.node in nodes.values(), "Node not found in nodes"
    return link

def create_nodes_from_links(links:list[LinkStr], nodes:Optional[list[Node]] = None) -> list[Node]:
    """parse link string and create nodes

    Args:
        links (list[LinkStr]): link list

    Returns:
        list[Node]: all nodes in the link list
    """ 
    if nodes is None:
        nodes = []

    def get_nodenames():
        return [node.name for node in nodes]

    for link in links:
        _, *node_names = link.split(" ")
        for node_name in node_names:
            if node_name not in get_nodenames():
                nodes.append(Node(name=node_name))
    return nodes
        
class Circuit(BaseModel):
    elements: list[Element]
    nodes: list[Node] = Field(default_factory=lambda:[])
    links: list[str]

    @root_validator(pre=True)
    def validate_nodes_from_links(cls, values: dict) -> dict:
        """create nodes from links if not provided

        Args:
            v (list[Node]): input nodes
            values (dict): contains links

        Returns:
            list[Node]: all nodes in the link list
        """
        nodes = [Node.parse_obj(node) for node in values["nodes"]] if "nodes" in values else None
        values["nodes"] = create_nodes_from_links(values["links"], nodes)
        return values

    @validator("elements", "nodes")
    def validate_names(cls, v:list[Element|Node], field:str) -> list[Element|Node]:
        names = [_.name for _ in v]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate {field} names found")
        return v

    @validator("elements", pre=True)
    def load_element_libs(cls, v:list[Element]) -> list[Element]:
        """convert element to corresponding Element subclass obj based on libname

        - Element subclasses info is managed in `LibManager`

        Args:
            v (list[Element]): element list

        Returns:
            list[Element]: Element list
        """
        elements = [parse_element(element) for element in v]
        return elements

    @validator("links")
    def validate_links(cls, v:list[str], values: dict) -> list[str]:
        """parse link string to Link Object and validate Link

        Args:
            v (list[str]): link list
            values (dict): contains set elements and nodes

        Returns:
            list[Link]: Link list
        """
        elements:dict[str, Element] = {element.name: element for element in values["elements"]}
        nodes: dict[str, Node] = {node.name: node for node in values["nodes"]}

        _links = []
        for link in v:
            _links.extend(parse_link_str(link, elements, nodes))
        for link in _links:
            validate_link(link, elements, nodes)
        return v

    def parse_links(self) -> list[Link]:
        elements:dict[str, Element] = {element.name: element for element in self.elements}
        nodes: dict[str, Node] = {node.name: node for node in self.nodes}

        _links = []
        for link in self.links:
            _links.extend(parse_link_str(link, elements, nodes))
        return _links

    def get_element(self, name:str) -> Element:
        for element in self.elements:
            if element.name == name:
                return element
        raise ValueError(f"Element {name} not found")

    def get_node(self, name:str) -> Node:
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError(f"Node {name} not found")

    def connect(self):
        """Connect elements and nodes, update fields respectively"""
        for link in self.parse_links():
            link.connect()
        self.check_connectivity()
        for element in self.elements:
            element.connect()

    def check_connectivity(self):
        """Check if all nodes are connected"""
        for node in self.nodes:
            if len(node.elements) == 0:
                raise ValueError(f"Node {node.name} is not connected")

    def set_f(self, frequency: AnyParamValue):
        """broadcast frequency to the elements"""
        if not isinstance(frequency, np.ndarray):
            frequency = np.array(frequency)
        for element in self.elements:
            element.set_f(frequency)
        return

        
Element.update_forward_refs()
from .lib.utils import parse_element, load_element_json







