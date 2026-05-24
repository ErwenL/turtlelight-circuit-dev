from __future__ import annotations
from .ntwk import NtwkNode
from .connection import (
    PortName, AnyPortName,
    NodeConnectionType, AnyNodeLink, SubNwtkPortsDef
)
from typing import Optional, Literal, Union, Sequence, Any, TypeVar
import numpy as np
from pydantic import BaseModel

EmptyNodeWithPortNames = tuple[int, str, Optional[Sequence[AnyPortName]]]
EmptyNode = Union[tuple[int, str, Optional[Sequence[str|int]]], EmptyNodeWithPortNames]

def validate_empty_node(empty_node: EmptyNode):
    assert isinstance(empty_node, Sequence), "The empty node must be a Sequence."
    assert isinstance(empty_node[0], int), "first element: num of port must be an integer."
    assert isinstance(empty_node[1], str), "second element: node name must be a string."
    if len(empty_node) > 2 and empty_node[2] is not None:
        assert isinstance(empty_node[2], Sequence), "third element: port names must be a Sequence."
        try:
            [PortName.validate(port) for port in empty_node[2]]
        except Exception:
            assert all(isinstance(port, str|int) for port in empty_node[2]), "third element: port names must be a list of strings or integers."

def convert_empty_node_2_with_port_names(empty_node: EmptyNode) -> EmptyNodeWithPortNames:
    if len(empty_node) < 3:
        return empty_node[0], empty_node[1], None
    assert isinstance(empty_node[2], Sequence), "The port names must be a Sequence."
    try:
        port_names = [PortName.validate(port) for port in empty_node[2]]
    except Exception:
        port_names = [PortName.validate((empty_node[1], port)) for port in empty_node[2]]
    return empty_node[0], empty_node[1], port_names
    
class NtwkStructure(BaseModel):
    name: str
    type: NodeConnectionType
    components: list[AnyStructure]
    links: list[AnyNodeLink] = []
    inherit_ports: Optional[FromStageInheritPortsDef|SubNwtkPortsDef] = None

    class Config:
        arbitrary_types_allowed = True
    
    def rename(self, name: str) -> "NtwkStructure":
        return self.copy(update={"name": name})

    def build(self, wavelength: np.ndarray) -> NtwkNode:
        return build_from_structure(self, wavelength)

def build_from_structure(structure: NtwkStructure, wavelength: np.ndarray) -> NtwkNode:
    # from stage
    if structure.type == "from_stage":
        from .connection.from_stage import from_transfer_matrix_stage, validate_from_stage_ports_def
        from ..transfer_matrix.structure import component_2_stage, validate_any_stage_structure
        stage = component_2_stage(validate_any_stage_structure(structure.components[0]), wavelength)
        assert isinstance(structure.inherit_ports, tuple), "The inherit_ports must be a tuple with length of 2."
        return from_transfer_matrix_stage(
            stage, 
            validate_from_stage_ports_def(structure.inherit_ports[0]), 
            validate_from_stage_ports_def(structure.inherit_ports[1]), 
            name=structure.name
        )

    # from ntwk structure
    node = component_2_ntwk(validate_any_ntwk_structure(structure.components[0]), wavelength)
    if structure.type == "ntwk":
        for next_node in structure.components[1:]:
            node = node.connect_to(
                component_2_ntwk( validate_any_ntwk_structure(next_node), wavelength), 
                structure.links
            )
        node.name = structure.name
    elif structure.type == "subntwk":
        assert isinstance(structure.inherit_ports, list|dict), "The inherit_ports must be a list or a dict."
        node = node.sub_network(structure.inherit_ports, name=structure.name)
    else:
        raise ValueError(f"Unknown network type: {structure.type}")
    return node

def component_2_ntwk(node: AnyNtwkStructure, wavelength: np.ndarray) -> NtwkNode:
    if isinstance(node, NtwkStructure):
        return build_from_structure(node, wavelength)
    # assume node has been validated as empty node before
    _node = convert_empty_node_2_with_port_names(node)
    return NtwkNode.empty(
        num_of_ports=node[0],
        wavelength=wavelength,
        name=node[1],
        ports=_node[2]
    )

def validate_any_ntwk_structure(structure: Any) -> AnyNtwkStructure:
    if isinstance(structure, NtwkStructure):
        return structure
    validate_empty_node(structure)
    return structure

AnyNtwkStructure = Union[NtwkStructure, EmptyNode]
EmptyStage = tuple[int, int, str]
AnyStructure = Union[AnyNtwkStructure, TypeVar("StageStructure", bound=BaseModel), EmptyStage]
FromStageInheritPortsDef = tuple["FromStagePortsDef", "FromStagePortsDef"]

from .connection.from_stage import FromStagePortsDef
NtwkStructure.update_forward_refs()