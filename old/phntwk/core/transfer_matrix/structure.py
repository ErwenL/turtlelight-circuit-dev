from __future__ import annotations

from .stage import Stage
from .connection import StageConnectionType
from typing import Any, Optional, Literal, TypeVar, Union, Sequence, get_args
import numpy as np
from pydantic import BaseModel

EmptyStage = tuple[int, int, str]

def validate_empty_stage(empty_stage: EmptyStage):
    assert isinstance(empty_stage, Sequence), "The empty stage must be a Sequence."
    assert isinstance(empty_stage[0], int), "first element: num of input port must be an integer."
    assert isinstance(empty_stage[1], int), "second element: num of output port must be an integer."
    assert isinstance(empty_stage[2], str), "third element: stage name must be a string."

class StageStructure(BaseModel):
    """example
    ```python
    mzi_through = StageStructure(
        name="mzi_through",
        type="substage",
        components=[
            StageStructure(
                name="mzi",
                type="cascade",
                components=[
                    (2, 2, "dc"),
                    StageStructure(
                        name="delaypair",
                        type="parallel",
                        components=[
                            (1, 1, "delay"),
                            (1, 1, "delay"),
                        ]
                    ),
                    (2, 2, "dc"),
                ]
            )
        ],
        inherit_ports=([0], [0])
    )

    mzi_through.dict() =>
    {
        "name": "mzi_through",
        "type": "substage",
        "components": [
            {
                "name": "mzi",
                "type": "cascade",
                "components": [
                    (2, 2, "dc"),
                    {
                        "name": "delaypair",
                        "type": "parallel",
                        "components": [
                            (1, 1, "delay"),
                            (1, 1, "delay"),
                        ]
                    },
                    (2, 2, "dc"),
                ]
            }
        ],
        "inherit_ports": ([0], [0])
    }

    ```
    """
    name: str
    type: StageConnectionType
    components: list[AnyStructure]
    inherit_ports: Optional[SubStageInheritPortsDef|FromNtwkInheritPortsDef] = None

    class Config:
        arbitrary_types_allowed = True

    def rename(self, name: str) -> "StageStructure":
        """return a new StageStructure with the same structure but different name"""
        return self.copy(update={"name": name})

    def build(self, wavelength: np.ndarray) -> Stage:
        """build the stage from the structure"""
        return build_from_structure(self, wavelength)

def build_from_structure(structure: StageStructure, wavelength: np.ndarray) -> Stage:
    # from ntwk
    if structure.type == "from_ntwk":
        from .connection.from_ntwk import from_ntwk_node_connect
        from ..network.structure import component_2_ntwk, validate_any_ntwk_structure
        node = component_2_ntwk(validate_any_ntwk_structure(structure.components[0]), wavelength)
        assert isinstance(structure.inherit_ports, tuple|list), "The inherit_ports must be a tuple or list."
        return from_ntwk_node_connect(node, structure.inherit_ports[0], structure.inherit_ports[1], name=structure.name)

    # from stage structure
    stage = component_2_stage(validate_any_stage_structure(structure.components[0]), wavelength)
    if structure.type == "cascade":
        for component in structure.components[1:]:
            stage = stage ** component_2_stage(validate_any_stage_structure(component), wavelength)
    elif structure.type == "parallel":
        for component in structure.components[1:]:
            stage = stage // component_2_stage(validate_any_stage_structure(component), wavelength)
    elif structure.type == "substage":
        assert isinstance(structure.inherit_ports, tuple|list), "The inherit_ports must be a tuple or list."
        assert all(isinstance(port, int) for ports in structure.inherit_ports for port in ports), "The input ports must be a list of integers."
        stage = stage.sub_stage(*structure.inherit_ports) # type: ignore
    else:
        raise ValueError(f"The type of the structure must be in {get_args(StageConnectionType)}.")
    stage.name = structure.name
    return stage

def component_2_stage(component: AnyStageStructure, wavelength: np.ndarray) -> Stage:
    if isinstance(component, StageStructure):
        return build_from_structure(component, wavelength)
    validate_empty_stage(component)
    return Stage.empty(
        num_of_input_port=component[0],
        num_of_output_port=component[1],
        wavelength=wavelength,
        name=component[2],
    )

def validate_any_stage_structure(structure: Any) -> AnyStageStructure:
    if isinstance(structure, StageStructure):
        return structure
    validate_empty_stage(structure)
    return structure

AnyStageStructure = Union[StageStructure, EmptyStage]
EmptyNode = tuple[int, str, Optional[Sequence[str|int]]]
AnyStructure = Union[AnyStageStructure, TypeVar("NtwkStructure", bound=BaseModel), EmptyNode]
SubStageInheritPortsDef = tuple[Sequence[int], Sequence[int]]
FromNtwkInheritPortsDef = tuple["SubNwtkPortsDef", "SubNwtkPortsDef"]
from ..network.connection.subntwk import SubNwtkPortsDef
StageStructure.update_forward_refs()

