from __future__ import annotations
import numpy as np
from typing import Any, Literal, Sequence
from pydantic import BaseModel, validator
from .port import PortName

class FromTransferMatrixStage(BaseModel):
    type: Literal["from_stage"] = "from_stage"
    super: Stage
    ports: list[PortName]

    class Config:
        arbitrary_types_allowed = True

    @validator("ports", always=True)
    def validate_ports(cls, v: list[PortName], values: dict[str, Any]) -> list[PortName]:
        for port in v:
            assert port.node in ["input", "output"], "The node name must be 'input' or 'output'."
            assert isinstance(port.port, int), "The port name must be an integer."
        max_input_index = max(v.index(port) for port in v if port.node == "input")
        min_output_index = min(v.index(port) for port in v if port.node == "output")
        assert max_input_index < min_output_index, "The input ports must be before the output ports."
        return v

    @property
    def input_s_ports(self) -> list[int]:
        return [port.port for port in self.ports if port.node == "input"] # type: ignore
    
    @property
    def output_s_ports(self) -> list[int]:
        return [port.port for port in self.ports if port.node == "output"] # type: ignore

    def get_s(self) -> np.ndarray:
        matrix = self.super.get_matrix()[:, self.output_s_ports, :][: ,:, self.input_s_ports]
        return np.block([
            [np.zeros((matrix.shape[0], matrix.shape[2], matrix.shape[2]), dtype=complex), matrix.transpose(0, 2, 1)],
            [matrix, np.zeros((matrix.shape[0], matrix.shape[1], matrix.shape[1]), dtype=complex)]
        ])

FromStagePortsDef = Sequence[int]|dict[str, int]

def from_transfer_matrix_stage(super: Stage, input_ports: FromStagePortsDef, output_ports: FromStagePortsDef, **kwargs) -> NtwkNode:
    connection = FromTransferMatrixStage(
        super=super, 
        ports=validate_from_stage_ports(input_ports, output_ports),
    )
    _name = kwargs.pop("name", f"{len(input_ports)+len(output_ports)}-ports Ntwk-from-Stage")
    assert isinstance(_name, str), "The name must be a string."
    if isinstance(input_ports, dict) and isinstance(output_ports, dict):
        _ports = [PortName.validate((_name, port)) for port in input_ports|output_ports]
    else:
        _ports = None

    new = NtwkNode(
        num_of_ports=len(input_ports)+len(output_ports),
        name=_name,
        wavelength=super.wavelength,
        connection=connection,
        ports=_ports,
    )
    super.parent = new
    new.children = [super]
    return new

def validate_from_stage_ports(input_ports: FromStagePortsDef, output_ports: FromStagePortsDef) -> list[PortName]:
    if isinstance(input_ports, dict):
        input_ports = list(input_ports.values())
    if isinstance(output_ports, dict):
        output_ports = list(output_ports.values())
    _input_ports = [PortName(node="input", port=port) for port in input_ports]
    _output_ports = [PortName(node="output", port=port) for port in output_ports]
    return _input_ports + _output_ports

def validate_from_stage_ports_def(ports: Any) -> FromStagePortsDef:
    if isinstance(ports, Sequence):
        assert all(isinstance(port, int) for port in ports), "The port names must be integers."
        return ports
    if isinstance(ports, dict):
        assert all(isinstance(port, int) for port in ports.values()), "The port names must be integers."
        return ports
    raise ValueError("The ports must be a dictionary or a sequence of integers.")


from ..ntwk import NtwkNode
from ...transfer_matrix.stage import Stage
FromTransferMatrixStage.update_forward_refs()