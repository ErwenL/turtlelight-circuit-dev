from __future__ import annotations
import numpy as np
from typing import Literal
from pydantic import BaseModel

class FromNtwkNodeConnect(BaseModel):
    type: Literal["from_ntwk"] = "from_ntwk"
    super: NtwkNode
    input_ports: list[PortName]
    output_ports: list[PortName]

    class Config:
        arbitrary_types_allowed = True

    def get_matrix(self) -> np.ndarray:
        assert isinstance(self.super.ports, list), "The ports of the super stage must be a list."
        _input_s_ports = [self.super.ports.index(port) for port in self.input_ports]
        _output_s_ports = [self.super.ports.index(port) for port in self.output_ports]
        return self.super.get_s()[:, _output_s_ports, :][:, :, _input_s_ports]

def from_ntwk_node_connect(super: NtwkNode, input_ports: SubNwtkPortsDef, output_ports: SubNwtkPortsDef, **kwargs) -> Stage:
    connection = FromNtwkNodeConnect(
        super=super, 
        input_ports=validate_subntwk_ports(super, input_ports),
        output_ports=validate_subntwk_ports(super, output_ports),
    )
    new = Stage(
        name=kwargs.pop("name", f"{len(input_ports)}x{len(output_ports)} Stage-from-Ntwk"),
        wavelength=super.wavelength,
        num_of_input_ports=len(input_ports),
        num_of_output_ports=len(output_ports),
        # connection=connection,
    )
    # assign connection in new Stage may cause recursive error in pydantic
    # maybe a bug in pydantic
    new.connection = connection
    super.parent = new
    new.children = [super]
    return new



from ..stage import Stage
from ...network.ntwk import NtwkNode
from ...network.connection.port import PortName
from ...network.connection.subntwk import validate_subntwk_ports, SubNwtkPortsDef
FromNtwkNodeConnect.update_forward_refs()