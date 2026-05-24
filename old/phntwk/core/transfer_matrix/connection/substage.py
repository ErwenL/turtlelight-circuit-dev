from __future__ import annotations
import numpy as np
from typing import Literal
from pydantic import BaseModel

class SubConnect(BaseModel):
    type: Literal["substage"] = "substage"
    super: Stage
    input_ports: list[int]
    output_ports: list[int]

    class Config:
        arbitrary_types_allowed = True

    def get_matrix(self) -> np.ndarray:
        return self.super.get_matrix()[ :,self.output_ports,: ][ :,:,self.input_ports]

def sub_connect(super: Stage, input_ports: list[int], output_ports: list[int]) -> Stage:
    validate_sub_connection(super, input_ports, output_ports)
    connection = SubConnect(
        super=super, 
        input_ports=input_ports,
        output_ports=output_ports,
    )
    new = Stage(
        name=f"{len(input_ports)}x{len(output_ports)} Substage",
        wavelength=super.wavelength,
        num_of_input_ports=len(input_ports),
        num_of_output_ports=len(output_ports),
        connection=connection,
    )
    super.parent = new
    new.children = [super]
    return new


def validate_sub_connection(super: Stage, input_ports: list[int], output_ports: list[int]):
    assert list(set(input_ports)) == input_ports, "The input ports must be unique."
    assert list(set(output_ports)) == output_ports, "The output ports must be unique."
    assert all(port < super.num_of_input_ports for port in input_ports), "The input ports must be less than the number of input ports of the super stage."
    assert all(port < super.num_of_output_ports for port in output_ports), "The output ports must be less than the number of output ports of the super stage."


from ..stage import Stage
SubConnect.update_forward_refs()