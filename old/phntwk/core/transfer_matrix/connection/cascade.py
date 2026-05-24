from __future__ import annotations
import numpy as np
from typing import Literal
from pydantic import BaseModel

class CascadeConnect(BaseModel):
    type: Literal["cascade"] = "cascade"
    previous: Stage
    next: Stage

    class Config:
        arbitrary_types_allowed = True

    def get_matrix(self) -> np.ndarray:
        return self.next.get_matrix() @ self.previous.get_matrix()

def cascade_connect(previous: Stage, next: Stage) -> Stage:
    validate_cascade_connection(previous, next)
    connection = CascadeConnect(
        previous=previous, 
        next=next
    )
    new = Stage(
        name=f"{previous.num_of_input_ports}x{next.num_of_output_ports} Cascade",
        wavelength=previous.wavelength,
        num_of_input_ports=previous.num_of_input_ports,
        num_of_output_ports=next.num_of_output_ports,
        connection=connection,
    )
    previous.parent = new
    next.parent = new
    new.children = [previous, next]
    return new

def validate_cascade_connection(previous: Stage, next: Stage):
    assert previous.num_of_output_ports == next.num_of_input_ports, "The number of output ports of the previous stage must be equal to the number of input ports of the next stage."
    if previous.wavelength is not next.wavelength:
        assert np.all(previous.wavelength == next.wavelength), "The wavelength of the previous stage must be equal to the wavelength of the next stage."



from ..stage import Stage
CascadeConnect.update_forward_refs()