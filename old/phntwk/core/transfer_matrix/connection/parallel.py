from __future__ import annotations
import numpy as np
from typing import Literal
from pydantic import BaseModel

class ParallelConnect(BaseModel):
    type: Literal["parallel"] = "parallel"
    previous: Stage
    next: Stage

    class Config:
        arbitrary_types_allowed = True

    def get_matrix(self) -> np.ndarray:
        num_of_wavelength = len(self.previous.wavelength)
        upper_right = np.zeros((num_of_wavelength, self.previous.num_of_output_ports, self.next.num_of_input_ports), dtype=complex)
        lower_left = np.zeros((num_of_wavelength, self.next.num_of_output_ports, self.previous.num_of_input_ports), dtype=complex)
        return np.block([
            [self.previous.get_matrix(), upper_right],
            [lower_left, self.next.get_matrix()]
        ])

def parallel_connect(previous: Stage, next: Stage) -> Stage:
    validate_parallel_connection(previous, next)
    connection = ParallelConnect(
        previous=previous, 
        next=next
    )
    num_of_input_port = previous.num_of_input_ports + next.num_of_input_ports
    num_of_output_port = previous.num_of_output_ports + next.num_of_output_ports
    new = Stage(
        name=f"{num_of_input_port}x{num_of_output_port} Parallel",
        wavelength=previous.wavelength,
        num_of_input_ports=num_of_input_port,
        num_of_output_ports=num_of_output_port,
        connection=connection,
    )
    previous.parent = new
    next.parent = new
    new.children = [previous, next]
    return new

def validate_parallel_connection(previous: Stage, next: Stage):
    if previous.wavelength is not next.wavelength:
        assert np.all(previous.wavelength == next.wavelength), "The wavelength of the previous stage must be equal to the wavelength of the next stage."


from ..stage import Stage
ParallelConnect.update_forward_refs()