from __future__ import annotations
from typing import Optional, Any, Union
import pandas as pd
import numpy as np
import pandera as pa
from pandera.typing import DataFrame, Series
from pydantic import BaseModel, Field, validator, root_validator
from .dataframe import TransferMatrixSegma, to_matrix

StageNode = Union["Stage", "NtwkNode"]

class Stage(BaseModel):
    num_of_input_ports: int
    num_of_output_ports: int
    name: Optional[str] = None
    wavelength: np.ndarray
    matrix: Optional[np.ndarray] = None
    parent: Optional[StageNode] = Field(default=None, exclude=True)
    children: list[StageNode] = []
    connection: Optional[StageConnection] = Field(default=None, exclude=True, )

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return f"{self.name} <Stage at {hex(id(self))}>"

    @validator("name", always=True)
    def default_name(cls, v, values):
        if v is None:
            return f"{values['num_of_input_port']}x{values['num_of_output_port']} Stage"
        return v

    @validator("matrix", always=True)
    def validate_matrix_shape(cls, v: np.ndarray|None, values):
        if v is not None:
            assert v.shape == (len(values["wavelength"]), values["num_of_output_port"], values["num_of_input_port"]), "The shape of the matrix must be (len(wavelength), num_of_output_port, num_of_input_port)."
        return v


    def __pow__(self, _o) -> "Stage":
        """cascade two TransferStage objects"""
        validate_transfer_stage(_o)
        return cascade_connect(self, _o)

    def __floordiv__(self, _o) -> "Stage":
        """parallel two TransferStage objects"""
        validate_transfer_stage(_o)
        return parallel_connect(self, _o)

    def sub_stage(self, input_ports: list[int], output_ports: list[int]) -> "Stage":
        """get a nxm substage from current stage (NxM) """
        return sub_connect(self, input_ports, output_ports)

    def get_matrix(self) -> np.ndarray:
        """get the transfer matrix of the stage"""
        if self.matrix is not None:
            return self.matrix
        if self.connection is not None:
            self.matrix = self.connection.get_matrix()
            return self.matrix
        raise ValueError("The matrix of the stage must be defined.")

    def set_matrix(self, matrix: np.ndarray|None) -> None:
        if matrix is not None:
            assert matrix.shape == (len(self.wavelength), self.num_of_output_ports, self.num_of_input_ports), "The shape of the matrix must be (len(wavelength), num_of_output_port, num_of_input_port)."
        self.matrix = matrix
        if self.parent is not None:
            if isinstance(self.parent, Stage):
                self.parent.set_matrix(None)
            else:
                self.parent.set_s(None)
        return

    @staticmethod
    def empty(
            num_of_input_port: int, 
            num_of_output_port: int, 
            wavelength: np.ndarray,
            name: Optional[str] = None,
        ) -> "Stage":
        return Stage(
            num_of_input_ports=num_of_input_port,
            num_of_output_ports=num_of_output_port,
            wavelength=wavelength,
            name=name,
        )

    def to_dataframe(self) -> pd.DataFrame:
        return to_dataframe(self)

def validate_transfer_stage(_o: Any):
    assert isinstance(_o, Stage), "The object must be a TransferStage object."

@pa.check_input(TransferMatrixSegma)
def to_stage(df: pd.DataFrame) -> Stage:
    matrix = to_matrix(df)
    return Stage(
        wavelength=df["wavelength"].to_numpy(),
        num_of_input_ports=matrix.shape[2],
        num_of_output_ports=matrix.shape[1],
        matrix=matrix
    )

@pa.check_output(TransferMatrixSegma)
def to_dataframe(stage: Stage) -> pd.DataFrame:
    df = pd.DataFrame()
    df["wavelength"] = stage.wavelength
    assert stage.matrix is not None, "The matrix of the stage must be defined."
    for to_port in range(stage.num_of_output_ports):
        for from_port in range(stage.num_of_input_ports):
            df[f"t_{to_port + 1}_{from_port + 1}"] = stage.matrix[:, to_port, from_port]
    return df

# from .connection import StageConnection, cascade_connect, parallel_connect, sub_connect
from .connection.cascade import CascadeConnect, cascade_connect
from .connection.parallel import ParallelConnect, parallel_connect
from .connection.substage import SubConnect, sub_connect
from .connection.from_ntwk import FromNtwkNodeConnect
StageConnection = Union[CascadeConnect, ParallelConnect, SubConnect, FromNtwkNodeConnect]
from ..network.ntwk import NtwkNode
Stage.update_forward_refs()