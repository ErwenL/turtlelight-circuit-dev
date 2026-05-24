from __future__ import annotations
from typing import Optional, Any, Sequence, Union
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator

StageNode = Union["Stage", "NtwkNode"]

class NtwkNode(BaseModel):
    num_of_ports: int
    name: Optional[str] = None
    ports: Optional[list[PortName]] = None
    wavelength: np.ndarray
    s: Optional[np.ndarray] = None
    parent: Optional[StageNode] = Field(default=None, exclude=True)
    children: list[StageNode] = []
    connection: Optional[NodeConnection] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        return f"{self.name} <NtwkNode at {hex(id(self))}>"

    @validator("name", pre=True, always=True)
    def validate_name(cls, v, values):
        if v is None:
            return cls.default_name(values["num_of_ports"])
        assert isinstance(v, str), "The name must be a string."
        return v

    @validator("ports")
    def validate_ports(cls, v, values):
        if isinstance(v, list):
            assert len(v) == values["num_of_ports"], "The number of port names must be equal to the number of ports."
            return v
        if v is None:
            return [(values["name"], i) for i in range(values["num_of_ports"])]
        raise ValueError("The port names must be a PortName list or None.")

    @staticmethod
    def default_name(num_of_ports: int) -> str:
        assert num_of_ports > 0, "The number of ports must be greater than 0."
        if num_of_ports == 1:
            return "1-port Network"
        return f"{num_of_ports}-ports Network"
    
    @property
    def ports_dict(self) -> dict[str|int, PortName]:
        _dict = {}
        assert isinstance(self.ports, list), "The port names must be a list."
        for port in self.ports:
            _dict[port.port] = port
        return _dict

    def connect_to(self, next: NtwkNode, links: list[AnyNodeLink]) -> NtwkNode:
        return connect_ntwks(self, next, links)

    def sub_network(self, ports: SubNwtkPortsDef, **kwargs) -> NtwkNode:
        return sub_ntwk(self, ports, **kwargs)

    def get_s(self) -> np.ndarray:
        """get the scattering matrix of the node"""
        if self.s is not None:
            return self.s
        if self.connection is not None:
            self.s = self.connection.get_s()
            return self.s
        raise ValueError("The scattering matrix of the node must be defined.")

    def set_s(self, s: np.ndarray|None) -> None:
        if s is not None:
            assert s.shape == (len(self.wavelength), self.num_of_ports, self.num_of_ports), "The shape of the scattering matrix must be (len(wavelength), num_of_ports, num_of_ports)."
        self.s = s
        if self.parent is not None:
            if isinstance(self.parent, NtwkNode):
                self.parent.set_s(None)
            else:
                self.parent.set_matrix(None)
    
    def get_matrix(self) -> np.ndarray:
        """alias of get_s"""
        return self.get_s()

    def set_matrix(self, s: np.ndarray|None) -> None:
        """alias of set_s"""
        self.set_s(s)

    @staticmethod
    def empty(
        num_of_ports: int,
        wavelength: np.ndarray,
        name: Optional[str] = None,
        ports: Optional[Sequence[AnyPortName]] = None,
    ) -> NtwkNode:
        return NtwkNode(
            num_of_ports=num_of_ports,
            wavelength=wavelength,
            name=name,
            ports=ports, # type: ignore
        )


from .connection.port import PortName, AnyPortName, AnyNodeLink
from .connection.connection import NtwkConnection, connect_ntwks
from .connection.subntwk import SubNtwk, sub_ntwk, SubNwtkPortsDef
from .connection.from_stage import FromTransferMatrixStage
NodeConnection = Union[NtwkConnection, SubNtwk, FromTransferMatrixStage]
from ..transfer_matrix.stage import Stage
NtwkNode.update_forward_refs()