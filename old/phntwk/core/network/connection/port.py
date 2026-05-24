from __future__ import annotations
from typing import Any, Sequence, Union
from pydantic import BaseModel

class PortName(BaseModel):
    node: str
    port: int|str

    def __str__(self) -> str:
        return f"{self.node}::{self.port}"

    def __hash__(self):
        return hash((self.node, self.port))

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: AnyPortName|Any) -> PortName:
        if isinstance(v, PortName):
            return v
        if isinstance(v, str):
            node, port = cls.validate_str(v)
            return cls(node=node, port=port)
        if isinstance(v, Sequence):
            node, port = cls.validate_sequence(v)
            return cls(node=node, port=port)
        raise ValueError("The port name must be a string or sequence.")

    @staticmethod
    def validate_sequence(v: Any):
        assert isinstance(v, tuple|list), "The port name must be a tuple or list."
        assert len(v) == 2, "The port name must be a sequence of length 2."
        assert isinstance(v[0], str), "The port name must be a tuple of (str, int|str)."
        assert isinstance(v[1], int|str), "The port name must be a tuple of (str, int|str)."
        return tuple(v)

    @staticmethod
    def validate_str(_name: str) -> tuple[str, int|str]:
        port_name = _name.split("::")
        assert len(port_name) == 2, "The port name must be a string of the form 'node_name::port_name'."
        node, port = port_name
        if port.isnumeric():
            port = int(port)
        return (node, port)

PortNameTuple = tuple[str, int|str]
AnyPortName = Union[PortName, str, PortNameTuple]
NodeLink = tuple[PortName, PortName]
AnyNodeLink = tuple[AnyPortName, AnyPortName]

def valiate_node_link(link: AnyNodeLink) -> NodeLink:
    assert isinstance(link, tuple), "The node link must be a tuple."
    assert len(link) == 2, "The node link must be a tuple of length 2."
    return PortName.validate(link[0]), PortName.validate(link[1])
