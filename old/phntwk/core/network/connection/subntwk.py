from __future__ import annotations
import numpy as np
from typing import Any, Literal, Sequence
from pydantic import BaseModel
from .port import PortName, AnyPortName

class SubNtwk(BaseModel):
    type: Literal["subntwk"] = "subntwk"
    super: NtwkNode
    ports: list[PortName]

    class Config:
        arbitrary_types_allowed = True

    def get_s(self) -> np.ndarray:
        assert isinstance(self.super.ports, list), "The port names of the super node must be a list."
        s_ports = [self.super.ports.index(port) for port in self.ports]
        return self.super.get_s()[ :,s_ports,: ][ :,:,s_ports]

SubNwtkPortsDef = Sequence[AnyPortName|int]|dict[str, AnyPortName|int]

def sub_ntwk(super: NtwkNode, ports: SubNwtkPortsDef, **kwargs) -> NtwkNode:
    connection = SubNtwk(
        super=super, 
        ports=validate_subntwk_ports(super, ports),
    )

    _name = kwargs.pop("name", f"{len(ports)}-ports Subnetwork")
    assert isinstance(_name, str), "The name must be a string."
    _ports = [PortName.validate((_name, port)) for port in ports] if isinstance(ports, dict) else None

    new = NtwkNode(
        num_of_ports=len(ports),
        name=_name,
        wavelength=super.wavelength,
        connection=connection,
        ports=_ports,
    )
    super.parent = new
    new.children = [super]
    return new


def validate_subntwk_ports(super: NtwkNode, ports: SubNwtkPortsDef) -> list[PortName]:
    if isinstance(ports, dict):
        ports = list(ports.values())
    assert isinstance(super.ports, list), "The port names of the super node must be a list."
    _ports: list[PortName] = []
    for port in ports:
        if isinstance(port, int):
            _ports.append(super.ports[port])
        elif isinstance(port, str) and "::" not in port: 
            # port name without node name
            _ports.append(super.ports_dict[port])
        else:
            _ports.append(PortName.validate(port))
    assert list(dict.fromkeys(_ports)) == _ports, "The ports must be unique."
    assert all(port in super.ports for port in _ports), "The ports must be in the super node."
    return _ports


from ..ntwk import NtwkNode
SubNtwk.update_forward_refs()
