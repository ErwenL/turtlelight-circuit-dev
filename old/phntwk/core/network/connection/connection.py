from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Any, Sequence
from pydantic import BaseModel
from .port import PortName, NodeLink, AnyNodeLink, valiate_node_link
from loguru import logger

SLink = tuple[int, int]

class NtwkConnection(BaseModel):
    type: Literal["ntwk"] = "ntwk"
    previous: NtwkNode
    next: NtwkNode
    links: list[NodeLink] = []

    class Config:
        arbitrary_types_allowed = True

    @property
    def internal_ports(self) -> list[PortName]:
        assert isinstance(self.previous.ports, list), "The port names of the previous node must be a list."
        assert isinstance(self.next.ports, list), "The port names of the next node must be a list."
        return self.previous.ports + self.next.ports
    
    @property
    def external_ports(self) -> list[PortName]:
        _ports = self.internal_ports.copy()
        for link in self.links:
            for port in link:
                if port not in _ports:
                    logger.warning(f"{port} not in {[str(p) for p in _ports]}")
                _ports.remove(port)
        return _ports
                
    def get_s(self) -> np.ndarray:
        return connect_s(
            self.previous.get_s(), 
            self.next.get_s(),
            self.internal_ports,
            self.links,
        )

def connect_ntwks(_previous: NtwkNode, _next: NtwkNode, links: list[AnyNodeLink]) -> NtwkNode:
    _links = validate_connection_links( _previous, _next, links,)
    
    connection = NtwkConnection(
        previous=_previous, 
        next=_next,
        links=_links,
    )
    num_of_external_ports = _previous.num_of_ports + _next.num_of_ports - 2 * len(_links)
    new = NtwkNode(
        num_of_ports=num_of_external_ports,
        ports=connection.external_ports,
        wavelength=_previous.wavelength,
        connection=connection,
    )
    _previous.parent = new
    _next.parent = new
    new.children = [_previous, _next]
    return new

def connect_s(previous: np.ndarray, next: np.ndarray, ports:list[PortName], links: list[NodeLink]) -> np.ndarray:
    _s = np.block([
        [
            previous, 
            np.zeros((previous.shape[0], previous.shape[1], next.shape[2]), dtype=complex)
        ],
        [
            np.zeros((previous.shape[0], next.shape[1], previous.shape[2]), dtype=complex), 
            next
        ]
    ])

    def get_s_link(_ports: list[PortName], _link: NodeLink) -> SLink:
        return _ports.index(_link[0]), _ports.index(_link[1])

    def update_ports(_ports: list[PortName], _link: NodeLink) -> list[PortName]:
        for port in _link:
            _ports.remove(port)
        return _ports

    def s_connect_ports(_s: np.ndarray, _ports: list[PortName], _link: NodeLink):
        k, l = get_s_link(_ports, _link)
        # import skrf as rf
        # _s = rf.network.innerconnect_s(_s, k, l)
        _s = innerconnect_s(_s, k, l)
        _ports = update_ports(_ports, _link)
        return _s, _ports

    for link in links:
        _s, ports = s_connect_ports(_s, ports, link)
    return _s


def innerconnect_s(s: np.ndarray, k: int, l:int) -> NDArray[np.complex_]:
    """reimplement skrf.network.innerconnect_s()
    """
    s = validate_s(s).copy()
    ds_items = [
        (s[:, :, [l]] @ s[:, [k], :]).astype(np.complex_), 
        (s[:, :, [k]] @ s[:, [l], :]).astype(np.complex_) ,
    ]
    norm = [np.ones_like(s[:, 0, 0]), np.zeros_like(s[:, 0, 0])]

    # i -> k(l) -> l(k) -> j
    if np.any(s[:, l, k] != 0):
        ds_items[0] *= (1 - s[:, l, k]).reshape(-1, 1, 1)
        norm[0] *= (1 - s[:, l, k])
    if np.any(s[:, k, l] != 0):
        ds_items[1] *= (1 - s[:, k, l]).reshape(-1, 1, 1)
        norm[0] *= (1 - s[:, k, l])
    ds = ds_items[0] + ds_items[1]

    # i -> k(l) -> l(k) -> k(l) -> j
    if np.any(s[:, k, k] != 0):
        ds += s[:, :, [l]] @ s[:, [l], :] * s[:, k, k].reshape(-1, 1, 1)
        norm[1] = s[:, k, k]
    if np.any(s[:, l, l] != 0):
        ds += s[:, :, [k]] @ s[:, [k], :] * s[:, l, l].reshape(-1, 1, 1)
        norm[1] *= s[:, l, l]

    s += ds / (norm[0] - norm[1]).reshape(-1, 1, 1)
    s = np.delete(s, [k, l], axis=1)
    s = np.delete(s, [k, l], axis=2)
    return s

def validate_s(s: np.ndarray) -> NDArray[np.complex_]:
    assert s.dtype == complex, "The dtype of the scattering matrix must be complex."
    assert s.ndim == 3, "The scattering matrix must be a 3-dimensional array."
    return s

def validate_connection_links(_previous: NtwkNode, _next: NtwkNode, links: list[AnyNodeLink]) -> list[NodeLink]:
    _links = []
    _ports = _previous.ports + _next.ports
    for link in links:
        _link = valiate_node_link(link)
        if _link[0] in _ports and _link[1] in _ports:
            _links.append(_link)
    return _links

from ..ntwk import NtwkNode
NtwkConnection.update_forward_refs()