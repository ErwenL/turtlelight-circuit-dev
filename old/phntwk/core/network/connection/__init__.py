from __future__ import annotations
from .port import PortName, AnyPortName, AnyNodeLink, PortNameTuple
from .connection import NtwkConnection, connect_ntwks, NodeLink
from .subntwk import SubNtwk, sub_ntwk, SubNwtkPortsDef
from .from_stage import FromTransferMatrixStage, from_transfer_matrix_stage, FromStagePortsDef
from typing import Union, Literal

NodeConnectionType = Literal["ntwk", "subntwk", "from_stage"]
NodeConnection = Union[NtwkConnection, SubNtwk, FromTransferMatrixStage]



