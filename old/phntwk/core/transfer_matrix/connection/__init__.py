from __future__ import annotations
from .cascade import CascadeConnect, cascade_connect
from .parallel import ParallelConnect, parallel_connect
from .substage import SubConnect, sub_connect
from .from_ntwk import FromNtwkNodeConnect, from_ntwk_node_connect
from typing import Protocol, runtime_checkable, Literal, Union
import numpy as np

StageConnectionType = Literal["cascade", "parallel", "substage", "from_ntwk"]

# @runtime_checkable
# class Connection(Protocol):
#     name: ConnectionType
#     def get_matrix(self) -> np.ndarray: ...

StageConnection = Union[CascadeConnect, ParallelConnect, SubConnect, FromNtwkNodeConnect]