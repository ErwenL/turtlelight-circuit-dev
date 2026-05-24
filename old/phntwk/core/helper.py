from __future__ import annotations
from .transfer_matrix.stage import Stage
from .transfer_matrix.connection import (
    CascadeConnect, ParallelConnect, SubConnect, FromNtwkNodeConnect,
    StageConnectionType
)
from .network.ntwk import NtwkNode
from .network.connection import (
    NtwkConnection, SubNtwk, FromTransferMatrixStage,
    NodeConnectionType
)
from typing import Union, Literal, Optional
import json
from functools import partial
from colorama import Fore, Style
from pydantic import BaseModel
from numpy.typing import NDArray

class PhntwkTree:
    def __init__(self, root: PhNode) -> None:
        self.root = root
        # legacy
        self.branch_list = get_branch_list(root)
        self.menu = get_leaf_menu(self.branch_list)
        self.branch_repr = get_branch_repr(self.branch_list, self.menu)
        # tree repr
        self.repr, self.node_type_dict, self.node_name_dict = analyze_phnode_tree(self.root)

    def print_branch(self) -> None:
        print(json.dumps(self.branch_repr, indent=4))
        return

    def set_leaf_matrix(self, leaf_name: str, index: int, matrix: NDArray) -> None:
        leaf = self.menu[leaf_name][index]
        leaf.set_matrix(matrix)
        return
    
    def get_leaf(self, leaf_name: str, index: int) -> PhNode:
        return self.menu[leaf_name][index]

    def print_tree(self) -> None:
        self.repr.print()
        return

    def _get_node_repr(self, name: str, index: int, by_type: bool = False) -> NodeRepr:
        if by_type:
            return self.node_type_dict[name][index]
        return self.node_name_dict[name][index]

    def get_node(self, name: str, index: int, by_type: bool = False) -> PhNode:
        _repr = self._get_node_repr(name, index, by_type)
        assert _repr.node is not None, "The node must be defined."
        return _repr.node
    
    def set_node_matrix(self, name: str, index: int, matrix: NDArray, by_type: bool = False) -> None:
        self.get_node(name, index, by_type).set_matrix(matrix)
        return

    def get_node_matrix(self, name: str, index: int = 0, by_type: bool = False) -> NDArray:
        return self.get_node(name, index, by_type).get_matrix()

    def print_node(self, name: str, index: int, by_type: bool = False) -> None:
        self._get_node_repr(name, index, by_type).print()

    def print_leaves_check(self) -> None:
        self._print_stage_leaves_check()
        self._print_ntwk_leaves_check
    
    def _print_stage_leaves_check(self) -> None:
        if "stage-leaf" in self.node_type_dict:
            for leaf in self.node_type_dict["stage-leaf"]:
                _node = leaf.node
                assert isinstance(_node, Stage), "The node must be a Stage object."
                if _node.matrix is None:
                    print(f"{leaf.name_repr()} matrix is {_mark_text('not set', color=Fore.LIGHTRED_EX, style=Style.BRIGHT)}.")
                else:
                    print(f"{leaf.name_repr()} matrix is {_mark_text('set', color=Fore.LIGHTGREEN_EX, style=Style.BRIGHT)}.")
        return
    
    def _print_ntwk_leaves_check(self) -> None:
        if "ntwk-leaf" in self.node_type_dict:
            for leaf in self.node_type_dict["ntwk-leaf"]:
                _node = leaf.node
                assert isinstance(_node, NtwkNode), "The node must be a NtwkNode object."
                if _node.s is None:
                    print(f"{leaf.name_repr()} s params is {_mark_text('not set', color=Fore.LIGHTRED_EX, style=Style.BRIGHT)}.")
                else:
                    print(f"{leaf.name_repr()} s params is {_mark_text('set', color=Fore.LIGHTGREEN_EX, style=Style.BRIGHT)}.")
        return


PhNode = Union[Stage, NtwkNode]
BranchList = list[Union[PhNode, "BranchList"]]
BranchReprList = list[Union[str, "BranchReprList"]]
LeafMenu = dict[str, list[PhNode]]

GroupConnection = Union[CascadeConnect, ParallelConnect, NtwkConnection]
InheritConnection = Union[SubConnect, SubNtwk, FromTransferMatrixStage, FromNtwkNodeConnect]

def get_branch_list(phnode: PhNode) -> BranchList:
    """convert a phnode to a list of phnodes
    - children in a connection group are put in a list
        - group connection: cascade, parallel, ntwk
    - inherit connection are ignored
        - inherit connection: substage, subntwk, from_ntwk, from_stage

    Args:
        phnode (PhNode): Stage or NtwkNode

    Returns:
        BranchList: a list of phnodes
    """

    # leaf node
    if phnode.connection is None:
        return [phnode]
    
    # branch node
    def get_child_connection_type(child:PhNode) -> type:
        if child.connection is None:
            return type(phnode.connection)
        if isinstance(child.connection, InheritConnection):
            return get_child_connection_type(child.connection.super)
        assert isinstance(child.connection, GroupConnection), "The connection must be a GroupConnection."
        return type(child.connection)

    def get_child_branch_list(child: PhNode) -> BranchList:
        if (
            isinstance(phnode.connection, GroupConnection)
            and type(phnode.connection) is not get_child_connection_type(child)
        ):
            return [get_branch_list(child)]
        return get_branch_list(child)

    if isinstance(phnode.connection, InheritConnection):
        return get_child_branch_list(phnode.connection.super)
    assert isinstance(phnode.connection, GroupConnection), "The connection must be a GroupConnection."
    return get_child_branch_list(phnode.connection.previous) + get_child_branch_list(phnode.connection.next)

def get_leaf_menu(branch: BranchList) -> LeafMenu:
    """convert a branch list to a leaf dict
    - the key is the name of the PhNode
    - the value is a list of PhNodes with the same name

    Args:
        branch (BranchList): A list of PhNodes

    Returns:
        LeafDict: A dict of PhNodes
    """
    menu: LeafMenu = {}

    def add_leaf_2_dict(leaf: PhNode) -> None:
        assert isinstance(leaf.name, str), "The name of the leaf must be a string."
        if leaf.name not in menu:
            menu[leaf.name] = []
        menu[leaf.name].append(leaf)

    def scan_branch(_root: BranchList|PhNode) -> None:
        for _branch in _root:
            if isinstance(_branch, list):
                scan_branch(_branch)
                continue
            assert isinstance(_branch, PhNode), "The leaf must be a Stage object."
            add_leaf_2_dict(_branch)
    
    scan_branch(branch)
    
    return menu

def get_branch_repr(branch: BranchList, menu: LeafMenu) -> BranchReprList:
    """replace the PhNode in a branch list with its name and its index in the leaf dict"""

    def get_node_index(stage: PhNode, _list: list[PhNode]) -> int:
        for i, _stage in enumerate(_list):
            if _stage is stage:
                return i
        raise ValueError("The stage must be in the list.")

    def node_repr(leaf: PhNode) -> str:
        assert isinstance(leaf.name, str), "The name of the leaf must be a string."
        return f"{leaf.name}[{get_node_index(leaf, menu[leaf.name])}]"

    def translate_branch(_root: BranchList|PhNode) -> BranchReprList|str:
        if isinstance(_root, PhNode):
            return node_repr(_root)
        assert isinstance(_root, list), "The branch must be a list."
        return [translate_branch(_branch) for _branch in _root]
    
    repr_list = translate_branch(branch)
    assert isinstance(repr_list, list), "The repr_list must be a list."
    return repr_list

class NodeRepr(BaseModel):
    node: Optional[PhNode] = None
    type: StageConnectionType|NodeConnectionType|Literal["stage-leaf", "ntwk-leaf"]
    name: str
    type_index: int = 0
    name_index: int = 0
    super_nodes: list[NodeRepr] = []
    """super nodes for inherit connection"""
    children: list[NodeRepr] = []
    """children nodes for group connection"""

    def get_children(self) -> list[NodeRepr]:
        if len(self.super_nodes) > 0:
            return self.super_nodes[-1].get_children()
        else:
            return self.children

    def name_repr(self) -> str:
        """mark up node `name[index]` with color and style

        - stage or ntwk:
            - stage: light blue
            - ntwk: light yellow
        - node_type:
            - leaf: bright
            - inherit: dim

        Returns:
            str: marked up node name
        """
        if isinstance(self.node, Stage):
            _markup = partial(_mark_text, color=Fore.LIGHTBLUE_EX)
        else:
            _markup = partial(_mark_text, color=Fore.LIGHTYELLOW_EX)
        if self.type in ["stage-leaf", "ntwk-leaf"]:
            _markup = partial(_markup, style=Style.BRIGHT)
        elif self.type in ["substage", "subntwk", "from_ntwk", "from_stage"]:
            _markup = partial(_markup, style=Style.DIM)
        return _markup(f"{self.name}[{self.name_index}]")

    def type_repr(self) -> str:
        if isinstance(self.node, Stage):
            _markup = partial(_mark_text, color=Fore.LIGHTGREEN_EX)
        else:
            _markup = partial(_mark_text, color=Fore.LIGHTRED_EX)
        if self.type in ["cascade", "parallel", "ntwk"]:
            _markup = partial(_markup, style=Style.BRIGHT)
        else:
            _markup = partial(_markup, style=Style.DIM)
        return _markup(f"<{self.type}[{self.type_index}]>")

    def repr(self) -> str:
        if len(self.super_nodes) > 0:
            return f"{self.name_repr()}{self.type_repr()} -> {self.super_nodes[-1].repr()}"
        else:
            return f"{self.name_repr()}{self.type_repr()}"
    
    def __repr__(self) -> str:
        return f"{self.name}[{self.name_index}]<{self.type}[{self.type_index}]>"

    def lines(self) -> list[str]:
        _lines = [self.repr()]
        children = self.get_children()
        if len(children) == 0:
            return _lines
        for idx, child in enumerate(children):
            if idx == len(children) - 1:
                _lines.append(f"└── {child.lines()[0]}")
                _lines.extend(f"    {line}" for line in child.lines()[1:])
            else:
                _lines.append(f"├── {child.lines()[0]}")
                _lines.extend(f"│   {line}" for line in child.lines()[1:])
        return _lines

    def print(self) -> None:
        for line in self.lines():
            print(line)
        return

def _mark_text(text: str, color: str, style:str = Style.NORMAL) -> str:
    return f"{style}{color}{text}{Fore.RESET}{Style.RESET_ALL}"

def analyze_phnode_tree(root: PhNode):

    node_type_dict: dict[str, list[NodeRepr]] = {}
    node_name_dict: dict[str, list[NodeRepr]] = {}

    def index_node_repr(_repr: NodeRepr, node: PhNode) -> NodeRepr:
        if _repr.type not in node_type_dict:
            node_type_dict[_repr.type] = [_repr]
        else:
            node_type_dict[_repr.type].append(_repr)
        if _repr.name not in node_name_dict:
            node_name_dict[_repr.name] = [_repr]
        else:
            node_name_dict[_repr.name].append(_repr)
        _repr.type_index = len(node_type_dict[_repr.type]) - 1
        _repr.name_index = len(node_name_dict[_repr.name]) - 1
        _repr.node = node
        return _repr
        

    def get_node_repr(node: PhNode) -> NodeRepr:
        assert isinstance(node.name, str), "The name of the node must be a string."
        # leaf node
        if node.connection is None:
            if isinstance(node, Stage):
                _type = "stage-leaf"
            else:
                _type = "ntwk-leaf"
            return index_node_repr(
                NodeRepr(type=_type, name=node.name),
                node
            )
        # inherit connection
        if isinstance(node.connection, InheritConnection):
            super_node = get_node_repr(node.connection.super)
            return index_node_repr(
                NodeRepr(
                    type=node.connection.type,
                    name=node.name,
                    super_nodes=[super_node] + super_node.super_nodes,
                ), node
            )
        assert isinstance(node.connection, GroupConnection), "The connection must be a GroupConnection."
        # group connection
        previous_node = get_node_repr(node.connection.previous)
        next_node = get_node_repr(node.connection.next)
        previous_children = previous_node.children if len(previous_node.children) > 0 else [previous_node]
        if previous_node.type == node.connection.type:
            children = previous_children + [next_node]
        else:
            children = [previous_node, next_node]

        return index_node_repr(
            NodeRepr(
                type=node.connection.type,
                name=node.name,
                children=children
            ), node
        )
    
    _repr = get_node_repr(root)
    return _repr, node_type_dict, node_name_dict
        







