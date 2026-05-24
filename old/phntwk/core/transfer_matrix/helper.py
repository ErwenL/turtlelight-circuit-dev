from .stage import Stage
from .connection import CascadeConnect, ParallelConnect, SubConnect
from typing import Union, Type
import json
import numpy as np

class StageHelper:
    def __init__(self, stage: Stage) -> None:
        self.stage = stage
        self.branch_list = get_branch_list(stage)
        self.menu = get_leaf_menu(self.branch_list)
        self.branch_repr = get_branch_repr(self.branch_list, self.menu)

    def print_branch(self) -> None:
        print(json.dumps(self.branch_repr, indent=4))
        return

    def set_leaf_matrix(self, leaf_name: str, leaf_num: int, matrix: np.ndarray) -> None:
        self.menu[leaf_name][leaf_num].set_matrix(matrix)
        return

BranchList = list[Union[Stage, "BranchList"]]
BranchReprList = list[Union[str, "BranchReprList"]]
LeafMenu = dict[str, list[Stage]]

def get_branch_list(stage: Stage) -> BranchList:
    """convert a stage to a list of stages
    - children in a series of cascade or parallel connection are put in a list
    - ignore substage

    Args:
        stage (Stage): Any stage

    Returns:
        StageList: A list of stages
    """
    # leaf node
    if stage.connection is None:
        return [stage]
    
    # branch node
    def get_child_connection_type(child:Stage) -> Type:
        if child.connection is None:
            return type(stage.connection)
        if isinstance(child.connection, SubConnect):
            return get_child_connection_type(child.connection.super)
        assert isinstance(child.connection, (CascadeConnect, ParallelConnect)), "The connection must be CascadeConnect or ParallelConnect."
        return type(child.connection)

    def get_child_branch_list(child: Stage) -> BranchList:
        if (
            isinstance(stage.connection, (CascadeConnect, ParallelConnect))
            and type(stage.connection) is not get_child_connection_type(child)
        ):
            return [get_branch_list(child)]
        return get_branch_list(child)

    if isinstance(stage.connection, SubConnect):
        return get_child_branch_list(stage.connection.super)
    assert isinstance(stage.connection, (CascadeConnect, ParallelConnect)), "The connection must be CascadeConnect or ParallelConnect."
    return get_child_branch_list(stage.connection.previous) + get_child_branch_list(stage.connection.next)

def get_leaf_menu(branch: BranchList) -> LeafMenu:
    """convert a branch list to a leaf dict
    - the key is the name of the stage
    - the value is a list of stages with the same name

    Args:
        branch (BranchList): A list of stages

    Returns:
        LeafDict: A dict of stages
    """
    menu: LeafMenu = {}

    def add_leaf_2_dict(leaf: Stage) -> None:
        assert isinstance(leaf.name, str), "The name of the leaf must be a string."
        if leaf.name not in menu:
            menu[leaf.name] = []
        menu[leaf.name].append(leaf)

    def scan_branch(_root: BranchList|Stage) -> None:
        for _branch in _root:
            if isinstance(_branch, list):
                scan_branch(_branch)
                continue
            assert isinstance(_branch, Stage), "The leaf must be a Stage object."
            add_leaf_2_dict(_branch)
    
    scan_branch(branch)
    
    return menu

def get_branch_repr(branch: BranchList, menu: LeafMenu) -> BranchReprList:
    """replace the stage in a branch list with its name and its index in the leaf dict"""

    def get_stage_index(stage: Stage, _list: list[Stage]) -> int:
        for i, _stage in enumerate(_list):
            if _stage is stage:
                return i
        raise ValueError("The stage must be in the list.")

    def stage_repr(stage: Stage) -> str:
        assert isinstance(stage.name, str), "The name of the leaf must be a string."
        return f"{stage.name}[{get_stage_index(stage, menu[stage.name])}]"

    def translate_branch(_root: BranchList|Stage) -> BranchReprList|str:
        if isinstance(_root, Stage):
            return stage_repr(_root)
        assert isinstance(_root, list), "The branch must be a list."
        return [translate_branch(_branch) for _branch in _root]
    
    repr_list = translate_branch(branch)
    assert isinstance(repr_list, list), "The repr_list must be a list."
    return repr_list