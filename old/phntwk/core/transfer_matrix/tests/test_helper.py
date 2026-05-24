from siluxApi.phntwk.core.transfer_matrix.structure import (
    StageStructure, build_from_structure
)
from siluxApi.phntwk.core.transfer_matrix.helper import (
    get_branch_list, get_leaf_menu, get_branch_repr, StageHelper,
)
import numpy as np

def get_mux_stage():
    delaypair = StageStructure(
        name="delaypair",
        type="parallel",
        components=[
            (1, 1, "delayline"),
            (1, 1, "delayline"),
        ]
    )
    mux_stage = StageStructure(
        name="mux_stage",
        type="cascade",
        components=[
            (1, 2, "trident"),
            delaypair,
            (2, 2, "dc"),
            delaypair,
            (2, 2, "dc"),
            delaypair,
            (2, 2, "dc"),
        ]
    )
    mux = StageStructure(
        name="mux",
        type="cascade",
        components=[
            mux_stage.rename("mux_stage_1"),
            StageStructure(
                name="mux_stage_2",
                type="parallel",
                components=[
                    mux_stage.rename("mux_stage_2_1"),
                    mux_stage.rename("mux_stage_2_2"),
                ]
            )
        ]
    )
    return build_from_structure(mux, np.linspace(1.2, 1.3, 101))

def get_sample_stage():
    delaypair = StageStructure(
        name="delaypair",
        type="parallel",
        components=[
            (1, 1, "delayline"),
            (1, 1, "delayline"),
        ]
    )
    mux_stage = StageStructure(
        name="mux_stage",
        type="cascade",
        components=[
            (2, 2, "dc"),
            delaypair,
            (2, 2, "dc"),
            delaypair,
            (2, 2, "dc"),
            delaypair,
            (2, 2, "dc"),
        ]
    )
    mux = StageStructure(
        name="mux",
        type="cascade",
        components=[
            StageStructure(
                name="mux_stage_1",
                type="substage",
                components=[mux_stage],
                inherit_ports=([0], [0, 1])
            ),
            StageStructure(
                name="mux_stage_2",
                type="parallel",
                components=[
                    StageStructure(
                        name="mux_stage_2_1",
                        type="substage",
                        components=[mux_stage],
                        inherit_ports=([1], [0, 1])
                    ),
                    StageStructure(
                        name="mux_stage_2_2",
                        type="substage",
                        components=[mux_stage],
                        inherit_ports=([0], [0, 1])
                    ),
                ]
            )
        ]
    )
    return build_from_structure(mux, np.linspace(1.2, 1.3, 101))

def test_get_branch_list():
    # stage = get_mux_stage()
    stage = get_sample_stage()
    return get_branch_list(stage)

def test_get_leaf_dict():
    branch = test_get_branch_list()
    return get_leaf_menu(branch)

def test_get_branch_repr():
    branch = test_get_branch_list()
    menu = get_leaf_menu(branch)
    return get_branch_repr(branch, menu)

def test_stage_helper():
    # stage = get_sample_stage()
    stage = get_mux_stage()
    helper = StageHelper(stage)
    helper.print_branch()
    return

def main():
    # result = test_get_branch_list()
    # result = test_get_leaf_dict()
    # result = test_get_branch_repr()
    test_stage_helper()
    return 

if __name__ == "__main__":
    main()