from siluxApi.phntwk.core import (
    StageStructure, NtwkStructure
)
from siluxApi.phntwk.core.tests.test_structure import test_build as get_mix_tree
from siluxApi.phntwk.core.helper import (
    PhntwkTree, get_branch_list,
    analyze_phnode_tree
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
    return mux.build(np.linspace(1.2, 1.3, 101))

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
    return mux.build(np.linspace(1.2, 1.3, 101))

def test_get_branch_list():
    stage = get_mix_tree()
    branch = get_branch_list(stage)
    return branch

def main():
    # branch = test_get_branch_list()
    tree = PhntwkTree(get_mix_tree())
    tree.print_tree()
    tree = PhntwkTree(get_sample_stage())
    tree.print_tree()
    return

if __name__ == "__main__":
    main()