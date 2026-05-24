from siluxApi.phntwk.core.transfer_matrix.structure import (
    StageStructure, build_from_structure
)
import numpy as np

def test_stage_structure():
    channel = StageStructure(
        name="mzi_through",
        type="substage",
        components=[
            StageStructure(
                name="mzi",
                type="cascade",
                components=[
                    (2, 2, "dc"),
                    StageStructure(
                        name="delaypair",
                        type="parallel",
                        components=[
                            (1, 1, "delay"),
                            (1, 1, "delay"),
                        ]
                    ),
                    (2, 2, "dc"),
                ]
            )
        ],
        inherit_ports=([0], [0])
    )
    return channel

def test_from_dict():
    channel = StageStructure.parse_obj(
        {
            "name": "mzi_through",
            "type": "substage",
            "components": [
                {
                    "name": "mzi",
                    "type": "cascade",
                    "components": [
                        (2, 2, "dc"),
                        {
                            "name": "delaypair",
                            "type": "parallel",
                            "components": [
                                (1, 1, "delay"),
                                (1, 1, "delay"),
                            ]
                        },
                        (2, 2, "dc"),
                    ]
                }
            ],
            "inherit_ports": ([0], [0])
        }
    )
    return channel

def test_mux_nested_structure():
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
    return mux

def test_build():
    wavelength = np.linspace(1.2, 1.3, 101)
    channel = test_stage_structure()
    stage = build_from_structure(channel, wavelength)
    return stage

def main():
    # channel = test_stage_structure()
    # channel = test_from_dict()
    # stage = test_build()
    mux_structure = test_mux_nested_structure()
    mux = build_from_structure(mux_structure, np.linspace(1.2, 1.3, 101))
    return

if __name__ == "__main__":
    from timeit import timeit
    print(timeit(main, number=100))