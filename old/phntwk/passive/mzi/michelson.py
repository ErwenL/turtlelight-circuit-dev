import numpy as np
import pandas as pd

from ...core import (
    Stage, StageStructure, NtwkStructure, PhntwkTree, df_2_matrix
)

def get_structure():
    splitter = NtwkStructure(
        name="splitter",
        type="from_stage",
        components=[
            (1, 2, "splitter")
        ],
        inherit_ports=(
            {"in":0}, 
            {"out1":0, "out2": 1}
        )
    )
    coupler = NtwkStructure(
        name="coupler",
        type="from_stage",
        components=[
            (2, 2, "coupler")
        ],
        inherit_ports=(
            {"in":0, "out": 1}, 
            {"ref":0, "dut": 1}
        )
    )

    def get_wg(name: str = "wg"):
        return NtwkStructure(
            name=name,
            type="from_stage",
            components=[
                (1, 1, name)
            ],
            inherit_ports=(
                {"in":0}, 
                {"out":0}
            )
        )

    mirror = NtwkStructure(
        name="mirror",
        type="subntwk",
        components=[
            NtwkStructure(
                name="mirror_ntwk",
                type="ntwk",
                components=[ splitter, get_wg() ],
                links=[
                    (("splitter", "out1"), ("wg", "in")),
                    (("splitter", "out2"), ("wg", "out"))
                ]
            )
        ],
        inherit_ports=(
            {"in": ("splitter", "in")}
        )
    )

    mi = NtwkStructure(
        name="mechilson_interferometer",
        type="ntwk",
        components=[
            coupler,
            get_wg("delayline"),
            get_wg("dut"),
            mirror.rename("ref_mirror"),
            mirror.rename("dut_mirror"),
        ],
        links=[
            (("coupler", "ref"), ("delayline", "in")),
            (("delayline", "out"), ("ref_mirror", "in")),
            (("coupler", "dut"), ("dut", "in")),
            (("dut", "out"), ("dut_mirror", "in")),
        ]
    )

    return StageStructure(
        name="measure",
        type="from_ntwk",
        components=[
            mi.rename("mi")
        ],
        inherit_ports=(
            [("coupler", "in")],
            [("coupler", "out")]
        )
    )
