from siluxApi.phntwk.core.transfer_matrix.structure import StageStructure
from siluxApi.phntwk.core.network.structure import NtwkStructure
import numpy as np

def sample_structure():
    tk = StageStructure(
        name="microring_ntwk_tk",
        type="from_ntwk",
        components=[
            NtwkStructure(
                name= "microring_ntwk",
                type= "ntwk",
                components=[
                    (4, "dc", ["in", "from_ring", "out", "to_ring"]),
                    NtwkStructure(
                        name="ring",
                        type="from_stage",
                        components=[
                            StageStructure(
                                name="ring_wg",
                                type="cascade",
                                components=[
                                    (1, 1, "bend"),
                                    StageStructure(
                                        name="dut_channel",
                                        type="substage",
                                        components=[
                                            (2, 2, "dut")
                                        ],
                                        inherit_ports=([0], [1])
                                    ),
                                    (1, 1, "bend"),
                                ],
                            )
                        ],
                        inherit_ports=({"in": 0}, {"out": 0})
                    )
                ],
                links=[
                    (("dc", "to_ring"), ("ring", "in")),
                    (("ring", "out"), ("dc", "from_ring"))
                ]
            )
        ],
        inherit_ports=(["in"], ["out"])
    )
    return tk

def test_mix_structure():
    coupler = NtwkStructure(
        name="dc_ntwk",
        type="from_stage",
        components=[
            (2, 2, "dc")
        ],
        inherit_ports=(
            {"in":0, "from_ring": 1}, 
            {"out":0, "to_ring": 1}
        )
    )

    ring = NtwkStructure(
        name="ring_ntwk",
        type="from_stage",
        components=[
            StageStructure(
                name="ring_wg",
                type="cascade",
                components=[
                    (1, 1, "bend"),
                    (1, 1, "dut"),
                ],
            )
        ],
        inherit_ports=(
            {"in": 0}, {"out": 0}
        )
    )

    return StageStructure(
        name="microring_tk",
        type="from_ntwk",
        components=[
            NtwkStructure(
                name="all-pass-microring",
                type="ntwk",
                components=[
                    coupler.rename("coupler"),
                    ring.rename("ring")
                ],
                links = [
                    (("coupler", "to_ring"), ("ring", "in")),
                    (("ring", "out"), ("coupler", "from_ring"))
                ]
            )
        ],
        inherit_ports=([("coupler", "in")], [("coupler", "out")])
    )

def test_build():
    # tk = sample_structure()
    tk = test_mix_structure()
    wavelength = np.linspace(1.2, 1.3, 101)
    stage = tk.build(wavelength)
    return stage


def main():
    stage = test_build()
    return

if __name__ == "__main__":
    main()