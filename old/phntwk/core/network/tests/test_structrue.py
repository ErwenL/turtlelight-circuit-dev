from siluxApi.phntwk.core.network.structure import (
    NtwkStructure, build_from_structure
)
import numpy as np

def test_ntwk_structure():
    tk = NtwkStructure(
        name="measure",
        type="subntwk",
        components=[
            NtwkStructure(
                name="microring_tk",
                type="ntwk",
                components=[
                    (4, "dc", ["in", "from_ring", "out", "to_ring"]),
                    (2, "ring", ["in", "out"]),
                ],
                links=[
                    (("dc", "from_ring"), ("ring", "in")),
                    (("dc", "to_ring"), ("ring", "out"))
                ]
            )
        ],
        links=[],
        # inherit_ports=[("dc", "in"), ("dc", "out")]
        inherit_ports={"in":("dc", "in"), "out":("dc", "out")}
    )
    return tk

def test_build():
    wavelength = np.linspace(1.2, 1.3, 101)
    tk = test_ntwk_structure()
    ntwk = build_from_structure(tk, wavelength)
    return ntwk

def main():
    ntwk = test_build()
    return

if __name__ == "__main__":
    main()