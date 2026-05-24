from ...core import StageStructure, NtwkStructure

base_structure = StageStructure(
    name="mzi",
    type="cascade",
    components=[
        (1, 2, "ref"),
        (2, 2, "delaypair"),
        (2, 1, "dut")
    ],
)