import numpy as np
import pandas as pd
from typing import Literal, Optional, Callable, TypedDict
from numpy.typing import NDArray
from ...core import (
    Stage, StageStructure, NtwkStructure, PhntwkTree, df_2_matrix
)
from pydantic import BaseModel
from ...utils import align_2_x, chain_transform
from ...passive import dc, waveguide as wg, abstract

def get_structure():
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
                name="ring_stage",
                type="cascade",
                components=[
                    (1, 1, "ring_wg"),
                    (1, 1, "ring_loss"),
                    (1, 1, "dut"),
                    (1, 1, "dut_wg"),
                ],
            )
        ],
        inherit_ports=(
            {"in": 0}, {"out": 0}
        )
    )

    return StageStructure(
        name="tk",
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

def create_tk_phntwk(wavelength: NDArray) -> tuple[Stage, PhntwkTree]:
    structure = get_structure()
    tk_phntwk = structure.build(wavelength)
    helper = PhntwkTree(tk_phntwk)
    return tk_phntwk, helper

class MicroringNtwkTestkeyMatrixStore(BaseModel):
    dc: NDArray
    ring_wg: NDArray
    ring_loss: NDArray
    dut: Optional[NDArray] = None
    dut_wg: Optional[NDArray] = None

    class Config:
        arbitrary_types_allowed = True

    def update_phntwk(self, phntwk: PhntwkTree, *nodes: str):
        if not nodes:
            nodes = ("dc", "ring_wg", "ring_loss", "dut", "dut_wg")
        for node in nodes:
            assert getattr(self, node) is not None, f"{node} matrix is not provided"
        for node in nodes:
            matrix = getattr(self, node)
            phntwk.set_node_matrix(node, 0, matrix)
        return

    def check_matrix(self):
        from colorama import Fore, Style
        for node in ("dc", "ring_wg", "ring_loss", "dut", "dut_wg"):
            matrix = getattr(self, node)
            if matrix is None:
                print(f"{Fore.RED}{node} matrix is not provided{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}{node} matrix is provided{Style.RESET_ALL}")
        return

class MicroringNtwkTestkeyHelper(BaseModel):
    get_dco_props: Callable[..., pd.DataFrame]
    get_dcc_props: Callable[..., pd.DataFrame]
    get_wgo_props: Callable[..., pd.DataFrame]
    get_wgc_props: Callable[..., pd.DataFrame]
    get_ridgewgo_props: Optional[Callable[..., pd.DataFrame]] = None
    get_ridgewgc_props: Optional[Callable[..., pd.DataFrame]] = None
    get_sinwgo_props: Optional[Callable[..., pd.DataFrame]] = None
    get_sinwgc_props: Optional[Callable[..., pd.DataFrame]] = None
    ring_loss: float = - 0.2
    
    class Config:
        arbitrary_types_allowed = True

    def get_dc_matrix(
        self,
        wavelength: NDArray,
        band: Literal["O", "C"],
        dc_couple_length: float,
    ):
        match band:
            case "O":
                df = self.get_dco_props()
            case "C":
                df = self.get_dcc_props()
        df["couple_length"] = dc_couple_length
        df = align_2_x(df, "wavelength", wavelength, ["couple_length"])
        df["couple_loss"] = 0
        df["fanout_loss"] = 0
        return chain_transform(df, dc.params_2_props, dc.props_2_trans_matrix, df_2_matrix)

    def get_wg_matrix(
        self,
        wavelength: NDArray,
        band: Literal["O", "C"],
        waveguide_type: Literal["si", "ridge", "sin"],
        length: float,
    ):
        match (band, waveguide_type):
            case ("O", "si"):
                get_props = self.get_wgo_props
            case ("C", "si"):
                get_props = self.get_wgc_props
            case ("O", "ridge"):
                get_props = self.get_ridgewgo_props
            case ("C", "ridge"):
                get_props = self.get_ridgewgc_props
            case ("O", "sin"):
                get_props = self.get_sinwgo_props
            case ("C", "sin"):
                get_props = self.get_sinwgc_props
            case _:
                raise ValueError(f"Unsupported band-waveguide_type combination: {band}-{waveguide_type}")
        assert get_props is not None, f"{waveguide_type}: get_wg_props not defined"
        df = get_props()
        df["length"] = length
        df = align_2_x(df, "wavelength", wavelength, ["length"])
        df["loss"] = 0
        return chain_transform(df, wg.params_2_props, wg.props_2_trans_matrix, df_2_matrix)
    
    def get_abstract_loss_matrix(
        self,
        wavelength: NDArray,
        loss: float,
    ):
        df = abstract.loss(wavelength, loss)
        return df_2_matrix(df)

    def get_matrix_store(
        self,
        wavelength: NDArray,
        band: Literal["O", "C"],
        ring_length: float,
        dc_couple_length: float,
        dut_length: float,
        dut_waveguide_type: Literal["si", "ridge", "sin"],
        estimate_dut_wg: bool = True,
    ) -> MicroringNtwkTestkeyMatrixStore:
        _dut_length = dut_length if estimate_dut_wg else 0
        return MicroringNtwkTestkeyMatrixStore(
            dc = self.get_dc_matrix(wavelength, band, dc_couple_length),
            ring_wg = self.get_wg_matrix(wavelength, band, "si", ring_length-dut_length),
            ring_loss = self.get_abstract_loss_matrix(wavelength, self.ring_loss),
            dut_wg = self.get_wg_matrix(wavelength, band, dut_waveguide_type, _dut_length),
        )





