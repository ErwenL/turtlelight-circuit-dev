from ..circuit import Element, LibName, AnyValueArray, ComplexArray, Circuit
from pydantic import Field, validator
from typing import Any, Optional
import numpy as np
from loguru import logger


class Composite(Element):
    """impedance or admittance of a composite element is calculated based on
    s params of the internal circuit.
    """

    libname: LibName = "Composite"

    def get_impedance(
        self, _from: str, _to: str, recursion_depth: int = 0
    ) -> AnyValueArray:
        """reverse the calculation of `extract_s_component`"""
        # ! this method may not be correct
        # TODO: need to refactor the arhitecture to include considering 'gnd' node for all element
        logger.warning(
            f"""`get_impedance({_from}, {_to})` of Composite is not implemented correctly. 
            It is only correct if there is no gnd node in the internal circuit."""
        )
        # i = self.get_s_component(_from, _to) / self.port_z0(_to)
        # v = self.get_s_component(_from, _from) + 1 - self.get_s_component(_from, _to)
        # return self.as_f(v / i)
        v_to = self.get_s_component(_from, _to) / 2
        i_to = v_to / self.port_z0(_to)
        i_s = 1 / self.port_z0(_from)
        i_from = i_s - i_to
        v_from = i_from * self.port_z0(_from)
        return self.as_f((v_from - v_to) / i_to)


class MixMode(Composite):
    libname: LibName = "MixMode"
    circuit_ports: list[str] = Field(default_factory=lambda: [])
    """ports of the internal circuit before mode mixing
    - the sequence of the ports should follow the definition of `skrf.Network.se2gmm()`
    """
    num_of_differential_ports: int
    se_z0: list[float] = Field(default=None)

    @validator("num_of_differential_ports", always=True)
    def validate_differential_ports(cls, v: int, values: dict) -> int:
        """validate and set_default mix mode port names

        - Defaults to 'd'/'c'str(index) of port number if not provided
        """
        assert "ports" in values, "ports must be provided"
        ports = [f"d{i}" for i in range(v)]
        ports += [f"c{i}" for i in range(v)]
        ports += values["ports"][2 * v :]
        values["ports"] = ports
        values["port_order"] = ports
        return v

    @validator("se_z0", always=True, pre=True)
    def validate_se_z0(cls, v: Any, values: dict) -> list[float]:
        return cls._validate_z0(v, values["num_of_ports"])

    def get_s(self) -> ComplexArray:
        if self.s is not None:
            # s is already calculated
            return self.s

        _s = super().get_s(z0=self.se_z0)
        import skrf as rf

        ntwk = rf.Network(
            frequency=rf.Frequency.from_f(self.frequency, unit="Hz"),
            s=_s,
            z0=self.se_z0,
        )
        ntwk.se2gmm(p=self.num_of_differential_ports)
        self.z0 = np.real(ntwk.z0[0, :]).tolist()
        self.s = ntwk.s
        return self.s


class SubElement(Composite):
    libname: LibName = "SubElement"
    circuit_ports: list[str] = Field(default_factory=lambda: [])
    """external ports of the super element"""

    @validator("circuit", always=True)
    def validate_super_element(cls, v: Circuit):
        """circuit should contain only one element, which is the super element of the sub element"""
        assert len(v.elements) == 1, "Only one element is supported"
        return v

    @validator("circuit_ports", always=True)
    def validate_circuit_ports(cls, v: list[str], values: dict) -> list[str]:
        super_element = values["circuit"].elements[0]
        for port in v:
            assert port in super_element.ports, (
                f"port {port} not found in super element. ports: {super_element.ports}"
            )
        return v

    @property
    def super_element(self) -> Element:
        return self.children[0]

    def get_s(self) -> ComplexArray:
        if self.s is not None:
            # s is already calculated
            return self.s

        _s = self.super_element.get_s()
        _z0 = self.super_element.z0
        external_ports_nums = [
            self.super_element.ports.index(port) for port in self.circuit_ports
        ]
        self.z0 = np.array(_z0)[external_ports_nums].tolist()
        self.s = _s[:, *np.meshgrid(external_ports_nums, external_ports_nums)]
        return self.s
