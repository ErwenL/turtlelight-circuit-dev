"""
f_domain Elements:
- frequencyDependent
    - Resistor
    - Inductor
    - Capacitor
    - Conductor

"""

from __future__ import annotations
from ..circuit import Element, LibName, ElementType, AnyValueArray, AnyParamValue
from .utils import create_lib
from pydantic import Field
import numpy as np
from loguru import logger


class Params(Element.Params):
    f: AnyParamValue = Field(default=None)

    def validate_f(self, f: AnyValueArray):
        """validate element frequency range
        - warn if the provided frequency range can not cover the element frequency range

        Args:
            f (AnyValueArray): element frequency range
        """
        assert self.f is not None, "frequency range of the element is not provided"
        if np.min(f) < np.min(self.f) or np.max(f) > np.max(self.f):
            logger.warning(
                f"frequency range of the element is not covered by the provided frequency range"
            )
        return f, self.f


class Resistor(Element):
    libname: LibName = "Resistor"
    num_of_ports: int = 2
    type: ElementType = "rlc"

    class Params(Params):
        r: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, f: AnyParamValue, r: AnyParamValue, **kwargs):
        return cls(name=name, params={"f": f, "r": r}, **kwargs)

    def get_impedance(
        self, _from: str | None = None, _to: str | None = None, recursion_depth: int = 0
    ) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        self.params.validate_f(self.f)
        return np.interp(self.f, self.params.f, self.params.r) # type: ignore


class Inductor(Element):
    libname: LibName = "Inductor"
    num_of_ports: int = 2
    type: ElementType = "rlc"

    class Params(Params):
        l: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, f: AnyParamValue, l: AnyParamValue, **kwargs):
        return cls(name=name, params={"f": f, "l": l}, **kwargs)

    def get_impedance(
        self, _from: str | None = None, _to: str | None = None, recursion_depth: int = 0
    ) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        self.params.validate_f(self.f)
        return self.jw * np.interp(self.f, self.params.f, self.params.l) # type: ignore


class Capacitor(Element):
    libname: LibName = "Capacitor"
    num_of_ports: int = 2
    type: ElementType = "rlc"

    class Params(Params):
        c: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, f: AnyParamValue, c: AnyParamValue, **kwargs):
        return cls(name=name, params={"f": f, "c": c}, **kwargs)

    def get_admittance(
        self, _from: str | None = None, _to: str | None = None, recursion_depth: int = 0
    ) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        self.params.validate_f(self.f)
        return self.jw * np.interp(self.f, self.params.f, self.params.c) # type: ignore

class Conductor(Element):
    libname: LibName = "Conductor"
    num_of_ports: int = 2
    type: ElementType = "rlc"

    class Params(Params):
        g: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, f: AnyParamValue, g: AnyParamValue, **kwargs):
        return cls(name=name, params={"f": f, "g": g}, **kwargs)

    def get_admittance(
        self, _from: str | None = None, _to: str | None = None, recursion_depth: int = 0
    ) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        self.params.validate_f(self.f)
        return np.interp(self.f, self.params.f, self.params.g) # type: ignore
    
frequencyDependentLib = create_lib(
    "frequencyDependent", Resistor, Inductor, Capacitor, Conductor
)

R = Resistor
L = Inductor
C = Capacitor
G = Conductor
