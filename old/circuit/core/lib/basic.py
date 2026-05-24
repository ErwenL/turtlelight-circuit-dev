"""
Basic Elements: 
    - Resistor
    - Capacitor
    - Inductor
    - Conductor
    - Voltage Source
    - Current Source
"""

from __future__ import annotations
from ..circuit import Element, LibName, ElementType, AnyValueArray, AnyParamValue
from pydantic import Field


class Resistor(Element):
    libname: LibName = "Resistor"
    num_of_ports: int = 2
    type: ElementType = "rlc"

    class Params(Element.Params):
        r: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, r: AnyParamValue, **kwargs):
        return cls(name=name, params={"r": r}, **kwargs)

    def get_impedance(
        self, _from: str | None = None, _to: str | None = None, recursion_depth: int = 0
    ) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        return self.as_f(self.params.r)


class Inductor(Element):
    libname: LibName = "Inductor"
    num_of_ports: int = 2
    type: ElementType = "rlc"

    class Params(Element.Params):
        l: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, l: AnyParamValue, **kwargs):
        return cls(name=name, params={"l": l}, **kwargs)

    def get_impedance(
        self, _from: str | None = None, _to: str | None = None, recursion_depth: int = 0
    ) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        return self.jw * self.as_f(self.params.l)


class Capacitor(Element):
    libname: LibName = "Capacitor"
    num_of_ports: int = 2
    type: ElementType = "rlc"

    class Params(Element.Params):
        c: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, c: AnyParamValue, **kwargs):
        return cls(name=name, params={"c": c}, **kwargs)

    def get_admittance(
        self, _from: str | None = None, _to: str | None = None, recursion_depth: int = 0
    ) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        return self.jw * self.as_f(self.params.c)

class Conductor(Element):
    libname: LibName = "Conductor"
    num_of_ports: int = 2
    type: ElementType = "rlc"

    class Params(Element.Params):
        g: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, g: AnyParamValue, **kwargs):
        return cls(name=name, params={"g": g}, **kwargs)

    def get_admittance(
        self, _from: str | None = None, _to: str | None = None, recursion_depth: int = 0
    ) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        return self.as_f(self.params.g)


class VoltageSource(Element):
    libname: LibName = "VoltageSource"
    num_of_ports: int = 2
    type: ElementType = "voltage_source"

    class Params(Element.Params):
        v: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, v: AnyParamValue, **kwargs):
        return cls(name=name, params={"v": v}, **kwargs)

    def get_voltage(self, _from: str, _to: str) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        return self.as_f(self.params.v) * self.get_voltage_direction(_from, _to)


class CurrentSource(Element):
    libname: LibName = "CurrentSource"
    num_of_ports: int = 2
    type: ElementType = "current_source"

    class Params(Element.Params):
        i: AnyParamValue = Field(default=None)

    @classmethod
    def new(cls, name: str, i: AnyParamValue, **kwargs):
        return cls(name=name, params={"i": i}, **kwargs)

    def get_current(self, _from: str, _to: str) -> AnyValueArray:
        assert isinstance(
            self.params, self.Params
        ), "params need to be of type cls.Params"
        return self.as_f(self.params.i) * self.get_current_direction(_from, _to)
