from ..circuit import Element
from .basic import (
    Resistor,
    Capacitor,
    Inductor,
    Conductor,
    VoltageSource,
    CurrentSource
)
from .composite import (
    Composite,
    MixMode,
    SubElement
)
from .f_domain import frequencyDependentLib
from .utils import (
    read_local_lib,
    LibManager
)

R = Resistor
C = Capacitor
L = Inductor
G = Conductor

LibManager.load_lib(read_local_lib(locals()))
LibManager.attach_lib("core", frequencyDependentLib)