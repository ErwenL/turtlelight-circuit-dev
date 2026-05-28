# eocircuit

Electro-optic circuit simulation and analysis framework.

## Installation

```bash
uv sync --package eocircuit
```

## Quick Start

```python
import numpy as np
from eocircuit.core import Port, PortDomain, FrequencyGrid, Network
from eocircuit.elec import Resistor, VoltageSource
from eocircuit.solvers import MNASolver

freq = FrequencyGrid(np.logspace(6, 12, 1000))
net = Network()
r1 = Resistor(name="R1", resistance=50.0)
vs = VoltageSource(name="V1", amplitude=1.0, frequency=1e9)
net.add_component(r1)
net.add_component(vs)
net.connect(vs.ports[0], r1.ports[0])
net.connect(vs.ports[1], r1.ports[1])

solver = MNASolver(net, freq)
result = solver.solve()
print(result.get_z_matrix())
```

## Features

- **Dual-domain simulation**: Separate electrical (MNA) and photonic (S-parameter) solvers
- **Electro-optic co-simulation**: Hybrid solver for mixed-domain networks
- **Rich component library**: RLCG, sources, waveguides, couplers, MZI, MRR, modulators, photodetectors
- **Pydantic v2 validation**: All parameters validated at creation time
- **Immutable components**: Thread-safe, predictable behavior
