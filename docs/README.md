# compact-common Documentation

Shared infrastructure for neutron star, black hole, and compact object physics.

## Modules

- `compact_common.eos`: Equation of State (EOS) models.
- `compact_common.structure`: Stellar structure solvers (TOV).
- `compact_common.spacetime`: General relativity metrics and utilities.
- `compact_common.constants`: Astrophysics constants.

## Quick Start

```python
from compact_common.eos import SigmaOmegaModel
from compact_common.structure import TOVSolver

model = SigmaOmegaModel(Gs=1.0, Gv=7.0)
e, p = model.compute_eos()
solver = TOVSolver(e, p)
r, m = solver.solve(central_density=1e15)
print(f"M = {m:.2f} M_sun, R = {r:.1f} km")
```
