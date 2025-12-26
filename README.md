# compact-common

Shared infrastructure for neutron star, black hole, and compact object physics.

## Overview

`compact-common` provides Python tools for modeling compact objects:

- **EOS Module**: Equation of state models (polytrope, Fermi gas, sigma-omega MFA, tabulated)
- **Structure Module**: TOV solver for stellar structure calculations
- **Spacetime Module**: Schwarzschild and Kerr metric functions, geodesic integration
- **Radiation Module**: Thermal and neutrino emission (planned)
- **Bayesian Module**: Parameter inference framework (planned)

## Installation

```bash
# Basic installation
pip install compact-common

# With all optional dependencies
pip install compact-common[all]

# Development installation
git clone https://github.com/Oichkatzelesfrettschen/compact-common.git
cd compact-common
pip install -e ".[dev]"
```

## Quick Start

### Compute neutron star mass-radius curve

```python
from compact_common.eos import Polytrope
from compact_common.structure import TOVSolver

# Create equation of state
eos = Polytrope(K=1e13, gamma=2.0)

# Solve TOV equations
solver = TOVSolver(eos)
star = solver.solve(central_density=1e15)

print(f"Mass: {star.mass_solar:.2f} M_sun")
print(f"Radius: {star.radius_km:.1f} km")
print(f"Compactness: {star.compactness:.3f}")

# Generate mass-radius curve
masses, radii, results = solver.mass_radius_curve(n_points=50)
```

### Sigma-Omega mean field EOS

```python
from compact_common.eos import SigmaOmegaModel

# Create MFA EOS with coupling constants
eos = SigmaOmegaModel(Gs=5.0, Gv=5.0)

# Get pressure at nuclear density
P = eos.pressure(2.8e14)  # dynes/cm^2
```

### Black hole spacetime calculations

```python
from compact_common.spacetime import isco_radius, photon_sphere_radius
from compact_common.constants import M_SUN

mass = 10 * M_SUN  # 10 solar mass black hole

r_isco = isco_radius(mass)
r_photon = photon_sphere_radius(mass)

print(f"ISCO: {r_isco / 1e5:.1f} km")
print(f"Photon sphere: {r_photon / 1e5:.1f} km")
```

## Modules

### EOS (Equation of State)

| Class | Description |
|-------|-------------|
| `Polytrope` | Simple P = K * rho^gamma |
| `PiecewisePolytrope` | Multi-segment polytrope |
| `FermiGas` | Non-relativistic degenerate gas |
| `RelFermiGas` | Relativistic Fermi gas |
| `SigmaOmegaModel` | Sigma-omega mean field approximation |
| `TabulatedEOS` | Interpolated tabular EOS |

### Structure

| Class/Function | Description |
|----------------|-------------|
| `TOVSolver` | Tolman-Oppenheimer-Volkoff integrator |
| `TOVResult` | Stellar structure result container |
| `StarModel` | Full stellar model with profiles |
| `MassRadiusCurve` | M-R relation utilities |
| `tidal_deformability` | Lambda calculation |

### Spacetime

| Function | Description |
|----------|-------------|
| `schwarzschild_radius` | r_s = 2GM/c^2 |
| `isco_radius` | Innermost stable circular orbit |
| `photon_sphere_radius` | Circular photon orbit |
| `gravitational_redshift` | Surface redshift |
| `kerr_isco` | ISCO for rotating BH |
| `frame_dragging` | Lense-Thirring effect |
| `integrate_geodesic` | Particle/photon trajectory |

## Related Projects

This library is part of the [OpenUniverse](https://github.com/Oichkatzelesfrettschen/openuniverse) collection:

- [grb-common](https://github.com/Oichkatzelesfrettschen/grb-common): GRB afterglow analysis
- [spandrel-core](https://github.com/Oichkatzelesfrettschen/spandrel-core): Cosmological calculations
- [CompactStar](https://github.com/Oichkatzelesfrettschen/CompactStar): Full C++ neutron star framework

## References

- Tolman, R. C. (1939). Phys. Rev. 55, 364
- Oppenheimer, J. R. & Volkoff, G. M. (1939). Phys. Rev. 55, 374
- Walecka, J. D. (1974). Ann. Phys. 83, 491
- Hinderer, T. (2008). ApJ 677, 1216

## License

GPL-3.0-or-later

## Citation

```bibtex
@software{compact_common,
  author = {Afrauthihinngreygaard, Deirikr Jaiusadastra},
  title = {compact-common: Neutron star and black hole physics tools},
  year = {2025},
  url = {https://github.com/Oichkatzelesfrettschen/compact-common}
}
```
