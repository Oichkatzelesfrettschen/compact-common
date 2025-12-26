"""
Stellar structure calculations for compact objects.

This module provides:

- TOVSolver: Tolman-Oppenheimer-Volkoff equation integrator
- StarModel: Container for stellar structure profiles
- MassRadius: Mass-radius relation calculator
- TidalDeformability: Tidal deformability (Lambda) calculator

The TOV equations describe hydrostatic equilibrium in general relativity:
    dP/dr = -(G/c^2) * (eps + P) * (m + 4*pi*r^3*P/c^2) / (r * (r - 2*G*m/c^2))
    dm/dr = 4*pi*r^2 * eps / c^2
"""

from compact_common.structure.tov import TOVSolver, TOVResult
from compact_common.structure.star import StarModel, MassRadiusCurve
from compact_common.structure.tidal import tidal_deformability, love_number_k2

__all__ = [
    "TOVSolver",
    "TOVResult",
    "StarModel",
    "MassRadiusCurve",
    "tidal_deformability",
    "love_number_k2",
]
