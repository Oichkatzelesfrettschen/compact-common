"""
Spacetime metrics and geodesic calculations.

This module provides:

- Schwarzschild metric functions
- Kerr metric functions
- Geodesic integration
- ISCO and photon sphere calculations
- Redshift calculations
"""

from compact_common.spacetime.geodesics import (
    integrate_geodesic,
    null_geodesic,
    timelike_geodesic,
)
from compact_common.spacetime.kerr import (
    frame_dragging,
    kerr_isco,
    kerr_photon_orbit,
)
from compact_common.spacetime.schwarzschild import (
    gravitational_redshift,
    isco_radius,
    photon_sphere_radius,
    proper_distance,
    schwarzschild_g_rr,
    schwarzschild_g_tt,
)

__all__ = [
    "schwarzschild_g_tt",
    "schwarzschild_g_rr",
    "isco_radius",
    "photon_sphere_radius",
    "gravitational_redshift",
    "proper_distance",
    "kerr_isco",
    "kerr_photon_orbit",
    "frame_dragging",
    "integrate_geodesic",
    "null_geodesic",
    "timelike_geodesic",
]
