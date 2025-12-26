"""
Spacetime metrics and geodesic calculations.

This module provides:

- Schwarzschild metric functions
- Kerr metric functions
- Geodesic integration
- ISCO and photon sphere calculations
- Redshift calculations
"""

from compact_common.spacetime.schwarzschild import (
    schwarzschild_g_tt,
    schwarzschild_g_rr,
    isco_radius,
    photon_sphere_radius,
    gravitational_redshift,
    proper_distance,
)
from compact_common.spacetime.kerr import (
    kerr_isco,
    kerr_photon_orbit,
    frame_dragging,
)
from compact_common.spacetime.geodesics import (
    integrate_geodesic,
    null_geodesic,
    timelike_geodesic,
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
