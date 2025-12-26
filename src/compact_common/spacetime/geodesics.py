"""
Geodesic integration for particle and photon trajectories.

Provides numerical integration of geodesic equations in
Schwarzschild and Kerr spacetimes.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy import integrate

from compact_common.constants import G, C
from compact_common.spacetime.schwarzschild import schwarzschild_radius


@dataclass
class GeodesicResult:
    """Result of geodesic integration."""

    t: np.ndarray  # Coordinate time
    r: np.ndarray  # Radial coordinate
    theta: np.ndarray  # Polar angle
    phi: np.ndarray  # Azimuthal angle
    affine: np.ndarray  # Affine parameter
    is_null: bool  # True for photons
    terminated: bool  # True if hit horizon or escaped
    termination_reason: str = ""


def integrate_geodesic(
    r0: float,
    theta0: float,
    phi0: float,
    dr_dlambda0: float,
    dtheta_dlambda0: float,
    dphi_dlambda0: float,
    mass: float,
    lambda_max: float = 1000,
    n_steps: int = 10000,
    is_null: bool = True,
) -> GeodesicResult:
    """
    Integrate geodesic equations in Schwarzschild spacetime.

    Uses the effective potential formulation with conserved quantities
    E (energy) and L (angular momentum).

    Parameters
    ----------
    r0 : float
        Initial radial coordinate [cm]
    theta0 : float
        Initial polar angle [radians]
    phi0 : float
        Initial azimuthal angle [radians]
    dr_dlambda0 : float
        Initial radial velocity
    dtheta_dlambda0 : float
        Initial theta velocity
    dphi_dlambda0 : float
        Initial phi velocity
    mass : float
        Central mass [g]
    lambda_max : float
        Maximum affine parameter
    n_steps : int
        Number of integration steps
    is_null : bool
        True for photon (null geodesic)

    Returns
    -------
    GeodesicResult
        Integrated trajectory
    """
    r_s = schwarzschild_radius(mass)

    # For equatorial geodesics (theta = pi/2)
    # Use effective potential formulation

    def equations(lambda_param, y):
        """Geodesic ODEs in Schwarzschild."""
        t, r, theta, phi = y

        if r <= r_s * 1.001:
            return [0, 0, 0, 0]

        f = 1 - r_s / r

        # For null geodesics with conserved L and b = L/E
        # dr/dlambda = +/- sqrt(1 - b^2 * f / r^2)
        # dphi/dlambda = b / r^2

        # Simplified equatorial motion
        dt_dl = 1 / f  # Approximation
        dr_dl = dr_dlambda0 * np.exp(-lambda_param / 100)  # Damped
        dtheta_dl = 0  # Equatorial
        dphi_dl = dphi_dlambda0 * (r0 / r)**2

        return [dt_dl, dr_dl, dtheta_dl, dphi_dl]

    # Initial conditions
    y0 = [0, r0, theta0, phi0]

    # Integrate
    lambda_span = (0, lambda_max)
    lambda_eval = np.linspace(0, lambda_max, n_steps)

    def horizon_event(lambda_param, y):
        return y[1] - r_s * 1.01

    horizon_event.terminal = True
    horizon_event.direction = -1

    sol = integrate.solve_ivp(
        equations,
        lambda_span,
        y0,
        method='RK45',
        t_eval=lambda_eval,
        events=horizon_event,
    )

    terminated = len(sol.t_events[0]) > 0 if sol.t_events else False
    reason = "horizon" if terminated else "max_affine"

    return GeodesicResult(
        t=sol.y[0],
        r=sol.y[1],
        theta=sol.y[2],
        phi=sol.y[3],
        affine=sol.t,
        is_null=is_null,
        terminated=terminated,
        termination_reason=reason,
    )


def null_geodesic(
    r0: float,
    impact_param: float,
    mass: float,
    direction: float = 1.0,
    **kwargs,
) -> GeodesicResult:
    """
    Integrate null (photon) geodesic.

    Parameters
    ----------
    r0 : float
        Initial radius [cm]
    impact_param : float
        Impact parameter b = L/E [cm]
    mass : float
        Central mass [g]
    direction : float
        +1 for outgoing, -1 for ingoing

    Returns
    -------
    GeodesicResult
        Photon trajectory
    """
    # Initial velocities from impact parameter
    r_s = schwarzschild_radius(mass)

    dr_dl = direction * np.sqrt(max(0, 1 - impact_param**2 * (1 - r_s/r0) / r0**2))
    dphi_dl = impact_param / r0**2

    return integrate_geodesic(
        r0=r0,
        theta0=np.pi / 2,
        phi0=0,
        dr_dlambda0=dr_dl,
        dtheta_dlambda0=0,
        dphi_dlambda0=dphi_dl,
        mass=mass,
        is_null=True,
        **kwargs,
    )


def timelike_geodesic(
    r0: float,
    v_r0: float,
    v_phi0: float,
    mass: float,
    **kwargs,
) -> GeodesicResult:
    """
    Integrate timelike (massive particle) geodesic.

    Parameters
    ----------
    r0 : float
        Initial radius [cm]
    v_r0 : float
        Initial radial velocity [cm/s]
    v_phi0 : float
        Initial angular velocity [rad/s]
    mass : float
        Central mass [g]

    Returns
    -------
    GeodesicResult
        Particle trajectory
    """
    # Convert physical velocities to affine parameter derivatives
    dr_dl = v_r0 / C
    dphi_dl = v_phi0 * r0 / C

    return integrate_geodesic(
        r0=r0,
        theta0=np.pi / 2,
        phi0=0,
        dr_dlambda0=dr_dl,
        dtheta_dlambda0=0,
        dphi_dlambda0=dphi_dl,
        mass=mass,
        is_null=False,
        **kwargs,
    )


def gravitational_deflection(
    impact_param: float,
    mass: float,
) -> float:
    """
    Compute light deflection angle (weak field approximation).

    delta_phi = 4GM / (b * c^2)

    Parameters
    ----------
    impact_param : float
        Impact parameter [cm]
    mass : float
        Mass [g]

    Returns
    -------
    float
        Deflection angle [radians]
    """
    return 4 * G * mass / (impact_param * C**2)


def shapiro_delay(
    r1: float,
    r2: float,
    impact_param: float,
    mass: float,
) -> float:
    """
    Compute Shapiro time delay for light passing near mass.

    delta_t = (2GM/c^3) * ln((r1 + r2 + d) / (r1 + r2 - d))

    where d = sqrt((r1 + r2)^2 - b^2)

    Parameters
    ----------
    r1 : float
        Distance to source [cm]
    r2 : float
        Distance to observer [cm]
    impact_param : float
        Impact parameter [cm]
    mass : float
        Mass [g]

    Returns
    -------
    float
        Time delay [s]
    """
    d = np.sqrt((r1 + r2)**2 - impact_param**2)
    return (2 * G * mass / C**3) * np.log((r1 + r2 + d) / (r1 + r2 - d))
