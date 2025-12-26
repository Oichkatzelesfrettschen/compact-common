"""
Schwarzschild spacetime functions.

The Schwarzschild metric describes spacetime around a non-rotating,
uncharged, spherically symmetric mass:

ds^2 = -(1-r_s/r)c^2 dt^2 + (1-r_s/r)^{-1} dr^2 + r^2 dOmega^2

where r_s = 2GM/c^2 is the Schwarzschild radius.
"""

import numpy as np

from compact_common.constants import G, C, M_SUN


def schwarzschild_radius(mass: float) -> float:
    """
    Compute Schwarzschild radius.

    Parameters
    ----------
    mass : float
        Mass [g]

    Returns
    -------
    float
        Schwarzschild radius r_s = 2GM/c^2 [cm]
    """
    return 2 * G * mass / C**2


def schwarzschild_g_tt(r: float, mass: float) -> float:
    """
    Schwarzschild metric component g_tt.

    Parameters
    ----------
    r : float
        Radial coordinate [cm]
    mass : float
        Mass [g]

    Returns
    -------
    float
        g_tt = -(1 - r_s/r) * c^2
    """
    r_s = schwarzschild_radius(mass)
    if r <= r_s:
        return np.nan
    return -(1 - r_s / r) * C**2


def schwarzschild_g_rr(r: float, mass: float) -> float:
    """
    Schwarzschild metric component g_rr.

    Parameters
    ----------
    r : float
        Radial coordinate [cm]
    mass : float
        Mass [g]

    Returns
    -------
    float
        g_rr = (1 - r_s/r)^{-1}
    """
    r_s = schwarzschild_radius(mass)
    if r <= r_s:
        return np.nan
    return 1 / (1 - r_s / r)


def isco_radius(mass: float, prograde: bool = True) -> float:
    """
    Innermost Stable Circular Orbit (ISCO) radius.

    For Schwarzschild: r_ISCO = 6 GM/c^2 = 3 r_s

    Parameters
    ----------
    mass : float
        Mass [g]
    prograde : bool
        Ignored for Schwarzschild (no spin)

    Returns
    -------
    float
        ISCO radius [cm]
    """
    return 3 * schwarzschild_radius(mass)


def photon_sphere_radius(mass: float) -> float:
    """
    Photon sphere radius where circular null orbits exist.

    For Schwarzschild: r_ph = 3 GM/c^2 = 1.5 r_s

    Parameters
    ----------
    mass : float
        Mass [g]

    Returns
    -------
    float
        Photon sphere radius [cm]
    """
    return 1.5 * schwarzschild_radius(mass)


def gravitational_redshift(r: float, mass: float) -> float:
    """
    Gravitational redshift factor at radius r.

    z = 1/sqrt(1 - r_s/r) - 1

    Parameters
    ----------
    r : float
        Radial coordinate [cm]
    mass : float
        Mass [g]

    Returns
    -------
    float
        Redshift z
    """
    r_s = schwarzschild_radius(mass)
    if r <= r_s:
        return np.inf
    return 1 / np.sqrt(1 - r_s / r) - 1


def surface_gravity(r: float, mass: float) -> float:
    """
    Surface gravity at radius r.

    kappa = GM / (r^2 * sqrt(1 - r_s/r))

    Parameters
    ----------
    r : float
        Radial coordinate [cm]
    mass : float
        Mass [g]

    Returns
    -------
    float
        Surface gravity [cm/s^2]
    """
    r_s = schwarzschild_radius(mass)
    if r <= r_s:
        return np.inf
    return G * mass / (r**2 * np.sqrt(1 - r_s / r))


def proper_distance(r1: float, r2: float, mass: float, n_steps: int = 1000) -> float:
    """
    Proper radial distance between two radii.

    L = integral from r1 to r2 of dr / sqrt(1 - r_s/r)

    Parameters
    ----------
    r1 : float
        Inner radius [cm]
    r2 : float
        Outer radius [cm]
    mass : float
        Mass [g]
    n_steps : int
        Number of integration steps

    Returns
    -------
    float
        Proper distance [cm]
    """
    r_s = schwarzschild_radius(mass)

    if r1 <= r_s or r2 <= r_s:
        return np.inf

    r_values = np.linspace(r1, r2, n_steps)
    integrand = 1 / np.sqrt(1 - r_s / r_values)
    dr = r_values[1] - r_values[0]

    return np.trapz(integrand, dx=dr)


def orbital_period(r: float, mass: float) -> float:
    """
    Coordinate orbital period for circular orbit.

    T = 2*pi * sqrt(r^3 / (GM))

    Parameters
    ----------
    r : float
        Orbital radius [cm]
    mass : float
        Mass [g]

    Returns
    -------
    float
        Orbital period [s]
    """
    if r <= schwarzschild_radius(mass):
        return np.nan
    return 2 * np.pi * np.sqrt(r**3 / (G * mass))


def orbital_velocity(r: float, mass: float) -> float:
    """
    Orbital velocity for circular orbit.

    v = sqrt(GM / r) (Newtonian approximation)

    Parameters
    ----------
    r : float
        Orbital radius [cm]
    mass : float
        Mass [g]

    Returns
    -------
    float
        Orbital velocity [cm/s]
    """
    r_s = schwarzschild_radius(mass)
    if r <= r_s:
        return np.nan
    return np.sqrt(G * mass / r)


def escape_velocity(r: float, mass: float) -> float:
    """
    Escape velocity at radius r.

    v_esc = sqrt(2GM/r) = c * sqrt(r_s/r)

    Parameters
    ----------
    r : float
        Radial coordinate [cm]
    mass : float
        Mass [g]

    Returns
    -------
    float
        Escape velocity [cm/s]
    """
    r_s = schwarzschild_radius(mass)
    if r <= r_s:
        return C
    return C * np.sqrt(r_s / r)
