"""
Kerr spacetime functions for rotating black holes.

The Kerr metric describes spacetime around a rotating, uncharged mass
characterized by mass M and spin parameter a = J/(Mc).
"""

import numpy as np

from compact_common.constants import C, G


def kerr_horizons(mass: float, spin_param: float) -> tuple:
    """
    Compute inner and outer horizons of Kerr black hole.

    r_+/- = GM/c^2 +/- sqrt((GM/c^2)^2 - a^2)

    Parameters
    ----------
    mass : float
        Mass [g]
    spin_param : float
        Spin parameter a = J/(Mc) [cm]

    Returns
    -------
    tuple
        (r_outer, r_inner) horizon radii [cm]
    """
    M_geom = G * mass / C**2  # Geometrized mass [cm]
    a = spin_param

    if abs(a) > M_geom:
        # Naked singularity
        return (float(np.nan), float(np.nan))

    discriminant = M_geom**2 - a**2
    r_outer = M_geom + np.sqrt(discriminant)
    r_inner = M_geom - np.sqrt(discriminant)

    return (float(r_outer), float(r_inner))


def kerr_isco(mass: float, spin_param: float, prograde: bool = True) -> float:
    """
    ISCO radius for Kerr black hole.

    Parameters
    ----------
    mass : float
        Mass [g]
    spin_param : float
        Spin parameter a [cm]
    prograde : bool
        True for prograde orbit, False for retrograde

    Returns
    -------
    float
        ISCO radius [cm]
    """
    M_geom = G * mass / C**2
    a = spin_param

    # Dimensionless spin
    a_star = a / M_geom

    if abs(a_star) > 1:
        return float(np.nan)

    # ISCO formula (Bardeen, Press, Teukolsky 1972)
    Z1 = 1 + (1 - a_star**2)**(1/3) * ((1 + a_star)**(1/3) + (1 - a_star)**(1/3))
    Z2 = np.sqrt(3 * a_star**2 + Z1**2)

    if prograde:
        r_isco = M_geom * (3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2)))
    else:
        r_isco = M_geom * (3 + Z2 + np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2)))

    return float(r_isco)


def kerr_photon_orbit(mass: float, spin_param: float, prograde: bool = True) -> float:
    """
    Photon orbit radius for Kerr black hole.

    Parameters
    ----------
    mass : float
        Mass [g]
    spin_param : float
        Spin parameter a [cm]
    prograde : bool
        True for prograde orbit, False for retrograde

    Returns
    -------
    float
        Photon orbit radius [cm]
    """
    M_geom = G * mass / C**2
    a = spin_param
    a_star = a / M_geom

    if abs(a_star) > 1:
        return float(np.nan)

    # Photon orbit formula
    if prograde:
        r_ph = 2 * M_geom * (1 + np.cos(2/3 * np.arccos(-a_star)))
    else:
        r_ph = 2 * M_geom * (1 + np.cos(2/3 * np.arccos(a_star)))

    return float(r_ph)


def frame_dragging(r: float, theta: float, mass: float, spin_param: float) -> float:
    """
    Frame-dragging angular velocity (omega).

    omega = -g_{t,phi} / g_{phi,phi}

    Parameters
    ----------
    r : float
        Boyer-Lindquist radial coordinate [cm]
    theta : float
        Polar angle [radians]
    mass : float
        Mass [g]
    spin_param : float
        Spin parameter a [cm]

    Returns
    -------
    float
        Frame-dragging angular velocity [rad/s]
    """
    M_geom = G * mass / C**2
    a = spin_param

    # Kerr metric functions
    Sigma = r**2 + a**2 * np.cos(theta)**2
    # Delta = r**2 - 2*M_geom*r + a**2

    # Frame-dragging (Lense-Thirring)
    omega = 2 * M_geom * a * r / (Sigma * (r**2 + a**2) + 2*M_geom*a**2*r*np.sin(theta)**2)
    omega *= C  # Convert to rad/s

    return float(omega)


def kerr_ergosphere(theta: float, mass: float, spin_param: float) -> float:
    """
    Ergosphere boundary radius.

    r_ergo = M + sqrt(M^2 - a^2 cos^2(theta))

    Parameters
    ----------
    theta : float
        Polar angle [radians]
    mass : float
        Mass [g]
    spin_param : float
        Spin parameter a [cm]

    Returns
    -------
    float
        Ergosphere radius [cm]
    """
    M_geom = G * mass / C**2
    a = spin_param

    return float(M_geom + np.sqrt(M_geom**2 - a**2 * np.cos(theta)**2))


def spin_from_dimensionless(mass: float, a_star: float) -> float:
    """
    Convert dimensionless spin to physical spin parameter.

    Parameters
    ----------
    mass : float
        Mass [g]
    a_star : float
        Dimensionless spin (0 to 1)

    Returns
    -------
    float
        Spin parameter a = a* * GM/c^2 [cm]
    """
    M_geom = G * mass / C**2
    return float(a_star * M_geom)
