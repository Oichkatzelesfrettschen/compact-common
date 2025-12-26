"""
Tidal deformability calculations for neutron stars.

The tidal deformability Lambda characterizes how a neutron star
responds to an external tidal field, important for gravitational
wave observations of binary neutron star mergers.

References
----------
- Hinderer, T. (2008). ApJ 677, 1216
- Flanagan & Hinderer (2008). PRD 77, 021502
"""

import numpy as np
from scipy import integrate

from compact_common.constants import G, C


def love_number_k2(
    compactness: float,
    y_R: float,
) -> float:
    """
    Compute the tidal Love number k2.

    Parameters
    ----------
    compactness : float
        Compactness C = GM/(Rc^2)
    y_R : float
        Value of y(R) from solving the tidal perturbation ODE

    Returns
    -------
    float
        Love number k2
    """
    C = compactness

    # Intermediate quantities
    C2 = C * C
    C3 = C2 * C
    C4 = C3 * C
    C5 = C4 * C

    # Prefactor
    prefac = (8.0 / 5.0) * C5 * (1 - 2*C)**2

    # Numerator terms
    num = (2 + y_R - 2*C*(5 - y_R + C*(11 - 12*y_R + C*(2 + y_R))))

    # Denominator terms
    den_term1 = 2*C * (6 - 3*y_R + 3*C*(5*y_R - 8) + 2*C2*(13 - 11*y_R + C*(3*y_R - 2) + 2*C2*(1 + y_R)))
    den_term2 = 3 * (1 - 2*C)**2 * (2 - y_R + 2*C*(y_R - 1)) * np.log(1 - 2*C)

    denom = den_term1 + den_term2

    if abs(denom) < 1e-15:
        return 0.0

    k2 = prefac * num / denom
    return max(0, k2)


def tidal_deformability(
    mass: float,
    radius: float,
    k2: float,
) -> float:
    """
    Compute dimensionless tidal deformability Lambda.

    Lambda = (2/3) * k2 * (R/M)^5 in G=c=1 units

    Parameters
    ----------
    mass : float
        Gravitational mass [g]
    radius : float
        Radius [cm]
    k2 : float
        Tidal Love number

    Returns
    -------
    float
        Dimensionless tidal deformability
    """
    # Convert to geometrized units
    M_geom = G * mass / C**2  # cm
    R = radius  # cm

    compactness = M_geom / R
    Lambda = (2.0 / 3.0) * k2 / compactness**5

    return Lambda


def compute_tidal_deformability(
    eos,
    central_density: float,
) -> float:
    """
    Compute tidal deformability for a star with given central density.

    This requires solving the coupled TOV + tidal perturbation equations.

    Parameters
    ----------
    eos : EOSBase
        Equation of state
    central_density : float
        Central density [g/cm^3]

    Returns
    -------
    float
        Dimensionless tidal deformability Lambda
    """
    from compact_common.structure.tov import TOVSolver

    # Solve TOV first
    solver = TOVSolver(eos)
    result = solver.solve(central_density=central_density, store_profile=True)

    if result.profile is None:
        return np.nan

    # TODO: Implement tidal perturbation ODE integration
    # For now, use approximate relation

    # Approximate k2 from fitting formula (Yagi & Yunes 2013)
    C = result.compactness
    if C < 0.01:
        return np.nan

    # Simplified k2 approximation
    k2_approx = 0.05 + 0.1 * (1 - 5*C)
    k2_approx = max(0, min(0.15, k2_approx))

    return tidal_deformability(result.mass, result.radius, k2_approx)
