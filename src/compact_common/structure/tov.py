"""
Tolman-Oppenheimer-Volkoff (TOV) equation solver.

Integrates the relativistic hydrostatic equilibrium equations to
determine neutron star structure from a given equation of state.

References
----------
- Tolman, R. C. (1939). Phys. Rev. 55, 364
- Oppenheimer, J. R. & Volkoff, G. M. (1939). Phys. Rev. 55, 374
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import integrate, interpolate

from compact_common.constants import C, G, M_SUN, PI, FOUR_PI
from compact_common.eos.base import EOSBase

# Try numba for performance
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    HAS_NUMBA = False


@dataclass
class TOVResult:
    """
    Result of TOV integration for a single star.

    Attributes
    ----------
    mass : float
        Gravitational mass [g]
    radius : float
        Circumferential radius [cm]
    central_density : float
        Central mass density [g/cm^3]
    central_pressure : float
        Central pressure [dynes/cm^2]
    central_energy : float
        Central energy density [erg/cm^3]
    baryon_mass : float
        Total baryon mass [g]
    compactness : float
        Compactness parameter M/R (dimensionless in G=c=1)
    surface_redshift : float
        Gravitational redshift at surface
    moment_of_inertia : float, optional
        Moment of inertia [g cm^2] (if computed)
    tidal_deformability : float, optional
        Dimensionless tidal deformability Lambda
    profile : dict, optional
        Radial profiles {r, m, P, eps, rho}
    """

    mass: float
    radius: float
    central_density: float
    central_pressure: float
    central_energy: float
    baryon_mass: float = 0.0
    compactness: float = 0.0
    surface_redshift: float = 0.0
    moment_of_inertia: Optional[float] = None
    tidal_deformability: Optional[float] = None
    profile: Optional[dict] = None

    @property
    def mass_solar(self) -> float:
        """Mass in solar masses."""
        return self.mass / M_SUN

    @property
    def radius_km(self) -> float:
        """Radius in kilometers."""
        return self.radius / 1e5

    def __repr__(self):
        return (
            f"TOVResult(M={self.mass_solar:.3f} M_sun, "
            f"R={self.radius_km:.2f} km, "
            f"C={self.compactness:.3f})"
        )


class TOVSolver:
    """
    TOV equation solver.

    Integrates the relativistic structure equations from center to surface
    for a given equation of state.

    Parameters
    ----------
    eos : EOSBase
        Equation of state model
    surface_pressure_ratio : float
        Stop integration when P < P_c * surface_pressure_ratio
    max_radius : float
        Maximum integration radius [cm]
    n_points : int
        Number of radial grid points for output profile

    Examples
    --------
    >>> from compact_common.eos import Polytrope
    >>> from compact_common.structure import TOVSolver
    >>>
    >>> eos = Polytrope(K=1e13, gamma=2.0)
    >>> solver = TOVSolver(eos)
    >>> star = solver.solve(central_density=1e15)
    >>> print(f"M = {star.mass_solar:.2f} M_sun")
    """

    def __init__(
        self,
        eos: EOSBase,
        surface_pressure_ratio: float = 1e-10,
        max_radius: float = 5e6,  # 50 km
        n_points: int = 1000,
    ):
        self.eos = eos
        self.surface_pressure_ratio = surface_pressure_ratio
        self.max_radius = max_radius
        self.n_points = n_points

        # Build interpolation table from EOS
        self._build_eos_table()

    def _build_eos_table(self, n_table: int = 500):
        """Build interpolation table for EOS lookups during integration."""
        table = self.eos.to_table(n_points=n_table)
        self._table_eps = table.energy_density
        self._table_P = table.pressure
        self._table_rho = table.density

        # P -> eps interpolator (log-log)
        log_P = np.log10(np.maximum(self._table_P, 1e-100))
        log_eps = np.log10(np.maximum(self._table_eps, 1e-100))

        # Remove duplicates and sort
        unique_idx = np.unique(log_P, return_index=True)[1]
        log_P_unique = log_P[unique_idx]
        log_eps_unique = log_eps[unique_idx]

        self._eps_from_P = interpolate.interp1d(
            log_P_unique, log_eps_unique,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )

    def _energy_from_pressure(self, P: float) -> float:
        """Get energy density from pressure using interpolation."""
        if P <= 0:
            return self._table_eps[0]
        return 10 ** self._eps_from_P(np.log10(P))

    def _tov_equations(self, r: float, y: np.ndarray) -> np.ndarray:
        """
        TOV differential equations.

        y = [P, m] where P is pressure and m is enclosed mass.

        Returns dy/dr = [dP/dr, dm/dr]
        """
        P, m = y

        if P <= 0 or r <= 0:
            return np.array([0.0, 0.0])

        # Energy density from EOS
        eps = self._energy_from_pressure(P)

        # Schwarzschild factor
        factor = 1 - 2 * G * m / (r * C**2)
        if factor <= 0:
            return np.array([0.0, 0.0])

        # TOV equation: dP/dr
        dP_dr = -(G / C**2) * (eps + P) * (m + FOUR_PI * r**3 * P / C**2)
        dP_dr /= r * (r - 2 * G * m / C**2)

        # Mass equation: dm/dr
        dm_dr = FOUR_PI * r**2 * eps / C**2

        return np.array([dP_dr, dm_dr])

    def _surface_event(self, r: float, y: np.ndarray) -> float:
        """Event function to detect surface (P -> 0)."""
        return y[0] - self._P_surface

    def solve(
        self,
        central_density: Optional[float] = None,
        central_pressure: Optional[float] = None,
        store_profile: bool = True,
    ) -> TOVResult:
        """
        Solve TOV equations for a single star.

        Parameters
        ----------
        central_density : float, optional
            Central mass density [g/cm^3]
        central_pressure : float, optional
            Central pressure [dynes/cm^2]
        store_profile : bool
            Store radial profile in result

        Returns
        -------
        TOVResult
            Stellar structure result
        """
        # Get central conditions
        if central_density is not None:
            rho_c = central_density
            P_c = self.eos.pressure(rho_c)
            eps_c = self.eos.energy_density(rho_c)
        elif central_pressure is not None:
            P_c = central_pressure
            eps_c = self._energy_from_pressure(P_c)
            rho_c = eps_c / C**2  # Approximate
        else:
            raise ValueError("Must specify central_density or central_pressure")

        # Surface pressure threshold
        self._P_surface = P_c * self.surface_pressure_ratio

        # Initial conditions at small radius r0
        r0 = 1e2  # 1 meter, small but finite
        m0 = FOUR_PI * r0**3 * eps_c / (3 * C**2)

        # Set up integration
        y0 = np.array([P_c, m0])

        # Surface detection event
        self._surface_event.terminal = True
        self._surface_event.direction = -1

        # Integrate outward
        sol = integrate.solve_ivp(
            self._tov_equations,
            t_span=(r0, self.max_radius),
            y0=y0,
            method='RK45',
            events=self._surface_event,
            dense_output=True,
            max_step=self.max_radius / 100,
        )

        # Extract surface values
        if sol.t_events[0].size > 0:
            R = sol.t_events[0][0]
            P_surf, M = sol.y_events[0][0]
        else:
            R = sol.t[-1]
            P_surf, M = sol.y[:, -1]

        # Compactness
        compactness = G * M / (R * C**2)

        # Surface redshift
        if compactness < 0.5:
            z_surf = 1 / np.sqrt(1 - 2 * compactness) - 1
        else:
            z_surf = np.inf

        # Build profile if requested
        profile = None
        if store_profile:
            r_grid = np.linspace(r0, R, self.n_points)
            y_grid = sol.sol(r_grid)
            P_grid = y_grid[0]
            m_grid = y_grid[1]
            eps_grid = np.array([self._energy_from_pressure(P) for P in P_grid])

            profile = {
                'r': r_grid,
                'm': m_grid,
                'P': P_grid,
                'eps': eps_grid,
            }

        # Compute baryon mass (simplified)
        baryon_mass = M  # Approximate; proper calculation needs integral

        return TOVResult(
            mass=M,
            radius=R,
            central_density=rho_c,
            central_pressure=P_c,
            central_energy=eps_c,
            baryon_mass=baryon_mass,
            compactness=compactness,
            surface_redshift=z_surf,
            profile=profile,
        )

    def mass_radius_curve(
        self,
        rho_min: float = 1e14,
        rho_max: float = 3e15,
        n_points: int = 50,
        log_spacing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[TOVResult]]:
        """
        Compute mass-radius curve over a range of central densities.

        Parameters
        ----------
        rho_min : float
            Minimum central density [g/cm^3]
        rho_max : float
            Maximum central density [g/cm^3]
        n_points : int
            Number of stellar models
        log_spacing : bool
            Use logarithmic spacing in density

        Returns
        -------
        masses : ndarray
            Masses [g]
        radii : ndarray
            Radii [cm]
        results : list
            Full TOVResult for each model
        """
        if log_spacing:
            rho_c_values = np.logspace(np.log10(rho_min), np.log10(rho_max), n_points)
        else:
            rho_c_values = np.linspace(rho_min, rho_max, n_points)

        masses = []
        radii = []
        results = []

        for rho_c in rho_c_values:
            try:
                result = self.solve(central_density=rho_c, store_profile=False)
                masses.append(result.mass)
                radii.append(result.radius)
                results.append(result)
            except Exception:
                # Skip failed integrations
                continue

        return np.array(masses), np.array(radii), results

    def maximum_mass(
        self,
        rho_min: float = 5e14,
        rho_max: float = 5e15,
        tol: float = 1e-3,
    ) -> TOVResult:
        """
        Find the maximum mass configuration.

        Uses golden section search to find the central density
        that maximizes gravitational mass.

        Parameters
        ----------
        rho_min : float
            Lower bound for central density search [g/cm^3]
        rho_max : float
            Upper bound for central density search [g/cm^3]
        tol : float
            Relative tolerance for density search

        Returns
        -------
        TOVResult
            Maximum mass stellar model
        """
        from scipy.optimize import minimize_scalar

        def neg_mass(log_rho):
            rho = 10 ** log_rho
            try:
                result = self.solve(central_density=rho, store_profile=False)
                return -result.mass
            except Exception:
                return 0

        opt = minimize_scalar(
            neg_mass,
            bounds=(np.log10(rho_min), np.log10(rho_max)),
            method='bounded',
            options={'xatol': tol}
        )

        rho_max_mass = 10 ** opt.x
        return self.solve(central_density=rho_max_mass, store_profile=True)
