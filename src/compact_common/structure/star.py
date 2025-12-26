"""
Star model containers and mass-radius utilities.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import interpolate

from compact_common.constants import M_SUN


@dataclass
class StarModel:
    """
    Container for stellar model data.

    Can be constructed from TOVResult or directly from profile data.
    """

    mass: float  # g
    radius: float  # cm
    central_density: float  # g/cm^3

    # Radial profile arrays
    r: Optional[np.ndarray] = None  # cm
    m: Optional[np.ndarray] = None  # g (enclosed mass)
    P: Optional[np.ndarray] = None  # dynes/cm^2
    eps: Optional[np.ndarray] = None  # erg/cm^3
    rho: Optional[np.ndarray] = None  # g/cm^3

    # Derived quantities
    compactness: Optional[float] = None
    surface_redshift: Optional[float] = None
    moment_of_inertia: Optional[float] = None
    tidal_deformability: Optional[float] = None

    @property
    def mass_solar(self) -> float:
        """Mass in solar masses."""
        return self.mass / M_SUN

    @property
    def radius_km(self) -> float:
        """Radius in kilometers."""
        return self.radius / 1e5

    def pressure_at_radius(self, r: float) -> float:
        """Interpolate pressure at given radius."""
        if self.r is None or self.P is None:
            raise ValueError("No profile data available")
        interp = interpolate.interp1d(self.r, self.P, bounds_error=False, fill_value=0)
        return float(interp(r))

    def mass_at_radius(self, r: float) -> float:
        """Interpolate enclosed mass at given radius."""
        if self.r is None or self.m is None:
            raise ValueError("No profile data available")
        interp = interpolate.interp1d(self.r, self.m, bounds_error=False, fill_value=self.mass)
        return float(interp(r))

    @classmethod
    def from_tov_result(cls, result) -> "StarModel":
        """Create StarModel from TOVResult."""
        profile = result.profile or {}
        return cls(
            mass=result.mass,
            radius=result.radius,
            central_density=result.central_density,
            r=profile.get('r'),
            m=profile.get('m'),
            P=profile.get('P'),
            eps=profile.get('eps'),
            compactness=result.compactness,
            surface_redshift=result.surface_redshift,
            moment_of_inertia=result.moment_of_inertia,
            tidal_deformability=result.tidal_deformability,
        )


@dataclass
class MassRadiusCurve:
    """
    Mass-radius relation for a given EOS.

    Contains a sequence of stellar models and provides
    interpolation and analysis utilities.
    """

    masses: np.ndarray  # g
    radii: np.ndarray  # cm
    central_densities: np.ndarray  # g/cm^3
    compactnesses: Optional[np.ndarray] = None
    eos_name: str = "unknown"

    @property
    def masses_solar(self) -> np.ndarray:
        """Masses in solar units."""
        return self.masses / M_SUN

    @property
    def radii_km(self) -> np.ndarray:
        """Radii in km."""
        return self.radii / 1e5

    @property
    def max_mass(self) -> float:
        """Maximum mass in solar masses."""
        return np.max(self.masses) / M_SUN

    @property
    def max_mass_radius(self) -> float:
        """Radius at maximum mass in km."""
        idx = np.argmax(self.masses)
        return self.radii[idx] / 1e5

    @property
    def max_mass_density(self) -> float:
        """Central density at maximum mass in g/cm^3."""
        idx = np.argmax(self.masses)
        return self.central_densities[idx]

    def radius_at_mass(self, mass_solar: float) -> float:
        """
        Interpolate radius for given mass.

        Parameters
        ----------
        mass_solar : float
            Mass in solar masses

        Returns
        -------
        float
            Radius in km
        """
        mass = mass_solar * M_SUN

        # Find stable branch (before maximum mass)
        idx_max = np.argmax(self.masses)
        stable_masses = self.masses[:idx_max + 1]
        stable_radii = self.radii[:idx_max + 1]

        if mass > np.max(stable_masses):
            raise ValueError(f"Mass {mass_solar:.2f} M_sun exceeds maximum mass")

        interp = interpolate.interp1d(stable_masses, stable_radii)
        return float(interp(mass)) / 1e5

    def is_stable(self) -> np.ndarray:
        """
        Determine stability of each configuration.

        Stable configurations satisfy dM/d(rho_c) > 0.

        Returns
        -------
        ndarray
            Boolean array indicating stability
        """
        dM = np.gradient(self.masses)
        drho = np.gradient(self.central_densities)
        return dM / drho > 0

    def canonical_radius(self) -> float:
        """Radius of 1.4 M_sun star in km."""
        try:
            return self.radius_at_mass(1.4)
        except ValueError:
            return np.nan

    @classmethod
    def from_solver(cls, solver, n_points: int = 50, **kwargs) -> "MassRadiusCurve":
        """
        Generate M-R curve from TOV solver.

        Parameters
        ----------
        solver : TOVSolver
            Configured TOV solver
        n_points : int
            Number of stellar models

        Returns
        -------
        MassRadiusCurve
            M-R relation
        """
        masses, radii, results = solver.mass_radius_curve(n_points=n_points, **kwargs)

        central_densities = np.array([r.central_density for r in results])
        compactnesses = np.array([r.compactness for r in results])

        return cls(
            masses=masses,
            radii=radii,
            central_densities=central_densities,
            compactnesses=compactnesses,
            eos_name=solver.eos.name,
        )
