"""
Base class for Equation of State models.

All EOS implementations should inherit from EOSBase and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from compact_common.constants import C, RHO_NUC


@dataclass
class EOSPoint:
    """Single point on the EOS curve."""

    density: float  # g/cm^3
    pressure: float  # dynes/cm^2
    energy_density: float  # erg/cm^3
    enthalpy: Optional[float] = None  # cm^2/s^2
    baryon_density: Optional[float] = None  # cm^-3
    sound_speed: Optional[float] = None  # cm/s


@dataclass
class EOSTable:
    """Tabulated EOS with arrays of thermodynamic quantities."""

    density: np.ndarray  # g/cm^3
    pressure: np.ndarray  # dynes/cm^2
    energy_density: np.ndarray  # erg/cm^3
    enthalpy: Optional[np.ndarray] = None  # cm^2/s^2
    baryon_density: Optional[np.ndarray] = None  # cm^-3
    sound_speed: Optional[np.ndarray] = None  # cm/s

    def __len__(self):
        return len(self.density)

    def save(self, filepath: str, format: str = "rns"):
        """
        Save EOS table to file.

        Parameters
        ----------
        filepath : str
            Output file path
        format : str
            Output format: 'rns' (4-column) or 'compose' (CompOSE format)
        """
        if format == "rns":
            # RNS format: energy_density, pressure, enthalpy, baryon_density
            # All in log10
            data = np.column_stack([
                np.log10(self.energy_density),
                np.log10(self.pressure),
                np.log10(self.enthalpy) if self.enthalpy is not None else np.zeros_like(self.density),
                np.log10(self.baryon_density) if self.baryon_density is not None else np.zeros_like(self.density),
            ])
            header = "# EOS table (log10): energy_density pressure enthalpy baryon_density"
        else:
            # Simple 2-column format
            data = np.column_stack([self.energy_density, self.pressure])
            header = "# EOS table: energy_density(erg/cm^3) pressure(dynes/cm^2)"

        np.savetxt(filepath, data, header=header)


class EOSBase(ABC):
    """
    Abstract base class for Equation of State models.

    Subclasses must implement:
        - pressure(density)
        - energy_density(density)

    Optional overrides:
        - sound_speed(density)
        - enthalpy(density)
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name

    @abstractmethod
    def pressure(self, density: float) -> float:
        """
        Compute pressure from mass density.

        Parameters
        ----------
        density : float
            Mass density [g/cm^3]

        Returns
        -------
        float
            Pressure [dynes/cm^2]
        """
        pass

    @abstractmethod
    def energy_density(self, density: float) -> float:
        """
        Compute energy density from mass density.

        Parameters
        ----------
        density : float
            Mass density [g/cm^3]

        Returns
        -------
        float
            Energy density [erg/cm^3]
        """
        pass

    def sound_speed(self, density: float) -> float:
        """
        Compute adiabatic sound speed.

        Default implementation uses numerical derivative.
        Override for analytic expressions.

        Parameters
        ----------
        density : float
            Mass density [g/cm^3]

        Returns
        -------
        float
            Sound speed [cm/s]
        """
        # c_s^2 = dP/d(epsilon)
        delta = density * 1e-6
        dP = self.pressure(density + delta) - self.pressure(density - delta)
        deps = self.energy_density(density + delta) - self.energy_density(density - delta)

        if deps == 0:
            return 0.0

        cs2 = C**2 * dP / deps
        return np.sqrt(max(0, cs2))

    def enthalpy(self, density: float) -> float:
        """
        Compute specific enthalpy h = (epsilon + P) / rho.

        Parameters
        ----------
        density : float
            Mass density [g/cm^3]

        Returns
        -------
        float
            Specific enthalpy [cm^2/s^2]
        """
        eps = self.energy_density(density)
        P = self.pressure(density)
        return (eps + P) / density

    def baryon_density(self, density: float) -> float:
        """
        Compute baryon number density.

        Default assumes n_B = rho / m_B where m_B is nucleon mass.
        Override for detailed composition.

        Parameters
        ----------
        density : float
            Mass density [g/cm^3]

        Returns
        -------
        float
            Baryon number density [cm^-3]
        """
        from compact_common.constants import M_NEUTRON
        return density / M_NEUTRON

    def point(self, density: float) -> EOSPoint:
        """Get all EOS quantities at a given density."""
        return EOSPoint(
            density=density,
            pressure=self.pressure(density),
            energy_density=self.energy_density(density),
            enthalpy=self.enthalpy(density),
            baryon_density=self.baryon_density(density),
            sound_speed=self.sound_speed(density),
        )

    def to_table(
        self,
        n_points: int = 200,
        rho_min: float = 1e10,
        rho_max: float = 1e16,
        log_spacing: bool = True,
    ) -> EOSTable:
        """
        Generate tabulated EOS.

        Parameters
        ----------
        n_points : int
            Number of table points
        rho_min : float
            Minimum density [g/cm^3]
        rho_max : float
            Maximum density [g/cm^3]
        log_spacing : bool
            Use logarithmic spacing (recommended)

        Returns
        -------
        EOSTable
            Tabulated EOS data
        """
        if log_spacing:
            densities = np.logspace(np.log10(rho_min), np.log10(rho_max), n_points)
        else:
            densities = np.linspace(rho_min, rho_max, n_points)

        pressures = np.array([self.pressure(rho) for rho in densities])
        energies = np.array([self.energy_density(rho) for rho in densities])
        enthalpies = np.array([self.enthalpy(rho) for rho in densities])
        baryons = np.array([self.baryon_density(rho) for rho in densities])
        sounds = np.array([self.sound_speed(rho) for rho in densities])

        return EOSTable(
            density=densities,
            pressure=pressures,
            energy_density=energies,
            enthalpy=enthalpies,
            baryon_density=baryons,
            sound_speed=sounds,
        )

    def is_causal(self, density: float) -> bool:
        """Check if EOS satisfies causality (c_s < c)."""
        cs = self.sound_speed(density)
        return cs < C

    def is_stable(self, density: float) -> bool:
        """Check thermodynamic stability (dP/drho > 0)."""
        delta = density * 1e-6
        dP = self.pressure(density + delta) - self.pressure(density - delta)
        return dP > 0

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
