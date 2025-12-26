"""
Fermi gas equation of state models.

Models for degenerate relativistic and non-relativistic Fermi gases,
useful as building blocks for more complex EOS.
"""

import numpy as np
from scipy import integrate

from compact_common.constants import (
    C,
    HBAR,
    HBARC,
    M_NEUTRON,
    M_NEUTRON_MEV,
    PI,
    PI2,
    FM_TO_CM,
)
from compact_common.eos.base import EOSBase


class FermiGas(EOSBase):
    """
    Non-relativistic degenerate Fermi gas.

    Valid when k_F << m*c (Fermi momentum much less than rest mass).

    Parameters
    ----------
    mass : float
        Particle mass [g]
    degeneracy : int
        Spin degeneracy factor (default: 2 for neutrons)
    name : str
        Model name
    """

    def __init__(
        self, mass: float = M_NEUTRON, degeneracy: int = 2, name: str = "fermi_gas"
    ):
        super().__init__(name=name)
        self.mass = mass
        self.g = degeneracy

    def fermi_momentum(self, density: float) -> float:
        """
        Compute Fermi momentum from density.

        k_F = (6 * pi^2 * n / g)^(1/3) where n = rho / m
        """
        n = density / self.mass  # number density
        return (6 * PI2 * n / self.g) ** (1 / 3) * HBAR

    def pressure(self, density: float) -> float:
        """
        Non-relativistic Fermi pressure.

        P = (1/5) * (6*pi^2/g)^(2/3) * hbar^2 * n^(5/3) / m
        """
        n = density / self.mass
        prefactor = (1 / 5) * (6 * PI2 / self.g) ** (2 / 3)
        return prefactor * HBAR**2 * n ** (5 / 3) / self.mass

    def energy_density(self, density: float) -> float:
        """
        Non-relativistic energy density (including rest mass).

        epsilon = rho * c^2 + (3/5) * n * E_F
        """
        n = density / self.mass
        k_F = self.fermi_momentum(density)
        E_F = k_F**2 / (2 * self.mass)  # Non-relativistic kinetic energy
        return density * C**2 + (3 / 5) * n * E_F


class RelFermiGas(EOSBase):
    """
    Relativistic degenerate Fermi gas.

    Full relativistic treatment valid for all k_F / (m*c).
    Based on the implementation in Generation-EOS repository.

    Parameters
    ----------
    mass_mev : float
        Particle mass [MeV/c^2]
    degeneracy : int
        Spin degeneracy factor
    name : str
        Model name
    """

    def __init__(
        self,
        mass_mev: float = M_NEUTRON_MEV,
        degeneracy: int = 2,
        name: str = "rel_fermi_gas",
    ):
        super().__init__(name=name)
        self.M = mass_mev / HBARC  # Mass in fm^-1
        self.g = degeneracy

    def _fermi_momentum_fm(self, density: float) -> float:
        """Fermi momentum in fm^-1 from CGS density."""
        # Convert g/cm^3 to fm^-3
        n_fm3 = density / M_NEUTRON * FM_TO_CM**3
        # k_F = (6 * pi^2 * n / g)^(1/3)
        return (6 * PI2 * n_fm3 / self.g) ** (1 / 3)

    def _energy_integrand(self, k: float) -> float:
        """Integrand for energy density: sqrt(M^2 + k^2) * k^2"""
        return np.sqrt(self.M**2 + k**2) * k**2

    def _pressure_integrand(self, k: float) -> float:
        """Integrand for pressure: k^4 / sqrt(M^2 + k^2)"""
        return k**4 / np.sqrt(self.M**2 + k**2)

    def pressure(self, density: float) -> float:
        """
        Relativistic Fermi pressure.

        P = (1 / 3*pi^2) * integral_0^k_F [k^4 / sqrt(M^2 + k^2)] dk
        """
        k_F = self._fermi_momentum_fm(density)

        if k_F < 1e-10:
            return 0.0

        # Integrate in MeV/fm^3 units
        result, _ = integrate.quad(self._pressure_integrand, 0, k_F)
        P_mev_fm3 = self.g / (6 * PI2) * result * HBARC  # MeV/fm^3

        # Convert to CGS
        from compact_common.constants import MEV_FM3_TO_DYNES
        return P_mev_fm3 * MEV_FM3_TO_DYNES

    def energy_density(self, density: float) -> float:
        """
        Relativistic energy density.

        epsilon = (1 / pi^2) * integral_0^k_F [sqrt(M^2 + k^2) * k^2] dk
        """
        k_F = self._fermi_momentum_fm(density)

        if k_F < 1e-10:
            return density * C**2

        # Integrate in MeV/fm^3 units
        result, _ = integrate.quad(self._energy_integrand, 0, k_F)
        eps_mev_fm3 = self.g / (2 * PI2) * result * HBARC  # MeV/fm^3

        # Convert to CGS
        from compact_common.constants import MEV_FM3_TO_CGS
        return eps_mev_fm3 * MEV_FM3_TO_CGS

    def sound_speed(self, density: float) -> float:
        """
        Relativistic sound speed.

        For ultra-relativistic limit: c_s -> c/sqrt(3)
        """
        # Numerical derivative
        delta = density * 1e-6
        if delta < 1e-20:
            return 0.0

        dP = self.pressure(density + delta) - self.pressure(density - delta)
        deps = self.energy_density(density + delta) - self.energy_density(density - delta)

        if deps == 0:
            return 0.0

        cs2 = C**2 * dP / deps
        return np.sqrt(max(0, min(cs2, C**2)))


def free_neutron_eos() -> RelFermiGas:
    """
    Free neutron gas EOS.

    Simple reference model: non-interacting relativistic neutrons.
    """
    return RelFermiGas(mass_mev=M_NEUTRON_MEV, degeneracy=2, name="free_neutron")


def free_electron_eos() -> RelFermiGas:
    """
    Free electron gas EOS.

    Useful for white dwarf modeling.
    """
    from compact_common.constants import M_ELECTRON_MEV
    return RelFermiGas(mass_mev=M_ELECTRON_MEV, degeneracy=2, name="free_electron")
