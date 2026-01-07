"""
Polytropic equation of state models.

Polytropes are simple analytic EOS models useful for testing and
understanding scaling relations in compact stars.
"""

from typing import List

import numpy as np

from compact_common.constants import C
from compact_common.eos.base import EOSBase


class Polytrope(EOSBase):
    """
    Single polytropic EOS: P = K * rho^gamma.

    The energy density includes rest mass:
        epsilon = rho * c^2 + P / (gamma - 1)

    Parameters
    ----------
    K : float
        Polytropic constant [cgs]
    gamma : float
        Adiabatic index (gamma > 1)
    name : str
        Model name

    Examples
    --------
    >>> eos = Polytrope(K=1e13, gamma=2.0)
    >>> P = eos.pressure(1e15)  # Pressure at 10^15 g/cm^3
    """

    def __init__(self, K: float, gamma: float, name: str = "polytrope"):
        super().__init__(name=name)
        if gamma <= 1:
            raise ValueError("Adiabatic index gamma must be > 1")
        self.K = K
        self.gamma = gamma

    def pressure(self, density: float) -> float:
        """P = K * rho^gamma"""
        return float(self.K * density**self.gamma)

    def energy_density(self, density: float) -> float:
        """epsilon = rho * c^2 + P / (gamma - 1)"""
        P = self.pressure(density)
        return float(density * C**2 + P / (self.gamma - 1))

    def sound_speed(self, density: float) -> float:
        """c_s^2 = gamma * P / (epsilon + P)"""
        P = self.pressure(density)
        eps = self.energy_density(density)
        cs2 = self.gamma * P * C**2 / (eps + P)
        return float(np.sqrt(float(cs2)))

    def enthalpy(self, density: float) -> float:
        """h = c^2 + gamma * P / ((gamma - 1) * rho)"""
        P = self.pressure(density)
        return float(C**2 + self.gamma * P / ((self.gamma - 1) * density))

    @classmethod
    def from_central_pressure(
        cls, P_c: float, rho_c: float, gamma: float
    ) -> "Polytrope":
        """
        Create polytrope from central conditions.

        Parameters
        ----------
        P_c : float
            Central pressure [dynes/cm^2]
        rho_c : float
            Central density [g/cm^3]
        gamma : float
            Adiabatic index

        Returns
        -------
        Polytrope
            Configured polytrope
        """
        K = P_c / rho_c**gamma
        return cls(K=K, gamma=gamma)

    def __repr__(self):
        return f"Polytrope(K={self.K:.3e}, gamma={self.gamma:.3f})"


class PiecewisePolytrope(EOSBase):
    """
    Piecewise polytropic EOS with multiple segments.

    Commonly used for realistic neutron star modeling where different
    regions (crust, core) have different stiffness.

    Parameters
    ----------
    dividing_densities : list
        Density boundaries between segments [g/cm^3]
    gammas : list
        Adiabatic indices for each segment (len = len(dividing_densities) + 1)
    K0 : float
        Polytropic constant for first segment
    name : str
        Model name

    Examples
    --------
    >>> # Two-segment polytrope (soft crust, stiff core)
    >>> eos = PiecewisePolytrope(
    ...     dividing_densities=[2e14],
    ...     gammas=[1.3, 2.5],
    ...     K0=1e12
    ... )
    """

    def __init__(
        self,
        dividing_densities: List[float],
        gammas: List[float],
        K0: float,
        name: str = "piecewise_polytrope",
    ):
        super().__init__(name=name)

        if len(gammas) != len(dividing_densities) + 1:
            raise ValueError("Need len(gammas) = len(dividing_densities) + 1")

        for g in gammas:
            if g <= 1:
                raise ValueError("All gamma values must be > 1")

        self.dividing_densities = np.array(dividing_densities)
        self.gammas = np.array(gammas)

        # Compute K values for continuity
        self.Ks = np.zeros(len(gammas))
        self.Ks[0] = K0

        for i in range(len(dividing_densities)):
            rho_div = dividing_densities[i]
            # Continuity: K_i * rho^gamma_i = K_{i+1} * rho^gamma_{i+1}
            P_at_div = self.Ks[i] * rho_div ** self.gammas[i]
            self.Ks[i + 1] = P_at_div / rho_div ** self.gammas[i + 1]

    def _get_segment(self, density: float) -> int:
        """Find which segment a density falls into."""
        for i, rho_div in enumerate(self.dividing_densities):
            if density < rho_div:
                return i
        return len(self.dividing_densities)

    def pressure(self, density: float) -> float:
        """Piecewise P = K_i * rho^gamma_i"""
        i = self._get_segment(density)
        return float(self.Ks[i] * density ** self.gammas[i])

    def energy_density(self, density: float) -> float:
        """
        Energy density with piecewise integration.

        epsilon = rho * c^2 + integral contributions from each segment.
        """
        i = self._get_segment(density)

        # Rest mass contribution
        eps = density * C**2

        # Thermal contribution from current segment
        P = self.pressure(density)
        eps += P / (self.gammas[i] - 1)

        # Note: For full accuracy, should integrate through all segments
        # This is a simplified form assuming continuous pressure

        return float(eps)

    def sound_speed(self, density: float) -> float:
        """c_s in current segment."""
        i = self._get_segment(density)
        P = self.pressure(density)
        eps = self.energy_density(density)
        cs2 = self.gammas[i] * P * C**2 / (eps + P)
        return float(np.sqrt(max(0.0, float(cs2))))


# Standard polytrope configurations
def sly4_approximation() -> PiecewisePolytrope:
    """
    Piecewise polytrope approximation to SLy4 EOS.

    Reference: Read et al. 2009, PRD 79, 124032
    """
    return PiecewisePolytrope(
        dividing_densities=[2.44e14, 3.78e14, 2.62e15],
        gammas=[1.358, 2.830, 3.445, 3.348],
        K0=6.80e12,
        name="SLy4_approx",
    )


def apr4_approximation() -> PiecewisePolytrope:
    """
    Piecewise polytrope approximation to APR4 EOS.

    Reference: Read et al. 2009, PRD 79, 124032
    """
    return PiecewisePolytrope(
        dividing_densities=[2.44e14, 3.78e14, 2.62e15],
        gammas=[1.358, 2.830, 3.445, 2.884],
        K0=6.80e12,
        name="APR4_approx",
    )
