"""
Sigma-Omega mean field approximation (MFA) model.

Quantum hadrodynamics with scalar (sigma) and vector (omega) meson exchange.
Extracted and refactored from Generation-EOS repository.

References
----------
- Walecka, J. D. (1974). Ann. Phys. 83, 491
- Serot & Walecka (1986). Adv. Nucl. Phys. 16, 1
- Lopes, L. L. arXiv:2508.16789v1
"""

from typing import Tuple, Optional
import numpy as np
from scipy import integrate, optimize

from compact_common.constants import (
    HBARC,
    M_NEUTRON_MEV,
    M_SIGMA_MEV,
    M_OMEGA_MEV,
    PI,
    PI2,
    C,
    FM_TO_CM,
    M_NEUTRON,
)
from compact_common.eos.base import EOSBase

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    # Fallback: identity decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    HAS_NUMBA = False


@jit(nopython=True, cache=True)
def _secant_method(
    x0: float, x1: float, M: float, ms: float, Gs: float, kf: float, y: float,
    tol: float = 1e-8, max_iter: int = 100
) -> float:
    """
    Secant method to solve sigma field equation.

    Solves: sigma = (sqrt(Gs) / ms) * (y / 2*pi^2) * integral term
    """
    def func(sigma):
        M_star = M - np.sqrt(Gs) * ms * sigma
        if M_star <= 0:
            return 1e10

        # Integration using RK4 (Numba compatible)
        n_steps = 500
        h = kf / n_steps
        integral = 0.0

        for i in range(n_steps):
            k = i * h
            k_mid = k + h / 2
            k_next = k + h

            def integrand(kk):
                return M_star * kk**2 / np.sqrt(kk**2 + M_star**2)

            # RK4 step
            k1 = integrand(k)
            k2 = integrand(k_mid)
            k3 = integrand(k_mid)
            k4 = integrand(k_next)
            integral += h * (k1 + 2*k2 + 2*k3 + k4) / 6

        rhs = (np.sqrt(Gs) / ms) * (y / (2 * PI2)) * integral
        return sigma - rhs

    # Secant iteration
    f0 = func(x0)
    for _ in range(max_iter):
        f1 = func(x1)
        if abs(f1) < tol:
            return x1
        if abs(f1 - f0) < 1e-15:
            break
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0 = x1, f1
        x1 = x_new

    return x1


@jit(nopython=True, cache=True)
def _compute_eos_point(
    kf: float, Gs: float, Gv: float, M: float, ms: float, mw: float, y: float
) -> Tuple[float, float]:
    """
    Compute energy density and pressure at a given Fermi momentum.

    Parameters
    ----------
    kf : float
        Fermi momentum [fm^-1]
    Gs : float
        Scalar coupling constant (dimensionless)
    Gv : float
        Vector coupling constant (dimensionless)
    M : float
        Nucleon mass [fm^-1]
    ms : float
        Sigma meson mass [fm^-1]
    mw : float
        Omega meson mass [fm^-1]
    y : float
        Degeneracy factor

    Returns
    -------
    epsilon : float
        Energy density [MeV/fm^3]
    pressure : float
        Pressure [MeV/fm^3]
    """
    if kf < 1e-10:
        return 0.0, 0.0

    # Solve for sigma field
    sigma = _secant_method(0.01, 0.1, M, ms, Gs, kf, y)

    # Effective mass
    M_star = M - np.sqrt(Gs) * ms * sigma
    if M_star <= 0:
        M_star = 0.01 * M

    # Baryon density
    n_B = y * kf**3 / (6 * PI2)

    # Omega field (vector mean field)
    omega0 = (np.sqrt(Gv) / mw) * n_B

    # Energy density components (RK4 integration)
    n_steps = 500
    h = kf / n_steps

    # Kinetic energy integral
    E_kin = 0.0
    for i in range(n_steps):
        k = i * h
        def integrand_kin(kk):
            return np.sqrt(M_star**2 + kk**2) * kk**2
        k1 = integrand_kin(k)
        k2 = integrand_kin(k + h/2)
        k3 = integrand_kin(k + h/2)
        k4 = integrand_kin(k + h)
        E_kin += h * (k1 + 2*k2 + 2*k3 + k4) / 6

    E_kin *= y / (2 * PI2)

    # Scalar field energy (attractive)
    E_scalar = 0.5 * ms**2 * sigma**2

    # Vector field energy (repulsive)
    E_vector = 0.5 * mw**2 * omega0**2

    # Total energy density
    epsilon = (E_kin + E_scalar + E_vector) * HBARC  # MeV/fm^3

    # Pressure integral
    P_kin = 0.0
    for i in range(n_steps):
        k = i * h
        def integrand_P(kk):
            return kk**4 / np.sqrt(M_star**2 + kk**2)
        k1 = integrand_P(k)
        k2 = integrand_P(k + h/2)
        k3 = integrand_P(k + h/2)
        k4 = integrand_P(k + h)
        P_kin += h * (k1 + 2*k2 + 2*k3 + k4) / 6

    P_kin *= y / (6 * PI2)

    # Total pressure
    pressure = (P_kin - E_scalar + E_vector) * HBARC  # MeV/fm^3

    return epsilon, pressure


class SigmaOmegaModel(EOSBase):
    """
    Sigma-Omega mean field approximation (MFA) EOS.

    Implements quantum hadrodynamics with scalar (sigma) and vector (omega)
    meson exchange. The scalar field provides attraction while the vector
    field provides repulsion.

    Parameters
    ----------
    Gs : float
        Scalar coupling constant (dimensionless, typical: 0-10)
    Gv : float
        Vector coupling constant (dimensionless, typical: 0-10)
    M_mev : float
        Nucleon mass [MeV/c^2]
    ms_mev : float
        Sigma meson mass [MeV/c^2]
    mw_mev : float
        Omega meson mass [MeV/c^2]
    degeneracy : int
        Nucleon degeneracy factor (2 for neutrons only, 4 for n+p)
    name : str
        Model name

    Examples
    --------
    >>> # Standard neutron matter parameters
    >>> eos = SigmaOmegaModel(Gs=5.0, Gv=5.0)
    >>> P = eos.pressure(3e14)  # Pressure at ~ nuclear density
    """

    def __init__(
        self,
        Gs: float = 5.0,
        Gv: float = 5.0,
        M_mev: float = M_NEUTRON_MEV,
        ms_mev: float = M_SIGMA_MEV,
        mw_mev: float = M_OMEGA_MEV,
        degeneracy: int = 2,
        name: str = "sigma_omega",
    ):
        super().__init__(name=name)
        self.Gs = Gs
        self.Gv = Gv

        # Convert to natural units [fm^-1]
        self.M = M_mev / HBARC
        self.ms = ms_mev / HBARC
        self.mw = mw_mev / HBARC
        self.y = degeneracy

        # Cache for computed EOS points
        self._cache = {}

    def _density_to_kf(self, density: float) -> float:
        """Convert CGS density to Fermi momentum [fm^-1]."""
        # n [fm^-3] = rho [g/cm^3] / m_n [g] * (fm/cm)^3
        n_fm3 = density / M_NEUTRON * FM_TO_CM**3
        # k_F = (6 * pi^2 * n / y)^(1/3)
        return (6 * PI2 * n_fm3 / self.y) ** (1 / 3)

    def _get_eos_point(self, density: float) -> Tuple[float, float]:
        """Get (epsilon, P) in MeV/fm^3 with caching."""
        # Round to avoid cache misses from floating point
        key = round(density, 10)
        if key not in self._cache:
            kf = self._density_to_kf(density)
            eps, P = _compute_eos_point(
                kf, self.Gs, self.Gv, self.M, self.ms, self.mw, self.y
            )
            self._cache[key] = (eps, P)
        return self._cache[key]

    def pressure(self, density: float) -> float:
        """
        Compute pressure from density.

        Parameters
        ----------
        density : float
            Mass density [g/cm^3]

        Returns
        -------
        float
            Pressure [dynes/cm^2]
        """
        if density < 1e5:
            return 0.0

        _, P_mev = self._get_eos_point(density)

        # Convert MeV/fm^3 to dynes/cm^2
        from compact_common.constants import MEV_FM3_TO_DYNES
        return max(0, P_mev * MEV_FM3_TO_DYNES)

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
        if density < 1e5:
            return density * C**2

        eps_mev, _ = self._get_eos_point(density)

        # Convert MeV/fm^3 to erg/cm^3
        from compact_common.constants import MEV_FM3_TO_CGS
        return eps_mev * MEV_FM3_TO_CGS

    def effective_mass(self, density: float) -> float:
        """
        Compute effective nucleon mass at given density.

        Returns
        -------
        float
            M* / M ratio
        """
        kf = self._density_to_kf(density)
        sigma = _secant_method(0.01, 0.1, self.M, self.ms, self.Gs, kf, self.y)
        M_star = self.M - np.sqrt(self.Gs) * self.ms * sigma
        return M_star / self.M

    def clear_cache(self):
        """Clear the internal EOS cache."""
        self._cache = {}

    def __repr__(self):
        return f"SigmaOmegaModel(Gs={self.Gs:.3f}, Gv={self.Gv:.3f})"


# Pre-configured models
def nm1_model() -> SigmaOmegaModel:
    """NM1 parameterization (soft EOS)."""
    return SigmaOmegaModel(Gs=3.0, Gv=3.0, name="NM1")


def nm2_model() -> SigmaOmegaModel:
    """NM2 parameterization (medium stiffness)."""
    return SigmaOmegaModel(Gs=5.0, Gv=5.0, name="NM2")


def nm3_model() -> SigmaOmegaModel:
    """NM3 parameterization (stiff EOS)."""
    return SigmaOmegaModel(Gs=7.0, Gv=8.0, name="NM3")
