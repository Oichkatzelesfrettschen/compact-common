from __future__ import annotations

from typing import Any, Literal, Optional, cast

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from compact_common.constants import M_SUN, C, G

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Conversion factors
MEV_FM3_TO_G_CM3 = 1.7827e12
MEV_FM3_TO_DYNE_CM2 = 1.6022e33

if HAS_NUMBA:
    @jit(nopython=True)
    def _linear_interp(x, x_arr, y_arr):
        """Simple linear interpolation for sorted x_arr."""
        if x <= x_arr[0]:
            return y_arr[0]
        if x >= x_arr[-1]:
            return y_arr[-1]

        # Binary search for index
        # For small arrays, linear scan might be faster, but let's do binary
        idx = np.searchsorted(x_arr, x)

        x0 = x_arr[idx-1]
        x1 = x_arr[idx]
        y0 = y_arr[idx-1]
        y1 = y_arr[idx]

        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    @jit(nopython=True)
    def _tov_rhs_numba(r, P, m, p_arr, rho_arr, G_val, C_val):
        if P <= 0:
            return 0.0, 0.0

        # Get mass density from pressure
        rho = _linear_interp(P, p_arr, rho_arr)

        term1 = G_val * (rho + P/C_val**2)
        term2 = m + (4 * np.pi * r**3 * P) / C_val**2
        term3 = r * (r - (2 * G_val * m) / C_val**2)

        if term3 <= 0: # Avoid division by zero or black hole interior
            return 0.0, 0.0

        dP_dr = -(term1 * term2) / term3
        dm_dr = 4 * np.pi * r**2 * rho

        return dP_dr, dm_dr

    @jit(nopython=True)
    def _solve_tov_numba(Pc, p_arr, rho_arr, G_val, C_val, max_r, r_min=100.0):
        # Initial conditions
        rho_c = _linear_interp(Pc, p_arr, rho_arr)
        m = (4.0/3.0) * np.pi * r_min**3 * rho_c
        P = Pc
        r = r_min

        # RK4 Integration
        dr = 1000.0 # Initial step size 10m

        while r < max_r and P > 0:
            if r > 1e5:
                dr = 5000.0  # 50m steps further out

            k1_P, k1_m = _tov_rhs_numba(r, P, m, p_arr, rho_arr, G_val, C_val)
            k2_P, k2_m = _tov_rhs_numba(r + 0.5*dr, P + 0.5*dr*k1_P, m + 0.5*dr*k1_m, p_arr, rho_arr, G_val, C_val)
            k3_P, k3_m = _tov_rhs_numba(r + 0.5*dr, P + 0.5*dr*k2_P, m + 0.5*dr*k2_m, p_arr, rho_arr, G_val, C_val)
            k4_P, k4_m = _tov_rhs_numba(r + dr, P + dr*k3_P, m + dr*k3_m, p_arr, rho_arr, G_val, C_val)

            P_new = P + (dr/6.0) * (k1_P + 2*k2_P + 2*k3_P + k4_P)
            m_new = m + (dr/6.0) * (k1_m + 2*k2_m + 2*k3_m + k4_m)

            if P_new <= 0:
                # Interpolate to find surface
                dP = (P_new - P) / dr
                dr_surf = -P / dP
                r_surf = r + dr_surf
                m_surf = m + dr_surf * (m_new - m)/dr
                return r_surf, m_surf

            P = P_new
            m = m_new
            r += dr

        return r, m


def _tov_equations_static(r, y, rho_func, G_val, C_val):
    P, m = y
    if P <= 0:
        return [0, 0]

    rho = rho_func(P)

    term1 = G_val * (rho + P/C_val**2)
    term2 = m + (4 * np.pi * r**3 * P) / C_val**2
    term3 = r * (r - (2 * G_val * m) / C_val**2)

    if term3 == 0:
        return [0, 0]

    dP_dr = -(term1 * term2) / term3
    dm_dr = 4 * np.pi * r**2 * rho

    return [dP_dr, dm_dr]

class TOVSolver:
    """
    Solver for Tolman-Oppenheimer-Volkoff equations.
    """

    def __init__(
        self,
        energy_density: np.ndarray,
        pressure: np.ndarray,
        *,
        units: Literal["mev_fm3", "cgs"] = "mev_fm3",
    ):
        """
        Initialize with EOS table.
        
        Args:
            energy_density: Energy density values.
            pressure: Pressure values.
            units:
                - "mev_fm3": inputs are in MeV/fm^3 and are converted internally to CGS.
                - "cgs": inputs are already CGS; energy_density is in erg/cm^3 and is converted to mass density via /c^2.
        """
        # Sort by pressure
        idx = np.argsort(pressure)
        self.p_arr = np.asarray(pressure)[idx]
        self.e_arr = np.asarray(energy_density)[idx]

        if units == "mev_fm3":
            # Unit conversion to cgs for calculation
            self.p_cgs = self.p_arr * MEV_FM3_TO_DYNE_CM2
            self.rho_cgs = self.e_arr * MEV_FM3_TO_G_CM3
        elif units == "cgs":
            # Pressure is dyn/cm^2; convert energy density [erg/cm^3] to mass density [g/cm^3]
            self.p_cgs = self.p_arr
            self.rho_cgs = self.e_arr / C**2
        else:
            raise ValueError(f"Unsupported units: {units}")

        # Python interpolation (fallback)
        self.rho_interp = interp1d(self.p_cgs, self.rho_cgs, kind='linear', bounds_error=False, fill_value=0)
        self.rho_to_p = interp1d(self.rho_cgs, self.p_cgs, kind='linear', bounds_error=False, fill_value='extrapolate')

        # Numba preparation
        if HAS_NUMBA:
            # Ensure arrays are contiguous float64 for Numba
            self.p_cgs_jit = np.ascontiguousarray(self.p_cgs, dtype=np.float64)
            self.rho_cgs_jit = np.ascontiguousarray(self.rho_cgs, dtype=np.float64)

    def _tov_equations(self, r, y):
        """TOV equations wrapper."""
        return _tov_equations_static(r, y, self.rho_interp, G, C)

    def solve(self, central_density: float, max_r: float = 20e5) -> Optional[tuple[float, float]]:
        """
        Solve TOV for a given central density.
        
        Args:
            central_density (float): Central density in g/cm^3.
            max_r (float): Max radius integration in cm.
            
        Returns:
            tuple: (Radius [km], Mass [M_sun])
        """
        # Convert central density to central pressure using EOS
        Pc = float(cast(Any, self.rho_to_p)(central_density))

        # Fast path
        if HAS_NUMBA:
            R_surf, M_surf = _solve_tov_numba(Pc, self.p_cgs_jit, self.rho_cgs_jit, G, C, float(max_r))
            return float(R_surf / 100000.0), float(M_surf / M_SUN)

        # Slow path
        # Initial conditions
        # At r -> 0: P = Pc, m = 0
        # To avoid singularity at r=0, start at small r_min
        r_min = 100.0 # 1 meter
        m0 = (4/3) * np.pi * r_min**3 * central_density
        y0 = [Pc, m0]

        def event_zero_pressure(t, y):
            return y[0]
        event_zero_pressure_typed = cast(Any, event_zero_pressure)
        event_zero_pressure_typed.terminal = True
        event_zero_pressure_typed.direction = -1

        sol = solve_ivp(self._tov_equations, [r_min, max_r], y0, events=event_zero_pressure_typed, rtol=1e-6)

        if sol.status == 0: # Reached max_r without P=0
            # Debug: print final state
            # print(f"TOV solver failed to close: r_max={sol.t[-1]:.2e}, P_surf={sol.y[0][-1]:.2e}")
            return None

        R_surf = sol.t[-1] # cm
        M_total = sol.y[1][-1] # g

        return float(R_surf / 100000.0), float(M_total / M_SUN)
