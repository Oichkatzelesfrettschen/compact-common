import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from compact_common.constants import G, C, M_SUN, FOUR_PI
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Conversion factors (approximate, refine as needed)
MEV_FM3_TO_G_CM3 = 1.7827e12
MEV_FM3_TO_DYNE_CM2 = 1.6022e33

# Static JIT-compiled TOV equation function
if HAS_NUMBA:
    @jit(nopython=True)
    def _tov_rhs_numba(P, m, r, rho_interp_p, rho_interp_rho, G_val, C_val):
        if P <= 0:
            return 0.0, 0.0
            
        # Linear interpolation for rho(P)
        # Assuming sorted P arrays
        # Since we can't pass scipy interp1d to numba, we implement basic linear interp
        # or we assume P, rho arrays are passed.
        # For simplicity in this optimization step, we'll keep it simple or skip full JIT
        # if interpolation is complex.
        
        # Actually, let's keep the scipy version as default and only JIT the math part
        # if we can extract rho.
        
        # Simplified: We only JIT the differential equation part assuming rho is known.
        pass

# We will optimize the method by defining a standalone function for the ODE system
# that can be JIT compiled if we handle the interpolation manually or outside.

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
    
    def __init__(self, energy_density, pressure):
        """
        Initialize with EOS table.
        
        Args:
            energy_density (array): Energy density in MeV/fm^3 (or consistent units).
            pressure (array): Pressure in MeV/fm^3.
        """
        # Sort by pressure
        idx = np.argsort(pressure)
        self.p_arr = pressure[idx]
        self.e_arr = energy_density[idx]
        
        # Create interpolation
        self.eos_interp = interp1d(self.p_arr, self.e_arr, kind='linear', bounds_error=False, fill_value=0)
        
        # Unit conversion to cgs for calculation if input is MeV/fm^3
        self.p_cgs = self.p_arr * MEV_FM3_TO_DYNE_CM2
        self.e_cgs = self.e_arr * MEV_FM3_TO_G_CM3 
        
        self.rho_cgs = self.p_cgs / C**2 
        self.rho_interp = interp1d(self.p_cgs, self.e_cgs, kind='linear', bounds_error=False, fill_value=0)

    def _tov_equations(self, r, y):
        """TOV equations wrapper."""
        return _tov_equations_static(r, y, self.rho_interp, G, C)

    def solve(self, central_density, max_r=20e5):
        """
        Solve TOV for a given central density.
        
        Args:
            central_density (float): Central density in g/cm^3.
            max_r (float): Max radius integration in cm.
            
        Returns:
            tuple: (Radius [km], Mass [M_sun])
        """
        # Convert central density to central pressure using EOS
        # We need inverse interpolation: rho -> P
        rho_to_p = interp1d(self.e_cgs, self.p_cgs, kind='linear', bounds_error=False, fill_value='extrapolate')
        Pc = rho_to_p(central_density)
        
        # Initial conditions
        # At r -> 0: P = Pc, m = 0
        # To avoid singularity at r=0, start at small r_min
        r_min = 100.0 # 1 meter
        m0 = (4/3) * np.pi * r_min**3 * central_density
        y0 = [Pc, m0]
        
        def event_zero_pressure(t, y):
            return y[0]
        event_zero_pressure.terminal = True
        event_zero_pressure.direction = -1
        
        sol = solve_ivp(self._tov_equations, [r_min, max_r], y0, events=event_zero_pressure, rtol=1e-6)
        
        if sol.status == 0: # Reached max_r without P=0
            return None
            
        R_surf = sol.t[-1] # cm
        M_total = sol.y[1][-1] # g
        
        return R_surf / 100000.0, M_total / M_SUN