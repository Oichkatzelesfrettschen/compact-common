import numpy as np
from scipy.optimize import root_scalar

# Constants (Natural Units)
HBARC = 197.327      # MeV·fm
MS = 500.0 / HBARC   # meson mass (fm-1)
MW = 783.0 / HBARC   # omega mass (fm-1)
M_NUCLEON = 939.0 / HBARC # nucleon mass (fm-1)
Y = 2.0              # degeneracy factor

class SigmaOmegaModel:
    """
    Relativistic Mean Field Theory (Sigma-Omega) EOS model.
    """

    def __init__(self, Gs: float, Gv: float):
        """
        Initialize model with coupling constants.
        
        Args:
            Gs (float): Scalar coupling constant.
            Gv (float): Vector coupling constant.
        """
        self.Gs = Gs
        self.Gv = Gv

    def _f_sigma(self, sigma, kf):
        """Self-consistency equation for sigma field."""
        M_star = M_NUCLEON - np.sqrt(self.Gs) * MS * sigma

        def integrand(k):
            return (M_star * k**2) / np.sqrt(k**2 + M_star**2)

        # Integration (Trapezoidal for speed/simplicity, or use scipy.integrate.quad)
        k = np.linspace(0, kf, 100)
        y_val = integrand(k)

        # Use numpy.trapezoid (NumPy 2.0) or fallback to trapz
        try:
            integral = np.trapezoid(y_val, k)
        except AttributeError:
            integral = np.trapz(y_val, k)

        rhs = (np.sqrt(self.Gs) / MS) * (Y / (2 * np.pi**2)) * integral
        return sigma - rhs

    def compute_eos(self, number_density_range=(1e-4, 5.0), num_points=100):
        """
        Compute EOS (Energy Density vs Pressure).
        
        Returns:
            tuple: (energy_density [MeV/fm^3], pressure [MeV/fm^3])
        """
        n_arr = np.logspace(np.log10(number_density_range[0]), np.log10(number_density_range[1]), num_points)
        energy_density = []
        pressure = []

        for n in n_arr:
            kf = (3 * np.pi**2 * n)**(1/3)

            # Solve for sigma
            try:
                sol = root_scalar(self._f_sigma, args=(kf,), bracket=[0, 2.0*M_NUCLEON], method='brentq')
                sigma = sol.root
            except ValueError:
                sigma = 0.0 # Fallback

            M_star = M_NUCLEON - np.sqrt(self.Gs) * MS * sigma

            # Calculate energy components
            # Scalar energy
            epsilon_scalar = 0.5 * MS**2 * sigma**2

            # Vector energy
            rho_B = Y * kf**3 / (6 * np.pi**2)
            g_w = MW * np.sqrt(self.Gv)
            omega_0 = (g_w / MW**2) * rho_B
            epsilon_vector = 0.5 * MW**2 * omega_0**2

            # Kinetic energy integral
            k = np.linspace(0, kf, 100)
            e_int_vals = np.sqrt(M_star**2 + k**2) * k**2

            try:
                epsilon_integral = np.trapezoid(e_int_vals, k)
            except AttributeError:
                epsilon_integral = np.trapz(e_int_vals, k)

            epsilon_total = (1 / (np.pi**2)) * epsilon_integral + epsilon_scalar + epsilon_vector

            # Pressure
            p_int_vals = (k**4) / np.sqrt(M_star**2 + k**2)

            try:
                p_integral = np.trapezoid(p_int_vals, k)
            except AttributeError:
                p_integral = np.trapz(p_int_vals, k)

            p_total = (1 / (3 * np.pi**2)) * p_integral - epsilon_scalar + epsilon_vector

            energy_density.append(epsilon_total * HBARC)
            pressure.append(p_total * HBARC)

        return np.array(energy_density), np.array(pressure)
