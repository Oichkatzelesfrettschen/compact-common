"""
Physical constants for compact object physics.

All constants are in CGS units unless otherwise noted.
Natural units (G = c = 1) conversions are also provided.

References
----------
- CODATA 2018 recommended values
- Particle Data Group 2022
- IAU 2015 nominal solar values
"""

import numpy as np

# Fundamental constants (CGS)
C = 2.99792458e10  # Speed of light [cm/s]
G = 6.67430e-8  # Gravitational constant [cm^3/(g s^2)]
HBAR = 1.054571817e-27  # Reduced Planck constant [erg s]
HBARC = 197.3269804  # hbar * c [MeV fm]
K_B = 1.380649e-16  # Boltzmann constant [erg/K]

# Conversion factors
MEV_TO_ERG = 1.602176634e-6  # MeV to erg
MEV_TO_G = MEV_TO_ERG / C**2  # MeV to g (mass equivalent)
FM_TO_CM = 1e-13  # fm to cm
MEV_FM3_TO_CGS = MEV_TO_ERG / FM_TO_CM**3  # MeV/fm^3 to erg/cm^3
MEV_FM3_TO_DYNES = MEV_FM3_TO_CGS  # MeV/fm^3 to dynes/cm^2

# Solar values (IAU 2015 nominal)
M_SUN = 1.98841e33  # Solar mass [g]
R_SUN = 6.957e10  # Solar radius [cm]
L_SUN = 3.828e33  # Solar luminosity [erg/s]

# Neutron star characteristic scales
M_NS_TYPICAL = 1.4 * M_SUN  # Typical NS mass
R_NS_TYPICAL = 1.2e6  # Typical NS radius [cm] = 12 km
RHO_NUC = 2.8e14  # Nuclear saturation density [g/cm^3]
N0_NUC = 0.16  # Nuclear saturation number density [fm^-3]

# Particle masses (CGS)
M_NEUTRON = 1.67493e-24  # Neutron mass [g]
M_PROTON = 1.67262e-24  # Proton mass [g]
M_ELECTRON = 9.10938e-28  # Electron mass [g]

# Particle masses (MeV/c^2)
M_NEUTRON_MEV = 939.565  # Neutron mass [MeV]
M_PROTON_MEV = 938.272  # Proton mass [MeV]
M_ELECTRON_MEV = 0.511  # Electron mass [MeV]

# Meson masses for mean-field models (MeV)
M_SIGMA_MEV = 500.0  # Sigma meson mass
M_OMEGA_MEV = 783.0  # Omega meson mass
M_RHO_MEV = 763.0  # Rho meson mass

# Schwarzschild radius scale
R_SCHW_SUN = 2 * G * M_SUN / C**2  # Schwarzschild radius of 1 M_sun [cm]

# Natural units conversion
# In natural units: G = c = hbar = 1
# Length scale: G M_sun / c^2 = 1.477 km
KSCALE = C**2 / (G * 1e15)  # Scale for dimensionless EOS (rns convention)

# Geometrized units conversions
def to_geometrized_density(rho_cgs):
    """Convert density from g/cm^3 to geometrized units (1/cm^2)."""
    return G * rho_cgs / C**2


def to_geometrized_pressure(P_cgs):
    """Convert pressure from dynes/cm^2 to geometrized units (1/cm^2)."""
    return G * P_cgs / C**4


def from_geometrized_mass(M_geom, length_scale=1e5):
    """Convert mass from geometrized to grams. Default scale: km."""
    return M_geom * C**2 * length_scale / G


def to_natural_units(value, unit_type):
    """
    Convert CGS to natural units (hbar = c = 1).

    Parameters
    ----------
    value : float
        Value in CGS units
    unit_type : str
        One of 'energy', 'length', 'mass', 'time', 'density', 'pressure'

    Returns
    -------
    float
        Value in natural units (MeV-based)
    """
    if unit_type == 'energy':
        return value / MEV_TO_ERG  # erg -> MeV
    elif unit_type == 'length':
        return value / FM_TO_CM  # cm -> fm
    elif unit_type == 'mass':
        return value * C**2 / MEV_TO_ERG  # g -> MeV
    elif unit_type == 'time':
        return value * C / FM_TO_CM  # s -> fm/c
    elif unit_type == 'density':
        return value / MEV_FM3_TO_CGS  # erg/cm^3 -> MeV/fm^3
    elif unit_type == 'pressure':
        return value / MEV_FM3_TO_CGS  # dynes/cm^2 -> MeV/fm^3
    else:
        raise ValueError(f"Unknown unit type: {unit_type}")


# Useful combinations
PI = np.pi
PI2 = np.pi**2
FOUR_PI = 4.0 * np.pi
FOUR_PI_OVER_3 = 4.0 * np.pi / 3.0

# TOV equation scale factors
TOV_MASS_SCALE = FOUR_PI * R_NS_TYPICAL**3 * RHO_NUC / M_SUN
TOV_PRESSURE_SCALE = G * M_SUN * RHO_NUC / R_NS_TYPICAL

# Degeneracy factors
G_NEUTRON = 2  # Neutron spin degeneracy
G_PROTON = 2  # Proton spin degeneracy
G_ELECTRON = 2  # Electron spin degeneracy
