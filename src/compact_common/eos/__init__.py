"""
Equation of State (EOS) models for dense nuclear matter.

This module provides various EOS implementations for neutron star modeling:

- Polytrope: Simple power-law P = K * rho^gamma
- FermiGas: Relativistic degenerate Fermi gas
- SigmaOmega: Mean-field approximation with meson exchange
- Tabulated: Interpolation from tabulated EOS files

All EOS classes implement the same interface:
    - pressure(density): P(rho) in CGS
    - energy_density(density): epsilon(rho) in CGS
    - sound_speed(density): c_s(rho)
    - to_table(n_points): Generate tabulated form
"""

from compact_common.eos.base import EOSBase
from compact_common.eos.polytrope import Polytrope, PiecewisePolytrope
from compact_common.eos.fermi_gas import FermiGas, RelFermiGas
from compact_common.eos.sigma_omega import SigmaOmegaModel
from compact_common.eos.tabulated import TabulatedEOS, load_eos_file

__all__ = [
    "EOSBase",
    "Polytrope",
    "PiecewisePolytrope",
    "FermiGas",
    "RelFermiGas",
    "SigmaOmegaModel",
    "TabulatedEOS",
    "load_eos_file",
]
