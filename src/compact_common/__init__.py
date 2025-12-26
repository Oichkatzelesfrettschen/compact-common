"""
compact-common: Shared infrastructure for compact object physics.

This library provides common tools for neutron star and black hole research:

- eos: Equation of state models and interpolation
- structure: TOV solver and stellar structure calculations
- spacetime: Metric functions and geodesic integration
- radiation: Thermal and neutrino emission
- bayesian: Parameter inference framework

Example
-------
>>> from compact_common.eos import Polytrope, SigmaOmegaModel
>>> from compact_common.structure import TOVSolver
>>> from compact_common.constants import M_SUN, C, G
>>>
>>> # Create polytropic EOS
>>> eos = Polytrope(K=100, gamma=2.0)
>>>
>>> # Solve TOV equations
>>> solver = TOVSolver(eos)
>>> star = solver.solve(central_density=1e15)
>>> print(f"M = {star.mass / M_SUN:.2f} M_sun, R = {star.radius / 1e5:.1f} km")
"""

from compact_common._version import __version__
from compact_common import constants

__all__ = [
    "__version__",
    "constants",
]
