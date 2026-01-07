"""
Tabulated equation of state with interpolation.

Loads EOS tables from files in various formats and provides
interpolated values for arbitrary densities.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy import interpolate

from compact_common.constants import C
from compact_common.eos.base import EOSBase, EOSTable


class TabulatedEOS(EOSBase):
    """
    Tabulated EOS with interpolation.

    Supports loading from files or direct array input.
    Uses cubic spline interpolation in log-log space for smoothness.

    Parameters
    ----------
    density : array-like
        Density values [g/cm^3]
    pressure : array-like
        Pressure values [dynes/cm^2]
    energy_density : array-like, optional
        Energy density values [erg/cm^3]. If not provided, assumes eps = rho*c^2 + P/(gamma-1)
    name : str
        Model name

    Examples
    --------
    >>> # Load from arrays
    >>> rho = np.logspace(10, 16, 100)
    >>> P = 1e13 * (rho / 1e14)**2
    >>> eos = TabulatedEOS(rho, P, name="custom")

    >>> # Load from file
    >>> eos = load_eos_file("eos_sly4.dat", format="rns")
    """

    def __init__(
        self,
        density: np.ndarray,
        pressure: np.ndarray,
        energy_density: Optional[np.ndarray] = None,
        enthalpy: Optional[np.ndarray] = None,
        baryon_density: Optional[np.ndarray] = None,
        name: str = "tabulated",
    ):
        super().__init__(name=name)

        self._density = np.asarray(density)
        self._pressure = np.asarray(pressure)

        # Sort by density
        idx = np.argsort(self._density)
        self._density = self._density[idx]
        self._pressure = self._pressure[idx]

        # Energy density: use provided or estimate
        if energy_density is not None:
            self._energy_density = np.asarray(energy_density)[idx]
        else:
            # Estimate: eps = rho * c^2 (rest mass only)
            self._energy_density = self._density * C**2

        # Optional arrays
        if enthalpy is not None:
            self._enthalpy = np.asarray(enthalpy)[idx]
        else:
            self._enthalpy = None

        if baryon_density is not None:
            self._baryon_density = np.asarray(baryon_density)[idx]
        else:
            self._baryon_density = None

        # Build interpolators in log space
        self._log_rho = np.log10(self._density)
        self._log_P = np.log10(np.maximum(self._pressure, 1e-100))
        self._log_eps = np.log10(np.maximum(self._energy_density, 1e-100))

        # Pressure interpolator
        self._P_interp = interpolate.interp1d(
            self._log_rho, self._log_P,
            kind='cubic', bounds_error=False, fill_value='extrapolate'
        )

        # Energy density interpolator
        self._eps_interp = interpolate.interp1d(
            self._log_rho, self._log_eps,
            kind='cubic', bounds_error=False, fill_value='extrapolate'
        )

        # Inverse: P -> eps for TOV integration
        self._eps_from_P_interp = interpolate.interp1d(
            self._log_P, self._log_eps,
            kind='cubic', bounds_error=False, fill_value='extrapolate'
        )

        # Store bounds
        self.rho_min = self._density[0]
        self.rho_max = self._density[-1]

    def pressure(self, density: float) -> float:
        """Interpolated pressure."""
        if density < self.rho_min:
            # Extrapolate low density as ideal gas
            return float(self._pressure[0] * (density / self.rho_min) ** (5 / 3))
        if density > self.rho_max:
            # Extrapolate high density (caution: may be unphysical)
            return float(10 ** float(self._P_interp(np.log10(density))))

        return float(10 ** float(self._P_interp(np.log10(density))))

    def energy_density(self, density: float) -> float:
        """Interpolated energy density."""
        if density < self.rho_min:
            return float(density * C**2)
        if density > self.rho_max:
            return float(10 ** float(self._eps_interp(np.log10(density))))

        return float(10 ** float(self._eps_interp(np.log10(density))))

    def energy_from_pressure(self, pressure: float) -> float:
        """Get energy density from pressure (for TOV integration)."""
        if pressure <= 0:
            return float(self._energy_density[0])
        log_P = np.log10(pressure)
        return float(10 ** float(self._eps_from_P_interp(log_P)))

    def enthalpy(self, density: float) -> float:
        """Interpolated specific enthalpy."""
        if self._enthalpy is not None:
            if density < self.rho_min or density > self.rho_max:
                return super().enthalpy(density)

            log_h_interp = interpolate.interp1d(
                self._log_rho, np.log10(self._enthalpy),
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            return float(10 ** float(log_h_interp(np.log10(density))))
        return super().enthalpy(density)

    def to_table(
        self,
        n_points: int = 200,
        rho_min: float = 1e10,
        rho_max: float = 1e16,
        log_spacing: bool = True,
    ) -> EOSTable:
        """Return the underlying table (or resampled version)."""
        if n_points == len(self._density) and rho_min <= self.rho_min and rho_max >= self.rho_max:
            return EOSTable(
                density=self._density.copy(),
                pressure=self._pressure.copy(),
                energy_density=self._energy_density.copy(),
                enthalpy=self._enthalpy.copy() if self._enthalpy is not None else None,
                baryon_density=self._baryon_density.copy() if self._baryon_density is not None else None,
            )
        return super().to_table(
            n_points=n_points,
            rho_min=rho_min,
            rho_max=rho_max,
            log_spacing=log_spacing,
        )

    def __len__(self):
        return len(self._density)


def load_eos_file(
    filepath: Union[str, Path],
    format: str = "auto",
    name: Optional[str] = None,
) -> TabulatedEOS:
    """
    Load EOS from file.

    Parameters
    ----------
    filepath : str or Path
        Path to EOS file
    format : str
        File format: 'auto', 'rns', 'compose', 'simple'
        - rns: 4 columns (log10): energy_density, pressure, enthalpy, baryon_density
        - compose: CompOSE format
        - simple: 2 columns: density, pressure (or energy_density, pressure)
    name : str, optional
        Model name (default: filename stem)

    Returns
    -------
    TabulatedEOS
        Loaded and interpolated EOS
    """
    filepath = Path(filepath)

    if name is None:
        name = filepath.stem

    # Read data
    data = np.loadtxt(filepath, comments='#')

    if format == "auto":
        # Guess format from number of columns
        if data.ndim == 1:
            raise ValueError("EOS file must have at least 2 columns")
        ncols = data.shape[1]
        if ncols == 2:
            format = "simple"
        elif ncols >= 4:
            format = "rns"
        else:
            format = "simple"

    if format == "rns":
        # RNS format: log10(eps), log10(P), log10(h), log10(n_B)
        log_eps, log_P, log_h, log_nB = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        energy_density = 10 ** log_eps
        pressure = 10 ** log_P
        enthalpy = 10 ** log_h
        baryon_density = 10 ** log_nB

        # Density from energy: rho ~ eps / c^2 (approximate)
        density = energy_density / C**2

        return TabulatedEOS(
            density=density,
            pressure=pressure,
            energy_density=energy_density,
            enthalpy=enthalpy,
            baryon_density=baryon_density,
            name=name,
        )

    elif format == "simple":
        # Simple 2-column: density (or eps), pressure
        col1, col2 = data[:, 0], data[:, 1]

        # Check if col1 is density or energy_density
        # Energy densities are typically >> density * c^2 at high density
        if np.max(col1) > 1e20:  # Likely energy density
            energy_density = col1
            density = energy_density / C**2
        else:
            density = col1
            energy_density = density * C**2

        return TabulatedEOS(
            density=density,
            pressure=col2,
            energy_density=energy_density,
            name=name,
        )

    elif format == "compose":
        # CompOSE format (multi-column, specific structure)
        # Simplified handling - assumes standard CompOSE columns
        # Full implementation would parse header
        raise NotImplementedError("CompOSE format parser not yet implemented")

    else:
        raise ValueError(f"Unknown EOS format: {format}")


# Built-in EOS data paths (relative to package)
def get_builtin_eos_path(name: str) -> Path:
    """Get path to built-in EOS file."""
    import compact_common
    package_dir = Path(compact_common.__file__).parent
    eos_dir = package_dir / "data" / "eos"
    return eos_dir / f"{name}.dat"
