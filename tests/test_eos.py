"""Tests for EOS module."""

import numpy as np
import pytest

from compact_common.constants import RHO_NUC, C
from compact_common.eos import FermiGas, PiecewisePolytrope, Polytrope, RelFermiGas


class TestPolytrope:
    """Tests for Polytrope EOS."""

    def test_init(self):
        """Test basic initialization."""
        eos = Polytrope(K=1e13, gamma=2.0)
        assert eos.K == 1e13
        assert eos.gamma == 2.0

    def test_invalid_gamma(self):
        """Test that gamma <= 1 raises error."""
        with pytest.raises(ValueError):
            Polytrope(K=1e13, gamma=0.5)

    def test_pressure_scaling(self):
        """Test P ~ rho^gamma."""
        eos = Polytrope(K=1e13, gamma=2.0)
        P1 = eos.pressure(1e14)
        P2 = eos.pressure(2e14)
        # P2/P1 should be 2^gamma = 4
        assert abs(P2 / P1 - 4.0) < 0.01

    def test_energy_density_positive(self):
        """Test energy density is positive."""
        eos = Polytrope(K=1e13, gamma=2.0)
        eps = eos.energy_density(1e15)
        assert eps > 0

    def test_sound_speed_causal(self):
        """Test sound speed < c."""
        eos = Polytrope(K=1e13, gamma=2.0)
        cs = eos.sound_speed(1e14)
        assert cs < C
        assert cs > 0

    def test_to_table(self):
        """Test table generation."""
        eos = Polytrope(K=1e13, gamma=2.0)
        table = eos.to_table(n_points=50)
        assert len(table) == 50
        assert len(table.pressure) == 50
        assert all(table.pressure > 0)


class TestPiecewisePolytrope:
    """Tests for PiecewisePolytrope EOS."""

    def test_init(self):
        """Test basic initialization."""
        eos = PiecewisePolytrope(
            dividing_densities=[2e14],
            gammas=[1.5, 2.5],
            K0=1e12
        )
        assert len(eos.gammas) == 2

    def test_continuity(self):
        """Test pressure continuity at segment boundaries."""
        eos = PiecewisePolytrope(
            dividing_densities=[2e14],
            gammas=[1.5, 2.5],
            K0=1e12
        )
        rho_div = 2e14
        P_below = eos.pressure(rho_div * 0.9999)
        P_above = eos.pressure(rho_div * 1.0001)
        # Should be continuous
        assert abs(P_below - P_above) / P_below < 0.01


class TestFermiGas:
    """Tests for Fermi gas EOS."""

    def test_pressure_scaling(self):
        """Test P ~ rho^(5/3) for non-relativistic gas."""
        eos = FermiGas()
        P1 = eos.pressure(1e10)
        P2 = eos.pressure(2e10)
        expected_ratio = 2 ** (5/3)
        assert abs(P2 / P1 - expected_ratio) < 0.1

    def test_fermi_momentum(self):
        """Test Fermi momentum calculation."""
        eos = FermiGas()
        kf = eos.fermi_momentum(1e14)
        assert kf > 0


class TestRelFermiGas:
    """Tests for relativistic Fermi gas."""

    def test_pressure_positive(self):
        """Test pressure is positive."""
        eos = RelFermiGas()
        P = eos.pressure(1e15)
        assert P > 0

    def test_energy_includes_rest_mass(self):
        """Test energy density > rest mass contribution."""
        eos = RelFermiGas()
        eps = eos.energy_density(1e15)
        assert eps > 1e15 * C**2 * 0.9  # Should be close to rho*c^2


class TestConstants:
    """Tests for physical constants."""

    def test_nuclear_density(self):
        """Test nuclear saturation density is reasonable."""
        assert 1e14 < RHO_NUC < 1e15

    def test_speed_of_light(self):
        """Test speed of light value."""
        assert abs(C - 2.998e10) < 1e8
