"""Validation of the adiabatic Bondi solution."""

import numpy as np
import pytest

import radbondi as rb
from radbondi.bondi import (
    adiabatic_profile,
    bondi_radius,
    bondi_rate,
    lambda_bondi,
)


def test_lambda_bondi_gamma_5_3():
    """For gamma=5/3 the Bondi eigenvalue is exactly 1/4."""
    assert lambda_bondi(5.0 / 3.0) == pytest.approx(0.25)


def test_lambda_bondi_gamma_4_3():
    """For gamma=4/3, lambda = (1/4) * 2^(3/2) = 1/sqrt(2) ~ 0.707.

    Derived from lambda(gamma) = (1/4) * (2/(5-3 gamma))^((5-3 gamma)/(2(gamma-1)))
    evaluated at gamma=4/3.
    """
    assert lambda_bondi(4.0 / 3.0) == pytest.approx(1.0 / np.sqrt(2.0))


def test_bondi_radius_scales_with_mass():
    amb = rb.presets.solar_core()
    M1, M2 = 1e-16 * rb.M_sun, 1e-13 * rb.M_sun
    assert bondi_radius(M2, amb) / bondi_radius(M1, amb) == pytest.approx(1e3)


def test_bondi_rate_scales_with_mass_squared():
    amb = rb.presets.solar_core()
    M1, M2 = 1e-16 * rb.M_sun, 1e-13 * rb.M_sun
    assert bondi_rate(M2, amb) / bondi_rate(M1, amb) == pytest.approx(1e6)


def test_adiabatic_profile_inner_power_laws():
    """At r << r_B, the adiabatic Bondi profile follows
        rho ~ r^(-3/2),  T ~ r^(-1),  v ~ r^(-1/2),
    each within a few percent."""
    amb = rb.presets.solar_core()
    x = np.array([1e-5, 1e-4, 1e-3])  # well inside r_B
    v, T, rho, _ = adiabatic_profile(x, amb)

    # Density slope: log(rho2/rho1) / log(x2/x1) ~ -3/2
    drho = np.log10(rho[-1] / rho[0]) / np.log10(x[-1] / x[0])
    dT = np.log10(T[-1] / T[0]) / np.log10(x[-1] / x[0])
    dv = np.log10(np.abs(v[-1]) / np.abs(v[0])) / np.log10(x[-1] / x[0])
    assert drho == pytest.approx(-1.5, abs=0.02)
    assert dT == pytest.approx(-1.0, abs=0.02)
    assert dv == pytest.approx(-0.5, abs=0.02)


def test_adiabatic_profile_subsonic_at_outer_grid():
    """At the typical outer grid edge (x = 3) the flow is highly subsonic.
    The flow only approaches ambient values asymptotically as x -> infinity;
    at x = 3 we already see ~35% density compression."""
    amb = rb.presets.solar_core()
    _, _, _, mach = adiabatic_profile(np.array([3.0]), amb)
    assert mach[0] < 0.1


def test_adiabatic_profile_subsonic_inside_r_B_for_5_3():
    """For gamma=5/3 the formal sonic point is at r = 0, so the adiabatic
    transonic flow is subsonic everywhere outside r=0. At r = r_B the Mach
    number is well below unity."""
    amb = rb.presets.solar_core()
    _, _, _, mach = adiabatic_profile(np.array([1.0]), amb)
    assert 0.05 < mach[0] < 0.5
