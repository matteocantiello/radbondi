"""Classical (adiabatic) Bondi solution — used as initial condition."""

from __future__ import annotations

import numpy as np

from radbondi.constants import G, c_light


def lambda_bondi(gamma: float) -> float:
    """Bondi accretion eigenvalue lambda(gamma).

    For the smooth transonic solution onto a point mass at rest:

        lambda = (1/4) * (2 / (5 - 3 gamma))^((5 - 3 gamma) / (2 (gamma - 1)))

    Common values: lambda = 0.25 for gamma = 5/3 (the limit is taken).
    """
    if abs(gamma - 5.0 / 3.0) < 1e-10:
        return 0.25
    a = 5.0 - 3.0 * gamma
    return 0.25 * (2.0 / a) ** (a / (2.0 * (gamma - 1.0)))


def bondi_radius(M_BH: float, ambient) -> float:
    """Bondi radius r_B = G M / c_inf^2 [cm]."""
    return G * M_BH / ambient.cs**2


def schwarzschild_radius(M_BH: float) -> float:
    """Schwarzschild radius r_S = 2 G M / c^2 [cm]."""
    return 2.0 * G * M_BH / c_light**2


def bondi_rate(M_BH: float, ambient) -> float:
    """Adiabatic Bondi accretion rate Mdot_B = 4 pi lambda rho_inf G^2 M^2 / c_inf^3."""
    r_B = bondi_radius(M_BH, ambient)
    return 4.0 * np.pi * lambda_bondi(ambient.gamma) * ambient.rho * ambient.cs * r_B**2


def _subsonic_mach(target, gam, n_iter=60):
    """Vectorized bisection for the subsonic Bondi Mach number.

    Solves ``g(M) = target`` for M in ``(0, 1)``, where
    ``g(M) = M^alpha * (M^2/2 + 1/(gam-1))`` and
    ``alpha = 2 (1-gam)/(gam+1)``. ``g`` is strictly decreasing on
    ``(0, 1)`` for ``gam > 1``, so plain bisection converges robustly.
    Sixty iterations give relative precision of ~1e-18.
    """
    alpha = 2.0 * (1.0 - gam) / (gam + 1.0)
    inv_gm1 = 1.0 / (gam - 1.0)
    target = np.asarray(target, dtype=float)
    M_lo = np.full_like(target, 1e-15)
    M_hi = np.full_like(target, 1.0 - 1e-15)
    for _ in range(n_iter):
        M_mid = 0.5 * (M_lo + M_hi)
        g_mid = M_mid**alpha * (0.5 * M_mid**2 + inv_gm1)
        # g is decreasing in M, so g > target means M is too small.
        too_small = g_mid > target
        M_lo = np.where(too_small, M_mid, M_lo)
        M_hi = np.where(too_small, M_hi, M_mid)
    return 0.5 * (M_lo + M_hi)


def adiabatic_profile(x_array, ambient):
    """Adiabatic Bondi profiles at x = r/r_B.

    Solves the Bernoulli + isentropic equations for the transonic solution
    (subsonic branch, appropriate for ``gamma = 5/3`` where the formal sonic
    point sits at ``r = 0``). Returns dimensional ``(v, T, rho, Mach)``
    arrays in CGS.

    Parameters
    ----------
    x_array : array_like
        Radii in units of r_B.
    ambient : AmbientMedium
        Ambient conditions (provides cs, T, rho, gamma).
    """
    gam = ambient.gamma
    beta_exp = 4.0 * (gam - 1.0) / (gam + 1.0)
    lam = lambda_bondi(gam)
    Theta = lam ** (2.0 * (1.0 - gam) / (gam + 1.0))

    x_array = np.atleast_1d(np.asarray(x_array, dtype=float))
    target = Theta * x_array**beta_exp * (1.0 / x_array + 1.0 / (gam - 1.0))
    M_array = _subsonic_mach(target, gam)

    rho_t = (lam / (x_array**2 * M_array)) ** (2.0 / (gam + 1.0))
    cs_t = rho_t ** ((gam - 1.0) / 2.0)
    v_t = M_array * cs_t
    return v_t * ambient.cs, cs_t**2 * ambient.T, rho_t * ambient.rho, M_array
