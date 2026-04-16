"""Classical (adiabatic) Bondi solution — used as initial condition."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

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


def adiabatic_profile(x_array, ambient):
    """Adiabatic Bondi profiles at x = r/r_B.

    Solves the Bernoulli + isentropic equations for the transonic solution.
    Returns dimensional ``(v, T, rho, Mach)`` arrays in CGS.

    Parameters
    ----------
    x_array : array_like
        Radii in units of r_B.
    ambient : AmbientMedium
        Ambient conditions (provides cs, T, rho, gamma).
    """
    gam = ambient.gamma
    alpha_exp = 2.0 * (1.0 - gam) / (gam + 1.0)
    beta_exp = 4.0 * (gam - 1.0) / (gam + 1.0)
    lam = lambda_bondi(gam)

    def g_func(M):
        return M**alpha_exp * (M**2 / 2.0 + 1.0 / (gam - 1.0))

    def f_func(x):
        return x**beta_exp * (1.0 / x + 1.0 / (gam - 1.0))

    Theta = lam**alpha_exp
    x_array = np.atleast_1d(np.asarray(x_array, dtype=float))
    M_array = np.zeros_like(x_array)
    for i, x in enumerate(x_array):
        target = Theta * f_func(x)
        try:
            M_array[i] = brentq(lambda M, t=target: g_func(M) - t, 1e-15, 1.0 - 1e-15)
        except ValueError:
            M_array[i] = 1e-10

    rho_t = (lam / (x_array**2 * M_array)) ** (2.0 / (gam + 1.0))
    cs_t = rho_t ** ((gam - 1.0) / 2.0)
    v_t = M_array * cs_t
    return v_t * ambient.cs, cs_t**2 * ambient.T, rho_t * ambient.rho, M_array
