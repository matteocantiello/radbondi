"""Steady-state residual checks and other diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radbondi.ambient import AmbientMedium
from radbondi.constants import G


@dataclass
class SteadyStateResiduals:
    """Output of :func:`check_steady_state`.

    Each ``*_residual`` array is the per-cell residual of the corresponding
    conservation equation, normalized by the dominant term at that radius.
    The ``*_rms`` and ``*_max`` scalars are computed over interior cells only
    (excluding ``n_boundary`` cells at each end).
    """

    x: np.ndarray                # r/r_B at cell centers
    Mdot: np.ndarray             # mass flux profile
    Mdot_median: float
    mass_residual: np.ndarray
    momentum_residual: np.ndarray
    energy_residual: np.ndarray

    mass_rms: float
    mass_max: float
    momentum_rms: float
    momentum_max: float
    energy_rms: float
    energy_max: float

    n_boundary: int


def check_steady_state(solution, cooling, n_boundary: int = 5) -> SteadyStateResiduals:
    """Evaluate the steady-state Euler equations on a converged profile.

    Uses integral (control-volume) form for momentum and energy: the residual
    at cell ``i`` is the net flux + source imbalance divided by the dominant
    term, evaluated over the cell volume. Mass conservation is checked
    pointwise (Mdot constancy).

    Parameters
    ----------
    solution : Solution
        Converged solution from :meth:`BondiProblem.solve`.
    cooling : Cooling
        Cooling prescription used in the simulation.
    n_boundary : int
        Number of cells to exclude at each boundary when computing
        RMS/max statistics.
    """
    rho = solution.rho
    v = solution.v
    P = solution.P
    T = solution.T
    r = solution.r
    M_BH = solution.M_BH
    gamma = solution.ambient_gamma
    N = len(r)

    # Reconstruct grid (faces, areas, volumes) from cell centers
    ratio = (r[1] / r[0]) ** 0.5
    r_face = np.empty(N + 1)
    r_face[:-1] = r / ratio
    r_face[-1] = r[-1] * ratio
    area = r_face**2
    vol = (r_face[1:] ** 3 - r_face[:-1] ** 3) / 3.0

    # Reconstruct ambient for cooling evaluation
    ambient = AmbientMedium(
        T=solution.ambient_T, rho=solution.ambient_rho, mu=solution.ambient_mu,
        gamma=solution.ambient_gamma, X=solution.ambient_X, Y=solution.ambient_Y,
    )

    # ── 1. Mass flux conservation ─────────────────────────────────────────
    Mdot = 4.0 * np.pi * r**2 * rho * np.abs(v)
    Mdot_median = float(np.median(Mdot[n_boundary:-n_boundary]))
    mass_resid = (Mdot - Mdot_median) / Mdot_median

    # ── 2. Momentum equation (integral form) ──────────────────────────────
    # In steady state:  -d(area * F_mom)/dr * dr  +  P (area_R - area_L)  -  rho GM/r^2 vol = 0
    F_mom_cen = rho * v**2 + P
    F_mom_face = np.zeros(N + 1)
    F_mom_face[1:-1] = 0.5 * (F_mom_cen[:-1] + F_mom_cen[1:])
    F_mom_face[0] = F_mom_cen[0]
    F_mom_face[-1] = F_mom_cen[-1]

    mom_flux_div = -(area[1:] * F_mom_face[1:] - area[:-1] * F_mom_face[:-1])
    mom_pressure_src = P * (area[1:] - area[:-1])
    mom_gravity = -rho * G * M_BH / r**2 * vol
    mom_sum = mom_flux_div + mom_pressure_src + mom_gravity
    mom_scale = np.maximum.reduce(
        [np.abs(mom_flux_div), np.abs(mom_pressure_src), np.abs(mom_gravity)]
    )
    mom_scale = np.maximum(mom_scale, 1e-30)
    mom_resid = mom_sum / mom_scale

    # ── 3. Energy equation (integral form) ────────────────────────────────
    # In steady state:  -d(area * F_e)/dr * dr  -  rho v GM/r^2 vol  -  eps_net vol = 0
    # where F_e = (E + P) v.
    E = 0.5 * rho * v**2 + P / (gamma - 1.0)
    F_e_cen = (E + P) * v
    F_e_face = np.zeros(N + 1)
    F_e_face[1:-1] = 0.5 * (F_e_cen[:-1] + F_e_cen[1:])
    F_e_face[0] = F_e_cen[0]
    F_e_face[-1] = F_e_cen[-1]

    eps_net = cooling.net_emissivity(rho, T, ambient)
    energy_flux_div = -(area[1:] * F_e_face[1:] - area[:-1] * F_e_face[:-1])
    energy_grav = -rho * v * G * M_BH / r**2 * vol
    energy_cool = -eps_net * vol
    energy_sum = energy_flux_div + energy_grav + energy_cool
    energy_scale = np.maximum.reduce(
        [np.abs(energy_flux_div), np.abs(energy_grav), np.abs(energy_cool)]
    )
    energy_scale = np.maximum(energy_scale, 1e-30)
    energy_resid = energy_sum / energy_scale

    # Statistics over interior cells only
    sl = slice(n_boundary, N - n_boundary)
    return SteadyStateResiduals(
        x=r / solution.r_B,
        Mdot=Mdot,
        Mdot_median=Mdot_median,
        mass_residual=mass_resid,
        momentum_residual=mom_resid,
        energy_residual=energy_resid,
        mass_rms=float(np.sqrt(np.mean(mass_resid[sl] ** 2))),
        mass_max=float(np.max(np.abs(mass_resid[sl]))),
        momentum_rms=float(np.sqrt(np.mean(mom_resid[sl] ** 2))),
        momentum_max=float(np.max(np.abs(mom_resid[sl]))),
        energy_rms=float(np.sqrt(np.mean(energy_resid[sl] ** 2))),
        energy_max=float(np.max(np.abs(energy_resid[sl]))),
        n_boundary=n_boundary,
    )
