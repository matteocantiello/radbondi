"""Primitive variables, slope limiters, and the HLL/Rusanov Riemann solvers.

The conservative state is ``U = [rho, rho*v, e]`` where ``e`` is the total
energy density ``e = (1/2) rho v^2 + P / (gamma - 1)``.
"""

from __future__ import annotations

import numpy as np

from radbondi.constants import G, kB, m_p


def get_primitives(U, gamma: float, mu: float, T_floor: float):
    """Recover primitive variables (rho, v, P, T, cs) from conservatives U.

    Floors are applied to keep the state physical: rho >= 1e-30,
    P >= 1e-30, and T >= T_floor.
    """
    rho = np.maximum(U[0], 1e-30)
    v = U[1] / rho
    P = np.maximum((gamma - 1.0) * (U[2] - 0.5 * rho * v**2), 1e-30)
    T = np.maximum(P * mu * m_p / (rho * kB), T_floor)
    cs = np.sqrt(gamma * P / rho)
    return rho, v, P, T, cs


def minmod(a, b):
    """Minmod slope limiter."""
    return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def mc_limiter(a, b):
    """Monotonized-central slope limiter (less diffusive than minmod)."""
    c = 0.5 * (a + b)
    return np.where(
        a * b > 0,
        np.sign(a) * np.minimum(np.minimum(2 * np.abs(a), 2 * np.abs(b)), np.abs(c)),
        0.0,
    )


def compute_fluxes(
    U,
    grid,
    gamma: float,
    mu: float,
    T_floor: float,
    *,
    U_eq=None,
    order: int = 2,
    limiter: str = "minmod",
    flux: str = "hll",
):
    """Finite-volume interface fluxes with optional MUSCL reconstruction.

    Parameters
    ----------
    U : ndarray, shape (3, N)
        Conservative state.
    grid : Grid
    gamma, mu, T_floor : float
        Equation-of-state parameters.
    U_eq : ndarray or None
        Equilibrium state for well-balanced reconstruction. If provided, MUSCL
        slopes are computed on the deviation ``U - U_eq`` (in primitive space)
        and the HLL/Rusanov diffusion uses the deviation rather than the raw
        state — this preserves the equilibrium to machine precision.
    order : 1 or 2
        Spatial order. ``order=2`` enables MUSCL.
    limiter : 'minmod' or 'mc'
    flux : 'hll' or 'rusanov'

    Returns
    -------
    F_iface : ndarray, shape (3, N+1)
        Interface fluxes.
    S_max : float
        Maximum wave speed (for diagnostic / global-CFL use).
    """
    N = grid.N
    rho, v, P, _, cs = get_primitives(U, gamma, mu, T_floor)
    S_max = float((np.abs(v) + cs).max())

    if order == 2 and N > 4:
        # MUSCL reconstruction on PRIMITIVE variables (rho, v, P)
        W = np.array([rho, v, P])

        if U_eq is not None:
            rho_eq, v_eq, P_eq, _, _ = get_primitives(U_eq, gamma, mu, T_floor)
            W_eq = np.array([rho_eq, v_eq, P_eq])
            dW_base = W - W_eq
        else:
            dW_base = W

        slopes = np.zeros_like(dW_base)
        lim_func = mc_limiter if limiter == "mc" else minmod
        for k in range(3):
            dL = dW_base[k, 1:-1] - dW_base[k, :-2]
            dR = dW_base[k, 2:] - dW_base[k, 1:-1]
            slopes[k, 1:-1] = lim_func(dL, dR)

        dW_L = dW_base[:, :-1] + 0.5 * slopes[:, :-1]
        dW_R = dW_base[:, 1:] - 0.5 * slopes[:, 1:]

        if U_eq is not None:
            W_face_L = dW_L + W_eq[:, :-1]
            W_face_R = dW_R + W_eq[:, 1:]
        else:
            W_face_L = dW_L
            W_face_R = dW_R

        rho_L = np.maximum(W_face_L[0], 1e-30)
        v_L = W_face_L[1]
        P_L = np.maximum(W_face_L[2], 1e-30)
        rho_R = np.maximum(W_face_R[0], 1e-30)
        v_R = W_face_R[1]
        P_R = np.maximum(W_face_R[2], 1e-30)

        cs_L = np.sqrt(gamma * P_L / rho_L)
        cs_R = np.sqrt(gamma * P_R / rho_R)

        UL = np.array([rho_L, rho_L * v_L, 0.5 * rho_L * v_L**2 + P_L / (gamma - 1.0)])
        UR = np.array([rho_R, rho_R * v_R, 0.5 * rho_R * v_R**2 + P_R / (gamma - 1.0)])

        if U_eq is not None:
            diff_RL = (UR - U_eq[:, 1:]) - (UL - U_eq[:, :-1])
        else:
            diff_RL = UR - UL
    else:
        # First-order: piecewise constant
        rho_L, v_L, P_L, cs_L = rho[:-1], v[:-1], P[:-1], cs[:-1]
        rho_R, v_R, P_R, cs_R = rho[1:], v[1:], P[1:], cs[1:]
        UL = U[:, :-1]
        UR = U[:, 1:]
        if U_eq is not None:
            dU = U - U_eq
            diff_RL = dU[:, 1:] - dU[:, :-1]
        else:
            diff_RL = UR - UL

    # Physical fluxes at left/right interface states
    F_L = np.array([rho_L * v_L, rho_L * v_L**2 + P_L, (UL[2] + P_L) * v_L])
    F_R = np.array([rho_R * v_R, rho_R * v_R**2 + P_R, (UR[2] + P_R) * v_R])

    F_iface = np.zeros((3, N + 1))

    if flux == "hll":
        S_l = np.minimum(v_L - cs_L, v_R - cs_R)
        S_r = np.maximum(v_L + cs_L, v_R + cs_R)
        all_right = S_l >= 0
        all_left = S_r <= 0
        dS = np.where(np.abs(S_r - S_l) < 1e-30, 1e-30, S_r - S_l)
        F_hll = (
            S_r[np.newaxis, :] * F_L
            - S_l[np.newaxis, :] * F_R
            + S_l[np.newaxis, :] * S_r[np.newaxis, :] * diff_RL
        ) / dS[np.newaxis, :]
        F_int = np.where(
            all_right[np.newaxis, :], F_L, np.where(all_left[np.newaxis, :], F_R, F_hll)
        )
        F_iface[:, 1:N] = F_int
    elif flux == "rusanov":
        S_max_iface = np.maximum(np.abs(v_L) + cs_L, np.abs(v_R) + cs_R)
        F_iface[:, 1:N] = 0.5 * (F_L + F_R) - 0.5 * S_max_iface[np.newaxis, :] * diff_RL
    else:
        raise ValueError(f"Unknown flux solver {flux!r}; use 'hll' or 'rusanov'.")

    # Boundary fluxes (first-order upwind, set from cell-center state)
    F_phys = np.array([rho * v, rho * v**2 + P, (U[2] + P) * v])
    F_iface[:, 0] = F_phys[:, 0]
    F_iface[:, N] = F_phys[:, -1]

    return F_iface, S_max


def hydro_rhs(
    U,
    grid,
    M_BH: float,
    gamma: float,
    mu: float,
    T_floor: float,
    *,
    U_eq=None,
    order: int = 2,
    limiter: str = "minmod",
    flux: str = "hll",
):
    """Right-hand side of the spherical Euler equations (hydro only).

    Returns
    -------
    dU : ndarray, shape (3, N)
        Time derivative dU/dt.
    dt_global : float
        CFL-limited global timestep [s].
    """
    rho, v, P, _, cs = get_primitives(U, gamma, mu, T_floor)
    F_iface, _ = compute_fluxes(
        U, grid, gamma, mu, T_floor, U_eq=U_eq, order=order, limiter=limiter, flux=flux
    )
    area = grid.area
    vol = grid.vol
    dU = -(area[1:] * F_iface[:, 1:] - area[:-1] * F_iface[:, :-1]) / vol[np.newaxis, :]
    # Geometric pressure source from spherical divergence
    dU[1] += P * (area[1:] - area[:-1]) / vol
    # Gravity
    dU[1] -= rho * G * M_BH / grid.r_cen**2
    dU[2] -= rho * v * G * M_BH / grid.r_cen**2
    dt_global = float((grid.dr / (np.abs(v) + cs)).min())
    return dU, dt_global
