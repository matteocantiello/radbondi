"""Stage 3b: Conservative kinetic source coupling.

Injects the Stage 3a kinetic closure sources S_p(r), S_E(r) into the
1D hydro solver as momentum and energy source terms, replacing the
Dirichlet temperature override.

Three passes:
  Pass 1: frozen closure from baseline, alpha-scan (0, 0.1, 0.25, 0.5, 0.75, 1)
  Pass 2: semi-coupled closure with slow relaxation
  Pass 3: fully coupled feedback (if Pass 2 is stable)

Run:
    python experiments/stage3b_coupled.py
"""

from __future__ import annotations

import time as _time

import numpy as np

import radbondi as rb
from experiments.kinetic_closure import KineticClosure
from radbondi.bondi import adiabatic_profile
from radbondi.constants import G, c_light, kB, m_p
from radbondi.cooling import Cooling
from radbondi.grid import Grid
from radbondi.hydro import get_primitives, hydro_rhs

# ── Configuration ────────────────────────────────────────────────────

M_BH = 1e-16 * rb.M_sun
AMBIENT = rb.presets.solar_core()
COOLING = Cooling.default()

N = 1200
X_MIN = 1e-5
X_MAX = 3.0
CFL = 0.3
COOLING_RAMP = 5_000

PHASE1_STEPS = 80_000   # baseline (no sources)
PHASE2_STEPS = 100_000  # with sources
SOURCE_RAMP = 10_000    # ramp alpha over this many steps
SNAPSHOT_INTERVAL = 10_000

# Alpha values for the frozen-closure scan
ALPHA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

# ── Helpers ──────────────────────────────────────────────────────────

GM = G * M_BH
rS = 2 * GM / c_light**2


def measure_mdot(r, rho, v, r_B, x_lo=0.1, x_hi=0.5):
    x = r / r_B
    mask = (x > x_lo) & (x < x_hi)
    if not np.any(mask):
        return float("nan")
    mdot = 4.0 * np.pi * r**2 * rho * np.abs(v)
    return float(np.median(mdot[mask]))


def _apply_cooling(U, dt_array, cooling, ambient, eps_ambient, gamma, mu):
    T_floor = ambient.T * 0.5
    rho, v, P, T, _ = get_primitives(U, gamma, mu, T_floor)
    N_cells = len(rho)
    coeff = rho * kB / (mu * m_p * (gamma - 1.0))
    e_th = P / (gamma - 1.0)
    skip = 2
    sl = slice(skip, N_cells - skip)
    T_new = T.copy()
    active = T_new[sl] > ambient.T * 1.01
    if not np.any(active):
        return U
    T_work = T_new[sl][active]
    rho_work = rho[sl][active]
    coeff_work = coeff[sl][active]
    e_th_work = e_th[sl][active]
    dt_work = dt_array[sl][active]
    for _ in range(20):
        eps_val = cooling.net_emissivity(rho_work, T_work, ambient, eps_ambient)
        R = coeff_work * T_work - e_th_work + dt_work * eps_val
        dT_fd = T_work * 1e-6 + 1.0
        eps_p = cooling.net_emissivity(
            rho_work, T_work + dT_fd, ambient, eps_ambient
        )
        dR_dT = coeff_work + dt_work * (eps_p - eps_val) / dT_fd
        dR_dT = np.where(np.abs(dR_dT) < 1e-50, 1.0, dR_dT)
        delta_T = -R / dR_dT
        T_work = np.maximum(T_work + delta_T, ambient.T)
        if np.max(np.abs(delta_T) / T_work) < 1e-8:
            break
    T_full = T_new.copy()
    temp = T_full[sl].copy()
    temp[active] = T_work
    T_full[sl] = temp
    U_new = U.copy()
    P_new = rho * kB * T_full / (mu * m_p)
    U_new[2] = 0.5 * rho * v**2 + P_new / (gamma - 1.0)
    return U_new


# ── Main solver loop ─────────────────────────────────────────────────


def run_with_sources(S_p_frozen, S_E_frozen, alpha_target, label=""):
    """Run the solver with frozen kinetic sources at strength alpha."""
    ambient = AMBIENT
    cooling = COOLING
    gamma = ambient.gamma
    mu = ambient.mu
    T_floor = ambient.T * 0.5

    problem = rb.BondiProblem(M_BH=M_BH, ambient=ambient, cooling=cooling)
    grid = Grid.log_spaced(problem.r_B, N=N, x_min=X_MIN, x_max=X_MAX)
    r = grid.r_cen

    has_cooling = bool(cooling.processes)
    eps_ambient = cooling.ambient_emissivity(ambient) if has_cooling else 0.0

    # Initial condition: adiabatic Bondi
    v_phys, T_phys, rho_phys, _ = adiabatic_profile(grid.x, ambient)
    v_phys = -np.abs(v_phys)
    P_init = rho_phys * kB * T_phys / (mu * m_p)
    mom_init = rho_phys * v_phys
    E_init = 0.5 * rho_phys * v_phys**2 + P_init / (gamma - 1.0)
    U = np.array([rho_phys, mom_init, E_init])

    # Well-balanced reference
    U_eq = U.copy()
    R_eq, _ = hydro_rhs(
        U_eq, grid, M_BH, gamma, mu, T_floor,
        U_eq=U_eq, order=1, limiter="minmod", flux="hll",
    )

    U_outer_m1 = U[:, -1].copy()
    U_outer_m2 = U[:, -2].copy()

    total_steps = PHASE1_STEPS + PHASE2_STEPS
    mdot_history = []

    for step in range(1, total_steps + 1):
        in_phase2 = step > PHASE1_STEPS
        phase2_step = step - PHASE1_STEPS if in_phase2 else 0

        # Hydro RHS
        dU, _ = hydro_rhs(
            U, grid, M_BH, gamma, mu, T_floor,
            U_eq=U_eq, order=1, limiter="minmod", flux="hll",
        )
        delta_U = U - U_eq
        scale_U = np.abs(U) + 1e-30
        frac = np.sqrt(np.mean((delta_U / scale_U) ** 2, axis=0))
        R_eq_weight = np.exp(-frac / 0.01)
        dU -= R_eq * R_eq_weight[np.newaxis, :]

        rho, v, P, T, cs = get_primitives(U, gamma, mu, T_floor)
        dt_local = CFL * grid.dr / (np.abs(v) + cs)

        cool_frac = step / COOLING_RAMP if step <= COOLING_RAMP else 1.0

        # Forward Euler
        U_new = U + dt_local[np.newaxis, :] * dU

        # ── KINETIC SOURCE TERMS (phase 2 only) ──
        if in_phase2 and alpha_target > 0:
            # Ramp alpha smoothly
            ramp = min(phase2_step / SOURCE_RAMP, 1.0)
            alpha = ramp * alpha_target

            # Momentum source: opposes infall (sign: positive S_p
            # resists negative v)
            U_new[1] += alpha * S_p_frozen * dt_local

            # Energy source: thermalization heating
            # (S_E goes into internal energy; the momentum work
            # v * S_p is already in the momentum equation)
            U_new[2] += alpha * S_E_frozen * dt_local

        # Implicit cooling
        if has_cooling:
            dt_cool = dt_local * cool_frac
            U_new = _apply_cooling(
                U_new, dt_cool, cooling, ambient, eps_ambient, gamma, mu
            )

        # Inner BC
        U_new[:, 0] = U_new[:, 1]

        # Outer BC
        U_new[:, -1] = U_outer_m1
        U_new[:, -2] = U_outer_m2

        # Floors
        U_new[0] = np.maximum(U_new[0], 1e-30)
        rho_n, v_n, P_n, _, _ = get_primitives(U_new, gamma, mu, T_floor)
        P_floor = rho_n * kB * T_floor / (mu * m_p)
        mask_f = P_n < P_floor
        if np.any(mask_f):
            U_new[2, mask_f] = (
                0.5 * rho_n[mask_f] * v_n[mask_f] ** 2
                + P_floor[mask_f] / (gamma - 1.0)
            )

        # Residual
        diff = U_new - U
        scale = np.abs(U) + 1e-30
        _residual = float(np.sqrt(np.mean((diff / scale) ** 2)))

        U = U_new

        # Snapshot
        if step % SNAPSHOT_INTERVAL == 0 or step == total_steps:
            rho_s, v_s, _, T_s, cs_s = get_primitives(U, gamma, mu, T_floor)
            mdot = measure_mdot(r, rho_s, v_s, problem.r_B)
            mdot_history.append((step, mdot))

    # Final
    rho_f, v_f, _, _, _ = get_primitives(U, gamma, mu, T_floor)
    mdot_final = measure_mdot(r, rho_f, v_f, problem.r_B)
    mdot_baseline = [m for s, m in mdot_history if s == PHASE1_STEPS][0]
    chi = mdot_final / mdot_baseline if mdot_baseline > 0 else float("nan")

    return chi, mdot_final, mdot_baseline, mdot_history


def main():
    print("Stage 3b: Conservative kinetic source coupling")
    print("=" * 65)

    # First, compute the frozen closure from the baseline
    print("\n  Computing kinetic closure from baseline profiles...")
    sol = rb.load("examples/paper_sweep_output/mbh_logM-16.00.npz")
    closure = KineticClosure(sol)

    i_mid = len(sol.r) // 2
    Mdot_est = (
        4 * np.pi * sol.r[i_mid]**2 * sol.rho[i_mid] * np.abs(sol.v[i_mid])
    )
    S_p, S_E, diag = closure.compute(
        sol.rho[i_mid], sol.T[i_mid], np.abs(sol.v[i_mid]), Mdot_est
    )

    print(f"  P_pile/P_0 = {diag.P_pile_over_P0:.2f}")
    print(f"  <xi> = {diag.xi_avg:.1f}")
    print(f"  Populations: cap={diag.f_cap:.3f} refl={diag.f_refl:.3f} "
          f"coll={diag.f_coll:.3f} marg={diag.f_marg:.3f} "
          f"res={diag.f_res:.3f}")
    print(f"  Total S_E: {np.sum(S_E * 4/3 * np.pi * np.gradient(sol.r**3)):.3e} erg/s")
    print()

    # Interpolate sources onto the solver grid
    grid = Grid.log_spaced(
        G * M_BH / (np.sqrt(AMBIENT.gamma * kB * AMBIENT.T / (AMBIENT.mu * m_p)))**2,
        N=N, x_min=X_MIN, x_max=X_MAX,
    )
    S_p_grid = np.interp(grid.r_cen, sol.r, S_p)
    S_E_grid = np.interp(grid.r_cen, sol.r, S_E)

    # Alpha scan with frozen closure
    print("  Pass 1: frozen closure, alpha-scan")
    print(f"  Phase 1: {PHASE1_STEPS} steps (baseline)")
    print(f"  Phase 2: {PHASE2_STEPS} steps (with sources)")
    print()
    print(f"  {'alpha':>7s}  {'chi':>7s}  {'Mdot_final':>12s}  {'Mdot_base':>12s}")
    print("  " + "-" * 45)

    results = []
    for alpha in ALPHA_VALUES:
        t0 = _time.time()
        chi, mdot_f, mdot_b, hist = run_with_sources(
            S_p_grid, S_E_grid, alpha, label=f"alpha={alpha}"
        )
        dt = _time.time() - t0
        results.append((alpha, chi, mdot_f, mdot_b))
        flag = "!" if np.isnan(chi) else " "
        print(
            f" {flag}{alpha:>7.2f}  {chi:>7.3f}  "
            f"{mdot_f:>12.4e}  {mdot_b:>12.4e}  ({dt:.0f}s)"
        )

    print("\n" + "=" * 65)
    print("RESULT: chi(alpha) = Mdot / Mdot_B")
    print("=" * 65)
    print()
    for alpha, chi, _, _ in results:
        bar = "#" * int(chi * 50) if not np.isnan(chi) and chi > 0 else "NaN"
        print(f"  alpha={alpha:.2f}: chi={chi:.3f}  {bar}")
    print()


if __name__ == "__main__":
    main()
