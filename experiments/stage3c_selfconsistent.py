"""Stage 3c: Self-consistent conservative kinetic source coupling.

Like Stage 3b, but the kinetic closure is slowly updated from the
evolving hydro state rather than frozen at baseline values.

The feedback loop:
  Mdot → kinetic pile-up → S_p(r), S_E(r) → hydro back-reaction → Mdot

The closure is under-relaxed:
  C_{n+1} = (1 - omega) C_n + omega C_kinetic[rho, T, v]_n

Run:
    python experiments/stage3c_selfconsistent.py
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

PHASE1_STEPS = 80_000
PHASE2_STEPS = 200_000
SOURCE_RAMP = 10_000
CLOSURE_UPDATE_INTERVAL = 1_000  # recompute closure every N steps
SNAPSHOT_INTERVAL = 20_000

# Under-relaxation for the closure update
OMEGA = 1e-3

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


def main():
    print("Stage 3c: Self-consistent conservative kinetic closure")
    print("=" * 70)

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

    # Initial condition
    v_phys, T_phys, rho_phys, _ = adiabatic_profile(grid.x, ambient)
    v_phys = -np.abs(v_phys)
    P_init = rho_phys * kB * T_phys / (mu * m_p)
    mom_init = rho_phys * v_phys
    E_init = 0.5 * rho_phys * v_phys**2 + P_init / (gamma - 1.0)
    U = np.array([rho_phys, mom_init, E_init])

    U_eq = U.copy()
    R_eq, _ = hydro_rhs(
        U_eq, grid, M_BH, gamma, mu, T_floor,
        U_eq=U_eq, order=1, limiter="minmod", flux="hll",
    )
    U_outer_m1 = U[:, -1].copy()
    U_outer_m2 = U[:, -2].copy()

    # Load the paper solution for the kinetic closure
    sol_paper = rb.load("examples/paper_sweep_output/mbh_logM-16.00.npz")
    closure = KineticClosure(sol_paper)

    # Initial (frozen) sources from baseline
    i_mid = len(sol_paper.r) // 2
    Mdot_est = (
        4 * np.pi * sol_paper.r[i_mid]**2
        * sol_paper.rho[i_mid] * np.abs(sol_paper.v[i_mid])
    )
    S_p_base, S_E_base, diag0 = closure.compute(
        sol_paper.rho[i_mid], sol_paper.T[i_mid],
        np.abs(sol_paper.v[i_mid]), Mdot_est,
    )

    # Interpolate to solver grid
    S_p_current = np.interp(r, sol_paper.r, S_p_base)
    S_E_current = np.interp(r, sol_paper.r, S_E_base)

    print(f"  Phase 1: {PHASE1_STEPS} steps (baseline, no sources)")
    print(f"  Phase 2: {PHASE2_STEPS} steps (self-consistent sources)")
    print(f"  omega = {OMEGA}")
    print(f"  Closure update every {CLOSURE_UPDATE_INTERVAL} steps")
    print(f"  Initial P_pile/P_0 = {diag0.P_pile_over_P0:.2f}")
    print()

    total_steps = PHASE1_STEPS + PHASE2_STEPS
    mdot_history = []
    chi_history = []
    Ppile_history = []

    mdot_baseline = None
    t0 = _time.time()

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

        rho_arr, v_arr, P_arr, T_arr, cs_arr = get_primitives(
            U, gamma, mu, T_floor
        )
        dt_local = CFL * grid.dr / (np.abs(v_arr) + cs_arr)

        cool_frac = step / COOLING_RAMP if step <= COOLING_RAMP else 1.0

        U_new = U + dt_local[np.newaxis, :] * dU

        # Source terms (phase 2)
        if in_phase2:
            alpha_ramp = min(phase2_step / SOURCE_RAMP, 1.0)
            U_new[1] += alpha_ramp * S_p_current * dt_local
            U_new[2] += alpha_ramp * S_E_current * dt_local

        # Cooling
        if has_cooling:
            dt_cool = dt_local * cool_frac
            U_new = _apply_cooling(
                U_new, dt_cool, cooling, ambient, eps_ambient, gamma, mu
            )

        U_new[:, 0] = U_new[:, 1]
        U_new[:, -1] = U_outer_m1
        U_new[:, -2] = U_outer_m2

        U_new[0] = np.maximum(U_new[0], 1e-30)
        rho_n, v_n, P_n, _, _ = get_primitives(U_new, gamma, mu, T_floor)
        P_floor = rho_n * kB * T_floor / (mu * m_p)
        mask_f = P_n < P_floor
        if np.any(mask_f):
            U_new[2, mask_f] = (
                0.5 * rho_n[mask_f] * v_n[mask_f] ** 2
                + P_floor[mask_f] / (gamma - 1.0)
            )

        U = U_new

        # Save baseline Mdot at end of phase 1
        if step == PHASE1_STEPS:
            rho_s, v_s, _, _, _ = get_primitives(U, gamma, mu, T_floor)
            mdot_baseline = measure_mdot(r, rho_s, v_s, problem.r_B)

        # Self-consistent closure update (phase 2 only)
        if (
            in_phase2
            and phase2_step > SOURCE_RAMP
            and phase2_step % CLOSURE_UPDATE_INTERVAL == 0
        ):
            rho_s, v_s, _, T_s, _ = get_primitives(U, gamma, mu, T_floor)
            mdot_now = measure_mdot(r, rho_s, v_s, problem.r_B)

            # Recompute closure from current hydro state
            i_ref = len(r) // 2
            S_p_new, S_E_new, diag_new = closure.compute(
                rho_s[i_ref], T_s[i_ref], np.abs(v_s[i_ref]), mdot_now,
            )
            S_p_interp = np.interp(r, sol_paper.r, S_p_new)
            S_E_interp = np.interp(r, sol_paper.r, S_E_new)

            # Under-relax
            S_p_current = (1 - OMEGA) * S_p_current + OMEGA * S_p_interp
            S_E_current = (1 - OMEGA) * S_E_current + OMEGA * S_E_interp

        # Snapshot
        if step % SNAPSHOT_INTERVAL == 0 or step == total_steps:
            rho_s, v_s, _, T_s, cs_s = get_primitives(U, gamma, mu, T_floor)
            mdot = measure_mdot(r, rho_s, v_s, problem.r_B)
            elapsed = _time.time() - t0

            if mdot_baseline and mdot_baseline > 0:
                chi = mdot / mdot_baseline
            else:
                chi = 1.0

            phase = "P2" if in_phase2 else "P1"

            # Current P_pile
            P0 = rho_s[len(r) // 2] * kB * T_s[len(r) // 2] / (mu * m_p)
            S_p_int = np.sum(
                S_p_current * 4 * np.pi * r**2 * np.gradient(r)
            )
            P_pile_ratio = S_p_int / (
                4 * np.pi * r[len(r) // 2]**2 * P0
            ) if P0 > 0 else 0

            mdot_history.append((step, mdot))
            chi_history.append(chi)
            Ppile_history.append(P_pile_ratio)

            print(
                f"  [{phase}] {step:>7d}: "
                f"chi={chi:.3f}, "
                f"Mdot={mdot:.4e}, "
                f"P_pile/P0={P_pile_ratio:.2f}, "
                f"t={elapsed:.0f}s"
            )

    # Final summary
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    rho_f, v_f, _, _, _ = get_primitives(U, gamma, mu, T_floor)
    mdot_final = measure_mdot(r, rho_f, v_f, problem.r_B)
    chi_final = mdot_final / mdot_baseline if mdot_baseline > 0 else 1.0

    print(f"\n  Mdot_baseline = {mdot_baseline:.4e} g/s")
    print(f"  Mdot_final    = {mdot_final:.4e} g/s")
    print(f"  chi_self      = {chi_final:.3f}")
    print()

    # Chi history
    p2_chis = [c for c in chi_history if c < 1.0]
    if p2_chis:
        print(f"  Phase 2 chi: mean={np.mean(p2_chis):.3f}, "
              f"min={np.min(p2_chis):.3f}, max={np.max(p2_chis):.3f}")
    print()

    # Comparison
    print("  Stage 3b frozen:        chi = 0.677")
    print(f"  Stage 3c self-consist.: chi = {chi_final:.3f}")
    if chi_final > 0.677:
        print("  -> Self-consistent value is HIGHER (less reduction)")
        print("     as expected: reduced Mdot weakens the pile-up.")
    print()


if __name__ == "__main__":
    main()
