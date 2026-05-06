"""Stage 4: Mass dependence of the self-consistent hydro-kinetic correction.

Runs the Stage 3c self-consistent closure for multiple PBH masses to
produce chi(M) = Mdot / Mdot_B across the mass range where the pile-up
could matter.

Expected: chi ~ 0.7 near 10^{-16} M_sun, chi → 1 as r_sonic > r_coll.

Run:
    python experiments/stage4_mass_scan.py
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np

import radbondi as rb
from experiments.kinetic_closure import KineticClosure
from radbondi.bondi import adiabatic_profile
from radbondi.constants import G, c_light, kB, m_p
from radbondi.cooling import Cooling
from radbondi.grid import Grid
from radbondi.hydro import get_primitives, hydro_rhs

# ── Configuration ────────────────────────────────────────────────────

AMBIENT = rb.presets.solar_core()
COOLING = Cooling.default()

N = 1200
X_MIN = 1e-5
X_MAX = 3.0
CFL = 0.3
COOLING_RAMP = 5_000

PHASE1_STEPS = 80_000
PHASE2_STEPS = 150_000
SOURCE_RAMP = 10_000
CLOSURE_UPDATE_INTERVAL = 1_000
OMEGA = 1e-3

# Mass grid
LOG_MASSES = [-16.1, -16.0, -15.8, -15.6, -15.3]

SWEEP_DIR = Path("examples/paper_sweep_output")


def _apply_cooling(U, dt_array, cooling, ambient, eps_ambient, gamma, mu):
    T_floor = ambient.T * 0.5
    rho_v, v_v, P_v, T_v, _ = get_primitives(U, gamma, mu, T_floor)
    N_cells = len(rho_v)
    coeff = rho_v * kB / (mu * m_p * (gamma - 1.0))
    e_th = P_v / (gamma - 1.0)
    skip = 2
    sl = slice(skip, N_cells - skip)
    T_new = T_v.copy()
    active = T_new[sl] > ambient.T * 1.01
    if not np.any(active):
        return U
    T_work = T_new[sl][active]
    rho_work = rho_v[sl][active]
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
    P_new = rho_v * kB * T_full / (mu * m_p)
    U_new[2] = 0.5 * rho_v * v_v**2 + P_new / (gamma - 1.0)
    return U_new


def measure_mdot(r, rho_arr, v_arr, r_B, x_lo=0.1, x_hi=0.5):
    x = r / r_B
    mask = (x > x_lo) & (x < x_hi)
    if not np.any(mask):
        return float("nan")
    mdot = 4.0 * np.pi * r**2 * rho_arr * np.abs(v_arr)
    return float(np.median(mdot[mask]))


def run_stage3c_for_mass(logM):
    """Run the full Stage 3c for one PBH mass. Returns chi, diagnostics."""
    M_BH = 10.0**logM * rb.M_sun
    GM_val = G * M_BH
    rS = 2 * GM_val / c_light**2

    ambient = AMBIENT
    cooling = COOLING
    gamma = ambient.gamma
    mu = ambient.mu
    T_floor = ambient.T * 0.5

    problem = rb.BondiProblem(M_BH=M_BH, ambient=ambient, cooling=cooling)
    grid = Grid.log_spaced(problem.r_B, N=N, x_min=X_MIN, x_max=X_MAX)
    r = grid.r_cen

    has_cooling = bool(cooling.processes)
    eps_ambient_val = cooling.ambient_emissivity(ambient) if has_cooling else 0.0

    # Initial condition
    v_phys, T_phys, rho_phys, _ = adiabatic_profile(grid.x, ambient)
    v_phys = -np.abs(v_phys)
    P_init = rho_phys * kB * T_phys / (mu * m_p)
    U = np.array([rho_phys, rho_phys * v_phys,
                   0.5 * rho_phys * v_phys**2 + P_init / (gamma - 1.0)])

    U_eq = U.copy()
    R_eq, _ = hydro_rhs(
        U_eq, grid, M_BH, gamma, mu, T_floor,
        U_eq=U_eq, order=1, limiter="minmod", flux="hll",
    )
    U_outer_m1 = U[:, -1].copy()
    U_outer_m2 = U[:, -2].copy()

    # Load paper solution for closure (find the closest mass)
    candidates = sorted(SWEEP_DIR.glob(f"mbh_logM{logM:+.2f}*.npz"))
    if not candidates:
        candidates = sorted(SWEEP_DIR.glob(f"mbh_logM{logM:+06.2f}*.npz"))
    if not candidates:
        # Try looser match
        all_npz = sorted(SWEEP_DIR.glob("mbh_logM*.npz"))
        best = min(all_npz, key=lambda p: abs(
            float(p.stem.split("logM")[1]) - logM
        ))
        candidates = [best]

    sol_paper = rb.load(str(candidates[0]))
    closure = KineticClosure(sol_paper)

    # Initial closure
    i_mid = len(sol_paper.r) // 2
    Mdot_est = (
        4 * np.pi * sol_paper.r[i_mid]**2
        * sol_paper.rho[i_mid] * np.abs(sol_paper.v[i_mid])
    )
    S_p_base, S_E_base, diag0 = closure.compute(
        sol_paper.rho[i_mid], sol_paper.T[i_mid],
        np.abs(sol_paper.v[i_mid]), Mdot_est,
    )

    S_p_current = np.interp(r, sol_paper.r, S_p_base)
    S_E_current = np.interp(r, sol_paper.r, S_E_base)

    # Find r_coll and r_sonic from paper profiles
    cs_paper = np.sqrt(gamma * kB * sol_paper.T / (mu * m_p))
    Mach_paper = np.abs(sol_paper.v) / cs_paper
    i_sonic = np.argmin(np.abs(Mach_paper - 1.0))

    E4 = (4.803e-10) ** 4
    n_e = (sol_paper.ambient_X + 0.5 * sol_paper.ambient_Y) * sol_paper.rho / m_p
    lam = (kB * sol_paper.T)**2 / (np.pi * n_e * E4 * 5.0)
    Kn = lam / sol_paper.r
    crossings = np.where(np.diff(np.sign(Kn - 1.0)))[0]
    r_coll_rS = sol_paper.r[crossings[-1]] / rS if len(crossings) > 0 else float("nan")
    r_sonic_rS = sol_paper.r[i_sonic] / rS

    total_steps = PHASE1_STEPS + PHASE2_STEPS
    mdot_baseline = None
    chi_final = float("nan")

    for step in range(1, total_steps + 1):
        in_phase2 = step > PHASE1_STEPS
        phase2_step = step - PHASE1_STEPS if in_phase2 else 0

        dU, _ = hydro_rhs(
            U, grid, M_BH, gamma, mu, T_floor,
            U_eq=U_eq, order=1, limiter="minmod", flux="hll",
        )
        delta_U = U - U_eq
        scale_U = np.abs(U) + 1e-30
        frac = np.sqrt(np.mean((delta_U / scale_U) ** 2, axis=0))
        R_eq_weight = np.exp(-frac / 0.01)
        dU -= R_eq * R_eq_weight[np.newaxis, :]

        rho_arr, v_arr, _, _, cs_arr = get_primitives(U, gamma, mu, T_floor)
        dt_local = CFL * grid.dr / (np.abs(v_arr) + cs_arr)
        cool_frac = step / COOLING_RAMP if step <= COOLING_RAMP else 1.0

        U_new = U + dt_local[np.newaxis, :] * dU

        if in_phase2:
            alpha_ramp = min(phase2_step / SOURCE_RAMP, 1.0)
            U_new[1] += alpha_ramp * S_p_current * dt_local
            U_new[2] += alpha_ramp * S_E_current * dt_local

        if has_cooling:
            U_new = _apply_cooling(
                U_new, dt_local * cool_frac, cooling, ambient,
                eps_ambient_val, gamma, mu,
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
                0.5 * rho_n[mask_f] * v_n[mask_f]**2
                + P_floor[mask_f] / (gamma - 1.0)
            )

        U = U_new

        if step == PHASE1_STEPS:
            rho_s, v_s, _, _, _ = get_primitives(U, gamma, mu, T_floor)
            mdot_baseline = measure_mdot(r, rho_s, v_s, problem.r_B)

        # Self-consistent closure update
        if (
            in_phase2
            and phase2_step > SOURCE_RAMP
            and phase2_step % CLOSURE_UPDATE_INTERVAL == 0
        ):
            rho_s, v_s, _, T_s, _ = get_primitives(U, gamma, mu, T_floor)
            mdot_now = measure_mdot(r, rho_s, v_s, problem.r_B)
            i_ref = len(r) // 2
            S_p_new, S_E_new, _ = closure.compute(
                rho_s[i_ref], T_s[i_ref], np.abs(v_s[i_ref]), mdot_now,
            )
            S_p_interp = np.interp(r, sol_paper.r, S_p_new)
            S_E_interp = np.interp(r, sol_paper.r, S_E_new)
            S_p_current = (1 - OMEGA) * S_p_current + OMEGA * S_p_interp
            S_E_current = (1 - OMEGA) * S_E_current + OMEGA * S_E_interp

    # Final
    rho_f, v_f, _, _, _ = get_primitives(U, gamma, mu, T_floor)
    mdot_final = measure_mdot(r, rho_f, v_f, problem.r_B)
    if mdot_baseline and mdot_baseline > 0:
        chi_final = mdot_final / mdot_baseline

    return {
        "logM": logM,
        "chi": chi_final,
        "Mdot_B": mdot_baseline,
        "Mdot_final": mdot_final,
        "r_coll_rS": r_coll_rS,
        "r_sonic_rS": r_sonic_rS,
        "P_pile_P0": diag0.P_pile_over_P0,
        "f_coll": diag0.f_coll,
        "f_res": diag0.f_res,
        "xi_avg": diag0.xi_avg,
    }


def main():
    print("Stage 4: Mass dependence of chi(M)")
    print("=" * 70)
    print(f"  Masses: {LOG_MASSES}")
    print(f"  N={N}, Phase1={PHASE1_STEPS}, Phase2={PHASE2_STEPS}")
    print(f"  omega={OMEGA}")
    print()

    print(f"  {'logM':>6s}  {'chi':>6s}  {'r_coll':>8s}  {'r_sonic':>8s}  "
          f"{'P/P0':>6s}  {'f_coll':>6s}  {'f_res':>6s}  {'xi':>5s}  "
          f"{'time':>6s}")
    print("  " + "-" * 65)

    results = []
    for logM in LOG_MASSES:
        t0 = _time.time()
        res = run_stage3c_for_mass(logM)
        dt = _time.time() - t0
        results.append(res)

        print(
            f"  {logM:>+6.1f}  {res['chi']:>6.3f}  "
            f"{res['r_coll_rS']:>8.0f}  {res['r_sonic_rS']:>8.0f}  "
            f"{res['P_pile_P0']:>6.1f}  {res['f_coll']:>6.3f}  "
            f"{res['f_res']:>6.3f}  {res['xi_avg']:>5.1f}  "
            f"{dt:>5.0f}s"
        )

    print("\n" + "=" * 70)
    print("RESULT: chi(M)")
    print("=" * 70)
    print()
    for res in results:
        bar = "#" * max(1, int(res["chi"] * 50)) if not np.isnan(res["chi"]) else "?"
        label = ""
        if res["r_coll_rS"] < res["r_sonic_rS"]:
            label = " (pile-up supersonic — no correction expected)"
        print(f"  logM={res['logM']:>+5.1f}: chi={res['chi']:.3f}  {bar}{label}")
    print()


if __name__ == "__main__":
    main()
