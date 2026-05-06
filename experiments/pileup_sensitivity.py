"""Measure the sensitivity of Mdot to a pressure enhancement at r_coll.

For each value of xi (pile-up factor), we:
1. Start from the converged baseline cooled Bondi solution
2. Re-evolve with the pressure (temperature) gradually ramped to
   xi * T_baseline in a Gaussian shell centered on r_coll
3. Measure the new Mdot in the subsonic region above r_coll

This directly answers: if the thermalization layer maintains a pressure
enhancement of order xi at r_coll, by how much does Mdot change?
"""

from __future__ import annotations

import numpy as np

import radbondi as rb
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
CFL_BASELINE = 0.4
CFL_PERTURB = 0.3
COOLING_RAMP = 5_000

BASELINE_STEPS = 80_000
PERTURB_STEPS = 60_000
OVERRIDE_RAMP = 5_000  # ramp the pressure override on gradually

XI_VALUES = [1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

# ── Helpers ──────────────────────────────────────────────────────────


def measure_mdot(r, rho, v, r_B, x_lo=0.1, x_hi=0.5):
    x = r / r_B
    mask = (x > x_lo) & (x < x_hi)
    if not np.any(mask):
        return float("nan")
    mdot = 4.0 * np.pi * r**2 * rho * np.abs(v)
    return float(np.median(mdot[mask]))


def find_r_coll(r, T, rho, X, Y):
    e4 = (4.803e-10) ** 4
    ln_Lambda = 5.0
    n_e = (X + 0.5 * Y) * rho / m_p
    lam = (kB * T) ** 2 / (np.pi * n_e * e4 * ln_Lambda)
    ratio = lam / r
    crossings = np.where(np.diff(np.sign(ratio - 1.0)))[0]
    if len(crossings) == 0:
        return r[len(r) // 2]
    return float(r[crossings[-1]])


# ── Baseline solve ───────────────────────────────────────────────────


def run_baseline():
    problem = rb.BondiProblem(M_BH=M_BH, ambient=AMBIENT, cooling=COOLING)
    cfg = rb.SolverConfig(
        N=N,
        x_min=X_MIN,
        x_max=X_MAX,
        n_steps=BASELINE_STEPS,
        cooling_ramp_steps=COOLING_RAMP,
        order=1,
        flux="hll",
        CFL=CFL_BASELINE,
        convergence_tol=0.0,
        snapshot_interval=BASELINE_STEPS,
        verbose=False,
    )
    sol = problem.solve(cfg)
    return problem, sol


# ── Implicit cooling (copied from solver.py) ─────────────────────────


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


# ── Perturbed evolution ──────────────────────────────────────────────


def evolve_with_pressure_override(problem, baseline_sol, xi, r_coll,
                                   n_steps=PERTURB_STEPS):
    ambient = problem.ambient
    cooling = problem.cooling
    gamma = ambient.gamma
    mu = ambient.mu
    T_floor = ambient.T * 0.5

    grid = Grid.log_spaced(problem.r_B, N=N, x_min=X_MIN, x_max=X_MAX)
    has_cooling = bool(cooling.processes)
    eps_ambient = cooling.ambient_emissivity(ambient) if has_cooling else 0.0

    T_baseline = baseline_sol.T

    # Narrower Gaussian shell (0.3 r_coll), only above r_coll
    sigma_shell = r_coll * 0.3
    weight = np.exp(-0.5 * ((grid.r_cen - r_coll) / sigma_shell) ** 2)
    weight[grid.r_cen < r_coll * 0.3] = 0.0
    # Normalize peak to 1
    weight /= weight.max() if weight.max() > 0 else 1.0

    U = baseline_sol.U.copy()

    # Well-balanced reference
    v_phys, T_phys, rho_phys, _ = adiabatic_profile(grid.x, ambient)
    v_phys = -np.abs(v_phys)
    P_ad = rho_phys * kB * T_phys / (mu * m_p)
    mom_ad = rho_phys * v_phys
    E_ad = 0.5 * rho_phys * v_phys**2 + P_ad / (gamma - 1.0)
    U_eq = np.array([rho_phys, mom_ad, E_ad])
    R_eq, _ = hydro_rhs(
        U_eq, grid, M_BH, gamma, mu, T_floor,
        U_eq=U_eq, order=1, limiter="minmod", flux="hll",
    )

    U_outer_m1 = U[:, -1].copy()
    U_outer_m2 = U[:, -2].copy()

    residuals = []

    for step in range(1, n_steps + 1):
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
        dt_local = CFL_PERTURB * grid.dr / (np.abs(v) + cs)

        U_new = U + dt_local[np.newaxis, :] * dU

        if has_cooling:
            U_new = _apply_cooling(
                U_new, dt_local, cooling, ambient, eps_ambient, gamma, mu
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

        # ── PRESSURE OVERRIDE (ramped) ──
        if xi > 1.0:
            ramp = min(step / OVERRIDE_RAMP, 1.0)
            xi_eff = 1.0 + ramp * (xi - 1.0)

            rho_cur, v_cur, _, T_cur, _ = get_primitives(
                U_new, gamma, mu, T_floor
            )
            T_target = T_baseline * xi_eff
            T_override = T_cur + weight * (T_target - T_cur)
            T_override = np.maximum(T_override, T_floor)
            P_new = rho_cur * kB * T_override / (mu * m_p)
            U_new[2] = 0.5 * rho_cur * v_cur**2 + P_new / (gamma - 1.0)

        diff = U_new - U
        scale = np.abs(U) + 1e-30
        residual = float(np.sqrt(np.mean((diff / scale) ** 2)))
        residuals.append(residual)

        U = U_new

    rho, v, _, _, _ = get_primitives(U, gamma, mu, T_floor)
    mdot = measure_mdot(grid.r_cen, rho, v, problem.r_B)
    return mdot, residuals


# ── Main ─────────────────────────────────────────────────────────────


def main():
    import time as _time

    print("Pile-up sensitivity experiment (v2: ramped, finer grid)")
    print("=" * 65)

    print("\n1. Running baseline solve...")
    t0 = _time.time()
    problem, baseline = run_baseline()
    dt_base = _time.time() - t0
    r_S = 2 * G * M_BH / c_light**2
    r_coll = find_r_coll(
        baseline.r, baseline.T, baseline.rho, AMBIENT.X, AMBIENT.Y
    )
    mdot_baseline = measure_mdot(
        baseline.r, baseline.rho, baseline.v, problem.r_B
    )

    print(f"   r_coll    = {r_coll:.3e} cm  ({r_coll / r_S:.0f} r_S)")
    print(f"   Mdot_base = {mdot_baseline:.4e} g/s")
    print(f"   eta_base  = {baseline.eta:.4e}")
    print(f"   time      = {dt_base:.0f} s")

    print(f"\n2. Perturbed solves ({PERTURB_STEPS} steps, "
          f"CFL={CFL_PERTURB}, ramp={OVERRIDE_RAMP})")
    print(f"\n   {'xi':>5s}  {'Mdot':>12s}  {'Mdot/Mdot_0':>12s}  "
          f"{'% change':>9s}  {'res':>10s}  {'time':>6s}")
    print("   " + "-" * 60)

    results = []
    for xi in XI_VALUES:
        t0 = _time.time()
        mdot, res = evolve_with_pressure_override(
            problem, baseline, xi, r_coll
        )
        dt = _time.time() - t0
        ratio = mdot / mdot_baseline if not np.isnan(mdot) else float("nan")
        pct = (ratio - 1.0) * 100 if not np.isnan(ratio) else float("nan")
        final_res = res[-1] if len(res) > 0 else float("nan")
        results.append((xi, mdot, ratio, final_res))
        flag = "!" if np.isnan(mdot) else " "
        print(
            f"  {flag}{xi:>5.1f}  {mdot:>12.4e}  {ratio:>12.4f}  "
            f"{pct:>+8.1f}%  {final_res:>10.3e}  {dt:>5.0f}s"
        )

    # Fit power law F(x) = x^{-alpha} to the valid points
    print("\n" + "=" * 65)
    valid = [(xi, r) for xi, _, r, res in results
             if not np.isnan(r) and xi > 1.0 and not np.isnan(res)]
    if len(valid) >= 2:
        xs = np.array([xi for xi, _ in valid])
        ms = np.array([r for _, r in valid])
        # F(xi) = Mdot(xi)/Mdot_0 = xi^{-alpha}
        # log(m) = -alpha log(xi)
        log_x = np.log(xs)
        log_m = np.log(ms)
        alpha, log_F0 = np.polyfit(log_x, log_m, 1)
        alpha = -alpha  # convention: F = x^{-alpha}
        print(f"Power-law fit: F(xi) = xi^{{-{alpha:.3f}}} "
              f"(from {len(valid)} points)")
        print(f"  (F_0 prefactor = {np.exp(log_F0):.3f}, "
              f"ideally 1.0)")
        print()

        # Self-consistent solution
        exp_sc = -alpha / (1 + alpha / 2)
        print("Self-consistent m = xi_0^{" + f"{exp_sc:.3f}" + "}")
        print()
        print(f"  {'xi_0':>6s}  {'m=Mdot/Mdot_B':>14s}  {'reduction':>10s}")
        print("  " + "-" * 35)
        for xi0 in [3, 5, 7, 10, 13]:
            m = xi0**exp_sc
            print(f"  {xi0:>6d}  {m:>14.3f}  {(1 - m) * 100:>9.1f}%")
    else:
        print("Not enough valid points for a fit.")

    print()


if __name__ == "__main__":
    main()
