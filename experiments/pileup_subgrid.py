"""Self-consistent sub-grid model for the collisionless pile-up.

Instead of imposing a fixed pressure enhancement, we:
1. Evolve the full 1D Euler+cooling equations (same as the standard solver)
2. After each step, compute the pile-up factor xi from the CURRENT flow
   profiles using the bremsstrahlung self-limitation formula
3. Apply the corresponding pressure enhancement at r_coll

The feedback loop is built in: if Mdot drops, rho drops, dE/E drops,
xi increases, but P_coll = xi * rho * kT/(mu mp) ~ sqrt(Mdot)
DECREASES, easing the pressure and letting Mdot recover.

Run:
    python experiments/pileup_subgrid.py
"""

from __future__ import annotations

import time as _time

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
CFL = 0.3
COOLING_RAMP = 5_000

# Run in two phases:
#   Phase 1: standard Bondi+cooling (no sub-grid) to get the cooled baseline
#   Phase 2: continue with the sub-grid pressure override
PHASE1_STEPS = 80_000
PHASE2_STEPS = 100_000
SUBGRID_RAMP = 10_000   # ramp the sub-grid on over this many steps in phase 2
XI_RELAXATION = 0.001   # under-relaxation: xi_smooth += omega * (xi_raw - xi_smooth)

SNAPSHOT_INTERVAL = 10_000

# ── Helpers ──────────────────────────────────────────────────────────

E4 = (4.803e-10) ** 4
LN_LAMBDA = 5.0


def compute_mfp(T, rho, X, Y):
    """Coulomb mean free path at each cell."""
    n_e = (X + 0.5 * Y) * rho / m_p
    return (kB * T) ** 2 / (np.pi * n_e * E4 * LN_LAMBDA)


def find_r_coll_index(r, mfp):
    """Index of the outermost cell where lambda_mfp / r crosses 1."""
    ratio = mfp / r
    crossings = np.where(np.diff(np.sign(ratio - 1.0)))[0]
    if len(crossings) == 0:
        return len(r) // 2
    return int(crossings[-1])


def compute_xi(rho, T, v, r, i_ref, cooling, ambient):
    """Compute the pile-up factor xi at a reference radius.

    xi = sqrt(2 / (dE/E)_0) where dE/E is the fractional kinetic
    energy loss per pericenter passage at the unperturbed density.
    """
    eps = float(cooling.total_emissivity(rho[i_ref], T[i_ref], ambient)[0])
    v_ref = max(abs(v[i_ref]), 1e-10)
    dEE = (eps / max(rho[i_ref], 1e-30)) * (r[i_ref] / v_ref) / (0.5 * v_ref**2)
    dEE = max(dEE, 1e-30)
    xi = np.sqrt(2.0 / dEE)
    return xi, dEE


def measure_mdot(r, rho, v, r_B, x_lo=0.1, x_hi=0.5):
    x = r / r_B
    mask = (x > x_lo) & (x < x_hi)
    if not np.any(mask):
        return float("nan")
    mdot = 4.0 * np.pi * r**2 * rho * np.abs(v)
    return float(np.median(mdot[mask]))


# ── Implicit cooling (from solver.py) ────────────────────────────────


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


def run():
    ambient = AMBIENT
    cooling = COOLING
    gamma = ambient.gamma
    mu = ambient.mu
    T_floor = ambient.T * 0.5
    GM = G * M_BH

    problem = rb.BondiProblem(M_BH=M_BH, ambient=ambient, cooling=cooling)
    grid = Grid.log_spaced(problem.r_B, N=N, x_min=X_MIN, x_max=X_MAX)
    r = grid.r_cen
    r_S = 2 * GM / c_light**2

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

    # These will be set at the end of phase 1
    T_baseline = None
    i_sonic_baseline = None
    i_coll_baseline = None
    xi_smooth = 1.0  # under-relaxed xi (starts at 1 = no enhancement)

    # Tracking
    mdot_history = []
    xi_history = []
    r_coll_history = []

    print("Sub-grid pile-up experiment")
    print("=" * 70)
    print(f"  Phase 1: {PHASE1_STEPS} steps (standard Bondi+cooling, no sub-grid)")
    print(f"  Phase 2: {PHASE2_STEPS} steps (sub-grid on, ramp={SUBGRID_RAMP})")
    print(f"  N={N}, CFL={CFL}, order=1")
    print()

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

        # Local dt
        rho, v, P, T, cs = get_primitives(U, gamma, mu, T_floor)
        dt_local = CFL * grid.dr / (np.abs(v) + cs)

        # Cooling ramp (phase 1 only)
        cool_frac = (
            step / COOLING_RAMP
            if step <= COOLING_RAMP
            else 1.0
        )

        # Forward Euler
        U_new = U + dt_local[np.newaxis, :] * dU

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

        # ── SAVE BASELINE AT END OF PHASE 1 ──
        if step == PHASE1_STEPS:
            rho_bl, v_bl, _, T_bl, cs_bl = get_primitives(
                U_new, gamma, mu, T_floor
            )
            T_baseline = T_bl.copy()
            # Baseline sonic point and r_coll (fixed for phase 2)
            Mach_bl = np.abs(v_bl) / cs_bl
            i_sonic_baseline = int(np.argmin(np.abs(Mach_bl - 1.0)))
            mfp_bl = compute_mfp(T_bl, rho_bl, ambient.X, ambient.Y)
            i_coll_baseline = find_r_coll_index(r, mfp_bl)
            # Baseline xi (for reference)
            xi_bl, dEE_bl = compute_xi(
                rho_bl, T_bl, v_bl, r, i_sonic_baseline, cooling, ambient
            )
            rho_sonic_baseline = rho_bl[i_sonic_baseline]
            T_sonic_baseline = T_bl[i_sonic_baseline]
            v_sonic_baseline = abs(v_bl[i_sonic_baseline])
            print(f"\n  Baseline frozen: r_coll={r[i_coll_baseline]/r_S:.0f} r_S, "
                  f"r_sonic={r[i_sonic_baseline]/r_S:.0f} r_S, "
                  f"xi_baseline={xi_bl:.2f}, dE/E={dEE_bl:.4f}")
            print(f"  At r_sonic: rho={rho_sonic_baseline:.3e}, "
                  f"T={T_sonic_baseline:.3e}, v={v_sonic_baseline:.3e}\n")

        # ── SUB-GRID: DIRICHLET INNER BC AT r_coll (phase 2 only) ──
        if in_phase2 and T_baseline is not None:
            rho_sg, v_sg, _, T_sg, _ = get_primitives(
                U_new, gamma, mu, T_floor
            )

            # Compute xi_raw from CURRENT density at the BASELINE sonic point
            xi_raw, dEE = compute_xi(
                rho_sg, T_sg, v_sg, r, i_sonic_baseline, cooling, ambient
            )

            # Under-relax: slowly track toward xi_raw
            xi_smooth = xi_smooth + XI_RELAXATION * (xi_raw - xi_smooth)

            # Ramp the sub-grid on gradually during the first SUBGRID_RAMP steps
            sg_frac = min(phase2_step / SUBGRID_RAMP, 1.0)
            xi_eff = 1.0 + sg_frac * (xi_smooth - 1.0)

            # Hold 2 cells at r_coll as a Dirichlet BC with enhanced T.
            # Only override energy (P/T) — let the solver set rho and v.
            ic = i_coll_baseline
            for j in [ic, max(ic - 1, 0)]:
                rho_j = rho_sg[j]
                v_j = v_sg[j]
                T_j = max(T_baseline[j] * xi_eff, T_floor)
                P_j = rho_j * kB * T_j / (mu * m_p)
                U_new[2, j] = 0.5 * rho_j * v_j**2 + P_j / (gamma - 1.0)

            # Track xi for output (use xi_smooth, which is what's applied)
            xi = xi_smooth

            # Record + diagnostics
            if step % SNAPSHOT_INTERVAL == 0 or step == total_steps:
                xi_history.append(xi_smooth)
                r_coll_history.append(r[ic] / r_S)
                # Diagnostic: raw xi and density/T ratios at r_sonic
                is_b = i_sonic_baseline
                rho_ratio = rho_sg[is_b] / rho_sonic_baseline
                T_ratio = T_sg[is_b] / T_sonic_baseline
                print(f"    diagnostic: xi_raw={xi_raw:.2f}, "
                      f"dE/E={dEE:.4f}, "
                      f"rho/rho_bl={rho_ratio:.2f}, "
                      f"T/T_bl={T_ratio:.2f}")
        else:
            xi = 1.0

        # Residual
        diff = U_new - U
        scale = np.abs(U) + 1e-30
        residual = float(np.sqrt(np.mean((diff / scale) ** 2)))

        U = U_new

        # Snapshot
        if step % SNAPSHOT_INTERVAL == 0 or step == total_steps:
            rho_s, v_s, _, T_s, cs_s = get_primitives(U, gamma, mu, T_floor)
            mdot = measure_mdot(r, rho_s, v_s, problem.r_B)
            mdot_history.append((step, mdot))
            Mach_s = np.abs(v_s) / cs_s
            phase = "P2" if in_phase2 else "P1"
            elapsed = _time.time() - t0
            print(
                f"  [{phase}] step {step:>6d}: "
                f"Mdot={mdot:.4e}, "
                f"res={residual:.3e}, "
                f"xi={xi:.1f}, "
                f"T_max/T_inf={T_s.max() / ambient.T:.0f}, "
                f"Mach_max={Mach_s.max():.1f}, "
                f"t={elapsed:.0f}s"
            )

    # Final results
    print("\n" + "=" * 70)
    rho_f, v_f, P_f, T_f, cs_f = get_primitives(U, gamma, mu, T_floor)
    mdot_final = measure_mdot(r, rho_f, v_f, problem.r_B)

    # Baseline Mdot (from phase 1 end)
    mdot_phase1 = [m for s, m in mdot_history if s == PHASE1_STEPS][0]

    print(f"\n  Mdot (end of phase 1, no sub-grid): {mdot_phase1:.4e} g/s")
    print(f"  Mdot (end of phase 2, with sub-grid): {mdot_final:.4e} g/s")
    print(f"  Ratio: {mdot_final / mdot_phase1:.4f}")
    print(f"  Change: {(mdot_final / mdot_phase1 - 1) * 100:+.1f}%")

    if xi_history:
        print(f"\n  Final xi: {xi_history[-1]:.1f}")
        print(f"  Final r_coll/r_S: {r_coll_history[-1]:.0f}")

    print("\n  Mdot history:")
    print(f"    {'step':>8s}  {'Mdot':>12s}  {'Mdot/Mdot_B':>12s}")
    for s, m in mdot_history:
        print(f"    {s:>8d}  {m:>12.4e}  {m / mdot_phase1:>12.4f}")

    print()


if __name__ == "__main__":
    run()
