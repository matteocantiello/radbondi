"""Stage 2c: Reservoir self-elimination timescale.

For each ell-bin in the reservoir population (Kn_eff > 1), compute:
  1. How much additional pile-up is needed to reach Kn_eff = 1
  2. How much mass that requires
  3. How long it takes to accumulate at the current trapping rate
  4. Whether the reservoir is self-eliminating or persistent

Run:
    python experiments/reservoir_buildup.py
"""

from __future__ import annotations

import numpy as np

import radbondi as rb
from radbondi.constants import G, c_light, kB, m_p
from radbondi.cooling import Cooling

# ── Load profiles ────────────────────────────────────────────────────

M_BH = 1e-16 * rb.M_sun
GM = G * M_BH
rS = 2 * GM / c_light**2
ell_crit = 4 * GM / c_light

sol = rb.load("examples/paper_sweep_output/mbh_logM-16.00.npz")
gamma = sol.ambient_gamma
mu = sol.ambient_mu
r = sol.r
rho = sol.rho
T = sol.T
v = np.abs(sol.v)
cs = np.sqrt(gamma * kB * T / (mu * m_p))

# Mfp profile
E4 = (4.803e-10) ** 4
n_e = (sol.ambient_X + 0.5 * sol.ambient_Y) * rho / m_p
lam_mfp = (kB * T) ** 2 / (np.pi * n_e * E4 * 5.0)
Kn_profile = lam_mfp / r

# Key radii
i_coll = np.where(np.diff(np.sign(Kn_profile - 1.0)))[0][-1]
r_coll = r[i_coll]
i_sonic = np.argmin(np.abs(v / cs - 1.0))
sigma_perp = cs[i_coll] / np.sqrt(gamma)
v_coll = v[i_coll]
E_coll = 0.5 * v_coll**2 - GM / r_coll
ell_max_bound = np.sqrt(GM**2 / (2 * abs(E_coll)))

# Bondi accretion rate
Mdot_B = 4 * np.pi * r[len(r) // 2] ** 2 * rho[len(r) // 2] * v[len(r) // 2]

# Emissivity for dE/E
amb = rb.AmbientMedium(
    T=sol.ambient_T, rho=sol.ambient_rho,
    mu=mu, gamma=gamma, X=sol.ambient_X, Y=sol.ambient_Y,
)
cool = Cooling.default()
eps_ff = np.array([
    float(cool.total_emissivity(rho[i], T[i], amb)[0])
    for i in range(len(r))
])


def Kn_at_radius(r_target):
    if r_target < r[0] or r_target > r[-1]:
        return 1e10
    return 10 ** float(np.interp(
        np.log10(r_target), np.log10(r),
        np.log10(np.maximum(Kn_profile, 1e-30)),
    ))


def rho_at_radius(r_target):
    if r_target < r[0] or r_target > r[-1]:
        return 0.0
    return float(np.interp(np.log10(r_target), np.log10(r), np.log10(rho)))


def compute_dEE(ell):
    if ell <= 0 or ell >= ell_max_bound:
        return 0.0
    a_c, b_c, c_c = E_coll, GM, -(ell**2) / 2
    disc = b_c**2 - 4 * a_c * c_c
    if disc < 0:
        return 0.0
    r1 = (-b_c + np.sqrt(disc)) / (2 * a_c)
    r2 = (-b_c - np.sqrt(disc)) / (2 * a_c)
    candidates = [x for x in [r1, r2] if x > 0]
    if not candidates:
        return 0.0
    r_peri = min(candidates)
    dr_range = r_coll - r_peri
    if dr_range < 1e-15:
        return 0.0
    r_inner = r_peri + 0.05 * dr_range
    r_outer = r_coll - 0.05 * dr_range
    mask = (r >= r_inner) & (r <= r_outer)
    if np.sum(mask) < 5:
        return 0.0
    r_orb = r[mask]
    rho_orb = rho[mask]
    eps_orb = eps_ff[mask]
    E_orb = ell**2 / (2 * r_coll**2) - GM / r_coll
    vr2 = 2 * (E_orb + GM / r_orb) - ell**2 / r_orb**2
    vr2 = np.maximum(vr2, 0.0)
    vr = np.sqrt(vr2)
    v_total = np.sqrt(vr2 + ell**2 / r_orb**2)
    vr = np.maximum(vr, 0.01 * v_total)
    KE = 0.5 * np.mean(v_total**2)
    cooling_rate = eps_orb / np.maximum(rho_orb, 1e-30)
    integrand = cooling_rate / vr
    dE_total = 2.0 * 1.1 * np.trapz(integrand, r_orb)
    return dE_total / max(KE, 1e-30)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    print("Stage 2c: Reservoir build-up / self-collisionalization")
    print("=" * 70)

    # Rayleigh distribution
    scale = r_coll * sigma_perp
    N_ELL = 300
    ell_grid = np.linspace(0.01 * ell_crit, ell_max_bound * 0.99, N_ELL)
    rayleigh_pdf = (ell_grid / scale**2) * np.exp(
        -(ell_grid**2) / (2 * scale**2)
    )
    rayleigh_pdf /= np.trapz(rayleigh_pdf, ell_grid)

    # Orbital period estimate: T_orb ~ 2 * pi * r_circ / v_circ
    # v_circ = sqrt(GM/r_circ)

    print(f"\n  r_coll = {r_coll / rS:.0f} r_S")
    print(f"  Mdot_B ~ {Mdot_B:.3e} g/s")
    print()

    print(f"  {'ell/lc':>7s}  {'r_circ':>8s}  {'xi':>5s}  "
          f"{'Kn_0':>6s}  {'Kn_eff':>6s}  {'A_req':>6s}  "
          f"{'M_need [g]':>10s}  {'t_build [s]':>11s}  "
          f"{'t_build/t_orb':>13s}  {'class':>15s}")
    print("  " + "-" * 100)

    # Accumulated reservoir metrics
    total_M_reservoir = 0.0
    total_M_to_coll = 0.0

    for ell_ratio in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]:
        ell = ell_ratio * ell_crit
        if ell >= ell_max_bound:
            continue

        r_circ = ell**2 / GM
        r_circ_rS = r_circ / rS

        dEE = compute_dEE(ell)
        xi = np.sqrt(2.0 / max(dEE, 1e-30)) if dEE > 0 else 1e10
        xi = min(xi, 1e6)

        Kn_0 = Kn_at_radius(r_circ)
        Kn_eff = Kn_0 / max(xi, 1)

        # Required enhancement for Kn_eff = 1
        A_req = Kn_eff  # need Kn_0 / (xi * A) = 1 → A = Kn_eff

        if A_req <= 1:
            classification = "already coll."
            M_need = 0.0
            t_build = 0.0
        else:
            # Mass needed: shell at r_circ with density A_req * xi * rho_Bondi
            # Volume: 4/3 pi (r_circ * 1.5)^3 - 4/3 pi (r_circ * 0.5)^3
            # (rough: particle occupies a shell of width ~ r_circ)
            shell_vol = (4 / 3) * np.pi * (
                (r_circ * 1.5) ** 3 - (r_circ * 0.5) ** 3
            )
            rho_bondi_local = 10 ** rho_at_radius(r_circ)
            rho_current = xi * rho_bondi_local
            rho_needed = A_req * rho_current
            M_current = rho_current * shell_vol
            M_needed = rho_needed * shell_vol
            M_need = M_needed - M_current

            # Supply rate: fraction of Mdot_B going into this ell bin
            # times the fraction that ends up as reservoir
            d_ell = ell_grid[1] - ell_grid[0]
            idx = np.argmin(np.abs(ell_grid - ell))
            f_bin = rayleigh_pdf[idx] * d_ell
            Mdot_trapped = f_bin * Mdot_B

            if Mdot_trapped > 0:
                t_build = M_need / Mdot_trapped
            else:
                t_build = np.inf

            if A_req < 3:
                classification = "self-elim."
            elif A_req < 10:
                classification = "slow build"
            else:
                classification = "persistent"

        # Orbital period at r_circ
        v_circ = np.sqrt(GM / r_circ)
        t_orb = 2 * np.pi * r_circ / v_circ

        t_ratio = t_build / t_orb if t_orb > 0 else np.inf

        print(
            f"  {ell_ratio:>7d}  {r_circ_rS:>8.0f}  {xi:>5.1f}  "
            f"{Kn_0:>6.1f}  {Kn_eff:>6.2f}  {A_req:>6.1f}  "
            f"{M_need:>10.2e}  {t_build:>11.3e}  "
            f"{t_ratio:>13.1e}  {classification:>15s}"
        )

        if Kn_eff > 1:
            total_M_reservoir += M_need if M_need > 0 else 0
            total_M_to_coll += M_need if M_need > 0 else 0

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  Total mass needed to collisionalize ALL reservoir: "
          f"{total_M_to_coll:.2e} g")
    print(f"  Mdot_B ~ {Mdot_B:.2e} g/s")
    print(f"  Total build-up time (all bins): "
          f"{total_M_to_coll / Mdot_B:.3e} s")
    print()

    # Compare to dynamical/flow timescales
    t_ff_rcoll = r_coll / v_coll
    t_flow_rB = sol.r_B / cs[0]
    print("  Timescale comparisons:")
    print(f"    Free-fall at r_coll:  {t_ff_rcoll:.3e} s")
    print(f"    Flow time at r_B:    {t_flow_rB:.3e} s")
    print(f"    Build-up / t_ff:     {total_M_to_coll / Mdot_B / t_ff_rcoll:.1e}")
    print(f"    Build-up / t_flow:   {total_M_to_coll / Mdot_B / t_flow_rB:.1e}")
    print()

    if total_M_to_coll / Mdot_B < 10 * t_ff_rcoll:
        print("  --> Reservoir self-eliminates on a dynamical timescale.")
        print("      The pile-up builds until it collisionalizes.")
    elif total_M_to_coll / Mdot_B < 100 * t_flow_rB:
        print("  --> Reservoir self-eliminates on a flow timescale.")
        print("      Transient, not permanent.")
    else:
        print("  --> Reservoir may be long-lived.")
        print("      Angular-momentum transport needed.")
    print()


if __name__ == "__main__":
    main()
