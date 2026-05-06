"""Stage 2b: Self-consistent collisionalization map.

For each angular-momentum bin, compute:
  - r_peri, r_circ (pericenter and circularization radius)
  - xi(ell) and dE/E(ell) from Stage 1
  - Kn_0 = lambda_mfp(r_circ) / r_circ at unperturbed density
  - Kn_eff = Kn_0 / xi (at pile-up-enhanced density)
  - Classification: captured / reflected / collisionalized / reservoir

Then integrate over the Rayleigh distribution to get the fraction
in each fate category.

The key question: does the pile-up make the circularization zone
collisional (Kn_eff < 1)?

Run:
    python experiments/collisionalization_map.py
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
Mach = v / cs

# Mfp profile
E4 = (4.803e-10) ** 4
LN_LAMBDA = 5.0
n_e = (sol.ambient_X + 0.5 * sol.ambient_Y) * rho / m_p
lam_mfp = (kB * T) ** 2 / (np.pi * n_e * E4 * LN_LAMBDA)
Kn_profile = lam_mfp / r  # Knudsen number at each radius

# Key radii
i_coll = np.where(np.diff(np.sign(Kn_profile - 1.0)))[0][-1]
r_coll = r[i_coll]
i_sonic = np.argmin(np.abs(Mach - 1.0))
sigma_perp = cs[i_coll] / np.sqrt(gamma)
v_coll = v[i_coll]
E_coll = 0.5 * v_coll**2 - GM / r_coll
ell_max_bound = np.sqrt(GM**2 / (2 * abs(E_coll)))

# Bremsstrahlung emissivity
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
    """Interpolate Kn_0 = lambda_mfp/r at a given radius."""
    if r_target < r[0] or r_target > r[-1]:
        return 1e10
    return float(np.interp(
        np.log10(r_target),
        np.log10(r),
        np.log10(np.maximum(Kn_profile, 1e-30)),
    ))


def compute_dEE(ell):
    """Orbit-averaged dE/E (from Stage 1 orbit_library)."""
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


# ── Main calculation ─────────────────────────────────────────────────


def main():
    print("Stage 2b: Collisionalization map")
    print("=" * 75)
    print(f"  r_coll   = {r_coll / rS:.0f} r_S")
    print(f"  r_sonic  = {r[i_sonic] / rS:.0f} r_S")
    print(f"  ell_crit = {ell_crit:.3f} cm^2/s")
    print(f"  ell_max  = {ell_max_bound:.1f} cm^2/s "
          f"({ell_max_bound / ell_crit:.1f} ell_crit)")
    print()

    # Angular momentum grid
    N_ELL = 300
    ell_grid = np.linspace(0.01 * ell_crit, ell_max_bound * 1.5, N_ELL)

    # Rayleigh distribution
    scale = r_coll * sigma_perp
    rayleigh_pdf = (ell_grid / scale**2) * np.exp(
        -(ell_grid**2) / (2 * scale**2)
    )
    rayleigh_pdf /= np.trapz(rayleigh_pdf, ell_grid)

    # Classification table
    print(f"  {'ell/ell_c':>9s}  {'r_peri':>8s}  {'r_circ':>8s}  "
          f"{'dE/E':>7s}  {'xi':>5s}  {'Kn_0':>6s}  {'Kn_eff':>6s}  "
          f"{'fate':>20s}")
    print("  " + "-" * 80)

    fates = []  # (ell, fraction, fate_label)

    for i, ell in enumerate(ell_grid):
        ell_ratio = ell / ell_crit

        if ell < ell_crit:
            fates.append("captured")
            if abs(ell_ratio - 0.5) < 0.3:
                print(f"  {ell_ratio:>9.1f}  {'—':>8s}  {'—':>8s}  "
                      f"{'—':>7s}  {'—':>5s}  {'—':>6s}  {'—':>6s}  "
                      f"{'direct GR capture':>20s}")
            continue

        if ell > ell_max_bound:
            fates.append("reflected")
            if abs(ell_ratio - 25) < 1 or abs(ell_ratio - 35) < 1:
                print(f"  {ell_ratio:>9.1f}  {'—':>8s}  {'—':>8s}  "
                      f"{'—':>7s}  {'—':>5s}  {'—':>6s}  {'—':>6s}  "
                      f"{'reflected':>20s}")
            continue

        # Recycling orbit
        r_peri = ell**2 / (2 * GM)  # Newtonian for marginally bound
        r_circ = ell**2 / GM  # circularization radius

        dEE = compute_dEE(ell)
        xi = np.sqrt(2.0 / max(dEE, 1e-30)) if dEE > 0 else 1e10
        xi = min(xi, 1e6)

        # Knudsen at circularization radius
        Kn_0 = 10 ** Kn_at_radius(r_circ)
        Kn_eff = Kn_0 / max(xi, 1)

        if Kn_eff < 1:
            fate = "collisionalized"
        elif Kn_eff < 3:
            fate = "marginal"
        else:
            fate = "reservoir"

        fates.append(fate)

        # Print selected rows
        show = (
            abs(ell_ratio - 2) < 0.5
            or abs(ell_ratio - 5) < 0.5
            or abs(ell_ratio - 8) < 0.5
            or abs(ell_ratio - 10) < 0.5
            or abs(ell_ratio - 12) < 0.5
            or abs(ell_ratio - 15) < 0.5
            or abs(ell_ratio - 18) < 0.5
            or abs(ell_ratio - 20) < 0.5
            or abs(ell_ratio - 22) < 0.5
        )
        if show:
            print(
                f"  {ell_ratio:>9.1f}  "
                f"{r_peri / rS:>8.0f}  "
                f"{r_circ / rS:>8.0f}  "
                f"{dEE:>7.4f}  "
                f"{xi:>5.1f}  "
                f"{Kn_0:>6.1f}  "
                f"{Kn_eff:>6.2f}  "
                f"{fate:>20s}"
            )

    # Integrate fractions
    print("\n" + "=" * 75)
    print("POPULATION FRACTIONS")
    print("=" * 75)

    for fate_label in ["captured", "reflected", "collisionalized",
                       "marginal", "reservoir"]:
        mask = np.array([f == fate_label for f in fates])
        if np.any(mask):
            frac = np.trapz(rayleigh_pdf[mask], ell_grid[mask])
        else:
            frac = 0.0
        marker = ""
        if fate_label == "collisionalized":
            marker = "  <-- pile-up makes zone collisional"
        elif fate_label == "reservoir":
            marker = "  <-- remains collisionless"
        print(f"  {fate_label:>20s}: {frac:.3f} ({frac * 100:.1f}%){marker}")

    # The key number
    f_coll = sum(
        np.trapz(rayleigh_pdf[np.array([f == fl for f in fates])],
                 ell_grid[np.array([f == fl for f in fates])])
        for fl in ["captured", "collisionalized", "marginal"]
    )
    f_res = np.trapz(
        rayleigh_pdf[np.array([f == "reservoir" for f in fates])],
        ell_grid[np.array([f == "reservoir" for f in fates])],
    )
    f_refl = np.trapz(
        rayleigh_pdf[np.array([f == "reflected" for f in fates])],
        ell_grid[np.array([f == "reflected" for f in fates])],
    )

    print()
    print(f"  Accreting (captured + collisionalized + marginal): "
          f"{f_coll:.3f} ({f_coll * 100:.1f}%)")
    print(f"  Reservoir (collisionless):  {f_res:.3f} ({f_res * 100:.1f}%)")
    print(f"  Reflected (never enters):   {f_refl:.3f} ({f_refl * 100:.1f}%)")

    # Required xi to collisionalize the reservoir
    print("\n" + "=" * 75)
    print("RESERVOIR COLLISIONALIZATION REQUIREMENTS")
    print("=" * 75)
    print()
    print(f"  For the reservoir population, how much additional pile-up")
    print(f"  is needed to reach Kn_eff = 1?")
    print()
    print(f"  {'ell/ell_c':>9s}  {'r_circ/r_S':>10s}  {'Kn_0':>6s}  "
          f"{'xi_needed':>9s}  {'xi_current':>10s}  {'ratio':>6s}")
    print("  " + "-" * 55)

    for ell_ratio in [3, 5, 8, 10, 12]:
        ell = ell_ratio * ell_crit
        if ell >= ell_max_bound:
            continue
        r_circ = ell**2 / GM
        Kn_0 = 10 ** Kn_at_radius(r_circ)
        xi_needed = Kn_0  # need Kn_0 / xi = 1 → xi = Kn_0
        dEE = compute_dEE(ell)
        xi_current = np.sqrt(2.0 / max(dEE, 1e-30)) if dEE > 0 else 0
        ratio = xi_needed / max(xi_current, 1e-10)
        print(f"  {ell_ratio:>9d}  {r_circ / rS:>10.0f}  {Kn_0:>6.1f}  "
              f"{xi_needed:>9.1f}  {xi_current:>10.1f}  {ratio:>6.1f}")

    print()
    print("  ratio < 1: already collisional (no additional pile-up needed)")
    print("  ratio ~ 1-3: marginally collisional (small additional pile-up)")
    print("  ratio >> 1: deep collisionless reservoir (hard to collisionalize)")
    print()


if __name__ == "__main__":
    main()
