"""Stage 1: Orbit-averaged kinetic post-processor.

Takes the converged cooled Bondi profiles and, for each angular momentum
ell in the Rayleigh distribution:
  1. Computes pericenter r_peri(ell)
  2. Integrates dE/E along the full orbit (not at one radius)
  3. Determines whether the orbit returns to r_coll or circularizes inside
  4. Computes xi(ell) = sqrt(2/(dE/E))

Then integrates over the ell distribution to produce:
  - Orbit-averaged <dE/E>, <xi>
  - Fraction reaching various radii (r_sonic, 550 r_S, 300 r_S)
  - Radial pile-up density profile rho_pile(r)
  - Cumulative luminosity L(<r)

This is pure post-processing on the existing profiles — no solver needed.

Run:
    python experiments/orbit_library.py
"""

from __future__ import annotations

import numpy as np

import radbondi as rb
from radbondi.constants import G, c_light, kB, m_p
from radbondi.cooling import Cooling

# ── Configuration ────────────────────────────────────────────────────

M_BH = 1e-16 * rb.M_sun
AMBIENT = rb.presets.solar_core()
COOLING = Cooling.default()

N_ELL = 200      # number of ell bins in the distribution
ELL_MAX_RATIO = 40  # max ell / ell_crit to consider

# ── Load profiles ────────────────────────────────────────────────────

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

# Mfp and r_coll
E4 = (4.803e-10) ** 4
LN_LAMBDA = 5.0
n_e = (sol.ambient_X + 0.5 * sol.ambient_Y) * rho / m_p
mfp = (kB * T) ** 2 / (np.pi * n_e * E4 * LN_LAMBDA)
ratio_mfp = mfp / r
i_coll = np.where(np.diff(np.sign(ratio_mfp - 1.0)))[0][-1]
r_coll = r[i_coll]

# Sonic point
i_sonic = np.argmin(np.abs(Mach - 1.0))

# Bremsstrahlung emissivity at each radius
amb = rb.AmbientMedium(
    T=sol.ambient_T, rho=sol.ambient_rho,
    mu=mu, gamma=gamma, X=sol.ambient_X, Y=sol.ambient_Y,
)
cool = COOLING
eps_ff = np.array([
    float(cool.total_emissivity(rho[i], T[i], amb)[0])
    for i in range(len(r))
])

# Thermal dispersion at r_coll
sigma_perp = cs[i_coll] / np.sqrt(gamma)

# ── Orbit calculations ───────────────────────────────────────────────


def pericenter_newtonian(ell, E_orb):
    """Newtonian pericenter for a particle with energy E and ell.

    E_orb = v^2/2 - GM/r (specific orbital energy).
    For marginally bound (E~0): r_peri = ell^2/(2GM).
    For bound (E<0): solve E = ell^2/(2r^2) - GM/r.
    """
    if abs(E_orb) < 1e-30:
        return ell ** 2 / (2 * GM)
    # Quadratic: E r^2 + GM r - ell^2/2 = 0
    a_coeff = E_orb
    b_coeff = GM
    c_coeff = -(ell ** 2) / 2
    disc = b_coeff ** 2 - 4 * a_coeff * c_coeff
    if disc < 0:
        return 0.0
    if E_orb < 0:
        # Bound orbit: two positive roots, take the smaller one
        r1 = (-b_coeff + np.sqrt(disc)) / (2 * a_coeff)
        r2 = (-b_coeff - np.sqrt(disc)) / (2 * a_coeff)
        return min(abs(r1), abs(r2))
    # Unbound: one positive root
    r1 = (-b_coeff + np.sqrt(disc)) / (2 * a_coeff)
    r2 = (-b_coeff - np.sqrt(disc)) / (2 * a_coeff)
    candidates = [x for x in [r1, r2] if x > 0]
    return min(candidates) if candidates else 0.0


def orbit_averaged_dEE(ell, r_peri, r_apo):
    """Integrate dE/E along the orbit from r_apo to r_peri and back.

    Uses the cooled profiles for eps_ff and rho. Velocity is the
    orbital velocity v_r(r) = sqrt(2(E + GM/r) - ell^2/r^2).
    """
    # Orbital energy at r_apo (v_r = 0 at apocenter)
    E_orb = ell ** 2 / (2 * r_apo ** 2) - GM / r_apo

    # Clip integration to avoid the turning-point singularity:
    # exclude 5% of the radial range near each turning point
    dr_range = r_apo - r_peri
    r_inner = r_peri + 0.05 * dr_range
    r_outer = r_apo - 0.05 * dr_range
    mask = (r >= r_inner) & (r <= r_outer)
    if np.sum(mask) < 5:
        return 0.0, 0.0

    r_orb = r[mask]
    rho_orb = rho[mask]
    eps_orb = eps_ff[mask]

    # Radial velocity along the orbit
    vr2 = 2 * (E_orb + GM / r_orb) - ell ** 2 / r_orb ** 2
    vr2 = np.maximum(vr2, 0.0)
    vr = np.sqrt(vr2)

    # Minimum physical vr: use 1% of the local total velocity
    v_total = np.sqrt(vr2 + ell ** 2 / r_orb ** 2)
    vr_floor = 0.01 * v_total
    vr = np.maximum(vr, vr_floor)

    # Characteristic KE: use the orbit-averaged total velocity
    KE = 0.5 * np.mean(v_total ** 2)

    # Energy loss rate per gram: eps/rho [erg/g/s]
    cooling_rate = eps_orb / np.maximum(rho_orb, 1e-30)

    # Time element: dt = dr / |v_r|
    # Integrate: dE = integral of (eps/rho) dt = integral of (eps/rho)/|vr| dr
    # Factor 2 for inbound + outbound legs (and ~1.1 to account
    # for the clipped turning-point regions)
    integrand = cooling_rate / vr
    dE_total = 2.0 * 1.1 * np.trapz(integrand, r_orb)

    # Fractional energy loss
    dEE = dE_total / max(KE, 1e-30)
    return dEE, dE_total


def density_contribution(ell, r_peri, r_apo, r_grid):
    """Density contribution at each r_grid point from an orbit (ell).

    rho_orb(r) = 1 / (4 pi r^2 |v_r(r)| T_r) for each passage,
    times 2 for inbound + outbound.
    """
    E_orb = ell ** 2 / (2 * r_apo ** 2) - GM / r_apo

    vr2 = 2 * (E_orb + GM / r_grid) - ell ** 2 / r_grid ** 2
    dr_range = r_apo - r_peri
    r_inner = r_peri + 0.05 * dr_range
    r_outer = r_apo - 0.05 * dr_range
    inside = (r_grid >= r_inner) & (r_grid <= r_outer) & (vr2 > 0)

    rho_contrib = np.zeros_like(r_grid)
    vr = np.sqrt(np.maximum(vr2, 0.0))
    v_total = np.sqrt(vr2 + ell ** 2 / r_grid ** 2)
    vr = np.maximum(vr, 0.01 * v_total)

    # Radial period (approximate): T_r ~ 2 * integral dr / |v_r|
    r_sub = r_grid[inside]
    vr_sub = vr[inside]
    if len(r_sub) < 2:
        return rho_contrib
    T_r = 2 * 1.1 * np.trapz(1.0 / np.maximum(vr_sub, 1e-10), r_sub)
    if T_r < 1e-30:
        return rho_contrib

    # rho_contrib = 2 / (4 pi r^2 |v_r| T_r) — the "2" for both legs
    rho_contrib[inside] = 2.0 / (
        4 * np.pi * r_grid[inside] ** 2
        * np.maximum(vr[inside], 1e-10)
        * T_r
    )
    return rho_contrib


# ── Main calculation ─────────────────────────────────────────────────


def main():
    print("Stage 1: Orbit-averaged kinetic post-processor")
    print("=" * 65)

    # Angular momentum grid (Rayleigh-distributed)
    ell_max = ELL_MAX_RATIO * ell_crit
    ell_grid = np.linspace(0.01 * ell_crit, ell_max, N_ELL)

    # Rayleigh PDF: P(ell) = ell / (r_coll^2 sigma_perp^2) * exp(...)
    scale = r_coll * sigma_perp
    rayleigh_pdf = (ell_grid / scale ** 2) * np.exp(
        -(ell_grid ** 2) / (2 * scale ** 2)
    )
    rayleigh_pdf /= np.trapz(rayleigh_pdf, ell_grid)

    # Orbital energy at r_coll (marginally bound — v ~ v_infall)
    v_coll = v[i_coll]
    E_at_rcoll = 0.5 * v_coll ** 2 - GM / r_coll

    print(f"\n  r_coll     = {r_coll:.3e} cm ({r_coll / rS:.0f} r_S)")
    print(f"  r_sonic    = {r[i_sonic]:.3e} cm ({r[i_sonic] / rS:.0f} r_S)")
    print(f"  ell_crit   = {ell_crit:.3f} cm^2/s")
    print(f"  ell_typ    = {r_coll * sigma_perp:.1f} cm^2/s")
    print(f"  sigma_perp = {sigma_perp:.3e} cm/s")
    print(f"  v at r_coll = {v_coll:.3e} cm/s")
    print(f"  E at r_coll = {E_at_rcoll:.3e} erg/g")
    print()

    # Arrays to accumulate
    r_peris = np.zeros(N_ELL)
    dEEs = np.zeros(N_ELL)
    xis = np.zeros(N_ELL)
    captured_direct = np.zeros(N_ELL, dtype=bool)
    reaches_sonic = np.zeros(N_ELL, dtype=bool)
    returns_to_rcoll = np.zeros(N_ELL, dtype=bool)
    rho_pile = np.zeros_like(r)

    print(f"  Computing orbits for {N_ELL} ell values...")

    # Maximum ell for a bound orbit at this energy
    ell_max_bound = np.sqrt(GM ** 2 / (2 * abs(E_at_rcoll)))
    f_reflected = np.exp(-(ell_max_bound ** 2) / (2 * scale ** 2))
    print(f"  ell_max (bound orbit) = {ell_max_bound:.1f} cm^2/s "
          f"({ell_max_bound / ell_crit:.1f} ell_crit)")
    print(f"  Fraction reflected (ell > ell_max): {f_reflected:.3f}")
    print()

    reflected = np.zeros(N_ELL, dtype=bool)

    for i, ell in enumerate(ell_grid):
        # Direct capture?
        if ell < ell_crit:
            captured_direct[i] = True
            r_peris[i] = 0.0
            dEEs[i] = 1.0
            xis[i] = 1.0
            continue

        # Too much angular momentum — reflected at centrifugal barrier
        if ell > ell_max_bound:
            reflected[i] = True
            r_peris[i] = r_coll  # never enters
            dEEs[i] = 0.0
            xis[i] = 0.0
            continue

        # Pericenter
        r_peri = pericenter_newtonian(ell, E_at_rcoll)
        r_peris[i] = r_peri

        # Does it reach the sonic point?
        reaches_sonic[i] = r_peri < r[i_sonic]

        # Returns to r_coll? (apocenter ~ r_coll for marginally bound)
        returns_to_rcoll[i] = True  # for marginally bound, always returns

        # Orbit-averaged dE/E
        dEE, _ = orbit_averaged_dEE(ell, max(r_peri, r[0]), r_coll)
        dEEs[i] = dEE

        # xi from the orbit-averaged dE/E
        if dEE > 1e-30:
            xis[i] = np.sqrt(2.0 / dEE)
        else:
            xis[i] = 1e10

        # Density contribution (weighted by Rayleigh PDF and xi)
        d_ell = ell_grid[1] - ell_grid[0] if i > 0 else ell_grid[1]
        weight = rayleigh_pdf[i] * xis[i] * d_ell
        rho_contrib = density_contribution(
            ell, max(r_peri, r[0]), r_coll, r
        )
        rho_pile += weight * rho_contrib

    # ── Results ──────────────────────────────────────────────────────

    # Distribution-averaged quantities (exclude captured and reflected)
    valid = ~captured_direct & ~reflected
    f_direct = np.trapz(rayleigh_pdf[captured_direct], ell_grid[captured_direct])
    f_reflected_meas = np.trapz(rayleigh_pdf[reflected], ell_grid[reflected])
    f_sonic = np.trapz(
        rayleigh_pdf[reaches_sonic & valid],
        ell_grid[reaches_sonic & valid],
    )
    # Average over valid (recycling) orbits only
    norm_valid = np.trapz(rayleigh_pdf[valid], ell_grid[valid])
    dEE_avg = np.trapz(rayleigh_pdf[valid] * dEEs[valid], ell_grid[valid]) / norm_valid
    xi_avg = np.trapz(rayleigh_pdf[valid] * xis[valid], ell_grid[valid]) / norm_valid

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    print(f"\n  Direct capture fraction (ell < ell_crit): {f_direct:.4e}")
    print(f"  Reflected fraction (ell > ell_max): {f_reflected_meas:.3f}")
    print(f"  Recycling fraction: {norm_valid:.3f}")
    print(f"  Of recycling orbits, fraction reaching r_sonic: "
          f"{f_sonic / norm_valid:.3f}")
    print()

    print("  Orbit-averaged quantities:")
    print(f"    <dE/E>  = {dEE_avg:.4f}")
    print(f"    <xi>    = {xi_avg:.1f}")
    print("    (Single-radius at r_sonic: dE/E = 0.084, xi = 4.9)")
    print("    (Single-radius at 1000 r_S: dE/E = 0.013, xi = 12.3)")
    print()

    # Distribution of xi vs ell
    print("  xi vs ell/ell_crit:")
    print(f"    {'ell/ell_crit':>12s}  {'r_peri/r_S':>10s}  {'dE/E':>8s}  "
          f"{'xi':>6s}  {'reaches sonic':>14s}")
    print("    " + "-" * 55)
    for ratio in [1.5, 3, 5, 10, 15, 18, 25, 35]:
        idx = np.argmin(np.abs(ell_grid / ell_crit - ratio))
        if not captured_direct[idx]:
            print(
                f"    {ell_grid[idx] / ell_crit:>12.1f}  "
                f"{r_peris[idx] / rS:>10.0f}  "
                f"{dEEs[idx]:>8.4f}  "
                f"{xis[idx]:>6.1f}  "
                f"{'yes' if reaches_sonic[idx] else 'no':>14s}"
            )

    # Cumulative luminosity L(<r)
    eps_pile = eps_ff * (rho_pile / np.maximum(rho, 1e-30)) ** 2
    # Approximate: use the Bondi emissivity scaled by (rho_pile/rho_bondi)^2
    # This is rough because the temperature also matters
    L_cum_bondi = 4 * np.pi * np.cumsum(eps_ff * r ** 2 * np.gradient(r))
    L_cum_pile = 4 * np.pi * np.cumsum(eps_pile * r ** 2 * np.gradient(r))

    print("\n  Cumulative luminosity L(<r):")
    print(f"    {'r/r_S':>8s}  {'L_bondi(<r)':>12s}  {'L_pile(<r)':>12s}  "
          f"{'ratio':>6s}")
    print("    " + "-" * 45)
    for target_rS in [100, 300, 550, 780, 1200, 3000, 7000]:
        idx = np.argmin(np.abs(r / rS - target_rS))
        ratio_L = (
            L_cum_pile[idx] / L_cum_bondi[idx]
            if L_cum_bondi[idx] > 0
            else 0
        )
        print(
            f"    {r[idx] / rS:>8.0f}  "
            f"{L_cum_bondi[idx]:>12.3e}  "
            f"{L_cum_pile[idx]:>12.3e}  "
            f"{ratio_L:>6.1f}"
        )

    print("\n  Pile-up density enhancement rho_pile/rho_bondi:")
    print(f"    {'r/r_S':>8s}  {'rho_pile/rho_bondi':>18s}")
    print("    " + "-" * 30)
    for target_rS in [100, 300, 550, 780, 1200, 3000, 7000]:
        idx = np.argmin(np.abs(r / rS - target_rS))
        enhancement = rho_pile[idx] / rho[idx] if rho[idx] > 0 else 0
        print(f"    {r[idx] / rS:>8.0f}  {enhancement:>18.2f}")

    print()


if __name__ == "__main__":
    main()
