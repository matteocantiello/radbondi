"""Stage 2: Loss-cone / recycling Markov model.

Answers the central question: what fraction of the supplied mass
reaches the BH as a function of angular-momentum transport efficiency?

Each particle packet starts at r_coll, drawn from the flux-weighted
thermal distribution. On each cycle:
  1. Draw ell from the Rayleigh distribution
  2. Classify: direct capture / reflected / recycling
  3. For recycling orbits, lose energy (orbit-averaged dE/E from Stage 1)
  4. Decide fate: return to r_coll, circularize, or diffuse into loss cone
  5. If returns: re-thermalize and redraw ell
  6. If circularizes: attempt angular-momentum diffusion toward loss cone

The free parameter is eta_J = t_E / t_J (angular-momentum transport
efficiency). Scan from no transport to rapid randomization.

Run:
    python experiments/recycling_markov.py
"""

from __future__ import annotations

import numpy as np

import radbondi as rb
from radbondi.constants import G, c_light, kB, m_p
from radbondi.cooling import Cooling

# ── Configuration ────────────────────────────────────────────────────

M_BH = 1e-16 * rb.M_sun
AMBIENT = rb.presets.solar_core()

N_PARTICLES = 50_000
MAX_CYCLES = 5000  # max cycles per particle before giving up

# Angular-momentum transport efficiency scan
# eta_J = t_E / t_J. Large eta_J = fast J transport = easy capture.
ETA_J_VALUES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# ── Load profiles and Stage 1 results ────────────────────────────────

GM = G * M_BH
rS = 2 * GM / c_light**2
ell_crit = 4 * GM / c_light

sol = rb.load("examples/paper_sweep_output/mbh_logM-16.00.npz")
gamma = sol.ambient_gamma
mu = sol.ambient_mu

r = sol.r
rho_prof = sol.rho
T_prof = sol.T
v_prof = np.abs(sol.v)
cs = np.sqrt(gamma * kB * T_prof / (mu * m_p))

# r_coll and sonic point
E4 = (4.803e-10) ** 4
n_e = (sol.ambient_X + 0.5 * sol.ambient_Y) * rho_prof / m_p
mfp = (kB * T_prof) ** 2 / (np.pi * n_e * E4 * 5.0)
i_coll = np.where(np.diff(np.sign(mfp / r - 1.0)))[0][-1]
r_coll = r[i_coll]
i_sonic = np.argmin(np.abs(v_prof / cs - 1.0))

# Thermal dispersion at r_coll
sigma_perp = cs[i_coll] / np.sqrt(gamma)

# Orbital energy at r_coll
v_coll = v_prof[i_coll]
E_coll = 0.5 * v_coll**2 - GM / r_coll

# Maximum ell for bound orbit
ell_max_bound = np.sqrt(GM**2 / (2 * abs(E_coll)))

# Bremsstrahlung emissivity
amb = rb.AmbientMedium(
    T=sol.ambient_T, rho=sol.ambient_rho,
    mu=mu, gamma=gamma, X=sol.ambient_X, Y=sol.ambient_Y,
)
cool = Cooling.default()
eps_ff = np.array([
    float(cool.total_emissivity(rho_prof[i], T_prof[i], amb)[0])
    for i in range(len(r))
])

# ── Orbit-averaged dE/E interpolant from Stage 1 ─────────────────────


def compute_dEE(ell):
    """Orbit-averaged fractional energy loss for a single ell value."""
    if ell <= 0 or ell >= ell_max_bound:
        return 0.0

    # Pericenter
    a_c = E_coll
    b_c = GM
    c_c = -(ell**2) / 2
    disc = b_c**2 - 4 * a_c * c_c
    if disc < 0:
        return 0.0

    r1 = (-b_c + np.sqrt(disc)) / (2 * a_c)
    r2 = (-b_c - np.sqrt(disc)) / (2 * a_c)
    candidates = [x for x in [r1, r2] if x > 0]
    if not candidates:
        return 0.0
    r_peri = min(candidates)
    r_apo = r_coll

    # Clip turning points
    dr_range = r_apo - r_peri
    if dr_range < 1e-15:
        return 0.0
    r_inner = r_peri + 0.05 * dr_range
    r_outer = r_apo - 0.05 * dr_range

    mask = (r >= r_inner) & (r <= r_outer)
    if np.sum(mask) < 5:
        return 0.0

    r_orb = r[mask]
    rho_orb = rho_prof[mask]
    eps_orb = eps_ff[mask]

    E_orb = ell**2 / (2 * r_apo**2) - GM / r_apo
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


# Pre-compute dE/E on a grid for fast lookup
_N_ELL_GRID = 500
_ell_grid = np.linspace(0.01 * ell_crit, ell_max_bound * 0.99, _N_ELL_GRID)
_dEE_grid = np.array([compute_dEE(el) for el in _ell_grid])


def dEE_interp(ell):
    """Interpolated orbit-averaged dE/E."""
    if ell <= _ell_grid[0]:
        return float(_dEE_grid[0])
    if ell >= _ell_grid[-1]:
        return float(_dEE_grid[-1])
    return float(np.interp(ell, _ell_grid, _dEE_grid))


# ── Markov model ─────────────────────────────────────────────────────


def draw_ell(rng):
    """Draw ell from the flux-weighted Rayleigh distribution."""
    scale = r_coll * sigma_perp
    return rng.rayleigh(scale)


def run_markov(eta_J, n_particles=N_PARTICLES, rng=None):
    """Run the Markov recycling model for a given eta_J.

    eta_J = t_E / t_J: angular-momentum transport efficiency.
    eta_J = 0: no angular-momentum transport (only direct capture).
    eta_J >> 1: rapid randomization (loss cone always fed).

    Returns a dict of diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_captured = 0
    n_reflected = 0
    n_circularized = 0  # ended up in reservoir
    n_captured_from_reservoir = 0
    n_max_cycles = 0  # hit MAX_CYCLES limit
    total_returns = 0
    total_orbits = 0

    for _ in range(n_particles):
        captured = False
        n_returns_this = 0

        for _cycle in range(MAX_CYCLES):
            # Draw angular momentum
            ell = draw_ell(rng)

            # Direct capture?
            if ell < ell_crit:
                n_captured += 1
                captured = True
                break

            # Reflected?
            if ell > ell_max_bound:
                n_reflected += 1
                # Reflected particles immediately re-thermalize.
                # They don't enter the collisionless zone, but they
                # DO get another chance (they fall back in with new ell).
                n_returns_this += 1
                total_returns += 1
                continue

            # Recycling orbit: track orbit-by-orbit.
            # Use the UNPERTURBED (dE/E)_0 per orbit — the pile-up
            # enhancement is an emergent property, not an input.
            dEE = dEE_interp(ell)
            total_orbits += 1

            # (a) Angular-momentum diffusion during this orbit:
            #     probability ~ (ell_crit/ell)^2 * eta_J * dEE
            p_diffuse = (ell_crit / ell) ** 2 * eta_J * max(dEE, 1e-30)
            p_diffuse = min(p_diffuse, 1.0)

            if rng.random() < p_diffuse:
                n_captured += 1
                n_captured_from_reservoir += 1
                captured = True
                break

            # (b) Energy loss this orbit: dE = dEE * |E|
            #     New energy: E_new = E_coll + dE (becomes more negative
            #     = more bound, since dE removes KE)
            E_current = E_coll  # reset each cycle (re-thermalized)
            dE_abs = dEE * abs(E_current)
            E_new = E_current - dE_abs  # more negative = more bound

            # (c) Does the particle return to r_coll?
            #     New apocenter from E_new and ell:
            a_c = E_new
            b_c = GM
            c_c = -(ell**2) / 2
            disc = b_c**2 - 4 * a_c * c_c
            if disc < 0:
                n_circularized += 1
                break

            r1 = (-b_c + np.sqrt(disc)) / (2 * a_c)
            r2 = (-b_c - np.sqrt(disc)) / (2 * a_c)
            candidates = [x for x in [r1, r2] if x > 0]
            if not candidates:
                n_circularized += 1
                break
            r_apo_new = max(candidates)

            if r_apo_new < r_coll * 0.9:
                # Apocenter shrank below r_coll — circularized
                n_circularized += 1
                break

            # Particle returns to r_coll, re-thermalizes, new cycle
            n_returns_this += 1
            total_returns += 1
            continue

        else:
            # Hit MAX_CYCLES
            n_max_cycles += 1

        if not captured:
            pass  # already counted as circularized or max_cycles

    total = n_particles
    f_cap = n_captured / total
    f_circ = n_circularized / total
    f_max = n_max_cycles / total
    avg_returns = total_returns / total

    return {
        "eta_J": eta_J,
        "n_captured": n_captured,
        "n_circularized": n_circularized,
        "n_max_cycles": n_max_cycles,
        "f_captured": f_cap,
        "f_circularized": f_circ,
        "f_max_cycles": f_max,
        "f_captured_from_reservoir": n_captured_from_reservoir / total,
        "avg_returns": avg_returns,
        "avg_orbits": total_orbits / total,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    print("Stage 2: Loss-cone / recycling Markov model")
    print("=" * 65)
    print(f"  N_particles = {N_PARTICLES}")
    print(f"  MAX_CYCLES  = {MAX_CYCLES}")
    print(f"  r_coll      = {r_coll / rS:.0f} r_S")
    print(f"  ell_crit    = {ell_crit:.3f} cm^2/s")
    print(f"  ell_max     = {ell_max_bound:.1f} cm^2/s "
          f"({ell_max_bound / ell_crit:.1f} ell_crit)")
    print(f"  sigma_perp  = {sigma_perp:.3e} cm/s")
    print()

    # Pre-compute check
    print("  Pre-computing dE/E grid...")
    dEE_check = dEE_interp(10 * ell_crit)
    print(f"  dE/E at 10 ell_crit: {dEE_check:.4f}")
    print()

    print(f"  {'eta_J':>7s}  {'f_cap':>7s}  {'f_circ':>7s}  "
          f"{'f_cap_res':>9s}  {'f_max':>6s}  "
          f"{'<returns>':>9s}  {'<orbits>':>8s}")
    print("  " + "-" * 62)

    results = []
    for eta_J in ETA_J_VALUES:
        res = run_markov(eta_J)
        results.append(res)
        print(
            f"  {eta_J:>7.3f}  "
            f"{res['f_captured']:>7.3f}  "
            f"{res['f_circularized']:>7.3f}  "
            f"{res['f_captured_from_reservoir']:>9.3f}  "
            f"{res['f_max_cycles']:>6.3f}  "
            f"{res['avg_returns']:>9.1f}  "
            f"{res['avg_orbits']:>8.1f}"
        )

    print("\n" + "=" * 65)
    print("INTERPRETATION")
    print("=" * 65)
    print()
    print("  f_cap = fraction captured by BH (direct + diffusive)")
    print("  f_circ = fraction circularized into bound reservoir")
    print("  f_cap_res = fraction captured via angular-momentum diffusion")
    print("  f_max = fraction that hit MAX_CYCLES (still recycling)")
    print()
    print("  The key plot: f_cap vs eta_J = t_E / t_J")
    print("  If f_cap ~ 1 for plausible eta_J, recycling works.")
    print("  If f_cap << 1, the gas accumulates in a reservoir.")
    print()

    # Summary
    print("  Summary:")
    for res in results:
        label = ""
        if res["f_captured"] > 0.9:
            label = " <-- efficient capture"
        elif res["f_captured"] < 0.1:
            label = " <-- reservoir-dominated"
        print(f"    eta_J = {res['eta_J']:>5.3f}: "
              f"Mdot_BH/Mdot_supplied = {res['f_captured']:.3f}{label}")
    print()


if __name__ == "__main__":
    main()
