"""Self-consistent radiative-feedback iteration with the MLT envelope model.

The radbondi solver takes the *ambient* medium as a fixed input. When the BH
luminosity is large enough to heat the surroundings, the effective ambient
temperature seen by the Bondi flow is no longer the unperturbed core
temperature ``T_core``: it is shifted upward by feedback. This example shows
the iterative recipe:

    1. Solve the Bondi+cooling problem with the unperturbed ambient.
    2. From the resulting luminosity ``L``, compute an effective ambient
       temperature ``T_eff`` using one of the feedback models.
    3. Build a new ``AmbientMedium`` with ``T = T_eff`` and re-solve.
    4. Iterate until ``T_eff`` (equivalently ``L``) converges.

We use the MLT envelope model (:class:`radbondi.feedback.MLTEnvelope`), which
properly handles the convective saturation that limits the temperature rise
when ``beta >> 1``. The simpler :class:`DiffusionFeedback` is a one-line
swap-in.

We pick ``M_BH = 1e-13 M_sun`` because that's where feedback bites: the
paper Table 1 has ``beta ~ 7e3`` here. Without feedback the run converges to
the unperturbed Bondi+cooling ``eta``; with feedback the ambient warming
suppresses the accretion rate and cools the inner flow, lowering ``eta``
modestly. The convergence is monotonic and takes only a few iterations.
"""

from __future__ import annotations

import time

import radbondi as rb
from radbondi.feedback import MLTEnvelope


def main():
    M_BH = 1e-13 * rb.M_sun
    ambient0 = rb.presets.solar_core()
    cooling = rb.Cooling.default()

    # Solver configuration: small enough to run in a few minutes on a laptop.
    cfg = rb.SolverConfig(
        N=400, x_min=1e-5, n_steps=30_000, cooling_ramp_steps=3_000,
        order=1, flux="hll",
        snapshot_interval=30_000, verbose=False,
    )

    # ── Step 1: baseline (no feedback) ────────────────────────────────
    print("─" * 70)
    print("Baseline (no feedback)")
    print("─" * 70)
    t0 = time.time()
    sol0 = rb.BondiProblem(M_BH=M_BH, ambient=ambient0, cooling=cooling).solve(cfg)
    print(
        f"  T_amb = {ambient0.T:.3e} K, eta = {sol0.eta:.3e}, "
        f"L = {sol0.L:.3e} erg/s, Mdot/Mdot_B = {sol0.mdot_ratio:.2f}, "
        f"wall = {time.time() - t0:.1f}s"
    )

    # ── Step 2-4: iterate (solve -> T_eff -> re-solve) ────────────────
    print()
    print("─" * 70)
    print("MLT feedback iteration")
    print("─" * 70)

    # MLTEnvelope uses an opacity for radiative transport in the envelope.
    # 'kappa_env' is for the bulk envelope (electron scattering by default).
    # 'kappa_BH' sets the photon-coupling radius r_c = 1/(kappa_BH * rho).
    # For the bremsstrahlung-cooled regime, thermal photons (~keV) couple via
    # roughly the Rosseland mean opacity ~ 1 cm^2/g.
    envelope = MLTEnvelope(
        ambient=ambient0, M_BH=M_BH, kappa_env=1.0, kappa_BH=1.0
    )

    ambient = ambient0
    sol = sol0
    L_prev = sol.L
    tol = 1e-3      # converge when |dL/L| < tol
    max_iter = 8

    print(
        "  iter  T_eff [K]    x = T_eff/T_core    eta         L [erg/s]   "
        "|dL|/L"
    )
    print(
        f"   0    {ambient.T:.3e}  {1.0:.3f}              "
        f"{sol.eta:.3e}  {sol.L:.3e}    -"
    )

    for it in range(1, max_iter + 1):
        # Step 2: compute the new effective T from the previous L.
        env_profile = envelope.integrate(L_BH=sol.L)
        T_eff = env_profile.T_eff

        # Step 3: rebuild the ambient with the new T and re-solve.
        ambient = ambient0.with_temperature(T_eff)
        sol = rb.BondiProblem(M_BH=M_BH, ambient=ambient, cooling=cooling).solve(cfg)

        # Step 4: check convergence.
        rel = abs(sol.L - L_prev) / L_prev
        print(
            f"  {it:2d}    {T_eff:.3e}  {env_profile.x:.3f}              "
            f"{sol.eta:.3e}  {sol.L:.3e}    {rel:.3e}"
        )
        if rel < tol:
            print(f"\n  Converged in {it} iterations (|dL|/L < {tol}).")
            break
        L_prev = sol.L
    else:
        print(f"\n  Did not converge in {max_iter} iterations.")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("Summary: baseline vs self-consistent")
    print("─" * 70)
    print(f"  T_amb         {ambient0.T:.3e} K  ->  {ambient.T:.3e} K")
    print(f"  eta           {sol0.eta:.3e}    ->  {sol.eta:.3e}")
    print(f"  L             {sol0.L:.3e}    ->  {sol.L:.3e}    erg/s")
    print(f"  Mdot/Mdot_B   {sol0.mdot_ratio:.2f}        ->  {sol.mdot_ratio:.2f}")
    print()
    print("Note: Mdot_B itself depends on T_amb via c_inf, so the two")
    print("Mdot_ratio values are normalized to *different* Bondi rates.")
    print("Compare physical Mdot in g/s if you need the absolute change.")


if __name__ == "__main__":
    main()
