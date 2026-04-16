"""Reproduce the Table 1 mass sweep from Cantiello et al. (in prep).

Solves the Bondi+cooling problem for a grid of BH masses embedded in the
current solar core, writes each ``Solution`` to disk as a ``.npz``, and
prints a summary table. At the end, if matplotlib is available, produces
a two-panel plot of the radiative efficiency ``eta`` and the accretion
enhancement ``Mdot/Mdot_B`` vs ``log10(M_BH / M_sun)``.

The paper's three canonical regimes (used by ``tests/test_validation.py``
as reference values):

    log10(M/M_sun) = -16.0  collisionless   eta ~ 1.2e-2   Mdot_ratio ~ 1.6
    log10(M/M_sun) = -13.5  bremsstrahlung  eta ~ 1.0e-2   Mdot_ratio ~ 7.2
    log10(M/M_sun) = -11.0  near-isothermal eta ~ 3.4e-1   Mdot_ratio ~ 7.0

Run with:

    python examples/02_paper_sweep.py               # fast demo
    RADBONDI_HI_RES=1 python examples/02_paper_sweep.py   # paper-resolution

Two modes:

* **Fast** (default): N=400, 1st-order HLL. 7 masses in a few minutes on
  a laptop. Reproduces the same numbers as ``tests/test_validation.py``,
  within ~30% of the paper's published values.
* **HI_RES**: N=1200, paper-style **adaptive-order** scheme — 2nd-order
  MUSCL/minmod for the weak-cooling regime (log M/M_sun <= -14.5) and
  1st-order HLL for the strongly-cooled regime (log M/M_sun > -14.5).
  This mirrors the paper's production strategy: MUSCL is less diffusive
  in the collisionless end where it helps, but 1st-order is used in the
  near-isothermal regime where MUSCL fundamentally cannot converge
  (the cooling front is steeper than any linear reconstruction can
  represent). ~15-45 min depending on hardware.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import radbondi as rb

# Flip to True (or `export RADBONDI_HI_RES=1`) for paper-resolution runs.
HI_RES = bool(int(os.environ.get("RADBONDI_HI_RES", "0")))

# Mass grid in log10(M/M_sun). Fine enough to resolve the transition from
# the collisionless regime (eta ~ 0.01) into near-isothermal accretion
# (eta ~ 0.3+).
LOG_MASSES = [-16.0, -15.0, -14.0, -13.5, -13.0, -12.0, -11.0]

OUT_DIR = Path(__file__).parent / "paper_sweep_output"


# Threshold in log10(M_BH/M_sun) separating the two regimes. Below this,
# cooling is weak and MUSCL helps; above it, the cooling front is too
# steep for any linear reconstruction and 1st-order is more stable.
STRONG_COOLING_LOGM = -14.5


def solve_one(logM: float, ambient, cooling) -> rb.Solution:
    M_BH = 10.0**logM * rb.M_sun
    problem = rb.BondiProblem(M_BH=M_BH, ambient=ambient, cooling=cooling)

    # Adaptive order: MUSCL in the weak-cooling regime, 1st-order in
    # the strongly-cooled regime (where MUSCL fails to converge).
    strong = logM > STRONG_COOLING_LOGM
    order = 1 if (strong or not HI_RES) else 2

    # More steps for the strongly-cooled end; the collisionless end
    # converges quickly once the cooling ramp finishes.
    if strong:
        n_steps = 80_000 if HI_RES else 40_000
    else:
        n_steps = 50_000 if HI_RES else 25_000

    cfg = rb.SolverConfig(
        N=1200 if HI_RES else 400,
        x_min=1e-5,
        x_max=3.0,
        n_steps=n_steps,
        cooling_ramp_steps=5_000,
        order=order,
        limiter="minmod",
        flux="hll",
        CFL=0.4,
        snapshot_interval=n_steps,  # silence per-step prints
        verbose=False,
    )
    return problem.solve(cfg)


def main():
    ambient = rb.presets.solar_core()
    cooling = rb.Cooling.default()

    OUT_DIR.mkdir(exist_ok=True)
    mode = (
        "HI_RES (N=1200, adaptive-order)" if HI_RES
        else "fast (N=400, HLL 1st-order)"
    )
    print(f"radbondi paper sweep — {mode}, {len(LOG_MASSES)} masses\n")
    print(f"  {'log M/Msun':>10s}  {'ord':>3s}  {'eta':>10s}  {'Mdot/Mdot_B':>12s}  "
          f"{'L [erg/s]':>12s}  {'T_max/T_inf':>11s}  {'res':>9s}  {'time [s]':>8s}")
    print("  " + "-" * 87)

    results = []
    nonconverged = []
    for logM in LOG_MASSES:
        t0 = time.time()
        sol = solve_one(logM, ambient, cooling)
        dt = time.time() - t0

        out_path = OUT_DIR / f"mbh_logM{logM:+.1f}.npz"
        sol.save(str(out_path))

        t_ratio = sol.T.max() / ambient.T
        order_used = 2 if (HI_RES and logM <= STRONG_COOLING_LOGM) else 1
        flag = " " if sol.converged else "!"
        if not sol.converged:
            nonconverged.append(logM)
        results.append((logM, sol.eta, sol.mdot_ratio, sol.L, t_ratio))
        print(
            f" {flag}{logM:>10.1f}  {order_used:>3d}  {sol.eta:>10.3e}  "
            f"{sol.mdot_ratio:>12.2f}  {sol.L:>12.3e}  {t_ratio:>11.1f}  "
            f"{sol.solver_residual:>9.2e}  {dt:>8.1f}"
        )

    if nonconverged:
        print(
            f"\nWARNING: {len(nonconverged)} runs did not reach the convergence "
            f"tolerance: logM = {nonconverged}"
        )
        print("  Their eta / Mdot values should not be trusted. Raise n_steps "
              "or switch to order=1 for those masses.")

    print(f"\nSolutions saved to: {OUT_DIR.resolve()}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(Install matplotlib to see the summary plot: pip install radbondi[plot])")
        return

    logM, eta, mdot, L, _ = zip(*results, strict=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(logM, eta, "o-")
    axes[0].set(
        xlabel=r"$\log_{10}(M_\bullet/M_\odot)$",
        ylabel=r"$\eta = L / (\dot M_B c^2)$",
        title="Radiative efficiency",
    )
    axes[1].plot(logM, mdot, "o-")
    axes[1].set(
        xlabel=r"$\log_{10}(M_\bullet/M_\odot)$",
        ylabel=r"$\dot M / \dot M_B$",
        title="Accretion enhancement",
    )
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = OUT_DIR / "paper_sweep.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Summary figure: {fig_path.resolve()}")


if __name__ == "__main__":
    main()
