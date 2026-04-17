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

    python examples/02_paper_sweep.py                 # fast demo
    RADBONDI_HI_RES=1 python examples/02_paper_sweep.py   # paper-like, single N
    RADBONDI_PAPER=1  python examples/02_paper_sweep.py   # full paper grid

Optional single-mass runs (useful for parallelising the PAPER sweep
across shells / cores):

    RADBONDI_PAPER=1 RADBONDI_LOGM=-14.0 python examples/02_paper_sweep.py

Three modes:

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
* **PAPER**: the full paper-grade configuration used to produce Table 1
  of Cantiello et al. — per-mass ``(order, N, x_min)`` from the hawking-
  stars development log, with ``n_steps=200_000`` and the early-exit
  convergence check disabled (the order=1 residual drops below useful
  tolerances right after the cooling ramp ends, long before η settles).
  Expect several hours wall time even when parallelised across masses.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import radbondi as rb

# Sweep mode: exactly one of PAPER / HI_RES / fast. PAPER wins if both are set.
PAPER = bool(int(os.environ.get("RADBONDI_PAPER", "0")))
HI_RES = bool(int(os.environ.get("RADBONDI_HI_RES", "0"))) and not PAPER

# Optional: run only a single mass. Handy for shell-backgrounding each mass
# on its own core when reproducing the paper grid.
_LOGM_SINGLE = os.environ.get("RADBONDI_LOGM", "")

# Default mass grid (fast + HI_RES modes). Fine enough to resolve the
# transition from the collisionless regime (eta ~ 0.01) into near-
# isothermal accretion (eta ~ 0.3+).
LOG_MASSES_DEFAULT = [-16.0, -15.0, -14.0, -13.5, -13.0, -12.0, -11.0]

# Paper-grade per-mass configuration from docs/paper_reproduction.md /
# hawking-stars-dev/log.md. Tuple is (logM, order, N, x_min).
# Order=2 is MUSCL/minmod, order=1 is plain HLL. x_min is in units of r_B.
PAPER_CONFIG: list[tuple[float, int, int, float]] = [
    (-16.1, 2, 6400, 3e-6),
    (-16.0, 2, 6400, 3e-6),
    (-15.6, 2, 6400, 3e-6),
    (-15.3, 2, 6400, 3e-6),
    (-15.1, 2, 6400, 3e-6),
    (-15.0, 2, 6400, 3e-6),
    (-14.52, 2, 6400, 5e-6),
    (-14.3, 1, 6400, 5e-6),
    (-14.0, 1, 6400, 5e-6),
    (-13.5, 1, 6400, 1e-5),
    (-13.3, 1, 6400, 1e-5),
    (-13.0, 1, 6400, 1e-5),
    (-12.5, 1, 3200, 1e-5),
    (-12.0, 1, 3200, 1e-5),
    (-11.5, 1, 3200, 1e-5),
    (-11.0, 1, 3200, 1e-5),
    (-10.5, 1, 3200, 1e-5),
    (-10.0, 1, 3200, 1e-5),
]

OUT_DIR = Path(__file__).parent / "paper_sweep_output"


# Threshold in log10(M_BH/M_sun) separating the two regimes. Below this,
# cooling is weak and MUSCL helps; above it, the cooling front is too
# steep for any linear reconstruction and 1st-order is more stable.
STRONG_COOLING_LOGM = -14.5


def _paper_entry(logM: float) -> tuple[int, int, float]:
    """Look up (order, N, x_min) for logM in PAPER_CONFIG."""
    for lm, order, N, x_min in PAPER_CONFIG:
        if abs(lm - logM) < 1e-6:
            return order, N, x_min
    raise KeyError(f"logM={logM} not in PAPER_CONFIG")


def solve_one(logM: float, ambient, cooling) -> rb.Solution:
    M_BH = 10.0**logM * rb.M_sun
    problem = rb.BondiProblem(M_BH=M_BH, ambient=ambient, cooling=cooling)

    if PAPER:
        order, N, x_min = _paper_entry(logM)
        # 200k steps is well past the residual-floor plateau on every mass
        # in the paper grid; the real paper used ~640k but most of that
        # is overkill.
        n_steps = 200_000
        # Disable the early-exit convergence check for the paper run. In
        # the order=1 isothermal regime the residual drops below any
        # useful tolerance (~5e-4) right after the cooling ramp ends —
        # before the flow has had time to equilibrate to the cooled
        # steady state. Always running the full n_steps gives the correct
        # eta (cf. the paper's 640k-step runs).
        convergence_tol = 0.0
    else:
        # Adaptive order: MUSCL in the weak-cooling regime, 1st-order in
        # the strongly-cooled regime (where MUSCL fails to converge).
        strong = logM > STRONG_COOLING_LOGM
        order = 1 if (strong or not HI_RES) else 2
        N = 1200 if HI_RES else 400
        x_min = 1e-5
        # More steps for the strongly-cooled end; the collisionless end
        # converges quickly once the cooling ramp finishes.
        if strong:
            n_steps = 80_000 if HI_RES else 40_000
        else:
            n_steps = 50_000 if HI_RES else 25_000
        convergence_tol = 1e-10  # historical default

    cfg = rb.SolverConfig(
        N=N,
        x_min=x_min,
        x_max=3.0,
        n_steps=n_steps,
        cooling_ramp_steps=5_000,
        order=order,
        limiter="minmod",
        flux="hll",
        CFL=0.4,
        convergence_tol=convergence_tol,
        snapshot_interval=n_steps,  # silence per-step prints
        verbose=False,
    )
    return problem.solve(cfg)


def _fmt_logM(logM: float) -> str:
    """Filename-safe formatting of logM that survives ``-14.52`` etc."""
    # Always keep two decimals so sorted order is stable across masses.
    return f"{logM:+06.2f}"


def main():
    ambient = rb.presets.solar_core()
    cooling = rb.Cooling.default()

    OUT_DIR.mkdir(exist_ok=True)

    # Resolve the mass list for the chosen mode.
    if PAPER:
        log_masses = [e[0] for e in PAPER_CONFIG]
    else:
        log_masses = list(LOG_MASSES_DEFAULT)
    if _LOGM_SINGLE:
        target = float(_LOGM_SINGLE)
        log_masses = [lm for lm in log_masses if abs(lm - target) < 1e-6]
        if not log_masses:
            raise SystemExit(
                f"RADBONDI_LOGM={_LOGM_SINGLE} not in the current mode's mass list"
            )

    if PAPER:
        mode = "PAPER (per-mass N, 200k steps, no early-exit)"
    elif HI_RES:
        mode = "HI_RES (N=1200, adaptive-order)"
    else:
        mode = "fast (N=400, HLL 1st-order)"
    print(f"radbondi paper sweep — {mode}, {len(log_masses)} masses\n")
    print(f"  {'log M/Msun':>10s}  {'ord':>3s}  {'N':>5s}  {'x_min':>8s}  "
          f"{'eta':>10s}  {'Mdot/Mdot_B':>12s}  {'L [erg/s]':>12s}  "
          f"{'T_max/T_inf':>11s}  {'res':>9s}  {'time [s]':>8s}")
    print("  " + "-" * 111)

    results = []
    nonconverged = []
    for logM in log_masses:
        t0 = time.time()
        sol = solve_one(logM, ambient, cooling)
        dt = time.time() - t0

        out_path = OUT_DIR / f"mbh_logM{_fmt_logM(logM)}.npz"
        sol.save(str(out_path))

        t_ratio = sol.T.max() / ambient.T
        if PAPER:
            order_used, N_used, xmin_used = _paper_entry(logM)
        else:
            strong = logM > STRONG_COOLING_LOGM
            order_used = 2 if (HI_RES and not strong) else 1
            N_used = 1200 if HI_RES else 400
            xmin_used = 1e-5
        flag = " " if sol.converged else "!"
        if not sol.converged:
            nonconverged.append(logM)
        results.append((logM, sol.eta, sol.mdot_ratio, sol.L, t_ratio))
        print(
            f" {flag}{logM:>10.2f}  {order_used:>3d}  {N_used:>5d}  "
            f"{xmin_used:>8.1e}  {sol.eta:>10.3e}  "
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
