"""Re-run a single PAPER_CONFIG mass with an override on n_steps.

Handy for pushing a borderline mass below the 5% tolerance when the paper
sweep's default 200 000 steps isn't quite enough.

Usage::

    RADBONDI_LOGM=-15.1 RADBONDI_NSTEPS=400000 \
        MPLBACKEND=Agg python examples/_rerun_tight.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import radbondi as rb

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import the PAPER_CONFIG lookup & formatting from 02_paper_sweep.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "sweep", str(SCRIPT_DIR / "02_paper_sweep.py")
)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

OUT_DIR = SCRIPT_DIR / "paper_sweep_output"


def main() -> int:
    logM = float(os.environ["RADBONDI_LOGM"])
    n_steps = int(os.environ.get("RADBONDI_NSTEPS", "400000"))
    order, N, x_min = sweep._paper_entry(logM)

    ambient = rb.presets.solar_core()
    cooling = rb.Cooling.default()
    M_BH = 10.0**logM * rb.M_sun
    problem = rb.BondiProblem(M_BH=M_BH, ambient=ambient, cooling=cooling)

    cfg = rb.SolverConfig(
        N=N, x_min=x_min, x_max=3.0,
        n_steps=n_steps, cooling_ramp_steps=5_000,
        order=order, limiter="minmod", flux="hll", CFL=0.4,
        convergence_tol=0.0,
        snapshot_interval=n_steps,
        verbose=False,
    )
    t0 = time.time()
    sol = problem.solve(cfg)
    dt = time.time() - t0

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"mbh_logM{sweep._fmt_logM(logM)}.npz"
    sol.save(str(out))
    print(
        f"logM={logM:+.2f}  order={order}  N={N}  x_min={x_min:.1e}  "
        f"n_steps={n_steps}  -> eta={sol.eta:.4e}  "
        f"Mdot/Mdot_B={sol.mdot_ratio:.3f}  res={sol.solver_residual:.2e}  "
        f"time={dt:.0f}s  -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
