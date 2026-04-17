"""Compare the PAPER sweep output against Cantiello et al. Table 1.

Reads every ``mbh_logM*.npz`` in ``examples/paper_sweep_output/``, looks up
the paper's published ``eta_best`` / ``Mdot_over_Mdot_B`` for that mass in
``sweep_summary_best.csv``, and prints a side-by-side table plus overall
pass/fail against a 5% tolerance on |Δη/η|.

Usage::

    MPLBACKEND=Agg python examples/_verify_paper_sweep.py \
        [PATH/TO/sweep_summary_best.csv]

The default reference CSV path is
``/mnt/home/mcantiello/work/pbh-accretion/sweep_results/sweep_summary_best.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import radbondi as rb

DEFAULT_REF = (
    "/mnt/home/mcantiello/work/pbh-accretion/sweep_results/sweep_summary_best.csv"
)

TOL = 0.05  # 5% on |eta/eta_paper - 1|


def main() -> int:
    ref_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_REF)
    if not ref_csv.exists():
        raise SystemExit(f"reference CSV not found: {ref_csv}")
    paper = pd.read_csv(ref_csv, comment="#")

    out_dir = Path(__file__).parent / "paper_sweep_output"
    npz_paths = sorted(out_dir.glob("mbh_logM*.npz"),
                       key=lambda p: float(p.stem.split("logM")[1]))
    if not npz_paths:
        raise SystemExit(f"no npz files in {out_dir}")

    rows = []
    for p in npz_paths:
        sol = rb.load(str(p))
        logM = float(p.stem.split("logM")[1])
        match = paper[np.isclose(paper.log_mass, logM, atol=1e-3)]
        eta_ref = float(match.eta_best.iloc[0]) if len(match) else float("nan")
        mdot_ref = (
            float(match.Mdot_over_Mdot_B.iloc[0]) if len(match) else float("nan")
        )
        rel_eta = sol.eta / eta_ref - 1.0 if eta_ref == eta_ref else float("nan")
        rel_mdot = (
            sol.mdot_ratio / mdot_ref - 1.0
            if mdot_ref == mdot_ref else float("nan")
        )
        rows.append(
            (logM, sol.eta, eta_ref, rel_eta,
             sol.mdot_ratio, mdot_ref, rel_mdot,
             sol.converged, sol.solver_residual)
        )

    header = (
        f"{'logM':>6s} {'eta':>10s} {'eta_paper':>10s} {'|Δ|/η':>7s}  "
        f"{'Mdot':>6s} {'Mdot_p':>6s} {'|Δ|':>6s}  "
        f"{'conv':>5s} {'res':>9s}"
    )
    print(header)
    print("-" * len(header))
    n_bad = 0
    worst = ("", 0.0)
    for r in rows:
        logM, eta, eta_ref, de, mdot, mdot_ref, dm, conv, res = r
        flag = "  " if abs(de) < TOL else " !"
        if abs(de) >= TOL:
            n_bad += 1
        if abs(de) > abs(worst[1]):
            worst = (f"{logM:+.2f}", de)
        print(
            f"{logM:>6.2f} {eta:>10.3e} {eta_ref:>10.3e} "
            f"{de * 100:>+6.1f}%{flag} "
            f"{mdot:>6.2f} {mdot_ref:>6.2f} {dm * 100:>+5.1f}%  "
            f"{str(conv):>5s} {res:>9.2e}"
        )

    print()
    if n_bad == 0:
        print(f"PASS: all {len(rows)} masses within {TOL * 100:.0f}% of paper η.")
    else:
        print(
            f"FAIL: {n_bad}/{len(rows)} masses exceed {TOL * 100:.0f}% on η; "
            f"worst at logM={worst[0]} ({worst[1] * 100:+.1f}%)."
        )
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
