"""Render examples/paper_sweep_output/SUMMARY.md from the current npz set.

Reads every ``mbh_logM*.npz`` and the paper reference CSV; writes a markdown
table comparing η and Ṁ/Ṁ_B, annotates outliers, and records the final
solver residual and step count for each mass.

Usage::

    MPLBACKEND=Agg python examples/_make_summary.py [PATH/TO/sweep_summary_best.csv]
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import radbondi as rb

DEFAULT_REF = (
    "/mnt/home/mcantiello/work/pbh-accretion/sweep_results/sweep_summary_best.csv"
)
TOL = 0.05


def main() -> int:
    ref_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_REF)
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
        rows.append({
            "logM": logM,
            "eta": sol.eta,
            "eta_ref": eta_ref,
            "rel_eta": rel_eta,
            "mdot": sol.mdot_ratio,
            "mdot_ref": mdot_ref,
            "rel_mdot": rel_mdot,
            "res": sol.solver_residual,
            "n_steps": len(sol.residuals),
        })

    n_bad = sum(1 for r in rows if abs(r["rel_eta"]) >= TOL)
    worst = max(rows, key=lambda r: abs(r["rel_eta"]))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Paper Table 1 reproduction — PAPER sweep",
        "",
        f"Generated: {now}",
        "",
        "This table compares each mass in the paper's production grid against",
        "Cantiello et al. Table 1 (`sweep_summary_best.csv`). The radbondi",
        "`PAPER` sweep mode uses per-mass (order, N, x_min) from the original",
        "hawking-stars production log, 200 000 steps per mass, and disables the",
        "early-exit convergence check (Lie splitting, HLL + minmod / MUSCL).",
        "",
        f"**Overall**: {len(rows) - n_bad} / {len(rows)} masses within "
        f"±{TOL * 100:.0f}% on η relative to the paper.",
        "",
        "| log M/M☉ | paper η | ours η | Δη/η | paper Ṁ/Ṁ_B | ours Ṁ/Ṁ_B | "
        "ΔṀ/Ṁ | n_steps | final residual |",
        "|---------:|--------:|-------:|-----:|------------:|-----------:|"
        "-----:|--------:|---------------:|",
    ]
    for r in rows:
        mark = " ❌" if abs(r["rel_eta"]) >= TOL else ""
        lines.append(
            f"| {r['logM']:+.2f}"
            f" | {r['eta_ref']:.4e}"
            f" | {r['eta']:.4e}"
            f" | {r['rel_eta'] * 100:+.1f}%{mark}"
            f" | {r['mdot_ref']:.3f}"
            f" | {r['mdot']:.3f}"
            f" | {r['rel_mdot'] * 100:+.2f}%"
            f" | {r['n_steps']}"
            f" | {r['res']:.2e} |"
        )

    lines += [
        "",
        (
            f"Worst η mismatch: **logM = {worst['logM']:+.2f}** at "
            f"{worst['rel_eta'] * 100:+.1f}%."
        ),
        "",
        "## Configuration",
        "",
        "- Ambient: `radbondi.presets.solar_core()` (solar-core composition,",
        "  T = 15.7 MK, ρ = 150 g/cm³, μ = 0.85, γ = 5/3).",
        "- Cooling: `radbondi.Cooling.default()` (bremsstrahlung + pair",
        "  annihilation + muon pair channel, with the ambient e⁻-scattering",
        "  floor subtracted).",
        "- Per-mass configuration: see `examples/02_paper_sweep.py :: PAPER_CONFIG`.",
        "- Sweep script: `examples/02_paper_sweep.py` with `RADBONDI_PAPER=1`.",
        "- Verification: `examples/_verify_paper_sweep.py`.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "# Single process, 18 masses sequentially (~10-15 h on one core):",
        "MPLBACKEND=Agg RADBONDI_PAPER=1 python examples/02_paper_sweep.py",
        "",
        "# Or parallel (recommended, ~2-3 h wall on a 16-core box):",
        "examples/_run_paper_sweep.sh 14",
        "```",
        "",
    ]

    out_path = out_dir / "SUMMARY.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path} — {len(rows) - n_bad}/{len(rows)} within {TOL * 100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
