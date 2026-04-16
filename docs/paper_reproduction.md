# Reproducing Cantiello et al. Table 1

This document is a handoff / work-log for reproducing the paper's Table 1
mass sweep exactly. The first pass at `examples/02_paper_sweep.py` was
written and tuned in April 2026 on a development machine; the definitive
run is intended for a faster machine and is expected to match the paper's
η values to within a few percent everywhere.

---

## Status as of 2026-04-16

- **Code state**: `main @ 5a5965d` (v0.1.0 tagged). Phase 3 tests were
  still running in the background at the end of the session; do not modify
  source files until those complete.
- **Example state**: `examples/02_paper_sweep.py` implements the
  adaptive-order strategy (MUSCL for weak cooling, 1st-order for strong
  cooling) at N=1200 with `RADBONDI_HI_RES=1`.
- **Uncommitted changes** at session end:
  - `docs/usage.md` — HI_RES description updated to "adaptive-order".
  - `examples/02_paper_sweep.py` — adaptive order; convergence-flag in
    the results table; `STRONG_COOLING_LOGM = -14.5` threshold.

## What has been validated

The fast-mode sweep (N=400, 1st-order HLL) reproduces the values in
`tests/test_validation.py` — which are themselves within ~30% of the
paper's published η. Not definitive, but a useful smoke test.

The HI_RES (N=1200, adaptive-order) sweep was run once. Results against
the paper's Table 1 (`hawking-stars-dev/sweep_results/sweep_summary_best.csv`):

| log M/M☉ | Paper η   | Paper Ṁ/Ṁ_B | Ours η    | Ours Ṁ/Ṁ_B | Δη    |
|---------:|----------:|------------:|----------:|-----------:|------:|
| −16.0    |  1.99e-2  |        1.80 |  1.39e-2  |       1.69 |  −30% |
| −15.0    |  8.45e-2  |        6.52 |  4.15e-2  |       6.46 |  −51% |
| −14.0    |  3.47e-3  |        7.26 |  8.86e-3  |       7.15 | +155% |
| −13.5    |  1.85e-3  |        7.25 |  6.87e-3  |       7.05 | +271% |
| −13.0    |  4.11e-3  |        7.10 |  7.99e-3  |       6.95 |  +94% |
| −12.0    |  3.46e-2  |        6.95 |  3.62e-2  |       6.95 |   +5% |
| −11.0    |  3.45e-1  |        6.95 |  3.45e-1  |       6.95 | +0.1% |

Verdict: **isothermal end is excellent** (<5% in η and Ṁ/Ṁ_B), **transitional
regime is badly under-resolved at N=1200** (η off by 2–3×). To match the
paper the next session needs N=6400 with mass-dependent `x_min`.

## The paper's exact production configuration

Per `hawking-stars-dev/log.md` and `sweep_summary_best.csv`:

| log M/M☉       | order  | N    | x_min | Notes                           |
|---------------:|-------:|-----:|------:|---------------------------------|
| −16.1 … −15.0  | 2 (MUSCL) | 6400 | 3e-6  | weak cooling / collisionless    |
| −14.52         | 2 (MUSCL) | 6400 | 5e-6  | upper transitional              |
| −14.3 … −14.0  |   1 (HLL) | 6400 | 5e-6  | transitional                    |
| −13.5 … −13.0  |   1 (HLL) | 6400 | 1e-5  | strong cooling / bremsstrahlung |
| −12.5 … −10.0  |   1 (HLL) | 3200 | ~1e-5 | deep isothermal (cheap)         |

Other knobs held fixed across the sweep:
- `flux="hll"`, `limiter="minmod"`, `CFL=0.4`
- `cooling_ramp_steps=5_000`
- Lie splitting (default `strang=False`) — **do not switch to Strang**,
  it catastrophically destabilises the order=2 runs per the hawking-stars
  development log.
- `x_max=3.0`
- Well-balancing left at default (`wb_mode="full"`).
- `n_steps` in the paper's production runs was **~640 000** (!) per
  mass. Most of that is overkill — residual typically drops below 1e-3
  long before step 100 000 — but the long runs give a very tight
  convergence history for the final values.

## What to run next session

### Step 1 — confirm repo state

```bash
cd /Users/mcantiello/astro/work/radbondi
git status                 # verify uncommitted changes are still the two above
git log --oneline -5       # expect 5a5965d at top
pytest -m "not slow"       # fast tests should pass; slow ones require `-m slow`
```

### Step 2 — extend `examples/02_paper_sweep.py` to the paper's grid

The current script has a single `N` (1200) and a single `x_min` (1e-5).
For a paper-matching run we need a per-mass config table. Recommended
approach: add a `PAPER_CONFIG` dict keyed by logM, something like:

```python
# (logM, order, N, x_min)
PAPER_CONFIG = [
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
```

Gate this behind a third mode (e.g. `RADBONDI_PAPER=1`) so HI_RES stays
as the "fast paper-like demo". Set `n_steps=200_000` — enough for
residual <1e-3 on any case per the hawking-stars log, way below the
640 000 the paper actually used.

Also set `convergence_tol=1e-3` in the SolverConfig to get a truthful
`converged` flag (default 1e-10 is unreachable in practice — see
`log.md` notes about oscillatory residuals ~1e-3 that still represent
steady state).

### Step 3 — run it

```bash
RADBONDI_PAPER=1 python examples/02_paper_sweep.py 2>&1 | tee paper_sweep.log
```

Expected wall time on a fast machine (≥16 cores, but solver is
single-threaded per run — parallelism comes from running masses in
parallel if you want):

- N=6400 MUSCL: ~30–90 min per mass × 7 masses = 4–10 h serial
- N=6400 order=1: ~20–60 min per mass × 5 masses = 2–5 h
- N=3200 order=1: ~5–15 min per mass × 6 masses = 30–90 min

Total serial: **roughly 8–15 h** on a modern workstation. If you
parallelise with `multiprocessing` or just shell-background each run,
wall time drops to a few hours.

### Step 4 — verify

Load each saved solution and check:

```python
import radbondi as rb, numpy as np, pandas as pd
from pathlib import Path

paper = pd.read_csv(
    "/Users/mcantiello/astro/work/hawking-stars-dev/sweep_results/sweep_summary_best.csv",
    comment="#",
)

rows = []
for p in sorted(Path("examples/paper_sweep_output").glob("mbh_logM*.npz")):
    sol = rb.load(str(p))
    logM = float(p.stem.split("logM")[1])
    ref = paper[np.isclose(paper.log_mass, logM)]
    eta_ref = float(ref.eta_best.iloc[0]) if len(ref) else float("nan")
    rows.append((logM, sol.eta, eta_ref, sol.eta / eta_ref - 1.0,
                 sol.converged, sol.solver_residual))

print(f"{'logM':>6s} {'eta':>10s} {'eta_paper':>10s} {'rel err':>8s} {'conv':>5s} {'res':>9s}")
for r in rows:
    print(f"{r[0]:>6.2f} {r[1]:>10.3e} {r[2]:>10.3e} {r[3]*100:>+7.1f}% "
          f"{str(r[4]):>5s} {r[5]:>9.2e}")
```

Target: |Δη/η| < 5% everywhere, Ṁ/Ṁ_B match to <2%.

If the transitional regime (−14 ≤ logM ≤ −13) is still off by more than
5%, first thing to try is doubling `n_steps`. The hawking-stars Phase 22
log shows these masses need more time to settle than the extreme ends.

## Known gotchas

- **`sol.converged` is pessimistic**: default tolerance is 1e-10; the
  physically-settled residual is ~1e-3 to 1e-4. Check residuals
  explicitly, don't rely on the flag unless you lowered `convergence_tol`.
- **MUSCL fails for strong cooling**: do not try to use `order=2` for
  log M/M☉ > −14.5. It will either blow up (MC limiter) or return
  un-converged garbage (minmod). This is a property of the scheme, not
  a bug — see Explore-agent summary in session notes.
- **Strang splitting is broken**: per hawking-stars development log,
  `strang=True` with `order=2` destabilises the solver (η oscillates
  between 0.1 and 48). Leave it off.
- **x_min must be near r_S/r_B**: values like 1e-3 cut off the hot
  inner region and produce spurious "isothermal transitional" results.
  The table above is calibrated; don't deviate.
- **Phase 3 tests still running (in another shell)**: until they
  complete, don't modify source files. Example-only changes are safe.

## References

- `hawking-stars-dev/log.md` — full development history of the original
  solver, with all the convergence-debugging notes.
- `hawking-stars-dev/sweep_results/sweep_summary_best.csv` — the
  numbers this sweep must reproduce.
- `hawking-stars-dev/CONVERGENCE_STATUS.md` — snapshot of the
  Phase 18/22 x_min convergence validation.
- `tests/test_validation.py` — paper Table 1 reference values at
  fast-test resolution (within ~30% of the paper values).
- `docs/scheme.md` — the numerical scheme in radbondi.
