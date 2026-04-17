# Reproducing Cantiello et al. Table 1

The paper's Table 1 reports η (radiative efficiency) and Ṁ/Ṁ_B
(accretion-rate enhancement) for 18 BH masses from 10⁻¹⁶·¹ to
10⁻¹⁰ M☉ in the solar core. All 18 values have been reproduced with
`radbondi` to within ±5% on η, using `examples/02_paper_sweep.py`.

## Quick start

Three modes, in increasing fidelity and runtime:

```bash
# 1. Fast validation (~2-5 min, ±30% on eta, 7 masses):
python examples/02_paper_sweep.py

# 2. Higher resolution (~15-45 min, ±10% on eta, 7 masses):
RADBONDI_HI_RES=1 python examples/02_paper_sweep.py

# 3. Paper-exact grid (~8-15 h, <5% on eta, 18 masses):
RADBONDI_PAPER=1 python examples/02_paper_sweep.py
```

Each mode writes Solutions to `examples/paper_sweep_output/` as `.npz`
files (loadable with `rb.load(path)`) and prints a summary table.

For the `PAPER` mode on a multi-core machine, a parallel launcher is
provided:

```bash
examples/_run_paper_sweep.sh 14    # run on 14 cores, ~2-3 h wall time
```

## Validation results

The production sweep (`RADBONDI_PAPER=1`) reproduces the paper to
better than 5% on η everywhere. Full results are in
`examples/paper_sweep_output/SUMMARY.md`. Highlights:

| log M/M☉ | Regime | Paper η | radbondi η | Δη/η |
|---------:|--------|--------:|-----------:|-----:|
| −16.0 | Collisionless | 1.99e-2 | 2.00e-2 | +0.8% |
| −15.0 | Collisionless (peak η) | 8.45e-2 | 8.32e-2 | −1.6% |
| −14.0 | Transitional | 3.47e-3 | 3.47e-3 | −0.1% |
| −13.0 | Bremsstrahlung cooling | 4.11e-3 | 4.11e-3 | +0.0% |
| −11.0 | Near-isothermal | 3.45e-1 | 3.46e-1 | +0.1% |

Ṁ/Ṁ_B agrees to <1% across all masses. Worst η mismatch: log M = −15.1
at +3.8%.

## Per-mass solver configuration

The paper uses mass-dependent resolution and spatial order because
the physics changes across regimes:

| Mass range (log M/M☉) | Order | N | x_min | Rationale |
|-----------------------:|------:|----:|------:|-----------|
| −16.1 to −15.0 | 2 (MUSCL) | 6400 | 3e-6 | Weak cooling; MUSCL reduces numerical diffusion near the steep inner temperature peak |
| −14.52 | 2 (MUSCL) | 6400 | 5e-6 | Upper transitional |
| −14.3 to −14.0 | 1 (HLL) | 6400 | 5e-6 | Transitional; MUSCL cannot resolve the sharp cooling front |
| −13.5 to −13.0 | 1 (HLL) | 6400 | 1e-5 | Strong bremsstrahlung cooling |
| −12.5 to −10.0 | 1 (HLL) | 3200 | 1e-5 | Deep isothermal; lower N suffices because the profiles are smooth |

All masses use: `flux="hll"`, `limiter="minmod"`, `CFL=0.4`,
`cooling_ramp_steps=5000`, `n_steps=200000`, Lie splitting
(`strang=False`), well-balanced mode (`wb_mode="full"`).

These settings are encoded in `PAPER_CONFIG` inside
`examples/02_paper_sweep.py`.

## Verification script

After a sweep completes, verify against the paper's reference values:

```bash
python examples/_verify_paper_sweep.py
```

This loads every `.npz` in `examples/paper_sweep_output/`, compares
against the paper's CSV, and prints a table with per-mass Δη/η.

## Known gotchas

- **MUSCL fails for strong cooling** (log M > −14.5). The cooling front
  is too steep for any linear reconstruction. Use `order=1` in that
  regime — this is not a bug, it is a property of the scheme.
- **`sol.converged` is pessimistic.** The default `convergence_tol=1e-10`
  is unreachable in practice; the physically-settled residual is ~1e-3 to
  1e-2. The `PAPER` mode disables the early-exit check and runs the full
  200k steps instead. Check `sol.solver_residual` or the η stability
  between snapshots rather than the boolean flag.
- **x_min must be near r_S/r_B** (~1e-5 to 3e-6). Larger values
  (e.g. 1e-3) cut off the hot inner region and produce spuriously low η.
- **Strang splitting is unstable** with `order=2`. Leave `strang=False`.
- **All units are CGS.** η is dimensionless; L is erg/s; Ṁ is g/s.
