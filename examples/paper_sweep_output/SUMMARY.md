# Paper Table 1 reproduction — PAPER sweep

Generated: 2026-04-17 00:56 UTC

This table compares each mass in the paper's production grid against
Cantiello et al. Table 1 (`sweep_summary_best.csv`). The radbondi
`PAPER` sweep mode uses per-mass (order, N, x_min) from the original
hawking-stars production log, 200 000 steps per mass, and disables the
early-exit convergence check (Lie splitting, HLL + minmod / MUSCL).

**Overall**: 18 / 18 masses within ±5% on η relative to the paper.

| log M/M☉ | paper η | ours η | Δη/η | paper Ṁ/Ṁ_B | ours Ṁ/Ṁ_B | ΔṀ/Ṁ | n_steps | final residual |
|---------:|--------:|-------:|-----:|------------:|-----------:|-----:|--------:|---------------:|
| -16.10 | 1.8304e-02 | 1.8099e-02 | -1.1% | 1.633 | 1.629 | -0.25% | 200000 | 1.24e-02 |
| -16.00 | 1.9889e-02 | 2.0047e-02 | +0.8% | 1.797 | 1.797 | +0.01% | 200000 | 1.20e-02 |
| -15.60 | 3.5659e-02 | 3.5875e-02 | +0.6% | 2.985 | 2.995 | +0.35% | 200000 | 1.43e-02 |
| -15.30 | 5.7733e-02 | 5.8844e-02 | +1.9% | 4.622 | 4.630 | +0.17% | 200000 | 1.44e-02 |
| -15.10 | 7.5858e-02 | 7.8773e-02 | +3.8% | 5.874 | 6.065 | +3.25% | 400000 | 1.45e-02 |
| -15.00 | 8.4534e-02 | 8.3213e-02 | -1.6% | 6.521 | 6.522 | +0.02% | 200000 | 1.59e-02 |
| -14.52 | 3.9211e-02 | 3.9224e-02 | +0.0% | 7.251 | 7.251 | +0.00% | 200000 | 7.13e-03 |
| -14.30 | 1.7471e-02 | 1.7382e-02 | -0.5% | 7.246 | 7.246 | -0.00% | 200000 | 5.62e-05 |
| -14.00 | 3.4747e-03 | 3.4723e-03 | -0.1% | 7.255 | 7.234 | -0.30% | 200000 | 9.00e-05 |
| -13.50 | 1.8524e-03 | 1.8536e-03 | +0.1% | 7.249 | 7.185 | -0.89% | 200000 | 1.69e-04 |
| -13.30 | 2.4589e-03 | 2.4633e-03 | +0.2% | 7.154 | 7.153 | -0.01% | 200000 | 2.16e-04 |
| -13.00 | 4.1100e-03 | 4.1120e-03 | +0.0% | 7.095 | 7.095 | +0.00% | 200000 | 2.87e-04 |
| -12.50 | 1.1644e-02 | 1.1652e-02 | +0.1% | 6.946 | 6.949 | +0.04% | 200000 | 4.42e-04 |
| -12.00 | 3.4568e-02 | 3.4594e-02 | +0.1% | 6.947 | 6.949 | +0.03% | 200000 | 4.89e-04 |
| -11.50 | 1.0918e-01 | 1.0928e-01 | +0.1% | 6.947 | 6.949 | +0.02% | 200000 | 3.93e-04 |
| -11.00 | 3.4533e-01 | 3.4554e-01 | +0.1% | 6.947 | 6.949 | +0.02% | 200000 | 4.73e-04 |
| -10.50 | 1.0918e+00 | 1.0927e+00 | +0.1% | 6.947 | 6.949 | +0.02% | 200000 | 3.75e-04 |
| -10.00 | 3.4530e+00 | 3.4552e+00 | +0.1% | 6.947 | 6.949 | +0.03% | 200000 | 4.31e-04 |

Worst η mismatch: **logM = -15.10** at +3.8%.

## Configuration

- Ambient: `radbondi.presets.solar_core()` (solar-core composition,
  T = 15.7 MK, ρ = 150 g/cm³, μ = 0.85, γ = 5/3).
- Cooling: `radbondi.Cooling.default()` (bremsstrahlung + pair
  annihilation + muon pair channel, with the ambient e⁻-scattering
  floor subtracted).
- Per-mass configuration: see `examples/02_paper_sweep.py :: PAPER_CONFIG`.
- Sweep script: `examples/02_paper_sweep.py` with `RADBONDI_PAPER=1`.
- Verification: `examples/_verify_paper_sweep.py`.

## Reproducing

```bash
# Single process, 18 masses sequentially (~10-15 h on one core):
MPLBACKEND=Agg RADBONDI_PAPER=1 python examples/02_paper_sweep.py

# Or parallel (recommended, ~2-3 h wall on a 16-core box):
examples/_run_paper_sweep.sh 14
```
