# Usage

A practical guide to running `radbondi`. For the physics behind the
equations, see [physics.md](physics.md); for the discretization, see
[scheme.md](scheme.md).

---

## 1. Install

```bash
pip install -e ".[dev]"      # development install
pip install -e ".[plot]"     # + matplotlib for plot_profiles
```

Python ≥ 3.10 required. Dependencies: `numpy`, `scipy`.

---

## 2. Quickstart

```python
import radbondi as rb

ambient = rb.presets.solar_core()
problem = rb.BondiProblem(
    M_BH=1e-16 * rb.M_sun,
    ambient=ambient,
    cooling=rb.Cooling.default(),
)

sol = problem.solve(rb.SolverConfig(N=400, x_min=1e-5, n_steps=30_000))

print(f"eta         = {sol.eta:.3e}")
print(f"Mdot/Mdot_B = {sol.mdot_ratio:.2f}")
print(f"L           = {sol.L:.3e} erg/s")
```

A runnable version of this is in `examples/01_quickstart.py`.

All inputs and outputs are CGS (g, cm, s, K, erg). Convert masses with
`rb.M_sun`; other constants live in `rb.constants`.

---

## 3. Workflow

The three objects you always work with:

1. **`AmbientMedium`** — the unperturbed gas (T, ρ, composition).
2. **`BondiProblem`** — pairs the accretor mass + ambient + cooling.
3. **`SolverConfig`** — numerical settings for the integrator.

Calling `problem.solve(config)` returns a **`Solution`** that carries the
converged profiles and derived scalars.

---

## 4. The `AmbientMedium`

```python
from radbondi import AmbientMedium

ambient = AmbientMedium(
    T=1.5e7,     # K
    rho=150.0,   # g cm^-3
    mu=0.85,     # mean molecular weight
    gamma=5/3,
    X=0.34,      # hydrogen mass fraction
    Y=0.64,      # helium
)
```

`AmbientMedium` is **frozen** (immutable). To change one property, use
the provided copy helpers — they are the intended pattern for feedback
iteration:

```python
hotter = ambient.with_temperature(2 * ambient.T)
denser = ambient.with_density(2 * ambient.rho)
```

Useful derived properties:

| attribute | meaning |
|---|---|
| `ambient.cs` | adiabatic sound speed [cm/s] |
| `ambient.n_e` | electron number density (full ionization) [cm⁻³] |
| `ambient.n_i` | ion number density (H+He) [cm⁻³] |

### Presets

```python
rb.presets.solar_core()                     # T=1.57e7, rho=150, mu=0.85
rb.presets.primordial_gas(T=1e4, rho=1e-24) # warm neutral medium default
```

---

## 5. `BondiProblem`

```python
problem = rb.BondiProblem(
    M_BH=1e-16 * rb.M_sun,  # g
    ambient=ambient,
    cooling=rb.Cooling.default(),  # optional; default = brem + pair
)

# Derived scales (read-only)
problem.r_B       # Bondi radius [cm]
problem.r_S       # Schwarzschild radius [cm]
problem.Mdot_B    # adiabatic Bondi rate [g/s]
```

`cooling` is optional. Omit or pass `rb.Cooling.adiabatic()` for a
cooling-free run (useful for testing — it should reproduce the classical
Bondi solution to within discretization error).

---

## 6. `SolverConfig`

```python
cfg = rb.SolverConfig(
    N=800,              # number of radial cells
    x_min=1e-3,         # inner boundary in r_B units
    x_max=3.0,          # outer boundary in r_B units
    n_steps=50_000,     # max iterations
    CFL=0.4,            # per-cell Courant number
    order=2,            # 1 = PCM, 2 = MUSCL (default)
    limiter="minmod",   # or "mc"
    flux="hll",         # or "rusanov"
    cooling_ramp_steps=5_000,
    convergence_tol=1e-10,
    verbose=True,
)
```

All knobs, with guidance on what to change and when:

### Grid
- **`N`** — resolution. Increase for better-resolved inner flow.
- **`x_min`** — push down toward the Schwarzschild radius
  (`problem.r_S / problem.r_B`) to capture the inner sonic transition.
  For stellar-interior compact-object problems we typically use `1e-5`
  to `3e-6`.
- **`x_max`** — 3 is enough to feel "ambient". Go larger only if the
  outer pressure balance matters for your problem.

### Time stepping
- **`CFL`** — default 0.4 is conservative. Can push to 0.8 with MUSCL +
  HLL; reduce to 0.2 if you see oscillations.
- **`n_steps`** — raise if you don't converge.
- **`convergence_tol`** — $10^{-10}$ is tight. Relax to $10^{-8}$ for
  quick scans.

### Spatial scheme
- **`order`** — start with `1` for debugging (robust, diffusive);
  production runs use `2`.
- **`limiter`** — minmod is more diffusive but rock-solid. "mc" is
  sharper; use if profiles are smooth.
- **`flux`** — HLL unless you have a specific reason to prefer Rusanov.

### Cooling
- **`cooling_ramp_steps`** — ramp cooling on smoothly over this many
  steps. Default 5000 works for most problems. Reduce to ~500 for very
  mild cooling; raise to ~20 000 if the flow oscillates early.
- **`strang`** — use Strang splitting. Second-order accurate in time.
  The default (Lie) is fine for steady-state targets.

### Well-balancing
- **`wb_mode`** — leave on `"full"`. Only set to `"off"` for debugging.
- **`inner_mach_threshold`** — when to switch from WB to
  free-extrapolation inner BC. Default 2 is fine.

### Stabilization (rarely needed)
- **`sponge_frac`** — outer damping. Off by default; can paper over
  genuine bugs, so use sparingly.
- **`relaxation`** — under-relaxation. Only drop below 1 if you see
  sustained oscillation.

### Diagnostics
- **`snapshot_interval`** — how often to print progress in verbose mode.
- **`verbose`** — toggle the per-step log.

---

## 7. The `Solution`

```python
sol = problem.solve(cfg)

# Profiles (cell-centered)
sol.r          # radii [cm]
sol.x          # r / r_B
sol.rho        # [g cm^-3]
sol.v          # [cm s^-1], negative = infall
sol.P          # [erg cm^-3]
sol.T          # [K]
sol.Mach       # |v| / cs

# Derived scalars
sol.Mdot          # accretion rate [g/s]
sol.mdot_ratio    # Mdot / Mdot_B
sol.L             # luminosity [erg/s]
sol.eta           # L / (Mdot_B c^2)

# Convergence info
sol.converged      # bool
sol.residuals      # per-step residual series
sol.solver_residual  # final value
```

Save and reload:

```python
sol.save("run1.npz")
sol = rb.load("run1.npz")
```

Quick-look plots (requires matplotlib):

```python
fig = sol.plot_profiles()    # T/T_inf, rho/rho_inf, Mach vs r/r_B
```

---

## 8. Verifying the steady state

```python
res = sol.check_steady_state(cooling)
print(res.mass_rms, res.momentum_rms, res.energy_rms)
```

This evaluates the time-independent Euler equations (mass flux constancy,
momentum/energy balance in integral form) on the converged profile.
Healthy runs reach RMS residuals $\lesssim 10^{-6}$ over interior cells.
See [scheme.md §10](scheme.md#10-post-processing).

You can also re-evaluate the luminosity with an alternative cooling
prescription — useful for decomposing $L$ into, e.g., bremsstrahlung vs
pair contributions:

```python
brem_only = rb.Cooling([rb.cooling.bremsstrahlung.RelativisticBremsstrahlung()])
L_brem = sol.recompute_luminosity(brem_only)
```

---

## 9. Plug-in cooling

The default cooling is bremsstrahlung + electron-positron + muon pair
annihilation. To build your own, subclass `CoolingProcess`:

```python
import numpy as np
from radbondi import CoolingProcess, Cooling
from radbondi.constants import m_p

class ConstantCooling(CoolingProcess):
    def __init__(self, Lambda0):
        self.Lambda0 = Lambda0  # [erg cm^3 s^-1]

    def emissivity(self, rho, T, ambient):
        n = rho / (ambient.mu * m_p)
        return self.Lambda0 * n**2 * np.ones_like(T)

problem = rb.BondiProblem(
    M_BH=..., ambient=...,
    cooling=Cooling([ConstantCooling(1e-23)]),
)
```

Pass a list of instances to combine arbitrarily many processes — their
emissivities sum. The solver only sees the **net** emissivity clipped at
the ambient value (see [physics.md §3](physics.md#3-cooling-microphysics)),
so constant returns in quiescent ambient gas do not bleed energy.

Special constructors:

```python
rb.Cooling.default()      # brem + e+e- pairs + mu+mu- pairs
rb.Cooling.adiabatic()    # no cooling (empty list)
```

---

## 10. Feedback: self-consistent iteration

Radiative feedback is **off by default**. The core solver takes a fixed
ambient. A feedback model takes the BH luminosity and returns a modified
effective ambient temperature $T_{\infty}'$; the user iterates
externally.

### Pure diffusion

```python
from radbondi.feedback import DiffusionFeedback

# kappa: opacity at the photon coupling radius [cm^2/g].
# Electron scattering: kappa_es = 0.2*(1 + X); for solar-core X=0.34 this is ~0.27.
feedback = DiffusionFeedback(ambient=ambient, kappa=0.27)

T_eff = ambient.T
for i in range(10):
    prob_i = rb.BondiProblem(M_BH, ambient.with_temperature(T_eff), cooling)
    sol_i = prob_i.solve(cfg)
    result = feedback.feedback_temperature(sol_i.L)
    if abs(result.T_eff / T_eff - 1) < 1e-3:
        break
    T_eff = result.T_eff
```

Valid when the dimensionless parameter $\beta \lesssim 1$ (returned in
`result.beta`). For $\beta \gg 1$ use the MLT envelope instead.

### MLT envelope

```python
from radbondi.feedback import MLTEnvelope

env = MLTEnvelope(
    ambient=ambient, M_BH=M_BH,
    kappa_env=None,   # default: electron scattering 0.2*(1+X) [cm^2/g]
    kappa_BH=0.4,     # coupling opacity for BH spectrum [cm^2/g]
    alpha_mlt=1.5,
)

T_eff = env.feedback_temperature(L_BH)
# Or for the full structure:
profile = env.integrate(L_BH)   # returns an EnvelopeProfile
```

MLT saturates the temperature enhancement once convection takes over,
keeping $\eta$ finite even for strongly radiating BHs. See
[physics.md §5.2](physics.md#52-mlt-envelope).

---

## 11. Recipes

### Solve adiabatic Bondi (no cooling)

```python
problem = rb.BondiProblem(M_BH, ambient, cooling=rb.Cooling.adiabatic())
sol = problem.solve()
# sol should reproduce the classical Bondi profile to discretization error
```

### Quick low-resolution scan

```python
cfg = rb.SolverConfig(
    N=200, x_min=1e-4, n_steps=10_000,
    cooling_ramp_steps=1_000,
    convergence_tol=1e-7,
    order=1, snapshot_interval=2_000,
)
```

### High-fidelity run

```python
cfg = rb.SolverConfig(
    N=1600, x_min=3e-6, n_steps=200_000,
    cooling_ramp_steps=20_000,
    convergence_tol=1e-10,
    order=2, limiter="mc", flux="hll",
    CFL=0.5,
)
```

### Sweep over BH mass (paper Table 1)

A runnable sweep that reproduces the paper's three canonical regimes —
collisionless, bremsstrahlung-dominated, and near-isothermal — lives in
[`examples/02_paper_sweep.py`](../examples/02_paper_sweep.py). It writes
each `Solution` to `examples/paper_sweep_output/` and produces a summary
figure with `eta(M_BH)` and `Mdot/Mdot_B(M_BH)`.

```bash
# Fast demo: N=400, 1st-order HLL, ~few minutes.
python examples/02_paper_sweep.py

# Paper config: N=1200, adaptive-order (MUSCL where stable, 1st-order
# where it isn't — see the script docstring). ~15-45 min.
RADBONDI_HI_RES=1 python examples/02_paper_sweep.py
```

Minimal inline version if you want to customize:

```python
for logM in [-16, -15, -14, -13.5, -13, -12, -11]:
    problem = rb.BondiProblem(10.**logM * rb.M_sun, ambient, cooling)
    sol = problem.solve(cfg)
    sol.save(f"mbh_logM{logM:+.1f}.npz")
    print(f"log M = {logM:+.1f}  eta = {sol.eta:.3e}  Mdot/Mdot_B = {sol.mdot_ratio:.2f}")
```

---

## 12. Troubleshooting

**Doesn't converge.** First check `sol.residuals` — is it decreasing,
stalling, or oscillating?
- *Decreasing*: raise `n_steps`.
- *Stalling*: tighten `CFL`, check `x_min` isn't inside $r_{S}$.
- *Oscillating*: reduce `CFL` to 0.2, set `relaxation=0.5`, or increase
  `cooling_ramp_steps`.

**Blows up early.** Usually a cooling-induced cliff at startup. Raise
`cooling_ramp_steps`, drop `CFL`, switch to `order=1` for diagnosis.

**Inner BC misbehaving.** Check the verbose output — the `BC=`
indicator shows whether the solver is using the WB or free-extrap
inner BC. For deep sonic transitions you want `BC=free`; if it's
stuck on `wb` try dropping `inner_mach_threshold` to 1.5.

**Steady-state residuals large.** If `check_steady_state` returns
RMS $\gg 10^{-4}$, the flow isn't actually at steady state — raise
`n_steps` or `convergence_tol` and try again.

**Pathological gradients near $r_{S}$.** Increase `N`, or push `x_min`
down (but never below $r_{S}/r_{B}$).

---

## 13. What's next

- `examples/01_quickstart.py` — single end-to-end run.
- `examples/02_paper_sweep.py` — reproduce the paper's Table 1 mass
  sweep and generate the `eta(M_BH)` figure.
- `tests/` — short, focused tests that double as usage examples
  (`test_solver.py` is the most illustrative).
- [physics.md](physics.md) — equations behind all of the above.
- [scheme.md](scheme.md) — how the solver works internally.
