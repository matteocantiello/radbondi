# Architecture

A guided tour of `radbondi`'s internals: how the package is organized,
how data flows through a solve, and where each piece of physics lives.

---

## Package layout

```
src/radbondi/
│
│  Constants & ambient medium
│  ─────────────────────────────────────────────────────────────
├── constants.py ············ CGS physical constants
│                             G  kB  c  m_e  m_p  M_sun  ...
│
├── ambient.py ·············· AmbientMedium (frozen dataclass)
│                             T  rho  mu  gamma  X  Y
│                             Properties:  cs   n_e   n_i
│                             Copies:      with_temperature()
│                                          with_density()
│
├── presets.py ·············· Ready-made ambients
│                             solar_core()  primordial_gas()
│
│  Adiabatic Bondi solution (initial condition)
│  ─────────────────────────────────────────────────────────────
├── bondi.py ················ lambda_bondi(gamma)
│                             bondi_radius()   schwarzschild_radius()
│                             bondi_rate()     adiabatic_profile()
│
│  Finite-volume grid
│  ─────────────────────────────────────────────────────────────
├── grid.py ················· Grid.log_spaced(r_B, N, x_min, x_max)
│                             Faces  r_{i+1/2}   (log-spaced)
│                             Centers  r_i        (geometric mean)
│                             Widths  dr_i    Volumes  V_i    Areas  A_{i+1/2}
│
│  Hydrodynamics (stateless, pure functions)
│  ─────────────────────────────────────────────────────────────
├── hydro.py ················ get_primitives()   rho, v, P, T, cs  from  U
│                             minmod()  mc_limiter()       slope limiters
│                             compute_fluxes()             MUSCL + HLL/Rusanov
│                             hydro_rhs()                  dU/dt (fluxes + geometry + gravity)
│
│  Cooling microphysics (plug-in)
│  ─────────────────────────────────────────────────────────────
├── cooling/
│   ├── base.py ············· CoolingProcess     abstract base class
│   │                         Cooling            collection; sums emissivities
│   │                           .default()       bremsstrahlung + e+e- + mu+mu-
│   │                           .adiabatic()     no cooling
│   │                           .net_emissivity()  excess above ambient floor
│   │
│   ├── bremsstrahlung.py ··· RelativisticBremsstrahlung
│   │                         Stepney & Guilbert 1983 (e-i + e-e)
│   │
│   └── pair_annihilation.py  PairAnnihilation
│                             Svensson 1982 / Stepney 1983
│                             species = "electron" | "muon"
│
│  Solver (time integration to steady state)
│  ─────────────────────────────────────────────────────────────
├── solver.py ··············· SolverConfig       all numerical knobs
│                             BondiProblem       ties M_BH + ambient + cooling
│                               .solve()         time-dependent PDE solver
│                               .solve_with_feedback()   iterative feedback loop
│
├── ode.py ·················· ODESolverConfig    (alternative ODE shooter)
│                             solve_ode()        three-zone shooting method
│
│  Output & diagnostics
│  ─────────────────────────────────────────────────────────────
├── solution.py ············· Solution           profiles + derived scalars
│                               .eta  .Mdot  .mdot_ratio  .L
│                               .save() / load()
│                               .plot_profiles()
│                               .recompute_luminosity()
│
├── diagnostics.py ·········· check_steady_state()
│                             Integral-form Euler residuals
│                             (mass flux, momentum, energy)
│
│  Feedback models (optional, external to the PDE)
│  ─────────────────────────────────────────────────────────────
└── feedback/
    ├── diffusion.py ········ DiffusionFeedback
    │                         Algebraic:  x^4 = 1 + beta * x^{-3/2}
    │
    └── mlt.py ·············· MLTEnvelope
                              1-D hydrostatic integration with
                              mixing-length theory for convection
```

---

## Import graph

Arrows show compile-time imports within the package. The layering is
strict: low-level modules (`constants`, `hydro`, `cooling`) never import
the solver or solution.

```
                        ┌──────────┐
                        │ __init__ │  (public API surface)
                        └────┬─────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                  ▼
      ┌─────────┐     ┌──────────┐       ┌──────────┐
      │ presets  │     │  solver  │       │   ode    │
      └────┬────┘     └──┬───┬──┘       └────┬─────┘
           │              │   │               │
           ▼              │   │               ▼
      ┌─────────┐         │   │          ┌─────────┐
      │ ambient │ ◄───────┘   │          │  bondi  │
      └────┬────┘             │          └────┬────┘
           │                  │               │
           │         ┌────────┼───────┐       │
           │         ▼        ▼       ▼       │
           │    ┌──────┐  ┌──────┐ ┌──────────┤
           │    │ grid │  │hydro │ │ solution │
           │    └──────┘  └──┬───┘ └──────────┘
           │                 │
           │                 ▼
           │           ┌───────────┐
           │           │ constants │
           │           └─────┬─────┘
           │                 │
           │         ┌───────┴───────┐
           ▼         ▼               ▼
      ┌──────────────────┐    ┌──────────┐
      │  cooling/base    │    │ feedback │
      │  bremsstrahlung  │    │ diffusion│
      │  pair_annihil.   │    │ mlt      │
      └──────────────────┘    └──────────┘
```

---

## Solve flow

What happens inside `problem.solve(config)`, step by step.

```
                         User code
                             │
                             ▼
               ┌─────────────────────────────┐
               │  BondiProblem.solve(config)  │
               └──────────────┬──────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                    ▼
   ┌─────────────┐   ┌───────────────┐   ┌───────────────────┐
   │  Build grid │   │ Bondi initial │   │ Pre-compute       │
   │  log-spaced │   │ condition U₀  │   │ ambient cooling   │
   │  (grid.py)  │   │ (bondi.py)    │   │ floor ε_ambient   │
   └──────┬──────┘   └───────┬───────┘   └────────┬──────────┘
          │                  │                     │
          │    Store U_eq = U₀ and R_eq = RHS(U₀)  │
          │    (for well-balanced correction)       │
          │                  │                     │
          └──────────────────┼─────────────────────┘
                             │
                             ▼
          ╔══════════════════════════════════════════╗
          ║        TIME-STEPPING  LOOP               ║
          ║     step = 1, 2, ..., n_steps            ║
          ╠══════════════════════════════════════════╣
          ║                                          ║
          ║  ┌─── 1. Hydro RHS ───────────────────┐  ║
          ║  │                                     │  ║
          ║  │  get_primitives(U)                  │  ║
          ║  │     U = [ρ, ρv, E]  →  ρ, v, P, T  │  ║
          ║  │                                     │  ║
          ║  │  MUSCL reconstruction (order=2)     │  ║
          ║  │     slopes on (W − W_eq) in         │  ║
          ║  │     primitive space (ρ, v, P)        │  ║
          ║  │     limiter: minmod or MC            │  ║
          ║  │                                     │  ║
          ║  │  HLL / Rusanov fluxes               │  ║
          ║  │     at each cell interface           │  ║
          ║  │     (WB-modified dissipation)        │  ║
          ║  │                                     │  ║
          ║  │  dU/dt = − (A·F)/V                  │  ║
          ║  │        + P·(A_R − A_L)/V   geometry  │  ║
          ║  │        − ρ GM/r²           gravity   │  ║
          ║  └─────────────────────────────────────┘  ║
          ║                    │                      ║
          ║                    ▼                      ║
          ║  ┌─── 2. Well-balanced correction ─────┐  ║
          ║  │                                     │  ║
          ║  │  dU  -=  w_i · R_eq                 │  ║
          ║  │                                     │  ║
          ║  │  w_i ≈ 1  near equilibrium          │  ║
          ║  │  w_i → 0  where cooling changed U   │  ║
          ║  └─────────────────────────────────────┘  ║
          ║                    │                      ║
          ║                    ▼                      ║
          ║  ┌─── 3. Forward Euler (local Δt) ─────┐  ║
          ║  │                                     │  ║
          ║  │  Δt_i = CFL · Δr_i / (|v_i| + cs)  │  ║
          ║  │  U_new = U + Δt_i · dU/dt           │  ║
          ║  │                                     │  ║
          ║  │  (each cell at its own pace —        │  ║
          ║  │   10³–10⁴× faster than global Δt)   │  ║
          ║  └─────────────────────────────────────┘  ║
          ║                    │                      ║
          ║                    ▼                      ║
          ║  ┌─── 4. Implicit cooling ─────────────┐  ║
          ║  │                                     │  ║
          ║  │  Operator-split (Lie or Strang).     │  ║
          ║  │  Hold ρ and v fixed; update T by     │  ║
          ║  │  Newton iteration:                   │  ║
          ║  │                                     │  ║
          ║  │    c_th · T − e_th + Δt · ε_net = 0 │  ║
          ║  │                                     │  ║
          ║  │  ε_net = Σ (cooling processes)       │  ║
          ║  │        − ε_ambient    (floor)        │  ║
          ║  │                                     │  ║
          ║  │  Ramp: ε scaled 0→1 over first      │  ║
          ║  │        cooling_ramp_steps             │  ║
          ║  └─────────────────────────────────────┘  ║
          ║                    │                      ║
          ║                    ▼                      ║
          ║  ┌─── 5. Boundary conditions ──────────┐  ║
          ║  │                                     │  ║
          ║  │  Inner (i=0):                        │  ║
          ║  │    Mach > 2  →  free extrapolation   │  ║
          ║  │    Mach < 2  →  WB extrapolation     │  ║
          ║  │                                     │  ║
          ║  │  Outer (i=N-1, N-2):                 │  ║
          ║  │    held to initial ambient values     │  ║
          ║  └─────────────────────────────────────┘  ║
          ║                    │                      ║
          ║                    ▼                      ║
          ║  ┌─── 6. Floors + convergence check ───┐  ║
          ║  │                                     │  ║
          ║  │  ρ ≥ 1e-30     T ≥ T_∞/2            │  ║
          ║  │                                     │  ║
          ║  │  residual = RMS( ΔU / |U| )         │  ║
          ║  │  if residual < tol:  exit loop       │  ║
          ║  └─────────────────────────────────────┘  ║
          ║                                          ║
          ╚══════════════════════════════════════════╝
                             │
                             ▼
          ┌──────────────────────────────────────────┐
          │  Post-process                             │
          │                                          │
          │  Final primitives:  ρ, v, P, T, Mach     │
          │  Luminosity:  L = 4π ∫ ε_net r² dr       │
          │                                          │
          │  Pack into Solution(...)                  │
          └──────────────────┬───────────────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │   Solution   │
                     │              │
                     │  .eta        │  η = L / (Ṁ_B c²)
                     │  .mdot_ratio │  Ṁ / Ṁ_B
                     │  .L          │  luminosity [erg/s]
                     │  .Mdot       │  accretion rate [g/s]
                     │              │
                     │  .save()     │  → .npz file
                     │  .plot_profiles()
                     │  .check_steady_state()
                     └──────────────┘
```

---

## Feedback loop (optional)

Feedback is external to the PDE solver. It iterates `solve()` with a
modified ambient temperature until the luminosity is self-consistent.

```
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   T_eff = T_core                                            │
    │                                                             │
    │   for iteration in range(max_iter):                         │
    │       │                                                     │
    │       ▼                                                     │
    │   ┌───────────────────────────────────────────┐             │
    │   │  problem = BondiProblem(M, ambient_i, cool) │             │
    │   │  sol = problem.solve(config)              │             │
    │   └───────────────────┬───────────────────────┘             │
    │                       │                                     │
    │                       ▼                                     │
    │   ┌───────────────────────────────────────────┐             │
    │   │  Feedback model:                          │             │
    │   │                                           │             │
    │   │  DiffusionFeedback   (algebraic, fast)    │             │
    │   │    x⁴ = 1 + β x⁻³/²                      │             │
    │   │                                           │             │
    │   │         ── or ──                           │             │
    │   │                                           │             │
    │   │  MLTEnvelope   (1-D integration)          │             │
    │   │    hydrostatic + mixing-length theory      │             │
    │   │    accounts for convective transport       │             │
    │   │                                           │             │
    │   │        T_eff  ←  feedback(sol.L)           │             │
    │   └───────────────────┬───────────────────────┘             │
    │                       │                                     │
    │                       ▼                                     │
    │   ambient_i = ambient.with_temperature(T_eff)               │
    │                                                             │
    │   if |T_eff − T_old| / T_old < tol:  ──── converged ────▶  │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    Also available as:  problem.solve_with_feedback(feedback, config)
```

---

## Cooling plug-in architecture

New cooling processes slot in without touching the solver.

```
                    CoolingProcess (ABC)
                    │
                    │  def emissivity(rho, T, ambient)
                    │      → erg cm⁻³ s⁻¹
                    │
         ┌──────────┼──────────────┐
         ▼          ▼              ▼
  Relativistic   Pair           Your custom
  Bremsstrahlung Annihilation   process
  (e-i + e-e)   (e⁺e⁻, μ⁺μ⁻)
         │          │              │
         └──────────┼──────────────┘
                    ▼
              Cooling([...])
              │
              │  .total_emissivity()    sum of all processes
              │  .net_emissivity()      excess above ambient floor
              │  .ambient_emissivity()  the floor itself
              │
              │  .default()      →  brem + e⁺e⁻ + μ⁺μ⁻
              │  .adiabatic()    →  empty list (no cooling)
              │
              └──── passed to BondiProblem + used in implicit step
```

---

## Data lifecycle

```
  ┌──────────┐     ┌───────────┐     ┌─────────┐
  │ Ambient  │     │   M_BH    │     │ Cooling │
  │ Medium   │     │  [g]      │     │ (plug-in)│
  └────┬─────┘     └─────┬─────┘     └────┬────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ BondiProblem │       r_B, r_S, Ṁ_B
                  └──────┬───────┘
                         │
                   .solve(config)
                         │
                         ▼
                  ┌──────────────┐
                  │   Solution   │ ── .save("run.npz")  ──▶  disk
                  └──────┬───────┘
                         │                                    │
            ┌────────────┼────────────┐              rb.load("run.npz")
            ▼            ▼            ▼                       │
       .eta  .L     .plot_profiles()  .check_steady_state()   ▼
       .Mdot        (matplotlib)      (diagnostics.py)     Solution
       .mdot_ratio
```
