# Changelog

All notable changes to `radbondi` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `radbondi.ode.solve_ode` and `ODESolverConfig`: three-zone ODE shooting
  solver (Born iteration + ballistic inner zone), faithfully ported from
  the original `sonic_solver.py`. Reproduces the paper's reference value
  `eta = 2.04e-2` at `M_BH = 1e-16 M_sun` exactly. Returns a `Solution`
  API-compatible with `BondiProblem.solve`.
- `tests/test_ode.py`: ODE solver smoke tests (3 fast + 1 slow ODE-vs-
  time-dependent cross-validation in the collisionless regime).
- `examples/03_feedback.py`: end-to-end recipe for the self-consistent
  feedback iteration loop (solve → compute T_eff → re-solve), using the
  MLT envelope model.

### Changed
- README now shows CI status, license, and Python version badges.
- `_initialize_from_bondi` (via `bondi.adiabatic_profile`): replaced the
  per-cell `brentq` Python loop with vectorized bisection. ~17x speedup
  at N=6400 (80 ms → 5 ms); identical results to within machine precision.
- CI: enforce minimum 85% coverage with `--cov-fail-under=85`
  (current: 88%).

### Removed
- `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` from the `[docs]`
  optional-dependency group. Documentation stays as flat markdown under
  `docs/`.

## [0.1.0] - 2026-04-16

First public release. The package reproduces the Cantiello et al. (in prep.)
paper Table 1 results within ~30 % at fast-test resolution and within a few
percent at the paper's production resolution. Steady-state mass-flux
conservation verified to better than 1 % across all four accretion regimes.

### Added
- `CITATION.cff` for machine-readable citation metadata (rendered as
  "Cite this repository" on GitHub).
- Initial package scaffold (Phase 1).
- Phase 2: working public API.
  - `AmbientMedium` (CGS, immutable, with `with_temperature` / `with_density`).
  - Plug-in cooling: `Cooling`, `CoolingProcess` ABC, `RelativisticBremsstrahlung`,
    `PairAnnihilation` (electron and muon species).
  - `Cooling.default()` and `Cooling.adiabatic()` constructors.
  - `Grid.log_spaced` finite-volume grid in r/r_B.
  - `BondiProblem` + `SolverConfig`: time-dependent Euler solver with HLL/Rusanov,
    MUSCL reconstruction, well-balanced scheme, local time stepping, implicit
    operator-split cooling.
  - `Solution` dataclass with `eta`, `mdot_ratio`, `luminosity`, `plot_profiles`,
    `check_steady_state`, plus `save`/`load`.
  - `radbondi.diagnostics.check_steady_state` (integral form).
  - Optional feedback: `feedback.DiffusionFeedback` and `feedback.MLTEnvelope`
    (off by default — must be applied explicitly via `ambient.with_temperature`).
  - Solar-core preset.
  - 19 smoke tests covering imports, ambient, cooling, solver, feedback,
    save/load round-trip, and steady-state diagnostics.
- Phase 3: physics validation tests.
  - `tests/test_bondi.py` — adiabatic Bondi solution: eigenvalue
    $\lambda(\gamma)$ at 5/3 and 4/3, $r_B$ and $\dot M_B$ scalings,
    interior power laws ($\rho\propto r^{-3/2}$, $T\propto r^{-1}$,
    $v\propto r^{-1/2}$).
  - `tests/test_microphysics.py` — bremsstrahlung limits (non-relativistic
    $T^{1/2}$ scaling, $\rho^2$, ultra-relativistic $T\log T$) and pair
    annihilation (sharp turn-on near $\theta_e\sim 1$, muon channel
    negligible at electron threshold), net-emissivity floor.
  - `tests/test_validation.py` — end-to-end paper Table 1 reproduction
    (parametrized across three BH masses), resolution convergence, tight
    steady-state residuals, save/load preserves $\eta$. Gated behind a
    `slow` pytest marker; run with `pytest -m slow`.
  - `tests/test_feedback.py` — MLT saturates below diffusion at $\beta\gg 1$.
  - `tests/conftest.py` and `slow` marker in `pyproject.toml`; default
    pytest run skips slow tests.
- Phase 4: user documentation.
  - `docs/physics.md` — governing equations, Bondi IC, cooling microphysics,
    and feedback models.
  - `docs/scheme.md` — finite-volume discretization, MUSCL, HLL/Rusanov,
    well-balancing, local time stepping, operator-split implicit cooling.
  - `docs/usage.md` — install, API walkthrough, full `SolverConfig`
    reference, plug-in cooling, feedback recipes, troubleshooting.
  - `docs/README.md` index and top-level README pointers.
