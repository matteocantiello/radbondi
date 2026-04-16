# Changelog

All notable changes to `radbondi` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
- Phase 4: user documentation.
  - `docs/physics.md` — governing equations, Bondi IC, cooling microphysics,
    and feedback models.
  - `docs/scheme.md` — finite-volume discretization, MUSCL, HLL/Rusanov,
    well-balancing, local time stepping, operator-split implicit cooling.
  - `docs/usage.md` — install, API walkthrough, full `SolverConfig`
    reference, plug-in cooling, feedback recipes, troubleshooting.
  - `docs/README.md` index and top-level README pointers.
