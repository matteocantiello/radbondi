# Documentation

- **[usage.md](usage.md)** — installation, quickstart, API walkthrough,
  `SolverConfig` knobs, plug-in cooling, feedback, troubleshooting.
- **[physics.md](physics.md)** — equations solved (Euler + cooling),
  Bondi initial condition, microphysics (bremsstrahlung + pair
  annihilation), feedback models (diffusion, MLT envelope).
- **[scheme.md](scheme.md)** — finite-volume discretization, MUSCL
  reconstruction, HLL/Rusanov Riemann solvers, well-balancing, local
  time stepping, operator-split implicit cooling.
- **[paper_reproduction.md](paper_reproduction.md)** — handoff /
  work-log for reproducing Cantiello et al. Table 1 on a fast machine:
  exact per-mass `(order, N, x_min)` config, verification recipe,
  known gotchas.
- **[custom_cooling.md](custom_cooling.md)** — recipe for adding your own
  cooling process by subclassing `CoolingProcess`, with a worked
  inverse-Compton example.

Runnable examples:

- [`examples/01_quickstart.py`](../examples/01_quickstart.py) — solve one
  mass, inspect the result.
- [`examples/02_paper_sweep.py`](../examples/02_paper_sweep.py) — paper
  Table 1 mass sweep.
- [`examples/03_feedback.py`](../examples/03_feedback.py) — self-consistent
  iteration with the MLT envelope feedback.
- [`examples/04_custom_cooling.py`](../examples/04_custom_cooling.py) —
  inverse-Compton plug-in cooling.
