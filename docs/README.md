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

A runnable end-to-end example is in [`examples/01_quickstart.py`](../examples/01_quickstart.py);
a multi-mass sweep in [`examples/02_paper_sweep.py`](../examples/02_paper_sweep.py).
