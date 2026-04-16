# Documentation

- **[usage.md](usage.md)** — installation, quickstart, API walkthrough,
  `SolverConfig` knobs, plug-in cooling, feedback, troubleshooting.
- **[physics.md](physics.md)** — equations solved (Euler + cooling),
  Bondi initial condition, microphysics (bremsstrahlung + pair
  annihilation), feedback models (diffusion, MLT envelope).
- **[scheme.md](scheme.md)** — finite-volume discretization, MUSCL
  reconstruction, HLL/Rusanov Riemann solvers, well-balancing, local
  time stepping, operator-split implicit cooling.

A runnable end-to-end example is in [`examples/01_quickstart.py`](../examples/01_quickstart.py).
