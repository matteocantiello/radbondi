# radbondi

[![tests](https://github.com/matteocantiello/radbondi/actions/workflows/test.yml/badge.svg)](https://github.com/matteocantiello/radbondi/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/radbondi.svg)](https://pypi.org/project/radbondi/)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

Time-dependent spherical Bondi accretion with radiative cooling.

`radbondi` solves the 1D spherical Euler equations with an implicit cooling
source term, evolving the flow from an initial adiabatic Bondi profile to
a self-consistent steady state. It is designed for problems where the standard
ODE shooting method fails — e.g., when cooling is strong enough that the sonic
point becomes a focus (complex eigenvalues) rather than a saddle.

The default microphysics (relativistic bremsstrahlung + e⁺e⁻ and μ⁺μ⁻
pair annihilation) targets accretion onto compact objects in hot, dense
environments such as stellar interiors, but the cooling module is plug-in:
users can add their own processes by subclassing `CoolingProcess`.

## Status

**Alpha.** API may change. See `CHANGELOG.md` for releases.

## Installation

```bash
pip install radbondi              # from PyPI
pip install radbondi[plot]        # + matplotlib for sol.plot_profiles()
pip install -e ".[dev]"           # development install (editable + test deps)
```

## Quickstart

```python
import radbondi as rb

ambient = rb.presets.solar_core()
problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=ambient)
sol = problem.solve(rb.SolverConfig(N=800, x_min=3e-6))

print(f"eta = {sol.eta:.3e}")
sol.plot_profiles()
```

## Documentation

See the [`docs/`](docs/README.md) directory:

- [**usage.md**](docs/usage.md) — installation, quickstart, API, and all
  `SolverConfig` knobs.
- [**physics.md**](docs/physics.md) — equations, Bondi solution,
  microphysics, and feedback models.
- [**scheme.md**](docs/scheme.md) — finite-volume discretization,
  well-balancing, and implicit cooling.

## Citation

If you use `radbondi` in published work, please cite both the software and
the paper describing the underlying physics. See [`CITATION.cff`](CITATION.cff)
for machine-readable metadata; GitHub renders a "Cite this repository" button
on the project page.

> Cantiello et al. (in prep.)

## License

MIT — see [LICENSE](LICENSE).
