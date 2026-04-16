# radbondi

Time-dependent spherical Bondi accretion with radiative cooling.

`radbondi` solves the 1D spherical Euler equations with an implicit cooling
source term, evolving the flow from an initial adiabatic Bondi profile to
a self-consistent steady state. It is designed for problems where the standard
ODE shooting method fails — e.g., when cooling is strong enough that the sonic
point becomes a focus (complex eigenvalues) rather than a saddle.

The default microphysics (relativistic bremsstrahlung + e⁺e⁻ pair annihilation)
targets accretion onto compact objects in hot, dense environments such as
stellar interiors, but the cooling module is plug-in: users can add their own
processes by subclassing `CoolingProcess`.

## Status

**Alpha.** API may change. See `CHANGELOG.md` for releases.

## Installation

```bash
pip install -e ".[dev]"   # development install
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

## Citation

If you use `radbondi` in published work, please cite:

> Cantiello et al. (in prep.)

A `CITATION.cff` will be added on first tagged release.

## License

MIT — see [LICENSE](LICENSE).
