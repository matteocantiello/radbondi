# Adding a custom cooling process

`radbondi` ships with relativistic bremsstrahlung and thermal pair
annihilation. Any other emission or absorption process you care about — line
cooling, inverse Compton, Comptonization off an external radiation field,
neutrino losses, dust thermal emission, photoionization heating (sign-flipped
into the same framework), … — can be plugged in by subclassing
[`CoolingProcess`](../src/radbondi/cooling/base.py).

The runnable companion to this guide is
[`examples/04_custom_cooling.py`](../examples/04_custom_cooling.py): an
inverse-Compton cooling term added to the default mix.

## The interface

```python
from radbondi.cooling import CoolingProcess

class MyProcess(CoolingProcess):
    def emissivity(self, rho, T, ambient):
        """Volumetric emissivity in erg cm^-3 s^-1.

        rho, T : ndarray of float, same shape (cell-center values)
        ambient: AmbientMedium (provides mu, gamma, X, Y, T, rho)

        Return: ndarray of the same shape as rho/T.
        """
        ...
```

That is the **only** method you have to implement. Three rules:

1. **Vectorize.** `rho` and `T` arrive as numpy arrays of identical shape and
   you must return an array of the same shape. The solver will call your
   process on every grid cell at every step, so use numpy broadcasting; a
   Python `for` loop will be slow.
2. **Return non-negative emissivity.** A "cooling" process removes energy
   from the gas. If you want to model heating, that is fine but you should
   subclass with a *negative* emissivity (the solver subtracts an ambient
   floor, see below) — or model it inside your process by returning the
   net signed value.
3. **Don't reach for module globals.** Take everything you need either from
   `ambient` (composition, μ, γ, T, ρ) or from instance attributes set in
   `__init__`. This is what keeps `radbondi` ambient-agnostic.

## Combining with the defaults

A `Cooling` instance is just a list of `CoolingProcess` instances whose
emissivities sum. Build it explicitly:

```python
from radbondi import Cooling
from radbondi.cooling import RelativisticBremsstrahlung, PairAnnihilation

cooling = Cooling([
    RelativisticBremsstrahlung(),
    PairAnnihilation(species="electron"),
    MyProcess(...),                        # your new one
])
```

or extend the defaults:

```python
cooling = Cooling.default()
cooling.processes.append(MyProcess(...))
```

The empty list (`Cooling.adiabatic()`) gives a purely adiabatic flow.

## Worked example: inverse Compton off a uniform radiation field

Inverse-Compton (IC) cooling off an external radiation bath of energy
density `U_rad` removes energy from the thermal electron population at the
rate

  ε_IC = (4 σ_T / m_e c) U_rad n_e k_B T   (T ≪ m_e c² / k_B)

The factor `(T - T_rad)` rather than just `T` would model the equilibrium at
the radiation temperature; we use the simpler form above for clarity.

```python
import numpy as np
from radbondi.cooling import CoolingProcess
from radbondi.constants import c_light, kB, m_e, m_p, sigma_T


class InverseComptonUniform(CoolingProcess):
    """Inverse-Compton cooling off a uniform-T_rad radiation bath.

    Parameters
    ----------
    U_rad : float
        Radiation energy density [erg cm^-3].
    """

    def __init__(self, U_rad):
        self.U_rad = float(U_rad)

    def emissivity(self, rho, T, ambient):
        n_e = (ambient.X + 0.5 * ambient.Y) * rho / m_p
        prefactor = 4.0 * sigma_T / (m_e * c_light)
        return prefactor * self.U_rad * n_e * kB * T
```

Whether this term *matters* depends on the problem regime. At solar-core
`M_BH = 1e-13 M_sun`, bremsstrahlung dominates by tens of orders of
magnitude in the inner Bondi sphere because the density compression near
`r_S` puts `rho^2` in the denominator of free-free. Even
`U_rad = 1e6 erg/cm^3` is invisible there. The example script shows you
how to diagnose this — and the API works the same way for *any* new
process whose physical importance varies across regimes.

Use it like any other process:

```python
import radbondi as rb
from radbondi import Cooling
from radbondi.cooling import RelativisticBremsstrahlung, PairAnnihilation

ambient = rb.presets.solar_core()
cooling = Cooling([
    RelativisticBremsstrahlung(),
    PairAnnihilation(species="electron"),
    InverseComptonUniform(U_rad=1e6),   # erg cm^-3 (illustrative)
])

problem = rb.BondiProblem(
    M_BH=1e-13 * rb.M_sun, ambient=ambient, cooling=cooling
)
sol = problem.solve(rb.SolverConfig(N=400, x_min=1e-5, n_steps=30_000))
print(f"eta = {sol.eta:.3e}")
```

## How `Cooling` uses your `emissivity`

There are two methods on `Cooling` that call `emissivity`:

- **`total_emissivity(rho, T, ambient)`** sums every process's contribution
  at the given (ρ, T). This is what you would use for a diagnostic plot of
  ε vs r.
- **`net_emissivity(rho, T, ambient, ambient_emissivity=None)`** returns
  `max(total - ambient_emissivity, 0)`. The solver uses *this*. The ambient
  floor prevents the solver from cooling gas below the ambient temperature
  — physically the surrounding medium is assumed to radiate at the same
  rate, so only the *excess* removes net energy.

The ambient floor is computed once (per solve) as
`Cooling.ambient_emissivity(ambient) = total_emissivity(ambient.rho,
ambient.T, ambient)`. This means: **if your process is non-zero in the
unperturbed ambient gas, its contribution will be subtracted out**. That is
usually what you want (e.g. bremsstrahlung at solar-core conditions is
non-zero but tiny; the subtraction removes a numerical seed). If you need
unsubtracted absolute values, call `total_emissivity` directly.

## Pitfalls

- **Numerical overflow** at the temperature/density extremes inside the
  Bondi sphere can produce `inf`/`nan` if your formula has powers of `T` or
  `rho` that grow unboundedly. The built-in `RelativisticBremsstrahlung`
  protects itself with `T_max = 1e12` and `rho_max = 1e15` clips. If your
  process can blow up, do the same.
- **Negative emissivity** (i.e., you wrote a heating term but the sign
  convention is energy *loss* per unit volume per unit time) will be
  silently ignored by `net_emissivity` because of the `max(., 0)` floor.
  If you want heating, model it inside the relevant process and ensure
  your `emissivity` returns the *signed* rate; or use a small positive
  baseline trick.
- **Composition assumptions.** If your process depends on metals, the
  current `AmbientMedium` exposes only `X` (H), `Y` (He), and the mean
  molecular weight `mu`. There is no `Z` field today; pass any extra
  composition data through your `__init__`.
- **Units.** Everything in `radbondi` is CGS. ρ in g cm⁻³, T in K, ε in
  erg cm⁻³ s⁻¹. The constants you need live in
  [`radbondi.constants`](../src/radbondi/constants.py).
