# Physics

`radbondi` solves the time-dependent spherically symmetric Euler equations
around a point-mass gravitational attractor, with optional volumetric
radiative cooling and optional radiative feedback on the ambient medium.
This document lists the equations as the code actually implements them and
points to the source files where each piece lives.

All quantities are in CGS. The fluid is an ideal gas with adiabatic index
$\gamma$ (default $5/3$) and constant mean molecular weight $\mu$.

---

## 1. Governing equations

The conservative state at cell centers is

$$
U = \bigl(\rho,\; \rho v,\; E\bigr),
\qquad
E = \tfrac{1}{2}\rho v^{2} + \frac{P}{\gamma - 1},
$$

with equation of state $P = \rho k_{B} T / (\mu m_{p})$. The 1D spherical
Euler equations with gravity and a volumetric sink $\varepsilon$ are

$$
\partial_{t}\rho + \frac{1}{r^{2}}\partial_{r}\!\bigl(r^{2}\rho v\bigr) = 0,
$$

$$
\partial_{t}(\rho v) + \frac{1}{r^{2}}\partial_{r}\!\bigl(r^{2}(\rho v^{2} + P)\bigr)
= \frac{2P}{r} - \rho\,\frac{GM_{\bullet}}{r^{2}},
$$

$$
\partial_{t}E + \frac{1}{r^{2}}\partial_{r}\!\bigl(r^{2}(E + P)v\bigr)
= -\rho v\,\frac{GM_{\bullet}}{r^{2}} - \varepsilon(\rho, T).
$$

The geometric pressure source $2P/r$ compensates for putting the scalar
pressure inside the spherical divergence:

$$
\frac{1}{r^{2}}\partial_{r}\!\bigl(r^{2}P\bigr) = \partial_{r}P + \frac{2P}{r}.
$$

The physical pressure gradient in the momentum equation is $\partial_{r}P$;
writing the momentum flux in fully conservative form as
$r^{2}(\rho v^{2} + P)$ therefore introduces a spurious $2P/r$ that is
moved to the right-hand side as a source. The code uses the discretely
flux-consistent finite-volume form $P\,(A_{i+1/2} - A_{i-1/2})/V_{i}$
rather than a pointwise $2P/r$, which exactly cancels the pressure
divergence at $v \to 0$ (preserves hydrostatic equilibrium — see
[scheme.md §2](scheme.md#2-finite-volume-discretization)).
See `hydro_rhs` in `src/radbondi/hydro.py`.

Gravity is Newtonian. There is no self-gravity; the accretor mass
$M_{\bullet}$ is fixed.

---

## 2. The adiabatic Bondi problem (initial condition)

With $\varepsilon \equiv 0$ the steady-state transonic solution is the
classical Bondi (1952) accretion flow. Define

$$
c_{\infty}^{2} = \frac{\gamma k_{B} T_{\infty}}{\mu m_{p}},
\qquad
r_{B} = \frac{G M_{\bullet}}{c_{\infty}^{2}},
\qquad
\dot M_{B} = 4\pi\,\lambda(\gamma)\,\rho_{\infty}\,c_{\infty}\,r_{B}^{2}.
$$

The Bondi eigenvalue is

$$
\lambda(\gamma) = \frac{1}{4}\left(\frac{2}{5 - 3\gamma}\right)^{(5 - 3\gamma)/(2(\gamma-1))},
$$

with the regular limit $\lambda = 1/4$ at $\gamma = 5/3$.

The transonic profile is obtained by combining the isentropic law and the
Bernoulli equation, yielding

$$
x^{\,\beta}\!\left(\frac{1}{x} + \frac{1}{\gamma-1}\right)
= \Theta\,
M^{\alpha}\!\left(\frac{M^{2}}{2} + \frac{1}{\gamma-1}\right),
$$

with $x = r/r_{B}$, $M$ the Mach number,
$\alpha = 2(1-\gamma)/(\gamma+1)$,
$\beta = 4(\gamma-1)/(\gamma+1)$, and $\Theta = \lambda^{\alpha}$. The code
solves this for the subsonic branch $M \in (0, 1)$ at each $x$ with
Brent's method (`adiabatic_profile` in `src/radbondi/bondi.py`). Density
and sound speed then follow from isentropic scaling:

$$
\frac{\rho}{\rho_{\infty}} = \left(\frac{\lambda}{x^{2} M}\right)^{2/(\gamma+1)},
\qquad
\frac{c_{s}}{c_{\infty}} = \left(\frac{\rho}{\rho_{\infty}}\right)^{(\gamma-1)/2}.
$$

This profile is used as the **initial condition** for the time-dependent
solver and as the **equilibrium reference** for the well-balanced scheme
(see [scheme.md](scheme.md)).

**Why time-dependent at all?** When cooling is strong, the inflow no longer
has a regular sonic point: linearizing the steady-state ODE at the critical
point yields complex eigenvalues (a focus, not a saddle), so shooting
methods fail. Evolving the PDEs to steady state sidesteps the ODE
singularity — the sonic point is resolved naturally by the Riemann solver.

---

## 3. Cooling microphysics

Cooling is a plug-in: `Cooling` holds a list of `CoolingProcess`
subclasses, and the total volumetric emissivity is the sum. Only the
*excess* emissivity over the ambient value acts on the flow:

$$
\varepsilon_{\rm net}(\rho, T) = \max\!\bigl[\varepsilon_{\rm tot}(\rho, T) - \varepsilon_{\rm tot}(\rho_{\infty}, T_{\infty}),\; 0\bigr].
$$

The ambient floor expresses the assumption that the surrounding medium
radiates at the same rate; only the extra emission from the shocked /
compressed inflow carries net energy away. See
`Cooling.net_emissivity` in `src/radbondi/cooling/base.py`.

The default package (`Cooling.default()`) sums three processes.

### 3.1 Relativistic bremsstrahlung

Implemented from Stepney & Guilbert (1983). Define
$\theta_{e} = k_{B}T/(m_{e}c^{2})$ and the prefactor
$q_{b} = \sigma_{T} c\,\alpha_{\rm fs}\,m_{e}c^{2}$. Electron-ion and
electron-electron contributions are summed:

$$
\varepsilon_{\rm br} = q_{b}\,n_{e}\left[\left(\sum_{i}Z_{i}^{2} n_{i}\right)F_{ei}(\theta_{e}) + n_{e} F_{ee}(\theta_{e})\right].
$$

$F_{ei}$ and $F_{ee}$ use the non-relativistic and ultra-relativistic
fitting forms for $\theta_{e} \lessgtr 1$ (eqs. in
`src/radbondi/cooling/bremsstrahlung.py`). For a fully ionized H+He plasma,
$\sum_{i}Z_{i}^{2} n_{i} = (X + Y)\rho/m_{p}$ and
$n_{e} = (X + Y/2)\rho/m_{p}$.

### 3.2 Electron-positron pair annihilation

Following Svensson (1982) and Stepney (1983). The thermal pair number
density from the Maxwell-Jüttner distribution is

$$
n_{\pm} = \frac{\theta}{\pi^{2}\lambda_{C}^{3}}\,K_{2}(1/\theta),
$$

with $\theta = k_{B}T/(m c^{2})$, $\lambda_{C} = \hbar/(m c)$, and
$K_{2}$ the modified Bessel function. The thermally-averaged
annihilation rate uses the Stepney (1983) interpolation:

$$
\langle \sigma v\rangle =
\sigma_{T}c\times
\begin{cases}
1 + \tfrac{1}{2}\theta, & \theta < 1,\\[2pt]
\dfrac{\pi}{2\theta}\bigl[\log(2\theta) + 0.5\bigr], & \theta \ge 1.
\end{cases}
$$

The emissivity is $\varepsilon = n_{\pm}^{2}\langle\sigma v\rangle\,(2 m c^{2})$.
Pair production turns on sharply near $\theta \sim 1$, i.e.
$T \sim 6\times 10^{9}\,\mathrm{K}$ for electrons.

### 3.3 Muon pair annihilation

Same form as 3.2 with $m \to m_{\mu}$,
$\lambda_{C}\to\hbar/(m_{\mu}c)$, and the classical cross-section scaled
by $(m_{e}/m_{\mu})^{2}$. Turns on at $T \sim 1.2\times 10^{12}\,\mathrm{K}$
— negligible for stellar-interior problems but included for completeness.

### 3.4 Adding your own process

Subclass `CoolingProcess` and implement

```python
def emissivity(self, rho, T, ambient) -> np.ndarray:
    ...   # return [erg cm^-3 s^-1]
```

Pass a list to `Cooling([...])` to use an arbitrary combination. See
[usage.md](usage.md#plug-in-cooling).

---

## 4. Derived scalars

From a converged `Solution`:

- **Accretion rate** $\dot M = 4\pi r^{2}\rho\,|v|$, reported as the
  median over the outer half of the domain (smooth, subsonic — avoids
  inner-boundary noise).
- **Mdot ratio** $\dot M / \dot M_{B}$: the cooling enhancement.
- **Luminosity**
  $L = 4\pi\int \varepsilon_{\rm net}(\rho, T)\,r^{2}\,dr$,
  integrated on the finite-volume mesh at solve time.
- **Radiative efficiency** $\eta = L / (\dot M_{B} c^{2})$.

---

## 5. Radiative feedback (optional)

Feedback models sit outside the PDE solver: they take a BH luminosity and
return a modified effective ambient temperature $T_{\infty}'$. The user
re-solves the Bondi problem with `ambient.with_temperature(T_eff)`.

### 5.1 Radiative diffusion

If the radiation that escapes from the BH must diffuse through an
optically thick surrounding medium of opacity $\kappa$, energy balance at
the photon-coupling radius gives

$$
x^{4} = 1 + \beta\,x^{-3/2},
\qquad
\beta = \frac{3(\kappa\rho)^{2} L_{\bullet}}{4\pi a c\,T_{\rm core}^{4}},
$$

where $x = T_{\infty}'/T_{\rm core}$. `DiffusionFeedback` solves this
algebraic equation for $x$ (`src/radbondi/feedback/diffusion.py`). Valid
when $\beta \lesssim 1$; breaks down once the induced gradient exceeds
the adiabatic gradient and convection kicks in.

### 5.2 MLT envelope

For $\beta \gg 1$ the medium becomes convectively unstable and pure
diffusion over-predicts the heating. `MLTEnvelope` integrates the 1D
hydrostatic envelope inward from $r \sim 200\,r_{B}$ using mixing-length
theory:

- Hydrostatic: $dP/dr = -\rho g$
- Radiative gradient
  $\nabla_{\rm rad} = F\,\big/\,[4 a c T^{4} / (3\kappa\rho H_{P})]$
- If $\nabla_{\rm rad} > \nabla_{\rm ad}$: solve
  $F_{\rm rad}(\nabla) + F_{\rm conv}(\nabla) = F_{\rm tot}$ with
  $F_{\rm conv} = \rho c_{p} T (\alpha_{\rm MLT}/2)\sqrt{g H_{P}}\,(\nabla - \nabla_{\rm ad})^{3/2}$
- Otherwise $\nabla = \nabla_{\rm rad}$ (purely radiative).

The integration stops at the photon coupling radius
$r_{c} = 1/(\kappa_{\rm BH}\rho)$, and $T_{\rm eff} = T(r_{c})$ is the
temperature felt by the Bondi flow. See
`src/radbondi/feedback/mlt.py`. Convection saturates the feedback so that
$\eta$ stays finite even for strongly radiating BHs.

---

## 6. Units and reference values

All inputs and outputs are CGS (g, cm, s, K, erg). Some useful reference
values used throughout the code live in `src/radbondi/constants.py`:

| symbol | value | |
|---|---|---|
| $G$ | $6.67430\times 10^{-8}$ | cm³ g⁻¹ s⁻² |
| $c$ | $2.998\times 10^{10}$ | cm s⁻¹ |
| $k_{B}$ | $1.381\times 10^{-16}$ | erg K⁻¹ |
| $\sigma_{T}$ | $6.652\times 10^{-25}$ | cm² |
| $a$ | $7.566\times 10^{-15}$ | erg cm⁻³ K⁻⁴ |
| $M_{\odot}$ | $1.989\times 10^{33}$ | g |

---

## References

- Bondi, H. 1952, MNRAS, 112, 195.
- Stepney, S. 1983, MNRAS, 202, 467 — pair annihilation rate.
- Stepney, S. & Guilbert, P. W. 1983, MNRAS, 204, 1269 — relativistic
  bremsstrahlung.
- Svensson, R. 1982, ApJ, 258, 321 — thermal pair production.
- Cantiello et al. (in prep.) — the feedback prescriptions and the overall
  framework of this code.
