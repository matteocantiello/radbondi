# Numerical scheme

`radbondi` is a **finite-volume, time-dependent** solver that evolves the
spherical Euler equations (see [physics.md](physics.md)) to steady state.
The design is driven by a single difficulty: when cooling is strong
enough, the steady-state ODE has a **complex** (focus-type) sonic point
rather than a saddle, and ODE shooting from either side fails to land on
the unique physical solution. Time evolution of the PDEs resolves this
automatically — the sonic transition is captured by the Riemann solver.

This document describes how the solver is built. All pieces referenced
live in `src/radbondi/{grid,hydro,solver}.py`.

---

## 1. Grid

Log-spaced spherical shells in $x = r/r_{B}$ from `x_min` to `x_max`
(`Grid.log_spaced`). Faces are geometrically spaced

$$
r_{i+1/2} = r_{\rm min}\!\left(\frac{r_{\rm max}}{r_{\rm min}}\right)^{i/N},
$$

and cell centers are the geometric mean $r_{i} = \sqrt{r_{i-1/2}r_{i+1/2}}$.
Cell volumes and face areas are computed exactly for spherical shells:

$$
V_{i} = \tfrac{1}{3}\bigl(r_{i+1/2}^{3} - r_{i-1/2}^{3}\bigr),
\qquad
A_{i+1/2} = r_{i+1/2}^{2}
$$

(both suppress a common $4\pi$ that cancels). Logarithmic spacing gives
good resolution near $r_{S}$ while still reaching the ambient at $r \sim r_{B}$.

**Default domain.** `x_min = 1e-3`, `x_max = 3`, `N = 800`. Push `x_min`
down toward the Schwarzschild radius if you need to resolve the inner
sonic transition. For compact-object accretion we routinely use
`x_min ~ 3e-6`.

---

## 2. Finite-volume discretization

The code stores cell-averaged conservatives $U_{i}$ and updates them as

$$
\frac{dU_{i}}{dt} =
-\frac{A_{i+1/2}F_{i+1/2} - A_{i-1/2}F_{i-1/2}}{V_{i}} + S_{i},
$$

where $F$ are the flux vectors of the Euler system and
$S$ carries the geometric pressure source, gravity, and (optionally)
cooling. The geometric pressure source is written in discretely
**flux-consistent** form

$$
S^{\rm geom}_{i} = P_{i}\,\frac{A_{i+1/2} - A_{i-1/2}}{V_{i}},
$$

which exactly cancels the divergence of the pressure piece of the
momentum flux when $v \to 0$ and $P = \mathrm{const}$, so a hydrostatic
atmosphere stays hydrostatic at machine precision. This is a key
ingredient for well-balancing (§6).

See `hydro_rhs` in `src/radbondi/hydro.py`.

---

## 3. Spatial reconstruction

**First order.** `order=1` uses piecewise-constant cells and a simple
upwind interface state. Diffusive but bulletproof.

**Second order (MUSCL).** `order=2` reconstructs primitives
$(W = (\rho, v, P))$ rather than conservatives, for robustness near
density or pressure gradients. Slopes are computed on primitive
*deviations* from an equilibrium when one is available (§6):

$$
\Delta W_{i}^{L} = W_{i} - W_{i-1},\qquad \Delta W_{i}^{R} = W_{i+1} - W_{i},
$$

$$
\sigma_{i} = \phi(\Delta W_{i}^{L}, \Delta W_{i}^{R}),
$$

with $\phi$ either **minmod** (default, TVD, more dissipative) or
**monotonized-central** (less diffusive, still TVD for smooth flows).
Face states are then

$$
W_{i+1/2}^{L} = W_{i} + \tfrac{1}{2}\sigma_{i},
\qquad
W_{i+1/2}^{R} = W_{i+1} - \tfrac{1}{2}\sigma_{i+1}.
$$

Primitive floors (`rho >= 1e-30`, `P >= 1e-30`) are applied before the
sound speed is taken, to keep the scheme robust through inner-boundary
transients.

---

## 4. Riemann solvers

Two choices via `flux="hll"` (default) or `flux="rusanov"`.

**HLL** (Harten-Lax-van Leer) uses two signal speeds
$S_{L} = \min(v_{L}-c_{L}, v_{R}-c_{R})$ and
$S_{R} = \max(v_{L}+c_{L}, v_{R}+c_{R})$:

$$
F^{\rm HLL} =
\begin{cases}
F_{L}, & S_{L}\ge 0,\\[3pt]
\dfrac{S_{R}F_{L} - S_{L}F_{R} + S_{L}S_{R}(U_{R}-U_{L})}{S_{R} - S_{L}}, & S_{L} < 0 < S_{R},\\[6pt]
F_{R}, & S_{R} \le 0.
\end{cases}
$$

**Rusanov** (local Lax-Friedrichs) is an even simpler fallback that uses
$\max(|v|+c)$ as a single dissipation speed. Use it if you suspect HLL
pathology (rare).

When a well-balanced reference state $U_{\rm eq}$ is in play, the HLL
dissipation term $S_{L}S_{R}(U_{R}-U_{L})$ is replaced by
$S_{L}S_{R}\bigl[(U_{R}-U_{\rm eq,R}) - (U_{L}-U_{\rm eq,L})\bigr]$ so that
the dissipation vanishes in equilibrium. See `compute_fluxes` in
`src/radbondi/hydro.py`.

---

## 5. Time stepping

### 5.1 Local time stepping

Because the characteristic CFL time scales as $\Delta r / (|v| + c_{s})$,
and $\Delta r$ grows logarithmically while $|v|+c_{s}$ grows fast near
$r_{S}$, the inner cells are often $10^{3}$–$10^{4}\times$ more restrictive
than the outer cells. Under a global CFL timestep this is catastrophic.

For **steady-state** problems we don't care about the time history — only
the fixed point — so `radbondi` uses cell-local time stepping:

$$
\Delta t_{i} = \mathrm{CFL}\,\frac{\Delta r_{i}}{|v_{i}| + c_{s,i}},
\qquad
U_{i}^{n+1} = U_{i}^{n} + \Delta t_{i}\left(\frac{dU}{dt}\right)_{\!i}.
$$

Each cell advances at its own pace. The fixed point of this iteration is
still the physical steady state (the RHS is zero), and convergence is
$\sim 10^{3}\times$ faster than a global $\Delta t$. Printing at
`verbose=True` shows the speedup vs global.

Local time stepping is **not** a time-accurate integrator — you cannot
interpret the transient as physical. If you need the time history, use
`CFL` so small that all cells share a $\Delta t$ (slow).

### 5.2 Operator splitting for cooling

Cooling is stiff. The solver splits each step into hydro and cooling
sub-steps:

- **Lie splitting** (default, `strang=False`):
  hydro $\to$ implicit cool.
- **Strang splitting** (`strang=True`):
  cool/2 $\to$ hydro $\to$ cool/2. Second-order accurate in time, useful
  if you actually care about transients.

### 5.3 Implicit cooling step

Given a hydro-updated state, the cooling sub-step holds $\rho$ and $v$
fixed (cooling is isochoric in the split) and updates $T$ by
backward Euler:

$$
\frac{\rho k_{B}}{\mu m_{p}(\gamma - 1)}\,\bigl(T^{n+1} - T^{n}\bigr)
= -\Delta t\,\varepsilon_{\rm net}(\rho, T^{n+1}).
$$

Let $c_{\rm th} = \rho k_{B}/[\mu m_{p}(\gamma-1)]$ and
$e_{\rm th}^{n} = P^{n}/(\gamma-1)$. The Newton iteration solves

$$
R(T) = c_{\rm th} T - e_{\rm th}^{n} + \Delta t\,\varepsilon_{\rm net}(\rho, T) = 0,
$$

with the Jacobian approximated by finite differences in $T$. The
iteration is vectorized — all active cells advance in lockstep — and is
clipped at a floor $T \ge T_{\infty}$. Two boundary cells on each side
are skipped (BCs handled separately). See `_apply_cooling_implicit` in
`src/radbondi/solver.py`.

### 5.4 Cooling ramp

To avoid a startup shock when the initial adiabatic profile is suddenly
exposed to fully-on cooling, $\varepsilon$ is linearly ramped from 0 to 1
over `cooling_ramp_steps` (default 5000). Tune down for mild cooling; up
if you see early oscillations.

---

## 6. Well-balancing

A naive finite-volume solver applied to the initial adiabatic Bondi
profile produces a spurious residual $R_{\rm eq} \ne 0$ because the
pressure, gravity, and flux terms don't cancel to machine precision on a
discrete grid. That residual is large (order of the gravitational source)
and swamps the small residual induced by cooling.

The fix: store the initial adiabatic state $U_{\rm eq}$ and its discrete
residual $R_{\rm eq}$, and **subtract** $R_{\rm eq}$ from every
subsequent RHS evaluation:

$$
\left(\frac{dU}{dt}\right)^{\!\rm WB}_{i}
= R[U_{i}]
- w_{i}\,R_{\rm eq,i},\qquad
w_{i} = \exp\!\left(-\frac{\|U_{i} - U_{\rm eq,i}\|}{0.01\,\|U_{i}\|}\right).
$$

Where the flow has stayed near equilibrium ($w \to 1$) the scheme is
well-balanced to machine precision. Where cooling has restructured the
flow ($w \to 0$) full physical dissipation is restored. The weight
$w$ uses the normalized RMS deviation across the three conservatives.

MUSCL slopes are also computed on $W - W_{\rm eq}$, and the HLL
dissipation uses the deviation (§4). Together these let the scheme hold
the adiabatic Bondi equilibrium at round-off while still capturing the
cooling-induced modification.

Modes:

- `wb_mode="full"` (default): full equilibrium subtraction.
- `wb_mode="adaptive"`: same as "full" in current implementation (reserved).
- `wb_mode="off"`: disable well-balancing. Only useful for debugging.

---

## 7. Boundary conditions

**Inner boundary.** Adaptive based on the Mach number in the first
interior cell:

- `M_inner > inner_mach_threshold` (default 2): free extrapolation,
  $U_{0} = U_{1}$. Once the flow is supersonic infalling the interior is
  causally disconnected from the boundary.
- `M_inner < threshold`: well-balanced extrapolation,
  $U_{0} = U_{{\rm eq},0} + (U_{1} - U_{{\rm eq},1})$. Preserves the
  equilibrium structure until the flow steepens.

**Outer boundary.** The two outermost cells are held to their initial
values $U_{N-1}, U_{N-2}$. This pins the ambient conditions — the
intended behavior for a Bondi problem embedded in a quasi-uniform medium.

---

## 8. Stabilization knobs

**Sponge layer.** Optional quadratic damping of the outer `sponge_frac`
cells toward the initial equilibrium:

$$
U_{i} \leftarrow (1 - \alpha_{i})U_{i} + \alpha_{i} U_{{\rm eq},i},
\quad \alpha_{i} \in [0, 1],
$$

with $\alpha_{i} = ((i - (N - n_{\rm sp}))/n_{\rm sp})^{2}$. Off by default.
Use sparingly — it can hide bugs.

**Under-relaxation.** `relaxation=omega` blends the update:

$$
U^{n+1} = U^{n} + \omega\,(U^{n+1}_{\rm computed} - U^{n}).
$$

Default $\omega=1$ (off). Reduce to ~0.5 only if the iteration oscillates.

**Floors.** A density floor `rho >= 1e-30` and pressure floor
$P \ge \rho k_{B} T_{\rm floor}/(\mu m_{p})$ with
$T_{\rm floor} = T_{\infty}/2$ are applied after each step. These never
trigger on healthy runs; their job is to prevent blow-ups during early
transients.

---

## 9. Convergence criterion

The solver reports the residual

$$
\rho^{n} = \sqrt{\frac{1}{3N}\sum_{k,i}\!\left(\frac{U^{n+1}_{k,i} - U^{n}_{k,i}}{|U^{n}_{k,i}| + 10^{-30}}\right)^{\!2}}.
$$

Termination: $\rho^{n}$ < `convergence_tol` (default $10^{-10}$),
**and** step number $>\max(100,$ `cooling_ramp_steps`$)$. The step-count
guard prevents early exit while cooling is still ramping. If the loop
hits `n_steps` without converging, the solver returns the last state with
`converged=False` and you can inspect `sol.residuals` to see what
happened.

---

## 10. Post-processing

Two diagnostic checks can be applied to a converged `Solution`:

- **`sol.check_steady_state(cooling)`** — evaluates the steady-state
  Euler equations in integral (control-volume) form and returns RMS /
  max residuals for mass flux, momentum, and energy. Excludes
  `n_boundary` cells at each end by default. Healthy runs hit
  $\lesssim 10^{-6}$ (machine + iteration tolerance).
- **`sol.recompute_luminosity(cooling)`** — re-integrates the
  luminosity with a possibly different cooling prescription (useful
  for decomposing L into bremsstrahlung vs pair contributions without
  re-solving).

See `src/radbondi/diagnostics.py`.

---

## 11. Putting the pieces together

One full solver step (Lie splitting, WB on):

1. Compute primitive state $(\rho, v, P, T, c_{s})$ with floors.
2. MUSCL-reconstruct face states on $(W - W_{\rm eq})$ with minmod/MC.
3. Evaluate HLL fluxes using the deviation for dissipation.
4. Form the RHS: flux divergence + geometric pressure + gravity.
5. Subtract the weighted equilibrium residual $w\,R_{\rm eq}$.
6. Advance with local $\Delta t_{i}$.
7. (If cooling on) Newton-implicit update to $T$ in active cells.
8. Apply inner/outer BCs; optionally apply sponge, relaxation, floors.
9. Record residual. Repeat.

Strang splitting (step 0: half-implicit cool; step 7: half-implicit cool)
is an opt-in variant.

---

## References

- Harten, A., Lax, P. D., & van Leer, B. 1983, SIAM Rev., 25, 35 — HLL
  Riemann solver.
- LeVeque, R. J. 2002, *Finite Volume Methods for Hyperbolic Problems*,
  Cambridge U.P. — finite-volume and MUSCL machinery.
- Käppeli, R. & Mishra, S. 2014, J. Comput. Phys., 259, 199 — the
  well-balanced strategy used here.
