# The collisionless pile-up problem in PBH accretion

A comprehensive description of the problem, our analytical and
numerical approaches, what we found, and what remains open. Written
for sharing with collaborators who haven't followed the development.

*Matteo Cantiello, April 2026*

---

## 1. Context: the paper

We have a paper (Cantiello et al., in prep.) on the accretion
luminosity of primordial black holes (PBHs) embedded in stellar
interiors. The code (`radbondi`) solves the 1D spherical Euler
equations with an implicit bremsstrahlung cooling source term, evolving
from an adiabatic Bondi initial condition to a self-consistent cooled
steady state. The code works well and reproduces the paper's Table 1
across four accretion regimes (collisionless, bremsstrahlung-dominated,
transitional, near-isothermal) for PBH masses $10^{-16.1}$ to
$10^{-10}\,M_\odot$ in the solar core.

## 2. The question

A collaborator (Andrei Gruzinov) raised a sharp point about the lightest
masses in our range ($M \lesssim 10^{-14}\,M_\odot$): if the gas
becomes collisionless near the BH, particles retain their thermal
angular momentum and most *miss* the GR capture cross-section. Couldn't
the accretion rate be catastrophically reduced?

For a **fully** collisionless gas, the accretion rate drops to the
gravitational-capture value:

$$\frac{\dot{M}_{\rm coll}}{\dot{M}_B} \sim \frac{c_s^2}{c^2}
\approx 3 \times 10^{-6}$$

a factor $\sim 4 \times 10^5$ reduction. If true, this would
invalidate the paper's results for the lightest masses.

## 3. Why this isn't fully collisionless accretion

The key distinction: our PBHs have $\lambda_{\rm mfp} \ll r_B$. The
gas IS collisional at the Bondi radius — the hydrodynamic flow is valid
there. The flow only becomes collisionless at small radii ($r_{\rm coll}
\approx 7000\,r_S$ for $M = 10^{-16}\,M_\odot$), deep inside the
Bondi sphere.

Reference scales for $M = 10^{-16}\,M_\odot$ in the solar core:
- $r_B \approx 5 \times 10^{-6}$ cm
- $r_S \approx 3 \times 10^{-11}$ cm
- $\lambda_{\rm mfp} \sim 10^{-7}$ cm (at ambient)
- $r_B / \lambda_{\rm mfp} \sim 50$ (deeply collisional at $r_B$)

The fluid approximation only breaks down for
$M \lesssim 10^{-18}\,M_\odot$, well outside our range.

## 4. But most particles DO miss

At $r_{\rm coll}$, individual particles have thermal angular momentum
$\ell_{\rm typ} \sim r_{\rm coll} \times c_s/\sqrt{\gamma} \sim 30$
cm$^2$/s. The GR capture threshold is $\ell_{\rm crit} = 4GM/c \approx
1.8$ cm$^2$/s. Since $\ell_{\rm typ}/\ell_{\rm crit} \approx 18$,
most particles swing through pericenter and come back out.

The capture fraction per pass is only $f \sim (\ell_{\rm crit}/\ell_
{\rm typ})^2/2 \sim 1.5 \times 10^{-3}$. The quadratic scaling comes
from the 2D velocity space: both transverse velocity components must
independently be small for capture.

**Important**: $\ell_{\rm crit}$ comes from the GR effective potential.
In Newtonian gravity there is *no capture* for $\ell \neq 0$ — the
centrifugal barrier always deflects the particle. It is the relativistic
$-GM\ell^2/(c^2 r^3)$ term that destroys the barrier and enables
accretion.

## 5. The recycling argument

Particles that miss bounce back to $r > r_{\rm coll}$, re-enter the
collisional zone, re-thermalize, and fall in again. In steady state,
the net accretion rate equals the mass flux supplied from above — not
the single-pass capture rate.

**Condition**: this requires an effective sink (direct GR capture,
angular-momentum redistribution at the thermalization boundary, or
collisional transport). Without a sink, the missed particles would
build up indefinitely into a bound reservoir.

The recycling creates a density enhancement $\xi = 2/f_{\rm eff}$
relative to the current supplied inflow, where $f_{\rm eff}$ is the
effective capture/transport fraction per crossing.

## 6. Bremsstrahlung self-limits the pile-up

Orbits are not conservative. Bremsstrahlung from binary encounters
drains orbital energy during each pericenter passage:

$$\left(\frac{\Delta E}{E}\right)_0 \approx \frac{2\varepsilon r}
{\rho v^3}$$

evaluated from the cooled Bondi profiles:

| $r/r_S$ | Mach | $\Delta E/E$ |
|---|---|---|
| 3000 | 0.86 | 1.1% |
| 1000 | 0.96 | 1.3% |
| 550  | 1.08 | 8.0% |
| 300  | 1.33 | 21%  |

At the enhanced density $\xi \rho$, the cooling per particle scales as
$\xi$. The self-consistent pile-up factor is:

$$\xi = \sqrt{\frac{2}{(\Delta E/E)_0}} \approx 5\text{–}13$$

depending on the evaluation radius.

**Critical caveat**: energy loss at fixed angular momentum does NOT
cause plunging — it causes **circularization**. The particle settles
into progressively tighter, roughly circular orbits. Actual BH capture
requires either $\ell < \ell_{\rm crit}$ (direct capture) or
angular-momentum loss/randomization (which only happens at $r_{\rm
coll}$ during re-thermalization).

**Pericenter caveat**: a typical particle with $\ell_{\rm typ} \sim
18\,\ell_{\rm crit}$ has Newtonian pericenter $r_{\rm peri} =
\ell^2/(2GM) \approx 1200\,r_S$, which is *above* the sonic point
($\sim 700$–$800\,r_S$). The large $\Delta E/E$ values at 300–550 $r_S$
apply only to the lower-$\ell$ tail (~10–30% of particles). For the
bulk population, $\xi \sim 12$–$13$.

## 7. But there IS a back-reaction on $\dot{M}$

Here's where the story gets interesting. We initially argued that the
pile-up doesn't affect $\dot{M}_B$ (the Bondi accretion rate). This
was **wrong**.

At $r_{\rm coll}$, the returning particles thermalize and create a
pressure enhancement $\sim \xi \times P_{\rm current}$. Since $r_{\rm
coll}$ is in the **subsonic** zone ($\mathcal{M} \approx 0.7$), this
pressure perturbation can communicate upstream all the way to $r_B$,
modifying the Bondi eigenvalue.

Inside the collisionless zone ($r < r_{\rm coll}$), the inbound and
outbound streams don't interact (two-stream, no single-fluid pressure).
But at $r_{\rm coll}$ itself, the thermalization is collisional and the
back-pressure is real.

## 8. Quantifying the back-reaction: numerical experiments

We attempted to quantify the $\dot{M}$ reduction through two
complementary numerical approaches.

### Option B: Fixed-$\xi$ sensitivity mapping

**Idea**: impose a fixed pressure enhancement ($\xi$ times the baseline
temperature) in a shell near $r_{\rm coll}$ and measure how $\dot{M}$
responds. No feedback — $\xi$ doesn't change with $\dot{M}$.

**Implementation**: Gaussian temperature override at $r_{\rm coll}$,
ramped on gradually over 5000 steps. N=1200, order=1, CFL=0.3.

**Results** (3 valid points; $\xi \geq 2$ blew up):
- $\xi = 1.0$: $\dot{M}/\dot{M}_0 = 1.000$ (sanity check)
- $\xi = 1.1$: 0.848 (−15%)
- $\xi = 1.2$: 0.715 (−29%)
- $\xi = 1.5$: 0.542 (−46%)

**Lesson**: the flow is VERY sensitive to inner boundary pressure. But
the Gaussian-shell implementation creates artificial interior pressure
gradients that cause blow-ups at $\xi \geq 2$.

### Option D: Self-consistent sub-grid model

**Idea**: let the solver compute the pile-up factor from the current
flow and apply it as a Dirichlet inner boundary condition at $r_{\rm
coll}$. The feedback loop is built in: if $\dot{M}$ drops, the pile-up
adjusts, the pressure changes, and $\dot{M}$ recovers.

**Implementation evolution** (5 iterations to get it right):
1. **v1**: $T_{\rm target} = T_{\rm current} \times \xi$ — compounded
   exponentially, blew up instantly
2. **v2**: $T_{\rm target} = T_{\rm baseline} \times \xi$ in Gaussian
   shell — interior pressure gradient drove gas inward (wrong
   direction), blew up
3. **v3**: Dirichlet BC at $r_{\rm coll}$ (hold 2 cells fixed) — ran
   but oscillated wildly ($\dot{M}$ range: 0.07–1.45)
4. **v4**: Added under-relaxation $\omega = 0.02$ — still oscillated
   ($\omega$ too fast)
5. **v5**: $\omega = 0.001$ — oscillation damped to factor ~4 amplitude

**Key lessons**:
- NEVER multiply $T_{\rm current}$ by $\xi$ each step (compounds)
- NEVER apply pressure enhancement as a volumetric source in the flow
  interior (creates spurious pressure gradients)
- USE Dirichlet BC (hold boundary cells fixed) to represent the
  thermalization pressure
- The under-relaxation timescale must be comparable to the acoustic
  travel time from $r_{\rm coll}$ to $r_{\rm sonic}$ (~1000 steps)

**Final result** (v5, $\omega = 0.001$, 400k phase 2 steps):
- $\dot{M}/\dot{M}_B$: mean ≈ 0.55, median ≈ 0.52
- $\Theta_{\rm bc}$ (boundary temperature multiplier): mean ≈ 2.5
- Still oscillating (factor ~4), but mean is stable

## 9. The diagnostic revelation: back-pressure slows the flow

We added diagnostics to measure what's happening at $r_{\rm sonic}$
during the sub-grid run.

| Quantity | Baseline | Phase 2 mean |
|---|---|---|
| $\rho/\rho_{\rm baseline}$ | 1.0 | 0.46 |
| $T/T_{\rm baseline}$ | 1.0 | 1.09 |
| $v/v_{\rm baseline}$ (inferred) | 1.0 | ~0.48 |
| $\Delta E/E$ | 0.084 | 0.37 |
| $\xi_{\rm raw}$ | 4.88 | 2.36 |

**The key finding**: the pressure BC at $r_{\rm coll}$ **slows** the
inflow at $r_{\rm sonic}$ to ~48% of baseline velocity. The density
drops (as expected from reduced $\dot{M}$) and T barely changes, but
$v$ drops by half.

Since $\Delta E/E \propto 1/v^3$, the 2× velocity reduction causes an
8× increase in $1/v^3$, which dominates the 2× density reduction. Net:
$\Delta E/E$ rises from 0.084 to 0.37 (4.4× increase), and
$\xi_{\rm raw} = \sqrt{2/0.37} \approx 2.4$.

This explains why $\Theta_{\rm bc} \approx 2.5 < \xi_0 \approx 4.9$:
the back-pressure slows the flow, increasing the bremsstrahlung
residence time, which reduces the pile-up factor. The naive free-fall
scaling $\xi \propto 1/\sqrt{m}$ doesn't apply in the subsonic zone
because $v$ also changes.

## 10. Mass dependence: the effect vanishes for heavier PBHs

The back-reaction requires $r_{\rm coll} > r_{\rm sonic}$ (pile-up in
the subsonic zone). From the converged profiles across the paper's mass
range:

| log M/M☉ | $r_{\rm coll}/r_S$ | $r_{\rm sonic}/r_S$ | ratio | effect |
|---|---|---|---|---|
| −16.0 | 7200 | 630 | 11.5 | ~2× reduction |
| −15.6 | 2500 | 2700 | 0.94 | marginal |
| −15.3 | 1000 | 17000 | 0.06 | none |
| −15.0 | 370 | 57000 | 0.01 | none |
| ≥ −14.5 | always collisional | — | — | none |

**Transition at $M_{\rm crit} \approx 10^{-15.6}\,M_\odot$**: above
this mass, the cooling-induced sonic point moves outward past $r_{\rm
coll}$, and the pile-up sits in the supersonic zone where it can't
communicate upstream. The 1D Bondi+cooling code is self-consistent for
$M \gtrsim 10^{-15.5}\,M_\odot$.

## 11. Three regimes of collisionality

| Mass range | Radius ordering | $\dot{M}$ | Physics |
|---|---|---|---|
| $M \gtrsim 10^{-15.6}\,M_\odot$ | $r_{\rm coll} < r_{\rm sonic} < r_B$ | $\dot{M}_B$ | Fully hydrodynamic; pile-up is supersonic |
| $10^{-18} \lesssim M \lesssim 10^{-15.6}$ | $r_{\rm sonic} < r_{\rm coll} < r_B$ | $\chi(M) \dot{M}_B$ | Collisionless inner zone; $\chi \sim 0.5$–1 |
| $M \lesssim 10^{-18}\,M_\odot$ | $r_B < r_{\rm coll}$ | $(c_s/c)^2 \dot{M}_B$ | Fully collisionless; no Bondi flow |

The paper covers regimes 1 and 2 only.

## 12. What we conclude

**The central claim is now:**

> For $M \sim 10^{-16}\,M_\odot$, the flow is collisional at $r_B$ but
> collisionless inside $r_{\rm coll}$. Single-pass GR capture is
> inefficient because most particles retain thermal angular momentum.
> However, missed particles recycle through the collisional boundary
> rather than being lost, so the accretion rate is not suppressed by the
> fully collisionless factor $(c_s/c)^2$. Bremsstrahlung and
> thermalization self-limit the residence time and create a pressure
> perturbation at $r_{\rm coll}$, which can reduce the hydrodynamic
> supply rate by an order-unity factor. A sub-grid experiment suggests
> $\dot{M}/\dot{M}_B \sim 0.5$ for $M = 10^{-16}\,M_\odot$, but the
> precise value depends on the kinetic boundary condition and
> angular-momentum transport.

## 13. Stage 1 result: orbit-averaged $\xi$

We computed the orbit-averaged energy loss $\Delta E/E(\ell)$ for 200
angular momentum values, integrating along the full Newtonian orbit
(apocenter at $r_{\rm coll}$, pericenter from the orbital energy
equation) using the converged N=6400 cooled profiles
(`experiments/orbit_library.py`).

### Population breakdown

The Rayleigh angular-momentum distribution at $r_{\rm coll}$ splits
into three populations:

| Population | Fraction | $\ell$ range | Fate |
|---|---|---|---|
| Direct capture | 0.1% | $\ell < \ell_{\rm crit}$ | Swallowed by BH |
| Reflected | 38% | $\ell > \ell_{\rm max} \approx 22.7\,\ell_{\rm crit}$ | Centrifugal barrier at $r_{\rm coll}$; never enters collisionless zone |
| Recycling | 61% | $\ell_{\rm crit} < \ell < \ell_{\rm max}$ | Forms the pile-up; orbits between $r_{\rm coll}$ and $r_{\rm peri}(\ell)$ |

The maximum $\ell$ for a bound orbit at the local energy is
$\ell_{\rm max} = GM/\sqrt{2|E|} \approx 40$ cm²/s. Particles with
$\ell > \ell_{\rm max}$ cannot form closed orbits — they bounce off
the centrifugal barrier near $r_{\rm coll}$ and immediately
re-thermalize. This 38% contributes to the Bondi flow pressure but
not to the inner pile-up.

### Orbit-averaged cooling

For the recycling population:

| $\ell/\ell_{\rm crit}$ | $r_{\rm peri}/r_S$ | $\Delta E/E$ (orbit-avg) | $\xi$ | reaches $r_{\rm sonic}$? |
|---|---|---|---|---|
| 1.4 | 8 | 8.7% | 4.8 | yes |
| 3.0 | 37 | 7.4% | 5.2 | yes |
| 5.0 | 103 | 5.1% | 6.3 | yes |
| 10.1 | 427 | 1.4% | 11.9 | yes |
| 15.1 | 1041 | 1.1% | 13.6 | no |
| 18.1 | 1633 | 1.1% | 13.2 | no |

Distribution-averaged over the recycling population:

$$\langle \Delta E/E \rangle_{\rm rec} = 0.018, \qquad
\langle \xi \rangle_{\rm rec} = 11.9$$

Cross-check: $\sqrt{2/0.018} = 10.5$, consistent with $\langle\xi\rangle = 11.9$.

### Interpretation

The orbit-averaged result confirms the analytic single-radius
estimates: the pile-up is controlled by the **high-$\ell$,
shallow-pericenter** population ($\ell \sim 10$–$20\,\ell_{\rm crit}$,
$r_{\rm peri} \sim 500$–$2000\,r_S$), not by the strongly-cooled
low-$\ell$ tail. The bulk population sees modest per-orbit cooling
($\Delta E/E \sim 1$–$2\%$) and the corresponding $\xi \sim 12$–$13$
is near the upper end of the single-radius range.

With $\langle\xi\rangle \approx 12$ and $N_{\rm orb} = \xi/2 \approx 6$
orbital passages before significant orbit shrinkage, and a direct
capture probability of $f \sim 1.5 \times 10^{-3}$ per passage, the
probability of direct GR capture during the residence time is only
$\sim 6 \times 1.5 \times 10^{-3} \approx 1\%$. **The remaining
99% of particles circularize or return to $r_{\rm coll}$ for
re-thermalization.** This sharpens the central open question: does
re-thermalization + angular-momentum randomization eventually feed
the loss cone, or does the gas accumulate in a bound reservoir?

---

## 14. Stage 2 result: self-induced collisionality resolves the reservoir problem

### 14.1 The Markov model (Stage 2a)

A Monte Carlo model tracking individual particle orbits
(`experiments/recycling_markov.py`) shows that 99% of particles
circularize inside $r_{\rm coll}$ after ~7 orbital passages.
Only ~1% are directly captured by GR ($\ell < \ell_{\rm crit}$).
Angular-momentum diffusion barely helps, even at high efficiency.

This means **direct GR capture is not the dominant sink.** The
question becomes: does the circularized gas form a permanent
collisionless reservoir, or does it self-collisionalize?

### 14.2 The collisionalization map (Stage 2b)

For each $\ell$-bin, we compute the effective Knudsen number at the
circularization radius $r_{\rm circ} = \ell^2/GM$, using the
pile-up-enhanced density:

$${\rm Kn}_{\rm eff}(\ell) = \frac{{\rm Kn}_0(r_{\rm circ})}{\xi(\ell)}$$

(`experiments/collisionalization_map.py`). The population fractions
among particles that enter the collisionless zone:

| Population | Fraction (of total) | Fraction (of non-reflected) | $\ell/\ell_{\rm crit}$ | ${\rm Kn}_{\rm eff}$ |
|---|---|---|---|---|
| Direct capture | 0.2% | 0.3% | < 1 | — |
| Reflected | 33% | — | > 22.7 | — |
| **Collisionalized** | **42%** | **63%** | 12–23 | < 1 |
| Marginal | 10% | 15% | 10–12 | 1–3 |
| Reservoir | 14% | 21% | 1.5–10 | 4–140 |

The transition from reservoir to collisionalized occurs at
$\ell \approx 12\,\ell_{\rm crit}$ ($r_{\rm circ} \approx 1200\,r_S$,
${\rm Kn}_{\rm eff} \approx 1$). For $\ell \gtrsim 12\,\ell_{\rm crit}$,
the pile-up-enhanced density is sufficient to make the circularization
region collisional, enabling angular-momentum redistribution.

### 14.3 The reservoir self-eliminates (Stage 2c)

The 14% reservoir fraction is not permanent. The additional mass
required to reduce ${\rm Kn}_{\rm eff}$ to unity in all reservoir
bins is only $\sim 6 \times 10^{-15}$ g
(`experiments/reservoir_buildup.py`). Even if supplied only by the
reservoir fraction of the reduced hydrodynamic inflow
($\dot{M}_{\rm res} \sim f_{\rm res} \times \chi \times \dot{M}_B
\sim 0.08$ g/s), this mass accumulates in $\sim 7 \times 10^{-14}$ s
— tens of local free-fall times. The reservoir grows until it becomes
collisional, after which angular-momentum redistribution and
hydrodynamic transport proceed.

### 14.4 The self-regulating feedback loop

$$\text{cooling} \to \text{circularization} \to \rho_{\rm pile}\!\uparrow
\to \lambda_{\rm mfp}/r\!\downarrow
\to \text{collisionalization} \to \text{angular-momentum transport}
\to \text{inward hydrodynamic flow}$$

This removes the main qualitative concern that missed particles form
a long-lived collisionless reservoir. The remaining uncertainty is
quantitative: the exact back-pressure and luminosity require a
conservative hydro–kinetic boundary condition.

### 14.5 Conservative hydro-kinetic source coupling (Stage 3b)

Using the orbit-averaged kinetic closure, we compute physical
momentum and energy source terms $S_p(r)$, $S_E(r)$ and inject
them into the 1D solver as conservative volumetric sources
(`experiments/stage3b_coupled.py`). This replaces the Dirichlet
temperature override with physically derived fluxes.

An $\alpha$-scan (source strength from 0 to 1) gives a smooth,
monotonic response:

| $\alpha$ | $\chi = \dot{M}/\dot{M}_B$ |
|---|---|
| 0.00 | 1.000 |
| 0.10 | 0.959 |
| 0.25 | 0.900 |
| 0.50 | 0.811 |
| 0.75 | 0.735 |
| 1.00 | 0.677 |

All runs converge cleanly — no blow-ups, no oscillations. At full
source strength: $\chi_{\rm frozen} \approx 0.68$ (32% reduction).

A **fully self-consistent** calculation, where the kinetic closure
is slowly updated from the evolving hydro state ($\omega = 10^{-3}$,
recomputed every 1000 steps), converges to a clean steady fixed
point:

$$\boxed{\chi_{\rm self} = \dot{M}/\dot{M}_B \approx 0.69}$$

The convergence is monotonic (0.88 → 0.75 → 0.70 → 0.69) with no
oscillations or limit cycle. The self-consistent value is only
marginally higher than the frozen estimate (0.69 vs 0.68), meaning
the pile-up pressure is insensitive to the ~31% $\dot{M}$ reduction.

### 14.6 Mass scan: chi(M) correction function (Stage 4)

Running the full self-consistent closure across multiple masses
(`experiments/stage4_mass_scan.py`) gives the production result:

| log M/M☉ | $\chi$ | reduction | $r_{\rm coll}/r_{\rm sonic}$ |
|---|---|---|---|
| −16.0 | 0.69 | 31% | 8.9 |
| −15.8 | 0.95 | 5% | 11.5 |
| −15.6 | 0.94 | 6% (marginal) | 0.94 |
| −15.3 | 1.00 | 0% | 0.06 |

The ~30% correction is sharply localized to $M \sim 10^{-16}\,M_\odot$.
By $10^{-15.8}$, it's already only ~5%. For $M \gtrsim 10^{-15.5}$,
$\chi = 1$ exactly.

Note: logM = −16.1 did not converge at N=1200; the −16.0 result is
adopted as the representative correction for the lightest masses.

### 14.7 Final central claim

> For $M \sim 10^{-16}\,M_\odot$, particles entering the collisionless
> region mostly miss the GR loss cone and circularize after
> bremsstrahlung energy losses. However, circularization produces a
> local density pile-up that shortens the Coulomb mean free path. For
> the majority of the mass, the circularization region becomes
> marginally or fully collisional, enabling angular-momentum
> redistribution and inward hydrodynamic transport. The residual
> low-$\ell$ reservoir self-eliminates on a dynamical timescale.
> A self-consistent hydro-kinetic coupling gives $\chi \sim 0.7$ at
> $10^{-16}\,M_\odot$, dropping to $\sim 0.95$ by $10^{-15.8}\,M_\odot$
> and unity for $M \gtrsim 10^{-15.5}\,M_\odot$. The collisionless
> pile-up does not invalidate the paper; it produces at most a modest
> correction at the lowest mass.

---

## 15. What remains open

### 15.1 Luminosity correction

The 1D code underestimates the emissivity in the collisionless zone.
A cumulative $L(<r)$ diagnostic is needed. The luminosity correction
should combine the reduced Mdot ($\chi \sim 0.7$) with the enhanced
pile-up density near circularization radii.

### 15.2 Luminosity

The 1D code underestimates the emissivity in the collisionless zone.
The physical pile-up density could enhance bremsstrahlung by factors
of several to ~10 locally. A cumulative $L(<r)$ diagnostic is needed.

### 15.3 Anomaly-based density profile

The orbit-integrated density profile $\rho_{\rm pile}(r)$ needs an
eccentric-anomaly parameterization to avoid the turning-point
singularity. This would also provide the self-consistent ${\rm Kn}(r)$
map rather than the per-$\ell$ estimate used in Stage 2b.

## 16. Next steps (prioritized)

1. **[DONE] Orbit-averaged $\xi$** (Stage 1, §13): confirms
   $\langle\xi\rangle \approx 12$, population breakdown.

2. **[DONE] Loss-cone / recycling Markov model** (Stage 2a):
   99% circularize, ~1% direct GR capture.

3. **[DONE] Collisionalization map** (Stage 2b): 52% collisionalized
   by pile-up, 14% reservoir.

4. **[DONE] Reservoir build-up timescale** (Stage 2c): self-eliminates
   in ~tens of free-fall times.

5. **[DONE] Conservative source coupling** (Stage 3b): frozen-closure
   $\chi \approx 0.68$. Smooth response, no instabilities.

6. **[DONE] Self-consistent source coupling** (Stage 3c): converges
   to $\chi \sim 0.7$. Clean steady state.

7. **[DONE] Mass scan** (Stage 4): chi(M) correction function.
   Correction localized to lightest masses; vanishes by 10^{-15.5}.

8. **Regularized density profile** (Deliverable B): anomaly-based
   $\rho_{\rm pile}(r)$ and $L(<r)$. Now informed by Stage 2 —
   the density profile is a quasi-steady recycling flow, not a
   long-lived reservoir.

6. **${\rm Kn}_{\rm eff}(\ell)$ figure**: shaded regions showing
   direct capture / reservoir / marginal / collisionalized / reflected.
   Single most useful summary figure.

7. **Hydro closure with kinetic fluxes** (Stage 3): replace the
   Dirichlet temperature BC with conservative flux matching.

8. **2D validation** (Stage 5): axisymmetric simulation to validate
   the reduced model.

---

## 17. Step-by-step numerical experiments

### Starting point

The 1D spherical Bondi+cooling solver (`radbondi`) produces converged
$\rho(r)$, $T(r)$, $v(r)$, $\mathcal{M}(r)$ profiles at N=6400 for
each PBH mass. These "baseline" profiles are the input to all
subsequent experiments.

### Stage 1: Orbit-averaged cooling (`orbit_library.py`)

Takes the N=6400 profiles for $M = 10^{-16}\,M_\odot$ and, for each
$\ell$ in a 200-bin Rayleigh distribution:
1. Computes Newtonian pericenter $r_{\rm peri} = \ell^2/(2GM)$
2. Checks whether a bound orbit exists ($\ell < \ell_{\rm max}$)
3. Integrates $\Delta E/E$ along the full orbit using the profile's
   bremsstrahlung emissivity
4. Computes $\xi(\ell) = \sqrt{2/(\Delta E/E)}$

**Issue encountered**: orbits with $\ell > \ell_{\rm max}$ (38% of
the distribution) can't form bound orbits — they contaminated the
average. Excluding them fixed the result: $\langle\Delta E/E\rangle = 0.018$,
$\langle\xi\rangle = 11.9$.

### Stage 2a: Markov recycling model (`recycling_markov.py`)

Monte Carlo of 50,000 particle packets. Each starts at $r_{\rm coll}$,
draws $\ell$, checks for direct capture ($\ell < \ell_{\rm crit}$) or
reflection ($\ell > \ell_{\rm max}$). Recycling particles lose energy
$\Delta E/E(\ell)$ per orbit (from Stage 1 interpolant), compute new
apocenter, and either circularize or return to $r_{\rm coll}$.

**Issue encountered**: first version used $\xi$-based residence blocks
where the total energy loss was $\Delta E/E \times N_{\rm orb} \times
\xi = 1$ by construction (circular formula). Particles never
circularized. Fixed by tracking individual orbits with the unperturbed
$(\Delta E/E)_0$ per orbit.

**Result**: 99% circularize after ~7 returns. Only ~1% direct GR
capture.

### Stage 2b: Collisionalization map (`collisionalization_map.py`)

For each $\ell$-bin: compute circularization radius $r_{\rm circ} =
\ell^2/GM$, look up ${\rm Kn}_0$ from profiles, divide by $\xi(\ell)$
to get ${\rm Kn}_{\rm eff}$. Classify: ${\rm Kn}_{\rm eff} < 1$ →
collisionalized; 1–3 → marginal; > 3 → reservoir.

**Result**: 63% collisionalized, 15% marginal, 21% reservoir.
Transition at $\ell \approx 12\,\ell_{\rm crit}$.

### Stage 2c: Reservoir build-up (`reservoir_buildup.py`)

For each reservoir $\ell$-bin: compute extra density needed for
${\rm Kn}_{\rm eff} = 1$, estimate mass and supply rate, compute
build-up time.

**Result**: total mass needed ~$6 \times 10^{-15}$ g. Build-up in
~4 free-fall times. Reservoir self-eliminates.

### Stage 3a: Kinetic closure module (`kinetic_closure.py`)

Reusable `KineticClosure` class. Given a hydro profile, computes
physical source terms $S_p(r)$ (momentum, opposes infall) and
$S_E(r)$ (energy, thermalization heating) by summing over all
$\ell$-bins with their deposition radii.

**Result**: $P_{\rm pile}/P_0 \approx 5$. Sources deposited across
148 cells from $9\,r_S$ to $r_{\rm coll}$.

### Stage 3b: Alpha scan (`stage3b_coupled.py`)

Injects Stage 3a sources into the solver with strength $\alpha$
(0 to 1). Phase 1 (80k steps): baseline. Phase 2 (100k steps):
sources on, $\alpha$ ramped over 10k steps. Sources are **frozen**
(not updated with the evolving flow).

Six runs: $\alpha = 0, 0.1, 0.25, 0.5, 0.75, 1.0$.

**Result**: smooth monotonic curve $\chi = 1.00, 0.96, 0.90, 0.81,
0.74, 0.68$. No blow-ups.

### Stage 3c: Self-consistent closure (`stage3c_selfconsistent.py`)

Like Stage 3b at $\alpha = 1$, but the closure is **updated** every
1000 steps from the current hydro state. Under-relaxed at
$\omega = 10^{-3}$ (e-folding ~1000 steps, comparable to acoustic
travel time).

**Result**: monotonic convergence to $\chi = 0.69$. Steady state by
step 160k. No oscillations.

### Stage 4: Mass scan (`stage4_mass_scan.py`)

Full Stage 3c for 5 masses: $\log M = -16.1, -16.0, -15.8, -15.6,
-15.3$.

**Result**: $\chi(M)$ correction function. ~30% at $10^{-16}$, ~5%
at $10^{-15.8}$, 0% by $10^{-15.5}$.

### Earlier experiments (superseded)

Before Stages 1–4, exploratory experiments taught key lessons:

**Option B** (`pileup_sensitivity.py`): Gaussian temperature shell
at $r_{\rm coll}$. Very sensitive (10% enhancement → 15% $\dot{M}$
drop), but blew up at $\xi \geq 2$ from interior pressure gradients.

**Sub-grid Dirichlet** (`pileup_subgrid.py`): five iterations:
- v1: $T \times \xi$ each step → compounded exponentially
- v2: Gaussian shell → drove gas wrong direction
- v3: Dirichlet BC → wild oscillations
- v4: $\omega = 0.02$ → still oscillated
- v5: $\omega = 0.001$ → damped, mean $\chi \sim 0.5$

The v5 diagnostic revealed the back-pressure **slows** the flow at
$r_{\rm sonic}$ (velocity drops to 48%), increasing bremsstrahlung
residence time. This explained why $\Theta_{\rm bc} < \xi_0$.

**Lessons learned** (applied in Stages 3b–3c):
- Never compound temperature overrides
- Never use interior Gaussian sources
- Always ramp slowly
- Match under-relaxation to acoustic travel time
- Use conservative source terms, not Dirichlet BC

---

## Files

- `docs/collisionless_pileup.tex` — formal LaTeX writeup (17 pages,
  3 figures, 7 tables)
- `docs/collisionless_pileup.pdf` — compiled PDF
- `docs/collisionless_analysis.md` — earlier detailed analysis
  (partially superseded)
- `docs/schematic_pileup.pdf` — Figure 1: pile-up geometry schematic
- `docs/kn_map.pdf` — Figure 2: Kn_eff collisionalization map
- `docs/chi_M.pdf` — Figure 3: chi(M) correction function
- `experiments/orbit_library.py` — Stage 1 orbit-averaged post-processor
- `experiments/recycling_markov.py` — Stage 2a loss-cone Markov model
- `experiments/collisionalization_map.py` — Stage 2b Kn_eff map
- `experiments/reservoir_buildup.py` — Stage 2c self-elimination timescale
- `experiments/kinetic_closure.py` — Stage 3a kinetic closure module
- `experiments/stage3b_coupled.py` — Stage 3b conservative source coupling
- `experiments/stage3c_selfconsistent.py` — Stage 3c self-consistent closure
- `experiments/stage4_mass_scan.py` — Stage 4 chi(M) mass scan
- `experiments/pileup_sensitivity.py` — Option B sensitivity mapping
- `experiments/pileup_subgrid.py` — Option D self-consistent sub-grid
