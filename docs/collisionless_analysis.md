# Collisionless inner region: self-regulation analysis

A detailed analysis of why the hydrodynamic Bondi accretion rate
survives even though the inner flow becomes collisionless. Prompted by
Andrei G. question, April 2026.

---

## The concern

For light PBHs in the solar core, the Coulomb mean free path
$\lambda_{\rm mfp}$ can exceed the local flow scale $r$ at small radii
near $r_S$, even though $\lambda_{\rm mfp} \ll r_B$ at the Bondi
radius. If the inner flow is collisionless, particles retain their
thermal angular momentum, most miss the BH on any single pass, and the
effective accretion rate could be severely reduced.

A naive estimate gives $\dot{M}_{\rm coll}/\dot{M}_B \sim (c_s/c)^2
\sim 3 \times 10^{-6}$ — a reduction by a factor $\sim 4 \times 10^5$.
This follows from the Schwarzschild capture cross-section
$\sigma_{\rm cap} = 16\pi(GM)^2/(c^2 v^2)$: averaging over a Maxwellian
gives $\langle\sigma v\rangle \propto (GM)^2/(c^2 c_s)$, while
$\dot{M}_B \propto (GM)^2/c_s^3$.

---

## Reference scales

For $M = 10^{-16}\,M_\odot$ in the solar core ($T_\infty = 1.57 \times
10^7$ K, $\rho_\infty = 150$ g/cm³, $\mu = 0.85$, $\gamma = 5/3$):

| quantity | value |
|---|---|
| $GM$ | $1.335 \times 10^{10}$ cm³/s² |
| $c_{s,\infty}$ | $5.0 \times 10^{7}$ cm/s |
| $r_B = GM/c_s^2$ | $5.3 \times 10^{-6}$ cm |
| $r_S = 2GM/c^2$ | $3.0 \times 10^{-11}$ cm |
| $\lambda_{\rm mfp}$ at ambient | $\sim 10^{-7}$ cm |

---

## 1. Adiabatic vs cooled profiles — which to use

The **adiabatic** Bondi solution for $\gamma = 5/3$ has its sonic point
at $r = 0$ — the flow is formally subsonic ($\mathcal{M} < 1$)
everywhere at finite $r$. This is the initial condition for the solver,
not the final answer.

The **cooled** steady-state solution (what the paper's code actually
computes) is qualitatively different: radiative cooling steepens the
temperature drop in the interior, lowers $c_s$, and creates a **sonic
point at finite radius**. For $M = 10^{-16}\,M_\odot$, the cooled
solution has:

$$r_{\rm sonic} \approx 780\,r_S \approx 2.3 \times 10^{-8}~\text{cm}$$

The flow is supersonic ($\mathcal{M} > 1$) below this and subsonic
above. This distinction is critical for the self-regulation argument
(§6).

Mach number profile from the converged cooled solution
(`02_paper_sweep.py`, PAPER mode, $N = 6400$):

| $r/r_S$ | $\mathcal{M}$ | $T$ [K] | $\rho$ [g/cm³] | region |
|---|---|---|---|---|
| $10^5$ | 0.24 | $2.9 \times 10^7$ | $6.4 \times 10^2$ | subsonic |
| $10^4$ | 0.68 | $1.7 \times 10^8$ | $9.4 \times 10^3$ | subsonic |
| $7000$ | 0.73 | — | — | subsonic ($\approx r_{\rm coll}$) |
| $3000$ | 0.86 | $5.2 \times 10^8$ | $5.1 \times 10^4$ | subsonic |
| $1000$ | 0.96 | $1.4 \times 10^9$ | $2.3 \times 10^5$ | subsonic |
| **780** | **1.00** | — | — | **sonic point** |
| $500$ | 1.11 | $2.6 \times 10^9$ | $5.9 \times 10^5$ | supersonic |
| $300$ | 1.33 | $3.8 \times 10^9$ | $1.1 \times 10^6$ | supersonic |
| $100$ | 2.11 | $7.0 \times 10^9$ | $4.8 \times 10^6$ | supersonic |
| $30$ | 3.08 | $1.4 \times 10^{10}$ | $2.6 \times 10^7$ | supersonic |
| $10$ | 3.91 | $3.1 \times 10^{10}$ | $1.5 \times 10^8$ | supersonic |

All subsequent analysis uses the **cooled** profiles.

---

## 2. Coulomb mean free path vs radius

The Coulomb mfp in the cooled Bondi flow, evaluated at each radius:

$$\lambda_{\rm mfp} = \frac{(k_B T)^2}{\pi\,n_e\,e^4\,\ln\Lambda}$$

with $n_e = (X + Y/2)\,\rho/m_p$ and $\ln\Lambda \approx 5$. The ratio
$\lambda_{\rm mfp}/r$ increases steeply inward (temperature rises faster
than density):

| $r/r_S$ | $\mathcal{M}$ | $\lambda_{\rm mfp}/r$ |
|---|---|---|
| $30000$ | 0.49 | 0.13 |
| $10000$ | 0.68 | 0.59 |
| **7000** | **0.73** | **1.0** |
| $5000$ | 0.78 | 1.6 |
| $3000$ | 0.86 | 3.5 |
| $1000$ | 0.96 | 17 |
| $500$ | 1.11 | 46 |
| $300$ | 1.33 | 82 |
| $100$ | 2.11 | 199 |

The collisionless transition
$\lambda_{\rm mfp}(r_{\rm coll}) = r_{\rm coll}$ occurs at:

$$r_{\rm coll} \approx 7000\,r_S \approx 2.1 \times 10^{-7}~\text{cm}$$

This is in the **subsonic** region ($\mathcal{M} = 0.73$), well above
the sonic point at $780\,r_S$.

---

## 3. Angular momentum budget at $r_{\rm coll}$

In the fluid frame, the bulk velocity is radial. Individual particles
have random transverse velocities drawn from a 2D Gaussian with
per-component dispersion $\sigma_\perp = \sqrt{k_B T/(\mu m_p)}
= c_s/\sqrt{\gamma}$.

At $r_{\rm coll} \approx 7000\,r_S$ (from the cooled profiles):

$$c_s \approx 1.9 \times 10^8~\text{cm/s},\qquad
\sigma_\perp = c_s/\sqrt{5/3} \approx 1.5 \times 10^8~\text{cm/s}$$

The transverse speed $v_\perp = \sqrt{v_\theta^2 + v_\phi^2}$ follows a
Rayleigh distribution with scale parameter $\sigma_\perp$. The
characteristic angular momentum (using the distribution scale):

$$\ell_{\rm typ} = r_{\rm coll} \times \sigma_\perp
\approx 2.1 \times 10^{-7} \times 1.5 \times 10^8
\approx 31~\text{cm}^2/\text{s}$$

The GR capture threshold for marginally bound particles on a
Schwarzschild BH (see §8 below):

$$\ell_{\rm crit} = \frac{4GM}{c}
= \frac{4 \times 1.33 \times 10^{10}}{3 \times 10^{10}} = 1.77~\text{cm}^2/\text{s}$$

$$\frac{\ell_{\rm typ}}{\ell_{\rm crit}} \approx 18 \gg 1$$

**Most particles miss the BH on any single pass.** The capture fraction
per pass (Rayleigh CDF at $v_\perp < \ell_{\rm crit}/r$):

$$f \approx \frac{\ell_{\rm crit}^2}{2\,r^2\sigma_\perp^2}
\approx 1.6 \times 10^{-3}$$

Using the deep-interior Bernoulli ($c_s^2 \approx GM/(2r)$ for
$\mathcal{M} \approx 1$) this scales as:

$$f(r) \approx \frac{8\gamma\,GM}{c^2\,r}
= \frac{40}{3}\,\frac{r_S}{r}
\approx 13\,\frac{r_S}{r}$$

(This analytic scaling is approximate; the numerical values above use
the actual cooled profiles.)

---

## 4. Recycling balance recovers $\dot{M}_B$

Particles that miss bounce back to $r > r_{\rm coll}$, re-thermalize
(fast — the gas is collisional there), and fall in again. In steady
state, mass balance at $r_{\rm coll}$:

$$J_{\rm down} = J_{\rm Bondi} + J_{\rm bounce}$$
$$J_{\rm bounce} = (1 - f)\,J_{\rm down}$$
$$J_{\rm acc} = f\,J_{\rm down}$$

Solving:

$$J_{\rm down} = \frac{J_{\rm Bondi}}{f},\qquad
\dot{M}_{\rm acc} = f \times \frac{\dot{M}_B}{f} = \dot{M}_B$$

**The Bondi rate is recovered regardless of how small $f$ is.**
Particles cycle $\sim 1/f$ times before capture, but the net throughput
equals the supply. This is analogous to a sieve with small holes: the
water level rises behind the sieve until the flow rate through the holes
matches the tap.

This derivation assumes $J_{\rm Bondi}$ is not modified by the pile-up
— i.e., the back-pressure from the density enhancement cannot
propagate upstream and choke the inflow. This is addressed in §6.

---

## 5. Self-consistent transition radius

The recycling amplifies the local density by $\eta \approx 1/f$. This
shortens $\lambda_{\rm mfp}$ and pushes the effective transition inward.
The self-consistent $r_*$ satisfies
$\lambda_{\rm mfp}(r_*,\,\eta\,\rho) = r_*$.

### Analytic estimate (adiabatic scaling)

Using the adiabatic power-law profiles
($\lambda_{\rm mfp} \propto r^{-1/2}$, $f \approx 13\,r_S/r$):

$$r_*^{5/2} = 13\,C\,r_S \implies r_* = (13\,C\,r_S)^{2/5}
\approx 2.1 \times 10^{-8}~\text{cm} \approx 700\,r_S$$

### Numerical result (cooled profiles)

Solving $\lambda_{\rm mfp}(r) \times f(r) = r$ using the actual cooled
$T(r)$, $\rho(r)$ profiles (where $f(r)$ is computed from the local
$\sigma_\perp$ and $\ell_{\rm crit}$):

$$\boxed{r_* \approx 550\,r_S \approx 1.6 \times 10^{-8}~\text{cm}}$$

| quantity | initial ($r_{\rm coll}$) | $\ell$-only ($r_*$) | with brems. (§7) |
|---|---|---|---|
| radius | $2.1 \times 10^{-7}$ cm | $1.6 \times 10^{-8}$ cm | see §7 |
| $r/r_S$ | $7000$ | $550$ | $\sim 1400$ |
| $\mathcal{M}$ | 0.73 (subsonic) | 1.08 (supersonic) | $\sim 0.94$ |
| density enhancement $\eta$ | 1 | ~40 | **5–13** |
| capture fraction $f$ | 0.16% | ~2.5% | 8–20% |
| $\ell_{\rm typ}/\ell_{\rm crit}$ | 18 | ~4.5 | — |
| $N_{\rm orbits}$ per particle | — | ~40 | **3–7** |
| **net accretion rate** | $\dot{M}_B$ | $\dot{M}_B$ | $\dot{M}_B$ |

The pure-angular-momentum estimate (middle column) over-predicts the
pile-up because it treats orbits as conservative. In reality,
bremsstrahlung drains a significant fraction of the orbital energy per
pass (§7), limiting $\eta$ to 5–13. The right column gives the
self-consistent values with energy dissipation included.

---

## 6. The cooling-induced sonic point (context for §7)

An important structural feature of the cooled flow, though **not** the
primary argument for why $\dot{M}_B$ is preserved (that comes from the
two-stream and net-flux arguments in §7).

**Adiabatic profiles ($\gamma = 5/3$):** the sonic point is at $r = 0$.
The flow is subsonic everywhere at finite $r$. At $r_{\rm coll}
\approx 7000\,r_S$, $\mathcal{M} = 0.73$ — perturbations propagate
upstream with signal speed $0.27\,c_s$.

**Cooled profiles:** radiative cooling creates a sonic point at
$r_{\rm sonic} \approx 780\,r_S$. Below this, the flow is supersonic
and causally disconnected from the upstream.

In the pure angular-momentum analysis (§5, middle column of the table),
the self-consistent $r_* \approx 550\,r_S$ sits below the sonic point
in the supersonic region — so causal disconnection alone would protect
$\dot{M}_B$. However, when bremsstrahlung energy losses are included
(§7), the pile-up is smaller ($\eta \sim 5$–$13$ instead of $\sim 40$)
and the revised transition stays at $\sim 1400\,r_S$ — **above** the
sonic point ($\mathcal{M} \approx 0.94$). Causal disconnection alone
does not suffice.

The accretion rate is instead protected by the two-stream nature of the
collisionless zone and the net-flux conservation at $r_{\rm coll}$
(§7.4–§7.5). The sonic point remains relevant as a structural feature
— it divides the collisionless zone into a subsonic outer part (two
interpenetrating streams) and a supersonic inner part (ballistic
orbits, rapid bremsstrahlung spiral-in) — but the argument for
$\dot{M}_{\rm acc} = \dot{M}_B$ does not depend on where $r_*$ sits
relative to $r_{\rm sonic}$.

---

## 7. Bremsstrahlung limits the pile-up

The pure angular-momentum analysis (§3–§5) assumed conservative orbits
— particles bounce indefinitely until they happen to have
$\ell < \ell_{\rm crit}$. Since capture requires both transverse
velocity components to be small (a tiny circle of radius
$\ell_{\rm crit}/r$ in 2D velocity space), the capture fraction scales
as $f \sim (\ell_{\rm crit}/\ell_{\rm typ})^2 \sim 1/18^2 \sim
3 \times 10^{-3}$, giving a pile-up $\eta = 2/f \sim 700$ with
conservative orbits. This over-predicts the pile-up. In reality,
**bremsstrahlung drains a significant fraction of the orbital energy on
each pass**, causing particles to spiral in regardless of their angular
momentum and cutting $\eta$ from $\sim 700$ to $\sim 5$–$13$.

### 7.1 Energy loss per orbit

A particle passing through radius $r$ in a gas of density $\rho$ loses
energy at a per-unit-mass rate $\varepsilon/\rho$, where $\varepsilon$
is the bremsstrahlung emissivity ($\propto \rho^2$ at fixed $T$). The
fractional energy loss per pericenter passage, spending time
$\sim r/v$ near the densest region:

$$\left(\frac{\Delta E}{E}\right)_{\!\rm Bondi}
\approx \frac{\varepsilon}{\rho}\,\frac{r}{v}\,\frac{1}{\tfrac{1}{2}v^2}$$

evaluated at the unperturbed Bondi density. From the converged cooled
profiles ($M = 10^{-16}\,M_\odot$):

| $r/r_S$ | $\mathcal{M}$ | $(\Delta E/E)_{\rm Bondi}$ |
|---|---|---|
| 3000 | 0.86 | 1.1% |
| 1000 | 0.96 | 1.3% |
| 550 | 1.08 | 8.0% |
| 300 | 1.33 | 21% |

These are NOT small. At the innermost radii, a single pass radiates a
substantial fraction of the orbital energy.

### 7.2 Self-consistent pile-up with energy dissipation

With density enhanced by $\eta$, the per-particle cooling rate scales
as $\eta$ (emissivity $\propto \rho^2$; per-particle rate
$\propto \rho$). The number of orbits before a particle spirals in is:

$$N_{\rm orbits} = \frac{1}{\eta\,(\Delta E/E)_{\rm Bondi}}$$

The density enhancement is $\eta \approx 2\,N_{\rm orbits}$ (inbound +
outbound streams at each radius). Self-consistency gives:

$$\eta = 2N = \frac{2}{\eta\,(\Delta E/E)_0}
\implies \boxed{\eta = \sqrt{\frac{2}{(\Delta E/E)_0}}}$$

| $r/r_S$ | $(\Delta E/E)_0$ | $\eta$ | $N_{\rm orbits}$ |
|---|---|---|---|
| 3000 | 1.1% | 13 | 7 |
| 1000 | 1.3% | 12 | 6 |
| 550 | 8.0% | 5 | 3 |
| 300 | 21% | 3 | 2 |

The pile-up is **self-limiting**: if $\eta$ grows, bremsstrahlung per
orbit increases, particles spiral in faster, and $\eta$ drops.
Bremsstrahlung acts as negative feedback on the pile-up.

### 7.3 Revised collisionless transition

With $\eta \approx 10$ (representative of the 1000–3000 $r_S$ region),
the enhanced density shortens $\lambda_{\rm mfp}$ by $10\times$. The
revised transition $\lambda_{\rm mfp}/(10\,r) = 1$ occurs at:

$$r_{\rm coll}^{\rm (revised)} \approx 1400\,r_S \qquad (\mathcal{M} \approx 0.94)$$

This is above the sonic point ($780\,r_S$) — the pile-up does NOT
cross into the supersonic region when bremsstrahlung is included. The
causal disconnection argument alone does not protect the Bondi rate.

Instead, the accretion rate is protected by the **two-stream** nature
of the collisionless zone and the **net-flux conservation** at
$r_{\rm coll}$ (§7.4 and §7.5).

### 7.4 Two-stream pressure

Between $r_{\rm coll}$ and the sonic point, the collisionless zone
contains two interpenetrating streams:

- **Inbound**: the Bondi flow, carrying mass flux $\dot{M}_B/f$
  (amplified by recycling).
- **Outbound**: bouncing particles, carrying $(1-f)\dot{M}_B/f$.

These streams **do not interact** ($\lambda_{\rm mfp} > r$ by
definition). The outbound stream adds kinetic density but NOT
thermodynamic pressure — it is a separate collisionless population.
The fluid pressure that the Bondi solution sees (and that could
communicate upstream) is the inbound stream's pressure alone, which is
the unperturbed Bondi value.

This is analogous to two beams of particles crossing in a vacuum: the
density at the crossing point is doubled, but neither beam exerts a
pressure force on the other.

### 7.5 Net flux conservation at $r_{\rm coll}$

At the collisionless boundary, returning particles re-thermalize
(over a distance $\sim \lambda_{\rm mfp} \sim r_{\rm coll}$) and rejoin
the inbound flow. In steady state:

- **Net mass flux**: $\dot{M}_{\rm in} - \dot{M}_{\rm out} = \dot{M}_B$
  (recycled mass cancels exactly).
- **Net energy flux**: Bondi energy flux minus the radiated luminosity
  $L$. The luminosity is what we compute; it is already accounted for
  by the cooling source term in the 1D code.
- **No net heating**: a returning particle left $r_{\rm coll}$ with
  kinetic energy $\frac{1}{2}v^2$ and returns with $\frac{1}{2}v^2
  - \Delta E_{\rm brems}$. The deficit $\Delta E_{\rm brems}$ was
  radiated as bremsstrahlung — it is the luminosity, not a heating
  term.

The Bondi flow at $r > r_{\rm coll}$ sees its own unperturbed mass
flux, pressure, and energy flux. The pile-up is invisible to it.

---

## 8. Derivation of $\ell_{\rm crit}$

For completeness: the GR capture threshold comes from the Schwarzschild
effective potential. A test particle with specific angular momentum
$\ell$ sees:

$$V_{\rm eff}(r) = -\frac{GM}{r} + \frac{\ell^2}{2r^2}
- \frac{GM\ell^2}{c^2\,r^3}$$

The last term is purely relativistic. It grows faster than the
centrifugal barrier at small $r$, destroying the turning point that
would otherwise deflect the particle.

**In Newtonian gravity there is no capture** — the centrifugal term
always wins at small $r$ and the particle swings past. It is the GR
$-1/r^3$ term that enables accretion of particles with nonzero $\ell$.

The barrier disappears when $V_{\rm eff}$ has no maximum. Setting
$dV_{\rm eff}/dr = 0$ gives a quadratic in $r$:

$$GM\,r^2 - \ell^2\,r + \frac{3GM\ell^2}{c^2} = 0$$

Real solutions (barrier exists) require $\ell^4 \geq 12\,G^2M^2
\ell^2/c^2$, i.e., $\ell \geq 2\sqrt{3}\,GM/c \approx 3.46\,GM/c$.
Below this value no barrier exists regardless of energy.

For marginally bound particles ($E \approx mc^2$, non-relativistic at
infinity — our case), the critical angular momentum is:

$$\ell_{\rm crit} = \frac{4GM}{c}$$

Our particles have $v \sim c_s \sim 10^{-3}c$ at $r_*$, so the
non-relativistic approximation is excellent.

---

## 9. What the paper's code gets right and wrong

**Correct:**
- The Bondi accretion rate $\dot{M}_B$ (set at $r_B$; pile-up at
  $r < r_{\rm coll}$ is invisible to the upstream flow via the
  two-stream and net-flux-conservation arguments of §7.4–§7.5).
- The dominance of bremsstrahlung over pair production in this regime.
- The qualitative density and velocity scalings in the deep interior
  (the free-fall Bondi profile and a collisionless free-fall profile
  give the same $\rho \propto r^{-3/2}$, $v \propto r^{-1/2}$).

**Approximate:**
- The density at $r < r_{\rm coll}$ is enhanced by a modest factor
  $\eta \approx 5$–$13$ (§7.2) due to angular-momentum recycling,
  limited by bremsstrahlung energy dissipation. The 1D code uses the
  unenhanced Bondi profile. Since $\varepsilon \propto n^2$, the
  emissivity could be boosted by $\eta^2 \sim 25$–$170$ locally.
  However, the luminosity integral is dominated by the hottest regions
  near $r_S$, and the volume element $r^2\,dr$ suppresses small-$r$
  contributions. The net effect is an $\mathcal{O}(1)$ uncertainty in
  $L$ — the 1D code likely **underestimates** $L$ (conservative).
- The velocity distribution at $r < r_{\rm coll}$ is not a local
  Maxwellian. The bremsstrahlung emissivity depends on relative
  velocities of colliding pairs, so the non-thermal distribution could
  modify the emissivity by $\mathcal{O}(1)$ factors.

**Net assessment:** the accretion rate is correct. The luminosity has an
$\mathcal{O}(1)$ uncertainty from the density enhancement and
non-thermal distribution, but is not wrong by orders of magnitude. The
$\eta \sim 10^{-2}$ quoted for this regime is robust in order of
magnitude.

---

## 10. Implications for the paper

1. **Rename the regime?** "Collisionless" invites the misreading that
   $\dot{M}$ is reduced. "Bremsstrahlung-dominated" or
   "sub-thermalized" describes the physics without suggesting the fluid
   picture is broken. TBD

2. **Add a clarifying paragraph.** Suggested text:

   > At radii where the Coulomb mean free path exceeds the local flow
   > scale ($r \lesssim 7000\,r_S$ for $M = 10^{-16}\,M_\odot$),
   > individual particles retain their thermal angular momentum and
   > most miss the GR capture cross-section on each inward pass
   > ($\ell_{\rm typ}/\ell_{\rm crit} \approx 18$). However, particles
   > that miss return to the collisional region, re-thermalize, and
   > fall in again. The resulting density enhancement is self-limited
   > by bremsstrahlung: at the enhanced density, each pericenter
   > passage drains a significant fraction of the orbital energy,
   > causing particles to spiral in after
   > $N \sim \sqrt{2/(\Delta E/E)_0} \approx 5$–$13$ orbits (where
   > $(\Delta E/E)_0$ is the fractional energy loss at the unperturbed
   > Bondi density). The self-consistent pile-up factor
   > $\eta = 2N$ is modest (5–13×), and the collisionless zone
   > contains two interpenetrating streams (inbound Bondi flow and
   > outbound bouncing particles) that do not interact as a single
   > fluid. The net mass and energy fluxes at the collisionless
   > boundary are therefore unmodified from the Bondi values, and the
   > accretion rate $\dot{M}_B$ is preserved. The luminosity may be
   > enhanced by an $\mathcal{O}(1)$ factor from the elevated inner
   > density, but the 1-D calculation is conservative (it
   > underestimates $\varepsilon \propto \rho^2$).

3. **Density enhancement caveat.** Note that the luminosity in the
   bremsstrahlung-dominated regime carries an $\mathcal{O}(1)$
   uncertainty from the enhanced inner density. This does not affect
   $\eta$ at the order-of-magnitude level but should be flagged.

4. **Regime boundary.** The hydro assumption genuinely breaks down at
   $M \lesssim 10^{-18}\,M_\odot$ (where $r_B \lesssim
   \lambda_{\rm mfp}$). Below this mass the full collisionless capture
   rate applies and $\dot{M}$ drops by $(c_s/c)^2 \sim 3 \times 10^{-6}$.
   This is outside the paper's mass range.

---

## Summary of the argument

```
  r_B (5.3e-6 cm)         Bondi rate set here. lambda_mfp << r_B.
       |                   Flow is deeply subsonic (M ~ 0.1).
       |
       |  Collisional, hydrodynamic Bondi flow.
       |  The pile-up below cannot modify this region:
       |  net mass flux = Mdot_B, net energy source = 0.
       |
  r_coll (7000 r_S)       lambda_mfp = r. M = 0.73 (subsonic).
       |                   Particles become collisionless below.
       |                   Returning particles thermalize here.
       |                   NET mass and energy flux = Bondi values.
       |
       |  Two interpenetrating streams (inbound + outbound).
       |  They do NOT interact — no single-fluid pressure.
       |  Density enhanced ~10x, but two-stream, not thermalized.
       |
  r_sonic (780 r_S)        Cooling-induced sonic point. M = 1.
       |
       |  Supersonic. Particles on ballistic orbits.
       |  Bremsstrahlung drains ~1-20% of KE per pass.
       |  Spiral-in after ~3-7 orbits.
       |
  r_S (1 r_S)             GR capture for ell < ell_crit = 4GM/c.
```

**Three mechanisms protect $\dot{M}_B$:**

1. **Bremsstrahlung self-limits the pile-up** ($\eta \approx 5$–$13$,
   not the $\sim 40$ from a conservative-orbit estimate).
2. **Two-stream pressure** — collisionless inbound and outbound
   streams do not exert thermodynamic pressure on each other.
3. **Net flux conservation** — at $r_{\rm coll}$, steady-state
   recycling produces zero net mass source and zero net energy source.

---

## References

- Bondi, H. 1952, MNRAS, 112, 195.
- Shapiro, S. L. & Teukolsky, S. A. 1983, *Black Holes, White Dwarfs,
  and Neutron Stars* — GR capture cross-sections.
- The self-regulation / recycling argument is structurally similar to
  loss-cone refilling in stellar-dynamical contexts (Lightman & Shapiro
  1977, ApJ, 211, 244), adapted here to a thermal gas with a Coulomb
  collisionality transition.
