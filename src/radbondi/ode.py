"""Three-zone ODE shooting solver for steady-state Bondi accretion with cooling.

This is a port of Cantiello et al.'s ``sonic_solver.py``. It complements the
default time-dependent solver (:class:`radbondi.BondiProblem`) by providing an
ODE-based reference solution in regimes where shooting works — typically the
collisionless / weak-cooling regime where the sonic point is a saddle.

The strategy splits the domain into three radial zones:

* **Outer (x > x_born_start)** — pure adiabatic Bondi. The gas velocity is
  small at large radii and cooling is dynamically negligible.
* **Intermediate (x_born_start > x > x_match)** — Born iteration. The
  coupled (v, T) ODEs are integrated inward with the emissivity evaluated on
  a *reference* profile (initially adiabatic, refined per iteration). This
  breaks the v-rho-eps feedback loop that would otherwise destabilize the
  ODE.
* **Inner (x < x_match)** — ballistic v(r) plus an explicit RK4 integration
  of the temperature equation. Cooling is no longer a small perturbation
  here, so v is prescribed analytically from energy conservation.

When cooling becomes strong (transitional / isothermal regimes), the ODE
shooting method fails because the sonic point becomes a focus rather than a
saddle. Use the time-dependent solver in that regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from radbondi.bondi import adiabatic_profile
from radbondi.constants import G, c_light, kB, m_p
from radbondi.solution import Solution


@dataclass
class ODESolverConfig:
    """Configuration for :func:`solve_ode`.

    Attributes
    ----------
    x_outer, x_inner : float
        Outer and inner boundaries of the integration domain in units of r_B.
    x_match : float
        Matching point between the Born and ballistic zones (in r_B units).
    x_born_start : float
        Initial guess for the outer edge of the Born zone (in r_B units).
        Reduced adaptively if the cooling correction is too large there.
    n_iterations : int
        Number of Born refinement iterations.
    n_profile_points : int
        Number of points per zone in the assembled output profile.
    n_inner_steps : int
        Number of explicit RK4 steps for the inner T equation.
    rtol, atol : float
        Tolerances for ``scipy.integrate.solve_ivp`` in the Born zone.
    gr_redshift : bool
        If True, apply the gravitational-redshift factor (1 - r_S/r) to the
        luminosity integral.
    verbose : bool
        Print progress to stdout.
    """

    x_outer: float = 50.0
    x_inner: float = 1e-5
    x_match: float = 1e-2
    x_born_start: float = 0.5
    n_iterations: int = 3
    n_profile_points: int = 1000
    n_inner_steps: int = 5000
    rtol: float = 1e-8
    atol: float = 1e-10
    gr_redshift: bool = True
    verbose: bool = True


# ── Helpers (private) ────────────────────────────────────────────────────


def _scalar_eps(cooling, rho, T, ambient, eps_ambient):
    """Wrap cooling.net_emissivity for scalar inputs (returns Python float)."""
    out = cooling.net_emissivity(
        np.array([rho]), np.array([T]), ambient, eps_ambient
    )
    return float(out[0])


def _cs2(T, ambient):
    """Sound speed squared at temperature T given the ambient mu, gamma."""
    return ambient.gamma * kB * T / (ambient.mu * m_p)


def _rho_from_continuity(r, v, Mdot):
    return Mdot / (4.0 * np.pi * r**2 * v)


def _adiabatic_term(v, T, r, M_BH, ambient):
    return v * abs(2.0 * _cs2(T, ambient) - G * M_BH / r)


def _pick_x_born_start(M_BH, ambient, cooling, eps_ambient,
                       x_match, x_initial=0.5, threshold=0.05):
    """Pick the outer edge of the Born zone adaptively.

    Returns the largest x in a fixed candidate list for which the cooling
    correction term is below ``threshold`` of the adiabatic dynamical term.
    """
    candidates = [x_initial, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.04,
                  0.03, 0.02, 0.015, 2.0 * x_match, 1.5 * x_match]
    for x_try in candidates:
        if x_try <= x_match:
            continue
        v, T, rho, _ = adiabatic_profile(np.array([x_try]), ambient)
        v_t = float(v[0])
        T_t = float(T[0])
        rho_t = float(rho[0])
        r_t = x_try * G * M_BH / ambient.cs**2
        eps_t = _scalar_eps(cooling, rho_t, T_t, ambient, eps_ambient)
        adiab = _adiabatic_term(v_t, T_t, r_t, M_BH, ambient)
        if adiab <= 0:
            continue
        corr = r_t * (ambient.gamma - 1.0) * eps_t / rho_t
        if abs(corr / adiab) < threshold:
            return x_try
    return x_match  # fallback: skip Born zone entirely


def _build_eps_table(M_BH, ambient, cooling, eps_ambient,
                     x_inner, x_outer, n_grid=3000):
    """Return an interpolator eps_ad(ln r) on the adiabatic profile.

    Used as the initial reference for Born iteration.
    """
    r_B = G * M_BH / ambient.cs**2
    x_grid = np.geomspace(x_inner, x_outer, n_grid)[::-1]   # outer -> inner
    v_ad, T_ad, rho_ad, _ = adiabatic_profile(x_grid, ambient)
    lnr_grid = np.log(x_grid * r_B)
    eps_ad = cooling.net_emissivity(rho_ad, T_ad, ambient, eps_ambient)
    return interp1d(lnr_grid, eps_ad, fill_value="extrapolate")


def _make_born_rhs(M_BH, ambient, eps_interp):
    """Build the Born-iteration RHS: (v, T) ODE with eps from a reference."""
    gam = ambient.gamma
    mu = ambient.mu

    def rhs(lnr, y, Mdot):
        v, T = y
        if v <= 0 or T <= 0:
            return [0.0, 0.0]
        r = np.exp(lnr)
        rho = _rho_from_continuity(r, v, Mdot)
        eps = max(float(eps_interp(lnr)), 0.0)
        c2 = _cs2(T, ambient)
        D = v**2 - c2
        N = v * (2.0 * c2 - G * M_BH / r) - r * (gam - 1.0) * eps / rho
        dv_dlnr = N / D
        dT_dlnr = (
            -(gam - 1.0) * T * (2.0 + dv_dlnr / v)
            + (gam - 1.0) * mu * m_p * r * eps / (kB * rho * v)
        )
        return [dv_dlnr, dT_dlnr]

    return rhs


def _ballistic_v(r, v_match, r_match, M_BH):
    """Ballistic velocity from energy conservation: v^2 = v_m^2 + 2GM(1/r-1/r_m)."""
    v2 = v_match**2 + 2.0 * G * M_BH * (1.0 / r - 1.0 / r_match)
    return np.sqrt(np.maximum(v2, 0.0))


def _make_inner_rhs(M_BH, ambient, cooling, eps_ambient, v_match, r_match, Mdot):
    """RK4-driven dT/dlnr in the inner ballistic zone."""
    gam = ambient.gamma
    mu = ambient.mu
    T_floor = ambient.T

    def rhs(lnr, T):
        T = max(T, T_floor)
        r = np.exp(lnr)
        v2 = v_match**2 + 2.0 * G * M_BH * (1.0 / r - 1.0 / r_match)
        if v2 <= 0:
            return 0.0
        v = np.sqrt(v2)
        dv_dlnr = G * M_BH / (r * v)        # ballistic: v dv/dr = GM/r^2
        rho = _rho_from_continuity(r, v, Mdot)
        eps = _scalar_eps(cooling, rho, T, ambient, eps_ambient)
        return (
            -(gam - 1.0) * T * (2.0 + dv_dlnr / v)
            + (gam - 1.0) * mu * m_p * r * eps / (kB * rho * v)
        )

    return rhs


def _integrate_inner_T(rhs, T0, lnr_start, lnr_end, n_steps, T_floor):
    """Explicit RK4 for the inner T equation (more robust than implicit)."""
    lnr = np.linspace(lnr_start, lnr_end, n_steps)
    dlnr = lnr[1] - lnr[0]
    T = np.empty(n_steps)
    T[0] = T0
    for i in range(1, n_steps):
        Ti = T[i - 1]
        ln = lnr[i - 1]
        k1 = rhs(ln, Ti)
        k2 = rhs(ln + 0.5 * dlnr, max(Ti + 0.5 * dlnr * k1, T_floor))
        k3 = rhs(ln + 0.5 * dlnr, max(Ti + 0.5 * dlnr * k2, T_floor))
        k4 = rhs(ln + dlnr, max(Ti + dlnr * k3, T_floor))
        T[i] = max(Ti + dlnr * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, T_floor)
    return lnr, T


def _stitch_zones(M_BH, ambient, cooling, eps_ambient, Mdot,
                  x_outer, x_match, x_born_start, born_ok, sol_born,
                  r_match, v_match, lnr_z3, T_z3, n_pts):
    """Assemble final r, v, T profile from the three zones."""
    r_B = G * M_BH / ambient.cs**2
    # Zone 1: adiabatic outer
    x_z1 = np.geomspace(x_outer, max(x_born_start, x_match), n_pts)
    v_z1, T_z1, _, _ = adiabatic_profile(x_z1, ambient)
    r_z1 = x_z1 * r_B

    # Zone 2: Born intermediate (if successful and non-degenerate)
    if born_ok and x_born_start > x_match * 1.1:
        lnr_z2 = np.linspace(sol_born.t[0], sol_born.t[-1], n_pts)
        y_z2 = sol_born.sol(lnr_z2)
        r_z2 = np.exp(lnr_z2)
        v_z2 = y_z2[0]
        T_z2 = y_z2[1]
    else:
        r_z2 = v_z2 = T_z2 = np.array([])

    # Zone 3: ballistic + T inner
    r_z3 = np.exp(lnr_z3)
    v_z3 = _ballistic_v(r_z3, v_match, r_match, M_BH)

    parts_r = [r_z1]
    parts_v = [v_z1]
    parts_T = [T_z1]
    if r_z2.size > 0:
        parts_r.append(r_z2[1:])
        parts_v.append(v_z2[1:])
        parts_T.append(T_z2[1:])
    parts_r.append(r_z3[1:])
    parts_v.append(v_z3[1:])
    parts_T.append(T_z3[1:])

    r_all = np.concatenate(parts_r)
    v_all = np.concatenate(parts_v)
    T_all = np.concatenate(parts_T)

    # Order from inner (small r) to outer (large r) for downstream consistency
    order = np.argsort(r_all)
    r_all = r_all[order]
    v_all = v_all[order]
    T_all = T_all[order]
    rho_all = _rho_from_continuity(r_all, v_all, Mdot)
    return r_all, v_all, T_all, rho_all


def _luminosity_integral(r, rho, T, M_BH, ambient, cooling, eps_ambient,
                         gr_redshift):
    """Trapezoid integral of net emissivity from r_S to r_B."""
    r_S = 2.0 * G * M_BH / c_light**2
    r_B = G * M_BH / ambient.cs**2
    mask = (r >= r_S) & (r <= r_B)
    rm = r[mask]
    if rm.size < 2:
        return 0.0
    rhom = rho[mask]
    Tm = T[mask]
    epsm = cooling.net_emissivity(rhom, Tm, ambient, eps_ambient)
    dr = np.abs(np.diff(rm))
    r_mid = 0.5 * (rm[:-1] + rm[1:])
    eps_mid = 0.5 * (epsm[:-1] + epsm[1:])
    integrand = 4.0 * np.pi * r_mid**2 * eps_mid * dr
    if gr_redshift:
        integrand *= np.maximum(1.0 - r_S / r_mid, 0.0)
    return float(np.sum(integrand))


# ── Public API ───────────────────────────────────────────────────────────


def solve_ode(problem, config: ODESolverConfig | None = None) -> Solution:
    """Solve a :class:`BondiProblem` with the three-zone ODE shooting method.

    Returns a :class:`~radbondi.Solution` API-compatible with the output of
    :meth:`BondiProblem.solve`.

    Parameters
    ----------
    problem : BondiProblem
        Defines M_BH, ambient, and cooling.
    config : ODESolverConfig, optional
        Solver configuration (defaults are reasonable for collisionless and
        weak-cooling regimes).

    Notes
    -----
    The ODE method is reliable in the regime where the sonic point is a
    saddle (typically ``M_BH <= 1e-14 M_sun`` for solar-core conditions).
    When cooling becomes strong, the sonic point becomes a focus (complex
    eigenvalues) and the Born iteration may diverge — use
    :meth:`BondiProblem.solve` (the time-dependent solver) instead.
    """
    if config is None:
        config = ODESolverConfig()
    cfg = config

    ambient = problem.ambient
    cooling = problem.cooling
    M_BH = problem.M_BH
    r_B = problem.r_B
    Mdot = problem.Mdot_B
    eps_ambient = cooling.ambient_emissivity(ambient) if cooling.processes else 0.0

    if cfg.verbose:
        print("=" * 65)
        print(f"radbondi ODE solver: M_BH = {M_BH:.3e} g")
        print("=" * 65)
        print(f"  r_B/r_S = {r_B / problem.r_S:.2e},  Mdot_B = {Mdot:.3e} g/s")

    # ── Adaptive Born start ──────────────────────────────────────────────
    x_born_start = _pick_x_born_start(
        M_BH, ambient, cooling, eps_ambient,
        x_match=cfg.x_match, x_initial=cfg.x_born_start,
    )

    # ── Born iteration ───────────────────────────────────────────────────
    born_ok = (x_born_start > cfg.x_match * 1.1)
    sol_born = None

    if born_ok:
        eps_interp = _build_eps_table(M_BH, ambient, cooling, eps_ambient,
                                      cfg.x_inner, cfg.x_outer)
        v_start, T_start, _, M_start = adiabatic_profile(
            np.array([x_born_start]), ambient
        )
        v0, T0 = float(v_start[0]), float(T_start[0])
        r_born_start = x_born_start * r_B
        r_match_phys = cfg.x_match * r_B

        if cfg.verbose:
            print(f"\n--- Born approximation (x={x_born_start:.3e} -> "
                  f"{cfg.x_match}) ---")
            print(f"  BC at outer edge: T={T0:.3e} K, v={v0:.3e}, "
                  f"Mach={float(M_start[0]):.3f}")

        for it in range(cfg.n_iterations):
            rhs = _make_born_rhs(M_BH, ambient, eps_interp)
            sol_born = solve_ivp(
                rhs,
                (np.log(r_born_start), np.log(r_match_phys)),
                [v0, T0], args=(Mdot,),
                method="Radau", rtol=cfg.rtol, atol=cfg.atol,
                dense_output=True, max_step=0.2,
            )
            if not sol_born.success or sol_born.y[0, -1] <= 0:
                if cfg.verbose:
                    print(f"  Iter {it + 1}: Born integration failed")
                born_ok = False
                break

            # Refine the reference epsilon
            n_eval = max(2000, len(sol_born.t) * 10)
            lnr_eval = np.linspace(sol_born.t[0], sol_born.t[-1], n_eval)
            y_eval = sol_born.sol(lnr_eval)
            v_new = np.maximum(y_eval[0], 1e-10)
            T_new = np.maximum(y_eval[1], ambient.T)
            rho_new = _rho_from_continuity(np.exp(lnr_eval), v_new, Mdot)
            eps_new = cooling.net_emissivity(rho_new, T_new, ambient, eps_ambient)
            eps_interp = interp1d(lnr_eval, eps_new, fill_value="extrapolate")

            if cfg.verbose:
                vm = sol_born.y[0, -1]
                Tm = sol_born.y[1, -1]
                print(f"  Iter {it + 1}: Mach_match={vm / np.sqrt(_cs2(Tm, ambient)):.3f}")

    # Matching values
    if born_ok and sol_born is not None and x_born_start > cfg.x_match:
        r_match = float(np.exp(sol_born.t[-1]))
        v_match = float(sol_born.y[0, -1])
        T_match = float(sol_born.y[1, -1])
    else:
        v_ad, T_ad, _, _ = adiabatic_profile(np.array([cfg.x_match]), ambient)
        r_match = cfg.x_match * r_B
        v_match = float(v_ad[0])
        T_match = float(T_ad[0])
    mach_match = v_match / np.sqrt(_cs2(T_match, ambient))

    # ── Inner ballistic + T equation ────────────────────────────────────
    r_S = problem.r_S
    r_inner_target = max(1.01 * r_S, cfg.x_inner * r_B)
    inner_rhs = _make_inner_rhs(M_BH, ambient, cooling, eps_ambient,
                                v_match, r_match, Mdot)
    lnr_z3, T_z3 = _integrate_inner_T(
        inner_rhs, T_match,
        np.log(r_match), np.log(r_inner_target),
        cfg.n_inner_steps, ambient.T,
    )
    if cfg.verbose:
        x_inner_final = np.exp(lnr_z3[-1]) / r_B
        v_inner = _ballistic_v(np.exp(lnr_z3[-1]), v_match, r_match, M_BH)
        cs_inner = np.sqrt(_cs2(T_z3[-1], ambient))
        print(f"\n--- Inner ballistic (x={cfg.x_match} -> {x_inner_final:.2e}) ---")
        print(f"  At inner edge: T={T_z3[-1]:.3e} K, "
              f"Mach={v_inner / cs_inner:.1f}")

    # ── Stitch zones into one profile ───────────────────────────────────
    r_all, v_all, T_all, rho_all = _stitch_zones(
        M_BH, ambient, cooling, eps_ambient, Mdot,
        cfg.x_outer, cfg.x_match, x_born_start, born_ok, sol_born,
        r_match, v_match, lnr_z3, T_z3, cfg.n_profile_points,
    )
    cs_all = np.sqrt(_cs2(T_all, ambient))
    P_all = rho_all * kB * T_all / (ambient.mu * m_p)
    Mach_all = np.abs(v_all) / cs_all

    # Conservatives (use negative v for infall, matching the time-dep solver)
    v_signed = -np.abs(v_all)
    E_all = 0.5 * rho_all * v_signed**2 + P_all / (ambient.gamma - 1.0)
    U = np.array([rho_all, rho_all * v_signed, E_all])

    # Luminosity
    L = _luminosity_integral(r_all, rho_all, T_all, M_BH, ambient, cooling,
                             eps_ambient, cfg.gr_redshift)

    return Solution(
        r=r_all, r_B=r_B,
        rho=rho_all, v=v_signed, P=P_all, T=T_all, Mach=Mach_all,
        U=U, L=L,
        residuals=np.array([]),
        converged=bool(born_ok or x_born_start <= cfg.x_match),
        M_BH=M_BH, Mdot_B=Mdot,
        ambient_T=ambient.T, ambient_rho=ambient.rho, ambient_mu=ambient.mu,
        ambient_gamma=ambient.gamma, ambient_X=ambient.X, ambient_Y=ambient.Y,
        metadata={
            "method": "ode",
            "x_match_actual": r_match / r_B,
            "x_born_start": x_born_start,
            "mach_match": mach_match,
            "born_zone_used": bool(born_ok and x_born_start > cfg.x_match * 1.1),
        },
    )
