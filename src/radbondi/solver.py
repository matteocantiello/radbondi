"""Time-dependent Bondi+cooling solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radbondi.ambient import AmbientMedium
from radbondi.bondi import (
    adiabatic_profile,
    bondi_radius,
    bondi_rate,
    schwarzschild_radius,
)
from radbondi.constants import kB, m_p
from radbondi.cooling import Cooling
from radbondi.grid import Grid
from radbondi.hydro import get_primitives, hydro_rhs
from radbondi.solution import Solution


@dataclass
class SolverConfig:
    """Configuration for :meth:`BondiProblem.solve`.

    Attributes
    ----------
    N : int
        Number of radial cells.
    x_min, x_max : float
        Inner / outer grid boundaries in units of r_B.
    n_steps : int
        Maximum number of integration steps.
    CFL : float
        Courant number (per-cell, for local time stepping).
    convergence_tol : float
        Stop when the relative residual falls below this value.
    order : int
        Spatial order: 1 (piecewise constant) or 2 (MUSCL).
    limiter : str
        Slope limiter for MUSCL: 'minmod' or 'mc'.
    flux : str
        Riemann solver: 'hll' or 'rusanov'.
    wb_mode : str
        Well-balancing: 'full' (default), 'adaptive', or 'off'.
    inner_mach_threshold : float
        Switch from well-balanced to free-extrapolation BC when the inner
        Mach number exceeds this value.
    cooling_ramp_steps : int
        Linearly ramp the cooling source from 0 to 1 over this many steps to
        avoid initial transients.
    strang : bool
        Use Strang splitting (cool/2 -> hydro -> cool/2) instead of Lie
        splitting (hydro -> cool).
    sponge_frac : float
        Fraction of outer cells used as a sponge layer (0 disables).
    relaxation : float
        Under-relaxation factor (1 = no relaxation).
    snapshot_interval : int
        Print progress every this many steps.
    verbose : bool
        Print configuration and progress.
    """

    # Grid
    N: int = 800
    x_min: float = 1e-3
    x_max: float = 3.0

    # Time stepping
    n_steps: int = 50_000
    CFL: float = 0.4
    convergence_tol: float = 1e-10

    # Spatial scheme
    order: int = 2
    limiter: str = "minmod"
    flux: str = "hll"

    # Well-balancing
    wb_mode: str = "full"

    # Boundary conditions
    inner_mach_threshold: float = 2.0

    # Cooling treatment
    cooling_ramp_steps: int = 5_000
    strang: bool = False

    # Sponge / relaxation
    sponge_frac: float = 0.0
    relaxation: float = 1.0

    # Diagnostics
    snapshot_interval: int = 10_000
    verbose: bool = True


class BondiProblem:
    """A spherical Bondi accretion problem with optional radiative cooling.

    Parameters
    ----------
    M_BH : float
        Black hole mass [g]. Use ``radbondi.M_sun`` to convert from solar masses.
    ambient : AmbientMedium
        Ambient medium specification.
    cooling : Cooling, optional
        Cooling prescription. Defaults to bremsstrahlung + pair annihilation.
        Use :meth:`Cooling.adiabatic` for an adiabatic flow.
    """

    def __init__(
        self,
        M_BH: float,
        ambient: AmbientMedium,
        cooling: Cooling | None = None,
    ):
        self.M_BH = float(M_BH)
        self.ambient = ambient
        self.cooling = cooling if cooling is not None else Cooling.default()

    # ── Derived quantities ────────────────────────────────────────────────

    @property
    def r_B(self) -> float:
        """Bondi radius [cm]."""
        return bondi_radius(self.M_BH, self.ambient)

    @property
    def r_S(self) -> float:
        """Schwarzschild radius [cm]."""
        return schwarzschild_radius(self.M_BH)

    @property
    def Mdot_B(self) -> float:
        """Adiabatic Bondi accretion rate [g s^-1]."""
        return bondi_rate(self.M_BH, self.ambient)

    # ── Driver ────────────────────────────────────────────────────────────

    def solve(self, config: SolverConfig | None = None) -> Solution:
        """Solve to steady state and return a :class:`Solution`."""
        if config is None:
            config = SolverConfig()
        return _evolve_local_dt(self, config)


# ──────────────────────────────────────────────────────────────────────────
# Internal: time integration loop
# ──────────────────────────────────────────────────────────────────────────


def _initialize_from_bondi(problem: BondiProblem, grid: Grid):
    """Build the initial conservative state from the adiabatic Bondi solution."""
    v_phys, T_phys, rho_phys, _Mach = adiabatic_profile(grid.x, problem.ambient)
    v_phys = -np.abs(v_phys)
    P = rho_phys * kB * T_phys / (problem.ambient.mu * m_p)
    mom = rho_phys * v_phys
    E = 0.5 * rho_phys * v_phys**2 + P / (problem.ambient.gamma - 1.0)
    return np.array([rho_phys, mom, E])


def _apply_cooling_implicit(
    U,
    dt_array,
    cooling: Cooling,
    ambient: AmbientMedium,
    eps_ambient: float,
    skip_boundary: int = 2,
):
    """Operator-split implicit cooling step (Newton iteration on T)."""
    rho, v, P, T, _ = get_primitives(U, ambient.gamma, ambient.mu, ambient.T * 0.5)
    N = len(rho)
    coeff = rho * kB / (ambient.mu * m_p * (ambient.gamma - 1.0))
    e_th = P / (ambient.gamma - 1.0)
    sl = slice(skip_boundary, N - skip_boundary)
    T_new = T.copy()
    active = T_new[sl] > ambient.T * 1.01
    if not np.any(active):
        return U
    T_work = T_new[sl][active]
    rho_work = rho[sl][active]
    coeff_work = coeff[sl][active]
    e_th_work = e_th[sl][active]
    dt_work = dt_array[sl][active]
    for _ in range(20):
        eps_val = cooling.net_emissivity(rho_work, T_work, ambient, eps_ambient)
        R = coeff_work * T_work - e_th_work + dt_work * eps_val
        dT_fd = T_work * 1e-6 + 1.0
        eps_p = cooling.net_emissivity(rho_work, T_work + dT_fd, ambient, eps_ambient)
        dR_dT = coeff_work + dt_work * (eps_p - eps_val) / dT_fd
        dR_dT = np.where(np.abs(dR_dT) < 1e-50, 1.0, dR_dT)
        delta_T = -R / dR_dT
        T_work = np.maximum(T_work + delta_T, ambient.T)
        if np.max(np.abs(delta_T) / T_work) < 1e-8:
            break
    T_full = T_new.copy()
    temp = T_full[sl].copy()
    temp[active] = T_work
    T_full[sl] = temp
    U_new = U.copy()
    P_new = rho * kB * T_full / (ambient.mu * m_p)
    U_new[2] = 0.5 * rho * v**2 + P_new / (ambient.gamma - 1.0)
    return U_new


def _evolve_local_dt(problem: BondiProblem, cfg: SolverConfig) -> Solution:
    """Steady-state solver with local time stepping.

    Faithful port of ``evolve_local_dt`` from the original ``run_timedep.py``.
    """
    ambient = problem.ambient
    cooling = problem.cooling
    M_BH = problem.M_BH
    gamma = ambient.gamma
    mu = ambient.mu
    T_floor = ambient.T * 0.5

    grid = Grid.log_spaced(problem.r_B, N=cfg.N, x_min=cfg.x_min, x_max=cfg.x_max)
    has_cooling = bool(cooling.processes)
    eps_ambient = cooling.ambient_emissivity(ambient) if has_cooling else 0.0

    U = _initialize_from_bondi(problem, grid)

    # Well-balanced setup: store the adiabatic equilibrium and its residual
    U_eq_adiabatic = U.copy()
    R_eq_adiabatic, _ = hydro_rhs(
        U_eq_adiabatic, grid, M_BH, gamma, mu, T_floor,
        U_eq=U_eq_adiabatic, order=cfg.order, limiter=cfg.limiter, flux=cfg.flux,
    )

    # Outer BC: hold to initial ambient values (preserves gradient)
    U_outer_m1 = U[:, -1].copy()
    U_outer_m2 = U[:, -2].copy()

    rho0, v0, _, _, cs0 = get_primitives(U, gamma, mu, T_floor)
    dt_local0 = grid.dr / (np.abs(v0) + cs0)

    if cfg.verbose:
        print(f"  Local dt range: {dt_local0.min():.3e} to {dt_local0.max():.3e} s")
        print(f"  Speedup vs global: {dt_local0.max() / dt_local0.min():.0f}x")
        if cfg.cooling_ramp_steps > 0:
            print(f"  Cooling ramp: 0->1 over {cfg.cooling_ramp_steps} steps")
        print(f"  WB mode: {cfg.wb_mode}")
        if cfg.relaxation < 1.0:
            print(f"  Under-relaxation: omega = {cfg.relaxation}")
        if cfg.sponge_frac > 0:
            print(f"  Sponge layer: {cfg.sponge_frac * 100:.0f}% of outer cells")
        split = "Strang (cool/2->hydro->cool/2)" if cfg.strang else "Lie (hydro->cool)"
        print(f"  Splitting: {split}")
        print(
            f"  Inner BC: well-balanced if Mach<{cfg.inner_mach_threshold}, "
            "else free extrap"
        )

    # Sponge layer setup
    n_sponge = int(cfg.N * cfg.sponge_frac) if cfg.sponge_frac > 0 else 0
    if n_sponge > 0:
        sponge_alpha = np.zeros(cfg.N)
        for i in range(cfg.N - n_sponge, cfg.N):
            sponge_alpha[i] = ((i - (cfg.N - n_sponge)) / n_sponge) ** 2
    else:
        sponge_alpha = None

    residuals: list[float] = []
    converged = False
    final_step = cfg.n_steps

    for step in range(1, cfg.n_steps + 1):
        # Well-balancing reference
        U_eq = None if cfg.wb_mode == "off" else U_eq_adiabatic

        dU, _ = hydro_rhs(
            U, grid, M_BH, gamma, mu, T_floor,
            U_eq=U_eq, order=cfg.order, limiter=cfg.limiter, flux=cfg.flux,
        )

        # R_eq subtraction with selective weighting (full where adiabatic,
        # zero where cooling has restructured the flow).
        if cfg.wb_mode != "off":
            delta_U = U - U_eq_adiabatic
            scale_U = np.abs(U) + 1e-30
            frac = np.sqrt(np.mean((delta_U / scale_U) ** 2, axis=0))
            R_eq_weight = np.exp(-frac / 0.01)
            dU -= R_eq_adiabatic * R_eq_weight[np.newaxis, :]

        rho, v, P, T, cs = get_primitives(U, gamma, mu, T_floor)
        dt_local = cfg.CFL * grid.dr / (np.abs(v) + cs)

        cool_frac = (
            step / cfg.cooling_ramp_steps
            if cfg.cooling_ramp_steps > 0 and step <= cfg.cooling_ramp_steps
            else 1.0
        )

        if cfg.strang and has_cooling:
            dt_cool_half = dt_local * cool_frac * 0.5
            U_half = _apply_cooling_implicit(
                U, dt_cool_half, cooling, ambient, eps_ambient
            )
            dU_s, _ = hydro_rhs(
                U_half, grid, M_BH, gamma, mu, T_floor,
                U_eq=U_eq, order=cfg.order, limiter=cfg.limiter, flux=cfg.flux,
            )
            if cfg.wb_mode != "off":
                delta_Us = U_half - U_eq_adiabatic
                scale_Us = np.abs(U_half) + 1e-30
                frac_s = np.sqrt(np.mean((delta_Us / scale_Us) ** 2, axis=0))
                R_eq_weight_s = np.exp(-frac_s / 0.01)
                dU_s -= R_eq_adiabatic * R_eq_weight_s[np.newaxis, :]
            U_new = U_half + dt_local[np.newaxis, :] * dU_s
            U_new = _apply_cooling_implicit(
                U_new, dt_cool_half, cooling, ambient, eps_ambient
            )
        else:
            U_new = U + dt_local[np.newaxis, :] * dU
            if has_cooling:
                dt_cool = dt_local * cool_frac
                U_new = _apply_cooling_implicit(
                    U_new, dt_cool, cooling, ambient, eps_ambient
                )

        # Inner BC
        rho_bc, v_bc, P_bc, T_bc, cs_bc = get_primitives(U_new, gamma, mu, T_floor)
        mach_inner = np.abs(v_bc[1]) / cs_bc[1]
        if mach_inner > cfg.inner_mach_threshold:
            U_new[:, 0] = U_new[:, 1]
        elif U_eq is not None:
            delta_1 = U_new[:, 1] - U_eq[:, 1]
            U_new[:, 0] = U_eq[:, 0] + delta_1
        else:
            U_new[:, 0] = U_new[:, 1]

        # Outer BC (held fixed)
        U_new[:, -1] = U_outer_m1
        U_new[:, -2] = U_outer_m2

        # Sponge layer
        if n_sponge > 0:
            for k in range(3):
                U_new[k] = (1 - sponge_alpha) * U_new[k] + sponge_alpha * U_eq_adiabatic[k]

        # Floor enforcement
        U_new[0] = np.maximum(U_new[0], 1e-30)
        rho_n, v_n, P_n, _, _ = get_primitives(U_new, gamma, mu, T_floor)
        P_floor = rho_n * kB * T_floor / (mu * m_p)
        mask = P_n < P_floor
        if np.any(mask):
            U_new[2, mask] = (
                0.5 * rho_n[mask] * v_n[mask] ** 2 + P_floor[mask] / (gamma - 1.0)
            )

        # Under-relaxation
        if cfg.relaxation < 1.0:
            U_new = U + cfg.relaxation * (U_new - U)

        # Residual
        diff = U_new - U
        scale = np.abs(U) + 1e-30
        residual = float(np.sqrt(np.mean((diff / scale) ** 2)))
        residuals.append(residual)

        U = U_new

        # Progress print
        if cfg.verbose and (step % cfg.snapshot_interval == 0 or step == cfg.n_steps):
            rho, v, _, T, cs = get_primitives(U, gamma, mu, T_floor)
            Mach = np.abs(v) / cs
            cool_info = ""
            if cfg.cooling_ramp_steps > 0 and step <= cfg.cooling_ramp_steps:
                cool_info = f", cool={step / cfg.cooling_ramp_steps:.2f}"
            bc_type = "free" if mach_inner > cfg.inner_mach_threshold else "wb"
            print(
                f"  Step {step:6d}: res={residual:.3e}, "
                f"T_max/T_inf={T.max() / ambient.T:.1f}, "
                f"Mach={Mach.max():.1f}, BC={bc_type}{cool_info}"
            )

        if residual < cfg.convergence_tol and step > max(100, cfg.cooling_ramp_steps):
            if cfg.verbose:
                print(f"  *** Converged at step {step}: residual = {residual:.3e}")
            converged = True
            final_step = step
            break

    rho, v, P, T, cs = get_primitives(U, gamma, mu, T_floor)
    Mach = np.abs(v) / cs

    return Solution(
        r=grid.r_cen, r_B=problem.r_B,
        rho=rho, v=v, P=P, T=T, Mach=Mach,
        U=U,
        residuals=np.array(residuals),
        converged=converged,
        M_BH=M_BH, Mdot_B=problem.Mdot_B,
        ambient_T=ambient.T, ambient_rho=ambient.rho, ambient_mu=ambient.mu,
        ambient_gamma=ambient.gamma, ambient_X=ambient.X, ambient_Y=ambient.Y,
        metadata={"final_step": final_step, "n_steps_requested": cfg.n_steps},
    )
