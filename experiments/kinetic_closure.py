"""Stage 3a: Conservative kinetic feedback closure.

Given a hydro state near r_coll, compute the return/circularization
fluxes from the kinetic model (Stages 1+2) and output conservative
source profiles S_p(r), S_E(r) for momentum and energy deposition.

This replaces the Dirichlet temperature override with physically
derived fluxes: mass, momentum, and energy deposited by returning
and circularizing particles.

Usage:
    # As a module:
    from experiments.kinetic_closure import KineticClosure
    closure = KineticClosure(sol)
    S_p, S_E, diag = closure.compute(rho_t, T_t, v_t, Mdot)

    # Standalone test:
    python experiments/kinetic_closure.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import radbondi as rb
from radbondi.constants import G, c_light, kB, m_p
from radbondi.cooling import Cooling


@dataclass
class ClosureDiagnostics:
    """Output diagnostics from the kinetic closure."""

    f_cap: float
    f_refl: float
    f_coll: float
    f_marg: float
    f_res: float
    r_coll: float
    r_coll_rS: float
    ell_max: float
    xi_avg: float
    dEE_avg: float
    P_pile_over_P0: float


class KineticClosure:
    """Compute conservative kinetic feedback from Stages 1+2.

    Given a hydrodynamic profile, computes the momentum and energy
    source terms that the recycling/circularization pile-up deposits
    back into the flow.
    """

    def __init__(self, sol, N_ell=200):
        self.sol = sol
        self.gamma = sol.ambient_gamma
        self.mu = sol.ambient_mu
        self.GM = G * sol.M_BH
        self.rS = 2 * self.GM / c_light**2
        self.ell_crit = 4 * self.GM / c_light

        self.r = sol.r
        self.N_ell = N_ell

        # Ambient for emissivity
        self.amb = rb.AmbientMedium(
            T=sol.ambient_T, rho=sol.ambient_rho,
            mu=self.mu, gamma=self.gamma,
            X=sol.ambient_X, Y=sol.ambient_Y,
        )
        self.cool = Cooling.default()

        # Pre-compute emissivity on the grid
        self._eps_ff = np.array([
            float(self.cool.total_emissivity(sol.rho[i], sol.T[i], self.amb)[0])
            for i in range(len(self.r))
        ])

        # Mfp profile
        E4 = (4.803e-10) ** 4
        n_e = (sol.ambient_X + 0.5 * sol.ambient_Y) * sol.rho / m_p
        self._lam_mfp = (kB * sol.T) ** 2 / (np.pi * n_e * E4 * 5.0)
        self._Kn = self._lam_mfp / self.r

    def _Kn_at(self, r_target):
        if r_target < self.r[0] or r_target > self.r[-1]:
            return 1e10
        return 10 ** float(np.interp(
            np.log10(r_target), np.log10(self.r),
            np.log10(np.maximum(self._Kn, 1e-30)),
        ))

    def _dEE(self, ell, E_coll, r_coll):
        """Orbit-averaged dE/E for a single ell value."""
        ell_max = np.sqrt(self.GM**2 / (2 * abs(E_coll)))
        if ell <= 0 or ell >= ell_max:
            return 0.0

        a_c, b_c, c_c = E_coll, self.GM, -(ell**2) / 2
        disc = b_c**2 - 4 * a_c * c_c
        if disc < 0:
            return 0.0

        r1 = (-b_c + np.sqrt(disc)) / (2 * a_c)
        r2 = (-b_c - np.sqrt(disc)) / (2 * a_c)
        candidates = [x for x in [r1, r2] if x > 0]
        if not candidates:
            return 0.0
        r_peri = min(candidates)

        dr_range = r_coll - r_peri
        if dr_range < 1e-15:
            return 0.0
        r_inner = r_peri + 0.05 * dr_range
        r_outer = r_coll - 0.05 * dr_range
        mask = (self.r >= r_inner) & (self.r <= r_outer)
        if np.sum(mask) < 5:
            return 0.0

        r_orb = self.r[mask]
        rho_orb = self.sol.rho[mask]
        eps_orb = self._eps_ff[mask]

        E_orb = ell**2 / (2 * r_coll**2) - self.GM / r_coll
        vr2 = 2 * (E_orb + self.GM / r_orb) - ell**2 / r_orb**2
        vr2 = np.maximum(vr2, 0.0)
        vr = np.sqrt(vr2)
        v_total = np.sqrt(vr2 + ell**2 / r_orb**2)
        vr = np.maximum(vr, 0.01 * v_total)

        KE = 0.5 * np.mean(v_total**2)
        cooling_rate = eps_orb / np.maximum(rho_orb, 1e-30)
        integrand = cooling_rate / vr
        dE_total = 2.0 * 1.1 * np.trapz(integrand, r_orb)

        return dE_total / max(KE, 1e-30)

    def compute(self, rho_t, T_t, v_t, Mdot):
        """Compute kinetic closure sources.

        Parameters
        ----------
        rho_t, T_t, v_t : float
            Hydro state at r_coll (density, temperature, infall speed).
        Mdot : float
            Current mass accretion rate [g/s].

        Returns
        -------
        S_p : ndarray
            Momentum source [g cm^-2 s^-2] at each grid radius.
        S_E : ndarray
            Energy source [erg cm^-3 s^-1] at each grid radius.
        diag : ClosureDiagnostics
        """
        GM = self.GM
        gamma = self.gamma

        # Find r_coll from current profiles
        cs_t = np.sqrt(gamma * kB * T_t / (self.mu * m_p))
        sigma_perp = cs_t / np.sqrt(gamma)
        E_coll = 0.5 * v_t**2 - GM / self.r[0]  # placeholder

        # Use the profile to find r_coll
        Kn_arr = self._lam_mfp / self.r
        crossings = np.where(np.diff(np.sign(Kn_arr - 1.0)))[0]
        if len(crossings) == 0:
            i_coll = len(self.r) // 2
        else:
            i_coll = crossings[-1]
        r_coll = self.r[i_coll]

        # Orbital energy at r_coll from the profile
        v_coll = np.abs(self.sol.v[i_coll])
        E_coll = 0.5 * v_coll**2 - GM / r_coll
        ell_max = np.sqrt(GM**2 / (2 * abs(E_coll)))
        sigma_perp_coll = np.sqrt(
            gamma * kB * self.sol.T[i_coll] / (self.mu * m_p)
        ) / np.sqrt(gamma)

        # Angular momentum grid
        ell_grid = np.linspace(
            0.01 * self.ell_crit, ell_max * 1.3, self.N_ell
        )
        scale = r_coll * sigma_perp_coll
        rayleigh_pdf = (ell_grid / scale**2) * np.exp(
            -(ell_grid**2) / (2 * scale**2)
        )
        rayleigh_pdf /= np.trapz(rayleigh_pdf, ell_grid)

        # Classify each ell bin and compute deposition radii
        S_p = np.zeros_like(self.r)
        S_E = np.zeros_like(self.r)

        n_cap, n_refl, n_coll, n_marg, n_res = 0.0, 0.0, 0.0, 0.0, 0.0
        xi_sum, dEE_sum, w_sum = 0.0, 0.0, 0.0

        d_ell = ell_grid[1] - ell_grid[0]

        for i_ell, ell in enumerate(ell_grid):
            w = rayleigh_pdf[i_ell] * d_ell

            # Direct capture
            if ell < self.ell_crit:
                n_cap += w
                continue

            # Reflected
            if ell > ell_max:
                n_refl += w
                # Reflected particles thermalize near r_coll.
                # Their kinetic energy (outward) is deposited as pressure.
                # Momentum: outward v ~ v_coll, deposited in 1-2 cells.
                i_dep = i_coll
                shell_vol = (4 / 3) * np.pi * (
                    self.r[min(i_dep + 1, len(self.r) - 1)]**3
                    - self.r[max(i_dep - 1, 0)]**3
                )
                if shell_vol > 0:
                    # Energy deposited: KE of returning particle
                    # ~ 0.5 * m * v_coll^2, at rate w * Mdot
                    E_dep = w * Mdot * 0.5 * v_coll**2 / shell_vol
                    S_E[i_dep] += E_dep
                    # Momentum: outward, opposes infall
                    p_dep = w * Mdot * v_coll / (4 * np.pi * self.r[i_dep]**2)
                    S_p[i_dep] += p_dep
                continue

            # Recycling: compute orbit properties
            dEE = self._dEE(ell, E_coll, r_coll)
            if dEE < 1e-30:
                n_refl += w
                continue

            xi = min(np.sqrt(2.0 / dEE), 1e6)
            r_circ = ell**2 / GM
            Kn_0 = self._Kn_at(r_circ)
            Kn_eff = Kn_0 / max(xi, 1)

            xi_sum += w * xi
            dEE_sum += w * dEE
            w_sum += w

            # Classify
            if Kn_eff < 1:
                fate = "coll"
                n_coll += w
            elif Kn_eff < 3:
                fate = "marg"
                n_marg += w
            else:
                fate = "res"
                n_res += w

            # Determine deposition radius
            if fate in ("coll", "marg"):
                # Collisionalized: thermalizes near r_circ
                r_dep = r_circ
            else:
                # Reservoir: delayed, but eventually thermalizes
                # near r_circ after self-collisionalization
                r_dep = r_circ

            # Find the grid cell closest to r_dep
            i_dep = np.argmin(np.abs(self.r - r_dep))
            i_dep = max(1, min(i_dep, len(self.r) - 2))

            shell_vol = (4 / 3) * np.pi * (
                self.r[i_dep + 1]**3 - self.r[i_dep - 1]**3
            )
            if shell_vol <= 0:
                continue

            # Energy deposited: the particle's orbital KE at r_dep
            v_at_dep = np.sqrt(max(2 * GM / r_dep, 0))
            E_dep = w * Mdot * 0.5 * v_at_dep**2 / shell_vol
            S_E[i_dep] += E_dep

            # Momentum: particles arriving from deeper radii have
            # outward radial velocity ~ v_at_dep after circularization.
            # The net momentum deposition opposes the infall.
            p_dep = w * Mdot * v_at_dep / (4 * np.pi * self.r[i_dep]**2)
            S_p[i_dep] += p_dep

        # Effective pressure from the pile-up
        P0_rcoll = (
            self.sol.rho[i_coll] * kB * self.sol.T[i_coll]
            / (self.mu * m_p)
        )
        P_pile = np.sum(S_p * self.r**2) / (
            self.r[i_coll]**2
        ) if P0_rcoll > 0 else 0
        P_ratio = P_pile / P0_rcoll if P0_rcoll > 0 else 0

        diag = ClosureDiagnostics(
            f_cap=n_cap,
            f_refl=n_refl,
            f_coll=n_coll,
            f_marg=n_marg,
            f_res=n_res,
            r_coll=r_coll,
            r_coll_rS=r_coll / self.rS,
            ell_max=ell_max,
            xi_avg=xi_sum / w_sum if w_sum > 0 else 0,
            dEE_avg=dEE_sum / w_sum if w_sum > 0 else 0,
            P_pile_over_P0=P_ratio,
        )

        return S_p, S_E, diag


# ── Standalone test ──────────────────────────────────────────────────


def main():
    print("Stage 3a: Conservative kinetic closure")
    print("=" * 65)

    sol = rb.load("examples/paper_sweep_output/mbh_logM-16.00.npz")
    closure = KineticClosure(sol)

    # Use the baseline hydro state
    i_mid = len(sol.r) // 2
    rho_t = sol.rho[i_mid]
    T_t = sol.T[i_mid]
    v_t = np.abs(sol.v[i_mid])
    Mdot = 4 * np.pi * sol.r[i_mid]**2 * rho_t * v_t

    S_p, S_E, diag = closure.compute(rho_t, T_t, v_t, Mdot)

    rS = closure.rS

    print(f"\n  r_coll = {diag.r_coll_rS:.0f} r_S")
    print(f"  ell_max = {diag.ell_max:.1f} cm^2/s")
    print(f"  <xi> = {diag.xi_avg:.1f}")
    print(f"  <dE/E> = {diag.dEE_avg:.4f}")
    print()
    print("  Population fractions:")
    print(f"    captured:       {diag.f_cap:.4f}")
    print(f"    reflected:      {diag.f_refl:.3f}")
    print(f"    collisionalized:{diag.f_coll:.3f}")
    print(f"    marginal:       {diag.f_marg:.3f}")
    print(f"    reservoir:      {diag.f_res:.3f}")
    print(f"    total:          {diag.f_cap + diag.f_refl + diag.f_coll + diag.f_marg + diag.f_res:.3f}")
    print()
    print(f"  P_pile / P_0 at r_coll: {diag.P_pile_over_P0:.2f}")
    print()

    # Show where the sources are deposited
    print("  Source deposition profile:")
    print(f"    {'r/r_S':>8s}  {'S_p':>12s}  {'S_E':>12s}")
    print("    " + "-" * 35)
    for target in [100, 300, 500, 800, 1200, 2000, 3000, 5000, 7000]:
        idx = np.argmin(np.abs(sol.r / rS - target))
        if S_p[idx] > 0 or S_E[idx] > 0:
            print(f"    {sol.r[idx] / rS:>8.0f}  {S_p[idx]:>12.3e}  {S_E[idx]:>12.3e}")

    print()


if __name__ == "__main__":
    main()
