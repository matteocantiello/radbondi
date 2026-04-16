"""Custom cooling process: inverse Compton off a uniform radiation bath.

Companion to ``docs/custom_cooling.md``. Demonstrates:

  1. Subclassing :class:`radbondi.CoolingProcess` to add a new emissivity.
  2. Combining the new process with the default mix
     (bremsstrahlung + pair annihilation).
  3. Inspecting per-process emissivity contributions on the converged
     profile to see which mechanism dominates at which radius.

Inverse-Compton (IC) cooling off a uniform radiation field of energy
density ``U_rad`` removes thermal-electron energy at the rate

    eps_IC = (4 sigma_T / (m_e c)) * U_rad * n_e * k_B T     (theta_e << 1)

Important physical caveat: at solar-core ``M_BH = 1e-13 M_sun``, the
inner-Bondi bremsstrahlung emissivity is enormous (``~1e45 erg/cm^3/s``)
because the density compression near ``r_S`` puts ``rho^2`` in the
denominator of free-free. To rival brems in this regime you would need
``U_rad >> 1e20 erg/cm^3``, far above any physical photon bath. For the
illustrative ``U_rad = 1e6 erg/cm^3`` chosen below, IC cooling is
present but utterly subdominant. The example still does its job: it
shows the plug-in mechanism works end-to-end, and the diagnostic at the
end makes the relative-magnitude story explicit. Crank ``U_rad`` up to
make IC visible, or replace it with a different physical process where
your problem's regime makes that process matter.
"""

from __future__ import annotations

import time

import numpy as np

import radbondi as rb
from radbondi import Cooling, CoolingProcess
from radbondi.constants import c_light, kB, m_e, m_p, sigma_T
from radbondi.cooling import PairAnnihilation, RelativisticBremsstrahlung


class InverseComptonUniform(CoolingProcess):
    """Inverse-Compton cooling off a uniform-density radiation bath.

    Parameters
    ----------
    U_rad : float
        Radiation energy density seen by the accreting plasma [erg cm^-3].
    """

    def __init__(self, U_rad: float):
        self.U_rad = float(U_rad)

    def emissivity(self, rho, T, ambient):
        n_e = (ambient.X + 0.5 * ambient.Y) * rho / m_p
        prefactor = 4.0 * sigma_T / (m_e * c_light)
        return prefactor * self.U_rad * n_e * kB * T


def main():
    M_BH = 1e-13 * rb.M_sun
    ambient = rb.presets.solar_core()
    cfg = rb.SolverConfig(
        N=400, x_min=1e-5, n_steps=30_000, cooling_ramp_steps=3_000,
        order=1, flux="hll",
        snapshot_interval=30_000, verbose=False,
    )

    print("─" * 70)
    print("Baseline: brem + pairs only")
    print("─" * 70)
    cool_base = Cooling.default()
    t0 = time.time()
    sol_base = rb.BondiProblem(M_BH=M_BH, ambient=ambient, cooling=cool_base).solve(cfg)
    print(f"  eta = {sol_base.eta:.3e},  T_max/T_inf = {sol_base.T.max() / ambient.T:.1f},"
          f"  wall = {time.time() - t0:.1f}s")

    print()
    print("─" * 70)
    print("Augmented: brem + pairs + Compton off U_rad = 1e6 erg/cm^3")
    print("─" * 70)
    cool_ic = Cooling(
        [
            RelativisticBremsstrahlung(),
            PairAnnihilation(species="electron"),
            PairAnnihilation(species="muon"),
            InverseComptonUniform(U_rad=1e6),
        ]
    )
    t0 = time.time()
    sol_ic = rb.BondiProblem(M_BH=M_BH, ambient=ambient, cooling=cool_ic).solve(cfg)
    print(f"  eta = {sol_ic.eta:.3e},  T_max/T_inf = {sol_ic.T.max() / ambient.T:.1f},"
          f"  wall = {time.time() - t0:.1f}s")

    print()
    print("─" * 70)
    print("Diagnostic: per-process emissivity at the inner cells")
    print("─" * 70)
    # Evaluate each process individually on the IC-augmented profile.
    rho = sol_ic.rho
    T = sol_ic.T
    eps_brems = RelativisticBremsstrahlung().emissivity(rho, T, ambient)
    eps_pair = PairAnnihilation(species="electron").emissivity(rho, T, ambient)
    eps_ic = InverseComptonUniform(U_rad=1e6).emissivity(rho, T, ambient)
    # Inner cell (deep): which process dominates?
    i = 5
    x = sol_ic.r[i] / sol_ic.r_B
    print(f"  At r/r_B = {x:.2e}:")
    print(f"    brems = {eps_brems[i]:.3e}")
    print(f"    pairs = {eps_pair[i]:.3e}")
    print(f"    IC    = {eps_ic[i]:.3e}")
    # Find the radius where IC just overtakes brems
    cross = np.where(eps_ic > eps_brems)[0]
    if cross.size > 0:
        x_cross = sol_ic.r[cross[0]] / sol_ic.r_B
        print(f"  Compton overtakes bremsstrahlung at r/r_B ~ {x_cross:.2e}")
    else:
        print("  Bremsstrahlung dominates everywhere on this profile.")


if __name__ == "__main__":
    main()
