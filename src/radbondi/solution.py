"""Solution object: profiles, derived quantities, save/load, plotting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from radbondi.constants import c_light


@dataclass
class Solution:
    """Steady-state solution returned by :meth:`BondiProblem.solve`.

    All arrays are on the cell-center grid. Derived scalars (eta, Mdot, L)
    are computed from the converged profiles.
    """

    # Grid
    r: np.ndarray             # cell-center radii [cm]
    r_B: float                 # Bondi radius [cm]

    # Primitive profiles
    rho: np.ndarray           # [g cm^-3]
    v: np.ndarray             # [cm s^-1] (negative = infall)
    P: np.ndarray             # [erg cm^-3]
    T: np.ndarray             # [K]
    Mach: np.ndarray          # |v| / cs

    # Conservatives (full state, useful for restart or further analysis)
    U: np.ndarray             # shape (3, N)

    # Convergence diagnostics
    residuals: np.ndarray     # per-step residual time series
    converged: bool

    # Problem context (not the live objects, just the values needed for derived
    # quantities and re-loading)
    M_BH: float               # [g]
    Mdot_B: float             # adiabatic Bondi rate [g s^-1]
    ambient_T: float          # [K]
    ambient_rho: float        # [g cm^-3]
    ambient_mu: float
    ambient_gamma: float
    ambient_X: float
    ambient_Y: float

    # Optional metadata (e.g., feedback iteration history)
    metadata: dict = field(default_factory=dict)

    # ── Derived scalars ──────────────────────────────────────────────────

    @property
    def x(self) -> np.ndarray:
        """Cell-center radii in units of r_B."""
        return self.r / self.r_B

    @property
    def Mdot(self) -> float:
        """Mass accretion rate [g s^-1], measured as the median of
        4*pi*r^2*rho*|v| over the outer half of the grid (subsonic, smooth)."""
        Mdot_profile = 4.0 * np.pi * self.r**2 * self.rho * np.abs(self.v)
        N = len(self.r)
        return float(np.median(Mdot_profile[N // 2 :]))

    @property
    def mdot_ratio(self) -> float:
        """Mdot / Mdot_Bondi (cooling enhancement factor)."""
        return self.Mdot / self.Mdot_B

    def luminosity(self, cooling) -> float:
        """Total luminosity L = integral of net emissivity over the domain.

        Requires a :class:`Cooling` instance to evaluate the emissivity. The
        ``Cooling`` is not stored on the Solution to keep it serialization-safe.
        """
        from radbondi.ambient import AmbientMedium
        ambient = AmbientMedium(
            T=self.ambient_T, rho=self.ambient_rho, mu=self.ambient_mu,
            gamma=self.ambient_gamma, X=self.ambient_X, Y=self.ambient_Y,
        )
        # Reconstruct cell volumes from the grid: faces are r_face[i] =
        # r_cen[i] / sqrt(r_cen[1]/r_cen[0]) for the log-spaced grid.
        ratio = (self.r[1] / self.r[0]) ** 0.5
        r_face = np.empty(len(self.r) + 1)
        r_face[:-1] = self.r / ratio
        r_face[-1] = self.r[-1] * ratio
        vol = (r_face[1:] ** 3 - r_face[:-1] ** 3) / 3.0
        eps = cooling.net_emissivity(self.rho, self.T, ambient)
        return float(4.0 * np.pi * np.sum(eps * vol))

    def eta(self, cooling) -> float:
        """Radiative efficiency eta = L / (Mdot c^2).

        Requires a :class:`Cooling` instance to evaluate L.
        """
        return self.luminosity(cooling) / (self.Mdot_B * c_light**2)

    @property
    def solver_residual(self) -> float:
        """Final solver residual (last entry of the convergence time series)."""
        return float(self.residuals[-1]) if len(self.residuals) > 0 else float("nan")

    # ── Save / load ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save solution to a .npz file (loadable via :func:`load`)."""
        np.savez(
            path,
            r=self.r, r_B=self.r_B,
            rho=self.rho, v=self.v, P=self.P, T=self.T, Mach=self.Mach,
            U=self.U,
            residuals=self.residuals,
            converged=self.converged,
            M_BH=self.M_BH, Mdot_B=self.Mdot_B,
            ambient_T=self.ambient_T, ambient_rho=self.ambient_rho,
            ambient_mu=self.ambient_mu, ambient_gamma=self.ambient_gamma,
            ambient_X=self.ambient_X, ambient_Y=self.ambient_Y,
        )

    # ── Plotting ─────────────────────────────────────────────────────────

    def plot_profiles(self, ax=None):
        """Quick-look plot of T/T_inf, density, and Mach number vs r/r_B.

        Requires matplotlib (install with ``pip install radbondi[plot]``).
        Returns the matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "plot_profiles requires matplotlib. "
                "Install with: pip install radbondi[plot]"
            ) from exc

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        x = self.x
        axes[0].loglog(x, self.T / self.ambient_T)
        axes[0].set(xlabel=r"$r/r_B$", ylabel=r"$T/T_\infty$", title="Temperature")
        axes[1].loglog(x, self.rho / self.ambient_rho)
        axes[1].set(xlabel=r"$r/r_B$", ylabel=r"$\rho/\rho_\infty$", title="Density")
        axes[2].loglog(x, self.Mach)
        axes[2].axhline(1.0, color="gray", ls=":", lw=0.5)
        axes[2].set(xlabel=r"$r/r_B$", ylabel="Mach", title="Mach number")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def check_steady_state(self, cooling, n_boundary: int = 5):
        """Verify the steady-state equations on the converged profile.

        See :func:`radbondi.diagnostics.check_steady_state`.
        """
        from radbondi.diagnostics import check_steady_state
        return check_steady_state(self, cooling, n_boundary=n_boundary)


def load(path: str) -> Solution:
    """Load a Solution previously saved with :meth:`Solution.save`."""
    data = np.load(path)
    return Solution(
        r=data["r"], r_B=float(data["r_B"]),
        rho=data["rho"], v=data["v"], P=data["P"], T=data["T"], Mach=data["Mach"],
        U=data["U"],
        residuals=data["residuals"],
        converged=bool(data["converged"]),
        M_BH=float(data["M_BH"]), Mdot_B=float(data["Mdot_B"]),
        ambient_T=float(data["ambient_T"]),
        ambient_rho=float(data["ambient_rho"]),
        ambient_mu=float(data["ambient_mu"]),
        ambient_gamma=float(data["ambient_gamma"]),
        ambient_X=float(data["ambient_X"]),
        ambient_Y=float(data["ambient_Y"]),
    )
