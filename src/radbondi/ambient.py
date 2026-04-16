"""Ambient medium specification."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from radbondi.constants import kB, m_p


@dataclass(frozen=True)
class AmbientMedium:
    """Ambient medium properties (CGS units).

    Parameters
    ----------
    T : float
        Temperature [K].
    rho : float
        Mass density [g cm^-3].
    mu : float
        Mean molecular weight (dimensionless).
    gamma : float, optional
        Adiabatic index. Default 5/3 (monatomic ideal gas).
    X : float, optional
        Hydrogen mass fraction. Default 0.7. Used by composition-dependent
        cooling processes (e.g. bremsstrahlung).
    Y : float, optional
        Helium mass fraction. Default 0.28.

    Notes
    -----
    The instance is frozen (immutable). Use :meth:`with_temperature` to obtain
    a copy with a modified temperature, e.g. for radiative-feedback iterations.
    """

    T: float
    rho: float
    mu: float
    gamma: float = 5.0 / 3.0
    X: float = 0.7
    Y: float = 0.28

    @property
    def cs(self) -> float:
        """Adiabatic sound speed [cm s^-1]."""
        return float(np.sqrt(self.gamma * kB * self.T / (self.mu * m_p)))

    @property
    def n_e(self) -> float:
        """Electron number density assuming full ionization [cm^-3]."""
        return (self.X + 0.5 * self.Y) * self.rho / m_p

    @property
    def n_i(self) -> float:
        """Ion number density assuming full ionization (H + He) [cm^-3]."""
        return (self.X + self.Y / 4.0) * self.rho / m_p

    def with_temperature(self, T: float) -> AmbientMedium:
        """Return a copy with a modified temperature."""
        return replace(self, T=T)

    def with_density(self, rho: float) -> AmbientMedium:
        """Return a copy with a modified density."""
        return replace(self, rho=rho)
