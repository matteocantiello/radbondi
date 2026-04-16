"""Relativistic bremsstrahlung emissivity."""

from __future__ import annotations

import numpy as np

from radbondi.constants import alpha_fs, c_light, kB, m_e, m_p, sigma_T
from radbondi.cooling.base import CoolingProcess


class RelativisticBremsstrahlung(CoolingProcess):
    """Relativistic bremsstrahlung (electron-ion + electron-electron).

    Uses the fitting functions of Stepney & Guilbert (1983) interpolating
    between the non-relativistic ``theta_e << 1`` and ultra-relativistic
    ``theta_e >> 1`` limits, where
    ``theta_e = k_B T / (m_e c^2)`` is the dimensionless electron temperature.

    The emissivity assumes a fully ionized H+He plasma with composition
    given by the ``ambient.X`` and ``ambient.Y`` mass fractions.

    References
    ----------
    Stepney & Guilbert 1983, MNRAS 204, 1269.
    """

    T_max: float = 1e12
    rho_max: float = 1e15

    def emissivity(self, rho, T, ambient) -> np.ndarray:
        rho_safe = np.clip(np.asarray(rho, dtype=float), 0.0, self.rho_max)
        T_safe = np.clip(np.asarray(T, dtype=float), 0.0, self.T_max)
        theta_e = kB * T_safe / (m_e * c_light**2)
        theta_e = np.clip(theta_e, 0.0, 20.0)

        n_e = (ambient.X + 0.5 * ambient.Y) * rho_safe / m_p
        # sum_i Z_i^2 n_i for fully ionized H + He: Z=1 for H, Z=2 for He
        # n_H = X*rho/m_p, n_He = Y*rho/(4*m_p), so
        # sum Z^2 n_i = X*rho/m_p + 4 * Y*rho/(4*m_p) = (X + Y) * rho / m_p
        sum_Z2_ni = (ambient.X + ambient.Y) * rho_safe / m_p

        q_b = sigma_T * c_light * alpha_fs * m_e * c_light**2

        # Electron-ion bremsstrahlung
        F_ei = np.where(
            theta_e < 1.0,
            4.0 * np.sqrt(2.0 * theta_e / np.pi**3) * (1.0 + 1.781 * theta_e**1.34),
            (9.0 * theta_e / (2.0 * np.pi))
            * (np.log(1.123 * theta_e + 0.48) + 1.5),
        )

        # Electron-electron bremsstrahlung
        gamma_E = 0.5772
        F_ee = np.where(
            theta_e < 1.0,
            (20.0 / (9.0 * np.sqrt(np.pi)))
            * (44.0 - 3.0 * np.pi**2)
            * theta_e**1.5
            * (1.0 + 1.1 * theta_e + theta_e**2 - 1.25 * theta_e**2.5),
            24.0 * theta_e * (np.log(2.0 * np.exp(-gamma_E) * theta_e) + 1.28),
        )

        return q_b * n_e * sum_Z2_ni * F_ei + q_b * n_e**2 * F_ee
