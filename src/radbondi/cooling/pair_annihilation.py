"""Thermal pair annihilation emissivity."""

from __future__ import annotations

import numpy as np
from scipy.special import kn

from radbondi.constants import c_light, hbar, kB, m_e, m_mu, sigma_T
from radbondi.cooling.base import CoolingProcess

_lambda_c_e = hbar / (m_e * c_light)
_lambda_c_mu = hbar / (m_mu * c_light)
_sigma_T_mu = sigma_T * (m_e / m_mu) ** 2


class PairAnnihilation(CoolingProcess):
    """Thermal lepton-antilepton pair annihilation, l+l- -> 2 gamma.

    The thermal pair density follows from the Maxwell-Juttner distribution and
    is approximated by

        n_pair = (theta / pi^2 lambda_c^3) * K_2(1/theta)

    where ``theta = k_B T / (m c^2)`` and ``K_2`` is the modified Bessel
    function of the second kind. The thermally averaged annihilation
    cross-section uses the Stepney (1983) interpolation.

    Two species are supported via the ``species`` keyword:

    * ``"electron"`` (default): e+ e- -> 2 gamma. Turns on sharply at
      ``T ~ 6e9`` K (``theta_e ~ 1``).
    * ``"muon"``: mu+ mu- -> 2 gamma. Turns on at ``T ~ 1.2e12`` K and is
      negligible at all temperatures reached in typical stellar-interior
      accretion problems, but included for completeness.

    References
    ----------
    Svensson 1982, ApJ 258, 321.
    Stepney 1983, MNRAS 202, 467.
    """

    T_max: float = 1e12

    def __init__(self, species: str = "electron"):
        if species == "electron":
            self.mass = m_e
            self.lambda_c = _lambda_c_e
            self.sigma_class = sigma_T
        elif species == "muon":
            self.mass = m_mu
            self.lambda_c = _lambda_c_mu
            self.sigma_class = _sigma_T_mu
        else:
            raise ValueError(f"Unknown species {species!r}; use 'electron' or 'muon'.")
        self.species = species

    def emissivity(self, rho, T, ambient) -> np.ndarray:
        T_safe = np.clip(np.asarray(T, dtype=float), 0.0, self.T_max)
        theta = kB * T_safe / (self.mass * c_light**2)
        theta = np.clip(theta, 0.0, 20.0)
        result = np.zeros_like(theta)
        mask = theta > 1e-4
        if not np.any(mask):
            return result
        th = theta[mask]
        n_pair = (1.0 / (np.pi**2 * self.lambda_c**3)) * th * kn(2, 1.0 / th)
        sigma_v = self.sigma_class * c_light * np.where(
            th < 1.0,
            1.0 + 0.5 * th,
            np.pi / (2.0 * th) * (np.log(2.0 * th) + 0.5),
        )
        result[mask] = n_pair**2 * sigma_v * 2.0 * self.mass * c_light**2
        return result
