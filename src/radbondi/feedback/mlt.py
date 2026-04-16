"""Mixing-length-theory envelope model for radiative feedback.

Computes the effective ambient temperature T_inf' for the Bondi problem by
integrating a 1D hydrostatic envelope from r >> r_B inward, using mixing
length theory for convective energy transport.

This is more physical than the pure-diffusion model (:mod:`.diffusion`) when
the radiative gradient exceeds the adiabatic gradient (``beta >> 1``) and
convection becomes important. Convection saturates the temperature
enhancement that pure diffusion would otherwise predict.

References
----------
See Section 2.4 of Cantiello et al. (in prep).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from radbondi.constants import G, a_rad, c_light, kB, m_p


@dataclass
class EnvelopeProfile:
    """Hydrostatic envelope profile from MLT integration.

    All arrays are on the radial integration grid (log-spaced from r_out
    inward).
    """

    r: np.ndarray            # radius [cm]
    T: np.ndarray            # temperature [K]
    rho: np.ndarray          # density [g cm^-3]
    P: np.ndarray            # pressure [erg cm^-3]
    nabla: np.ndarray        # actual d ln T / d ln P
    nabla_rad: np.ndarray    # radiative d ln T / d ln P
    f_conv: np.ndarray       # convective flux fraction (0 if radiative)
    T_eff: float             # T at the photon coupling radius [K]
    x: float                 # T_eff / T_core
    r_B: float               # Bondi radius [cm]
    r_c: float               # coupling radius [cm] (NaN if no kappa_BH)
    r_inner: float           # actual inner integration boundary [cm]


class MLTEnvelope:
    """MLT envelope feedback model.

    Parameters
    ----------
    ambient : AmbientMedium
        Unperturbed ambient medium.
    M_BH : float
        Black hole mass [g].
    kappa_env : float
        Opacity for radiative transport in the envelope [cm^2 g^-1].
        Typically electron scattering, ``0.2 * (1 + X)``.
    kappa_BH : float, optional
        Opacity for the BH photon spectrum [cm^2 g^-1]. Determines the
        coupling radius ``r_c = 1 / (kappa_BH * rho)`` where photons
        thermalize. If ``None``, integration goes all the way to ``r_B``.
    alpha_mlt : float
        Mixing-length parameter. Default 1.5.
    r_out_factor : float
        Outer integration boundary in units of r_B. Default 200.
    n_points : int
        Number of radial grid points (log-spaced). Default 2000.
    """

    def __init__(
        self,
        ambient,
        M_BH: float,
        kappa_env: float | None = None,
        kappa_BH: float | None = None,
        alpha_mlt: float = 1.5,
        r_out_factor: float = 200.0,
        n_points: int = 2000,
    ):
        self.ambient = ambient
        self.M_BH = float(M_BH)
        if kappa_env is None:
            # Electron scattering opacity for fully ionized H+He
            kappa_env = 0.2 * (1.0 + ambient.X)
        self.kappa_env = float(kappa_env)
        self.kappa_BH = float(kappa_BH) if kappa_BH is not None else None
        self.alpha_mlt = float(alpha_mlt)
        self.r_out_factor = float(r_out_factor)
        self.n_points = int(n_points)

        # Derived
        self._nabla_ad = (ambient.gamma - 1.0) / ambient.gamma
        self._c_p = ambient.gamma / (ambient.gamma - 1.0) * kB / (ambient.mu * m_p)
        cs = ambient.cs
        self._r_B = G * self.M_BH / cs**2

    def feedback_temperature(self, L_BH: float) -> float:
        """Convenience: integrate the envelope and return ``T_eff`` [K]."""
        return self.integrate(L_BH).T_eff

    def integrate(self, L_BH: float) -> EnvelopeProfile:
        """Integrate the envelope and return the full profile."""
        T_core = self.ambient.T
        rho_core = self.ambient.rho
        mu = self.ambient.mu
        r_B = self._r_B

        # Estimate coupling radius from ambient density
        r_c_estimate = (
            1.0 / (self.kappa_BH * rho_core) if self.kappa_BH is not None else 0.0
        )
        r_inner = max(r_c_estimate, r_B)
        r_out = self.r_out_factor * r_B

        if r_inner >= r_out:
            # No envelope to integrate — coupling is outside the domain
            P_core = rho_core * kB * T_core / (mu * m_p)
            return EnvelopeProfile(
                r=np.array([r_out]), T=np.array([T_core]),
                rho=np.array([rho_core]), P=np.array([P_core]),
                nabla=np.array([0.0]), nabla_rad=np.array([0.0]),
                f_conv=np.array([0.0]),
                T_eff=T_core, x=1.0, r_B=r_B,
                r_c=r_c_estimate if self.kappa_BH else float("nan"),
                r_inner=r_inner,
            )

        N = self.n_points
        r = np.logspace(np.log10(r_out), np.log10(r_inner), N)
        T_arr = np.zeros(N)
        rho_arr = np.zeros(N)
        P_arr = np.zeros(N)
        nabla_arr = np.zeros(N)
        nabla_rad_arr = np.zeros(N)
        f_conv_arr = np.zeros(N)

        T_arr[0] = T_core
        rho_arr[0] = rho_core
        P_arr[0] = rho_core * kB * T_core / (mu * m_p)

        r_c_local = float("nan")
        i_couple = N - 1

        for i in range(N - 1):
            ri = r[i]
            Ti = T_arr[i]
            rhoi = rho_arr[i]
            Pi = P_arr[i]

            if self.kappa_BH is not None and np.isnan(r_c_local):
                r_c_here = 1.0 / (self.kappa_BH * rhoi)
                if ri <= r_c_here:
                    r_c_local = ri
                    i_couple = i

            nab, nab_r, fc = self._nabla(Ti, rhoi, Pi, ri, L_BH)
            nabla_arr[i] = nab
            nabla_rad_arr[i] = nab_r
            f_conv_arr[i] = fc

            ri1 = r[i + 1]
            r_mid = 0.5 * (ri + ri1)
            g_mid = G * self.M_BH / r_mid**2
            dr = ri - ri1                 # positive (stepping inward)
            dP = rhoi * g_mid * dr
            P_new = Pi + dP
            dlnP = np.log(P_new / Pi)
            T_new = Ti * np.exp(nab * dlnP)
            rho_new = P_new * mu * m_p / (kB * T_new)
            T_arr[i + 1] = T_new
            rho_arr[i + 1] = rho_new
            P_arr[i + 1] = P_new

        # Final-point gradient (for diagnostics)
        nab, nab_r, fc = self._nabla(T_arr[-1], rho_arr[-1], P_arr[-1], r[-1], L_BH)
        nabla_arr[-1] = nab
        nabla_rad_arr[-1] = nab_r
        f_conv_arr[-1] = fc

        T_eff = float(T_arr[i_couple])
        if self.kappa_BH is None:
            r_c_local = float("nan")
        return EnvelopeProfile(
            r=r, T=T_arr, rho=rho_arr, P=P_arr,
            nabla=nabla_arr, nabla_rad=nabla_rad_arr, f_conv=f_conv_arr,
            T_eff=T_eff, x=T_eff / T_core,
            r_B=r_B,
            r_c=r_c_local if not np.isnan(r_c_local) else r_c_estimate,
            r_inner=r_inner,
        )

    def _nabla(self, T, rho, P, r, L):
        """Compute the actual gradient via MLT (private helper)."""
        g = G * self.M_BH / r**2
        H_P = P / (rho * g)
        F_total = L / (4.0 * np.pi * r**2)
        F_rad_coeff = 4.0 * a_rad * c_light * T**4 / (3.0 * self.kappa_env * rho * H_P)
        nabla_r = F_total / F_rad_coeff
        if nabla_r <= self._nabla_ad:
            return nabla_r, nabla_r, 0.0
        # Convective: solve F_rad(nabla) + F_conv(nabla) = F_total
        A_conv = rho * self._c_p * T * (self.alpha_mlt / 2.0) * np.sqrt(g * H_P)
        nabla_ad = self._nabla_ad

        def residual(nab):
            F_r = F_rad_coeff * nab
            dn = max(nab - nabla_ad, 0.0)
            F_c = A_conv * dn**1.5
            return F_r + F_c - F_total

        try:
            nab = brentq(residual, nabla_ad, nabla_r, xtol=1e-10, rtol=1e-10)
        except ValueError:
            nab = nabla_ad
        F_r = F_rad_coeff * nab
        F_c = F_total - F_r
        f_conv = max(F_c / F_total, 0.0)
        return nab, nabla_r, f_conv
