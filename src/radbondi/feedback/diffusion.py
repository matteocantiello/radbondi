"""Pure radiative diffusion feedback model.

Solves the algebraic self-consistency equation

    x^4 = 1 + beta * x^(-3/2)

where ``x = T_inf' / T_core`` is the fractional temperature enhancement at the
photon coupling radius and

    beta = 3 (kappa rho)^2 L0 / (4 pi a c T_core^4)

is the dimensionless feedback parameter (see Section 2.4 of Cantiello et al.).

This is the simplest feedback prescription — it assumes purely radiative
diffusion. The MLT envelope model (:mod:`radbondi.feedback.mlt`) handles
strong feedback (beta >> 1) more physically by accounting for convective
energy transport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from radbondi.constants import a_rad, c_light


@dataclass
class DiffusionFeedbackResult:
    """Output of :class:`DiffusionFeedback.feedback_temperature`."""

    T_eff: float    # effective ambient temperature seen by Bondi flow [K]
    x: float        # T_eff / T_core
    beta: float     # dimensionless feedback parameter


class DiffusionFeedback:
    """Radiative-diffusion feedback model.

    Parameters
    ----------
    ambient : AmbientMedium
        The unperturbed ambient medium (provides T_core and rho).
    kappa : float
        Opacity at the photon coupling radius [cm^2 g^-1].
    """

    def __init__(self, ambient, kappa: float):
        self.ambient = ambient
        self.kappa = float(kappa)

    def feedback_temperature(self, L_BH: float) -> DiffusionFeedbackResult:
        """Compute the effective T_inf given the BH luminosity.

        Returns ``T_eff`` such that the Bondi flow is to be re-solved with
        ``ambient.with_temperature(T_eff)``.
        """
        T_core = self.ambient.T
        beta = (
            3.0 * (self.kappa * self.ambient.rho) ** 2 * L_BH
            / (4.0 * np.pi * a_rad * c_light * T_core**4)
        )
        if beta < 1e-6:
            return DiffusionFeedbackResult(T_eff=T_core, x=1.0, beta=beta)
        x_max = max(np.sqrt(beta), 10.0)
        x_sol = brentq(lambda x: x**4 - 1.0 - beta * x ** (-1.5), 1.0, x_max)
        return DiffusionFeedbackResult(T_eff=T_core * x_sol, x=x_sol, beta=beta)
