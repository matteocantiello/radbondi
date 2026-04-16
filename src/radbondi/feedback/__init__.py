"""Optional radiative-feedback models (off by default).

The core :class:`~radbondi.BondiProblem` solves the accretion problem for a
fixed ambient medium. Feedback models compute a modified effective ambient
temperature given the BH luminosity, allowing iterative self-consistent
solutions:

    1. Solve the Bondi problem -> get L
    2. Apply feedback model -> get T_eff
    3. Re-solve with ambient.with_temperature(T_eff)
    4. Repeat until L converges
"""

from radbondi.feedback.diffusion import DiffusionFeedback, DiffusionFeedbackResult
from radbondi.feedback.mlt import EnvelopeProfile, MLTEnvelope

__all__ = [
    "DiffusionFeedback",
    "DiffusionFeedbackResult",
    "EnvelopeProfile",
    "MLTEnvelope",
]
