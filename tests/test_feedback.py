"""Smoke tests for the feedback modules."""

import pytest

import radbondi as rb
from radbondi.feedback import DiffusionFeedback, MLTEnvelope


def test_diffusion_feedback_zero_luminosity():
    """Zero luminosity -> no temperature enhancement."""
    amb = rb.presets.solar_core()
    fb = DiffusionFeedback(amb, kappa=1.0)
    res = fb.feedback_temperature(L_BH=0.0)
    assert res.x == pytest.approx(1.0)
    assert res.T_eff == pytest.approx(amb.T)


def test_diffusion_feedback_positive_luminosity():
    """Positive luminosity raises T_eff."""
    amb = rb.presets.solar_core()
    fb = DiffusionFeedback(amb, kappa=1.0)
    res = fb.feedback_temperature(L_BH=1e30)
    assert res.x > 1.0
    assert res.T_eff > amb.T


def test_mlt_envelope_construction():
    """MLT envelope can be constructed and integrated without error."""
    amb = rb.presets.solar_core()
    env = MLTEnvelope(ambient=amb, M_BH=1e-13 * rb.M_sun)
    # Low luminosity: T_eff ~ T_core
    T_eff = env.feedback_temperature(L_BH=1e15)
    assert 0.95 * amb.T < T_eff < 1.5 * amb.T
