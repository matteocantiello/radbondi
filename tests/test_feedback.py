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


def test_mlt_saturates_below_diffusion_at_high_beta():
    """When beta >> 1, the MLT envelope saturates the temperature
    enhancement that pure diffusion would predict, because convection
    carries most of the outward flux."""
    amb = rb.presets.solar_core()
    M_BH = 1e-13 * rb.M_sun
    L_BH = 1e25  # high enough to drive beta >> 1 with thermal opacity
    kappa = 1.0  # Rosseland mean for thermal photons

    diff = DiffusionFeedback(amb, kappa=kappa).feedback_temperature(L_BH=L_BH)
    mlt = MLTEnvelope(
        ambient=amb, M_BH=M_BH, kappa_env=kappa, kappa_BH=kappa
    ).integrate(L_BH=L_BH)

    assert diff.beta > 1.0, "Test setup error: need beta >> 1"
    # MLT temperature enhancement should be smaller than diffusion's
    assert mlt.x < diff.x, (
        f"MLT x = {mlt.x:.2f} should be < diffusion x = {diff.x:.2f}"
    )
