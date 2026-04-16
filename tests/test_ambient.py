"""Smoke tests for AmbientMedium and presets."""

import numpy as np
import pytest

import radbondi as rb


def test_solar_core_preset():
    amb = rb.presets.solar_core()
    assert amb.T == pytest.approx(1.57e7)
    assert amb.rho == pytest.approx(150.0)
    assert amb.mu == pytest.approx(0.85)


def test_sound_speed():
    amb = rb.presets.solar_core()
    # cs = sqrt(gamma kT / mu m_p) ~ 5e7 cm/s for solar core
    assert 4.9e7 < amb.cs < 5.1e7


def test_with_temperature_immutable():
    amb = rb.presets.solar_core()
    amb2 = amb.with_temperature(2 * amb.T)
    # Original unchanged
    assert amb.T == pytest.approx(1.57e7)
    # Copy has new T
    assert amb2.T == pytest.approx(2 * 1.57e7)
    # Sound speed scales with sqrt(T)
    assert amb2.cs == pytest.approx(amb.cs * np.sqrt(2.0), rel=1e-10)
