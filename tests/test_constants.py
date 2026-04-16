"""Smoke test for the constants module — verifies the package imports."""

import radbondi as rb
from radbondi import constants


def test_M_sun_value():
    """Solar mass is roughly 2e33 g."""
    assert 1.98e33 < rb.M_sun < 1.99e33


def test_M_sun_reexport():
    """rb.M_sun and rb.constants.M_sun are the same value."""
    assert rb.M_sun == constants.M_sun


def test_speed_of_light():
    """c ~ 3e10 cm/s."""
    assert 2.99e10 < constants.c_light < 3.00e10


def test_version():
    """Package exposes __version__."""
    assert isinstance(rb.__version__, str)
