"""Smoke tests for the cooling module."""

import numpy as np
import pytest

import radbondi as rb
from radbondi.cooling import (
    Cooling,
    PairAnnihilation,
    RelativisticBremsstrahlung,
)


def test_adiabatic_returns_zero():
    cool = Cooling.adiabatic()
    amb = rb.presets.solar_core()
    eps = cool.total_emissivity(amb.rho, amb.T, amb)
    assert np.all(eps == 0.0)


def test_default_combines_three_processes():
    cool = Cooling.default()
    assert len(cool.processes) == 3
    types = {type(p).__name__ for p in cool.processes}
    assert "RelativisticBremsstrahlung" in types
    assert "PairAnnihilation" in types


def test_bremsstrahlung_positive():
    proc = RelativisticBremsstrahlung()
    amb = rb.presets.solar_core()
    # Hot, dense gas: emissivity should be positive and substantial
    eps = proc.emissivity(np.array([1e3]), np.array([1e10]), amb)
    assert eps[0] > 0


def test_pair_annihilation_negligible_below_threshold():
    """Pair production is exponentially suppressed for theta_e << 1."""
    proc = PairAnnihilation(species="electron")
    amb = rb.presets.solar_core()
    # T = 1e7 K -> theta_e ~ 1.7e-3 -> pair density vanishes
    eps_cold = proc.emissivity(np.array([150.0]), np.array([1e7]), amb)
    # T = 1e10 K -> theta_e ~ 1.7 -> significant pair production
    eps_hot = proc.emissivity(np.array([150.0]), np.array([1e10]), amb)
    assert eps_cold[0] < eps_hot[0] * 1e-10


def test_net_emissivity_floors_at_ambient():
    cool = Cooling.default()
    amb = rb.presets.solar_core()
    # Evaluating net emissivity at ambient (rho, T) should return 0
    eps_net = cool.net_emissivity(amb.rho, amb.T, amb)
    assert eps_net[0] == pytest.approx(0.0)
