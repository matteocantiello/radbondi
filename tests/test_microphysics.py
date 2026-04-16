"""Microphysics validation: known limits of bremsstrahlung and pair emissivity."""

import numpy as np
import pytest

import radbondi as rb
from radbondi.constants import c_light, kB, m_e
from radbondi.cooling import PairAnnihilation, RelativisticBremsstrahlung


def _theta_e(T_K: float) -> float:
    return kB * T_K / (m_e * c_light**2)


# ── Bremsstrahlung ────────────────────────────────────────────────────────


def test_brems_nonrelativistic_scaling():
    """In the non-relativistic limit, electron-ion bremsstrahlung scales as
    n_e * n_i * T^(1/2). Doubling T should multiply the eps by ~sqrt(2)
    (with weak corrections from theta_e^1.34 terms at theta_e ~ 1e-3)."""
    proc = RelativisticBremsstrahlung()
    amb = rb.presets.solar_core()
    rho = np.array([100.0])
    # T = 1e6 K -> theta_e ~ 1.7e-4: deeply non-relativistic
    eps1 = proc.emissivity(rho, np.array([1e6]), amb)
    eps2 = proc.emissivity(rho, np.array([4e6]), amb)
    # T scales by 4 -> eps scales by sqrt(4) = 2 (within ~5% from theta corrections)
    ratio = eps2[0] / eps1[0]
    assert ratio == pytest.approx(2.0, rel=0.05)


def test_brems_density_squared_at_low_T():
    """Bremsstrahlung is dominated by e-i (proportional to n_e * n_i, both
    ~rho), so eps scales as rho^2 at fixed T. e-e contribution adds an
    additional rho^2 piece, so the rho^2 scaling is exact."""
    proc = RelativisticBremsstrahlung()
    amb = rb.presets.solar_core()
    T = np.array([1e8])
    eps1 = proc.emissivity(np.array([10.0]), T, amb)
    eps2 = proc.emissivity(np.array([20.0]), T, amb)
    assert eps2[0] / eps1[0] == pytest.approx(4.0, rel=1e-6)


def test_brems_ultra_relativistic_scaling():
    """In the ultra-relativistic limit (theta_e >> 1) both F_ei and F_ee
    scale as theta * log(theta), so doubling T should give a factor of
    roughly 2.0-2.4. We probe theta_e ~ 5-10 (T ~ 3e10-6e10 K) to stay
    inside the formula's domain (clipped at theta_e = 20)."""
    proc = RelativisticBremsstrahlung()
    amb = rb.presets.solar_core()
    rho = np.array([100.0])
    eps1 = proc.emissivity(rho, np.array([3e10]), amb)
    eps2 = proc.emissivity(rho, np.array([6e10]), amb)
    ratio = eps2[0] / eps1[0]
    assert 1.9 < ratio < 2.5


# ── Pair annihilation ────────────────────────────────────────────────────


def test_pair_subdominant_at_3e8_K():
    """Below ~3e8 K (theta_e ~ 0.05), thermal pair production via
    Maxwell-Juttner is several orders of magnitude smaller than
    bremsstrahlung."""
    pair = PairAnnihilation(species="electron")
    brems = RelativisticBremsstrahlung()
    amb = rb.presets.solar_core()
    rho = np.array([150.0])
    eps_pair = pair.emissivity(rho, np.array([3e8]), amb)
    eps_brems = brems.emissivity(rho, np.array([3e8]), amb)
    assert eps_pair[0] / eps_brems[0] < 1e-5


def test_pair_sharp_turn_on_near_pair_threshold():
    """Pair emissivity increases by many orders of magnitude across the
    pair-production threshold, reflecting the K_2(1/theta) ~ exp(-1/theta)
    suppression at theta_e << 1. From T = 3e8 to 1e9 K (a factor of 3),
    pair eps grows by > 10 orders of magnitude."""
    pair = PairAnnihilation(species="electron")
    amb = rb.presets.solar_core()
    rho = np.array([150.0])
    eps_low = pair.emissivity(rho, np.array([3e8]), amb)
    eps_high = pair.emissivity(rho, np.array([1e9]), amb)
    assert eps_high[0] / eps_low[0] > 1e10


def test_pair_turns_on_above_threshold():
    """Pair production turns on sharply near theta_e ~ 1 (T ~ 6e9 K)."""
    proc = PairAnnihilation(species="electron")
    amb = rb.presets.solar_core()
    eps_low = proc.emissivity(np.array([150.0]), np.array([3e9]), amb)
    eps_high = proc.emissivity(np.array([150.0]), np.array([3e10]), amb)
    # 10x increase in T should produce many orders of magnitude in pair eps
    assert eps_high[0] / max(eps_low[0], 1e-300) > 1e6


def test_pair_muon_negligible_at_pair_threshold():
    """Muon pair annihilation requires T ~ m_mu c^2 / k_B ~ 1.2e12 K, so
    at the electron pair threshold (T ~ 6e9 K) the muon channel is
    completely negligible."""
    e_proc = PairAnnihilation(species="electron")
    mu_proc = PairAnnihilation(species="muon")
    amb = rb.presets.solar_core()
    rho = np.array([150.0])
    T = np.array([1e10])  # past electron threshold, far below muon threshold
    eps_e = e_proc.emissivity(rho, T, amb)
    eps_mu = mu_proc.emissivity(rho, T, amb)
    assert eps_mu[0] / eps_e[0] < 1e-15


# ── Net emissivity ────────────────────────────────────────────────────────


def test_net_emissivity_no_negative_values():
    """net_emissivity should never go below zero, even where total < ambient."""
    cool = rb.Cooling.default()
    amb = rb.presets.solar_core()
    # Cooler than ambient -> total emissivity below ambient -> floored at 0
    rho = np.array([amb.rho])
    T = np.array([amb.T * 0.9])
    eps_net = cool.net_emissivity(rho, T, amb)
    assert eps_net[0] >= 0.0


def test_cooling_default_matches_explicit_construction():
    """Cooling.default() should give the same results as building it manually."""
    explicit = rb.Cooling(
        [
            RelativisticBremsstrahlung(),
            PairAnnihilation(species="electron"),
            PairAnnihilation(species="muon"),
        ]
    )
    default = rb.Cooling.default()
    amb = rb.presets.solar_core()
    rho = np.array([1e3, 1e5])
    T = np.array([1e9, 1e10])
    e1 = explicit.total_emissivity(rho, T, amb)
    e2 = default.total_emissivity(rho, T, amb)
    np.testing.assert_array_almost_equal(e1, e2)
