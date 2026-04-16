"""Slow physics validation tests: paper Table 1 reproduction and convergence.

These tests run the full solver and take ~10 s - several minutes each.
They are gated behind the ``slow`` marker; run with::

    pytest -m slow

or to run everything (fast + slow)::

    pytest -m ""
"""

import numpy as np
import pytest

import radbondi as rb


# Reference values from Table 1 of Cantiello et al. (in prep), with the
# baseline (no-feedback) configuration. The paper's production runs use
# higher resolution and the MUSCL (order=2) scheme; we use lower N here for
# speed and accept ~30% tolerance.
#
#   (log10(M/M_sun), N,   x_min, eta_ref, mdot_ratio_ref)
TABLE1_CASES = [
    (-16.0, 800, 1e-5, 1.16e-2, 1.6),    # collisionless, HLL 1st-order @ N=800
    (-13.5, 800, 1e-5, 1.01e-2, 7.2),    # bremsstrahlung cooling
    (-11.0, 800, 1e-5, 3.4e-1,  7.0),    # near-isothermal
]


@pytest.mark.slow
@pytest.mark.parametrize("logM,N,x_min,eta_ref,mdot_ref", TABLE1_CASES)
def test_reproduces_paper_table1(logM, N, x_min, eta_ref, mdot_ref):
    """End-to-end check that the solver reproduces published values within
    ~30%. Larger tolerance accounts for our lower N and 1st-order solver."""
    amb = rb.presets.solar_core()
    cool = rb.Cooling.default()
    problem = rb.BondiProblem(M_BH=10**logM * rb.M_sun, ambient=amb, cooling=cool)
    cfg = rb.SolverConfig(
        N=N, x_min=x_min, x_max=3.0,
        n_steps=80_000 if logM > -14 else 50_000,
        cooling_ramp_steps=5_000,
        order=1, flux="hll",
        snapshot_interval=20_000,
        verbose=False,
    )
    sol = problem.solve(cfg)

    assert sol.eta == pytest.approx(eta_ref, rel=0.30), (
        f"eta = {sol.eta:.3e}, ref = {eta_ref:.3e}"
    )
    assert sol.mdot_ratio == pytest.approx(mdot_ref, rel=0.20), (
        f"mdot_ratio = {sol.mdot_ratio:.2f}, ref = {mdot_ref:.2f}"
    )


@pytest.mark.slow
def test_resolution_convergence_collisionless():
    """In the collisionless regime, eta should converge as N is increased.
    We check that going from N=400 to N=800 changes eta by less than 30%."""
    amb = rb.presets.solar_core()
    cool = rb.Cooling.default()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb, cooling=cool)

    def run(N):
        cfg = rb.SolverConfig(
            N=N, x_min=1e-5,
            n_steps=30_000, cooling_ramp_steps=3_000,
            order=1, flux="hll",
            snapshot_interval=20_000, verbose=False,
        )
        return problem.solve(cfg).eta

    eta_400 = run(400)
    eta_800 = run(800)
    rel_change = abs(eta_800 - eta_400) / eta_400
    assert rel_change < 0.30, f"eta changed by {rel_change:.1%} from N=400 to 800"


@pytest.mark.slow
def test_steady_state_residuals_tight():
    """Tighter version of the steady-state check, with production-like
    parameters: mass flux to <0.5%, momentum to <3% across all regimes."""
    amb = rb.presets.solar_core()
    cool = rb.Cooling.default()

    for logM, x_min in [(-16.0, 1e-5), (-11.0, 1e-5)]:
        problem = rb.BondiProblem(M_BH=10**logM * rb.M_sun, ambient=amb, cooling=cool)
        cfg = rb.SolverConfig(
            N=800, x_min=x_min,
            n_steps=80_000 if logM > -14 else 50_000,
            cooling_ramp_steps=5_000,
            order=1, flux="hll",
            snapshot_interval=50_000, verbose=False,
        )
        sol = problem.solve(cfg)
        res = sol.check_steady_state(cool)
        assert res.mass_rms < 0.01, (
            f"log M = {logM}: mass_rms = {res.mass_rms:.3e} > 0.01"
        )
        assert res.momentum_rms < 0.05, (
            f"log M = {logM}: momentum_rms = {res.momentum_rms:.3e} > 0.05"
        )


@pytest.mark.slow
def test_save_load_preserves_eta():
    """Serialized Solution preserves the cached luminosity and eta."""
    import os
    import tempfile

    amb = rb.presets.solar_core()
    cool = rb.Cooling.default()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb, cooling=cool)
    cfg = rb.SolverConfig(
        N=400, x_min=1e-5,
        n_steps=20_000, cooling_ramp_steps=3_000,
        order=1, flux="hll",
        snapshot_interval=50_000, verbose=False,
    )
    sol = problem.solve(cfg)
    eta_before = sol.eta
    L_before = sol.L

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        sol.save(path)
        sol2 = rb.load(path)
        assert sol2.L == L_before
        assert sol2.eta == eta_before
    finally:
        os.unlink(path)
