"""Tests for the ODE shooting solver and ODE vs time-dependent cross-checks."""

import numpy as np
import pytest

import radbondi as rb
from radbondi.ode import ODESolverConfig, solve_ode


# ── Fast tests ────────────────────────────────────────────────────────────


def test_ode_solver_runs_collisionless():
    """ODE solver completes for a small BH and returns sensible values."""
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb)
    sol = solve_ode(problem, ODESolverConfig(verbose=False))
    assert sol.converged
    # eta is in the right ballpark for the collisionless regime
    assert 1e-3 < sol.eta < 0.1
    # Mass conservation: Mdot ratio is 1.0 by construction (adiabatic Bondi)
    assert sol.mdot_ratio == pytest.approx(1.0, rel=1e-3)
    # Solution metadata records the method
    assert sol.metadata["method"] == "ode"


def test_ode_adiabatic_recovers_paper_value():
    """At M=1e-16 in solar core, ODE solver reproduces paper Table 1
    GR-corrected eta to within 5% (paper: 2.04e-2)."""
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb)
    sol = solve_ode(problem, ODESolverConfig(verbose=False))
    assert sol.eta == pytest.approx(2.04e-2, rel=0.05)


def test_ode_solution_save_load_roundtrip():
    """Saving and loading an ODE Solution preserves L and eta."""
    import os
    import tempfile

    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb)
    sol = solve_ode(problem, ODESolverConfig(verbose=False))

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        sol.save(path)
        sol2 = rb.load(path)
        assert sol2.L == sol.L
        assert sol2.eta == sol.eta
        assert np.array_equal(sol2.T, sol.T)
    finally:
        os.unlink(path)


# ── Slow tests: ODE vs time-dependent cross-check ─────────────────────────


@pytest.mark.slow
def test_ode_vs_timedep_collisionless():
    """In the collisionless regime where both methods are valid, eta agrees
    to within 30%. They should NOT agree exactly: the time-dependent solver
    has finite resolution and includes operator-splitting cooling artefacts;
    the ODE includes GR redshift in the luminosity integral. ~30% tolerance
    accommodates these differences."""
    amb = rb.presets.solar_core()
    cool = rb.Cooling.default()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb, cooling=cool)

    sol_ode = solve_ode(problem, ODESolverConfig(verbose=False))

    cfg_td = rb.SolverConfig(
        N=800, x_min=1e-5, n_steps=30_000, cooling_ramp_steps=3_000,
        order=1, flux="hll", snapshot_interval=30_000, verbose=False,
    )
    sol_td = problem.solve(cfg_td)
    assert sol_td.converged

    rel = abs(sol_td.eta - sol_ode.eta) / sol_ode.eta
    assert rel < 0.30, (
        f"eta_td = {sol_td.eta:.3e}, eta_ode = {sol_ode.eta:.3e}, "
        f"rel diff = {rel:.1%}"
    )
    # Both should give Mdot/Mdot_B ~ 1 in this near-adiabatic regime.
    assert abs(sol_td.mdot_ratio - 1.0) < 0.5
    assert abs(sol_ode.mdot_ratio - 1.0) < 0.05
