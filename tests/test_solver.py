"""End-to-end smoke tests for the solver."""

import os
import tempfile

import numpy as np

import radbondi as rb


def test_adiabatic_preserves_bondi():
    """With well-balancing on and no cooling, the adiabatic Bondi profile
    must be preserved to machine precision."""
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(
        M_BH=1e-16 * rb.M_sun, ambient=amb, cooling=rb.Cooling.adiabatic()
    )
    cfg = rb.SolverConfig(
        N=200, n_steps=2000, cooling_ramp_steps=0,
        snapshot_interval=10000, verbose=False,
    )
    sol = problem.solve(cfg)
    assert sol.converged
    assert sol.solver_residual < 1e-12
    # Mass flux exactly conserved
    assert abs(sol.mdot_ratio - 1.0) < 1e-6


def test_problem_derived_quantities():
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb)
    # r_B / r_S is mass-independent at fixed ambient: c^2 / (2 c_inf^2) ~ 1.77e5
    ratio = problem.r_B / problem.r_S
    assert 1.7e5 < ratio < 1.8e5


def test_save_load_roundtrip():
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(
        M_BH=1e-16 * rb.M_sun, ambient=amb, cooling=rb.Cooling.adiabatic()
    )
    sol = problem.solve(rb.SolverConfig(
        N=100, n_steps=200, cooling_ramp_steps=0,
        snapshot_interval=10000, verbose=False,
    ))
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        sol.save(path)
        sol2 = rb.load(path)
        assert np.array_equal(sol.rho, sol2.rho)
        assert np.array_equal(sol.T, sol2.T)
        assert sol.M_BH == sol2.M_BH
    finally:
        os.unlink(path)


def test_steady_state_diagnostics():
    amb = rb.presets.solar_core()
    cool = rb.Cooling.default()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb, cooling=cool)
    sol = problem.solve(rb.SolverConfig(
        N=200, x_min=1e-4, x_max=3.0,
        n_steps=10000, cooling_ramp_steps=2000,
        order=1, snapshot_interval=10000, verbose=False,
    ))
    res = sol.check_steady_state(cool)
    # Mass flux is the cleanest diagnostic
    assert res.mass_rms < 0.05, f"mass_rms = {res.mass_rms:.3e}"
    # Momentum should be at most a few percent
    assert res.momentum_rms < 0.10, f"momentum_rms = {res.momentum_rms:.3e}"
