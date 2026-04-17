"""Tests for the unified solve() API and solve_with_feedback()."""

import pytest

import radbondi as rb
from radbondi.feedback import DiffusionFeedback


def test_solve_default_is_time_dependent():
    """problem.solve() without arguments uses the time-dependent solver."""
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(
        M_BH=1e-16 * rb.M_sun, ambient=amb, cooling=rb.Cooling.adiabatic()
    )
    sol = problem.solve(rb.SolverConfig(
        N=100, n_steps=200, cooling_ramp_steps=0, verbose=False,
    ))
    assert sol.converged
    assert sol.metadata.get("method", "time_dependent") == "time_dependent"


def test_solve_method_ode():
    """problem.solve(method='ode') dispatches to the ODE solver."""
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb)
    sol = problem.solve(method="ode", config=rb.ODESolverConfig(verbose=False))
    assert sol.metadata["method"] == "ode"
    assert 1e-3 < sol.eta < 0.1


def test_solve_auto_detects_ode_config():
    """Passing an ODESolverConfig auto-selects the ODE method."""
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb)
    sol = problem.solve(rb.ODESolverConfig(verbose=False))
    assert sol.metadata["method"] == "ode"


def test_solve_unknown_method_raises():
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(M_BH=1e-16 * rb.M_sun, ambient=amb)
    with pytest.raises(ValueError, match="Unknown method"):
        problem.solve(method="magic")


def test_solve_with_feedback_converges():
    """solve_with_feedback iterates and converges on a feedback-dominated mass."""
    amb = rb.presets.solar_core()
    problem = rb.BondiProblem(M_BH=1e-13 * rb.M_sun, ambient=amb)
    feedback = DiffusionFeedback(amb, kappa=1.0)
    cfg = rb.SolverConfig(
        N=200, x_min=1e-5, n_steps=15_000, cooling_ramp_steps=2_000,
        order=1, flux="hll", snapshot_interval=50_000, verbose=False,
    )
    sol = problem.solve_with_feedback(feedback, config=cfg, verbose=False)
    assert sol.metadata["feedback_converged"]
    assert sol.metadata["feedback_iterations"] <= 8
    assert sol.metadata["feedback_T_eff"] > amb.T
