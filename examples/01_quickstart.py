"""Quickstart example: solve a Bondi+cooling problem and print the result.

Run with:

    python examples/01_quickstart.py
"""

import radbondi as rb


def main():
    # 1. Ambient medium (solar core)
    ambient = rb.presets.solar_core()

    # 2. Default cooling (bremsstrahlung + e+e- pair annihilation + mu+mu-)
    cooling = rb.Cooling.default()

    # 3. Define the problem: a 1e-16 Msun primordial black hole
    problem = rb.BondiProblem(
        M_BH=1e-16 * rb.M_sun,
        ambient=ambient,
        cooling=cooling,
    )

    print(f"Bondi radius:    r_B = {problem.r_B:.3e} cm")
    print(f"Schwarzschild:   r_S = {problem.r_S:.3e} cm")
    print(f"Adiabatic Mdot:  Mdot_B = {problem.Mdot_B:.3e} g/s")
    print()

    # 4. Solver configuration
    config = rb.SolverConfig(
        N=400, x_min=1e-5, x_max=3.0,
        n_steps=30_000, cooling_ramp_steps=3_000,
        order=1, flux="hll",
        snapshot_interval=10_000,
    )

    # 5. Solve to steady state
    sol = problem.solve(config)

    # 6. Inspect results
    print()
    print(f"Converged:       {sol.converged}")
    print(f"Final residual:  {sol.solver_residual:.3e}")
    print(f"eta:             {sol.eta(cooling):.3e}")
    print(f"Mdot/Mdot_B:     {sol.mdot_ratio:.2f}")
    print(f"T_max / T_inf:   {sol.T.max() / ambient.T:.1f}")
    print(f"Mach_max:        {sol.Mach.max():.1f}")

    # 7. Verify steady state
    res = sol.check_steady_state(cooling)
    print()
    print(f"Mass flux RMS:   {res.mass_rms:.3e}")
    print(f"Momentum RMS:    {res.momentum_rms:.3e}")
    print(f"Energy RMS:      {res.energy_rms:.3e}")


if __name__ == "__main__":
    main()
