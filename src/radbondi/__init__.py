"""radbondi: time-dependent spherical Bondi accretion with radiative cooling."""

from radbondi import constants, presets
from radbondi.ambient import AmbientMedium
from radbondi.constants import M_sun
from radbondi.cooling import Cooling, CoolingProcess
from radbondi.grid import Grid
from radbondi.ode import ODESolverConfig, solve_ode
from radbondi.solution import Solution, load
from radbondi.solver import BondiProblem, SolverConfig

__version__ = "0.2.0"

__all__ = [
    "AmbientMedium",
    "BondiProblem",
    "Cooling",
    "CoolingProcess",
    "Grid",
    "M_sun",
    "ODESolverConfig",
    "Solution",
    "SolverConfig",
    "constants",
    "load",
    "presets",
    "solve_ode",
    "__version__",
]
