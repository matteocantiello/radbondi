"""Cooling processes (pluggable microphysics).

Add a new process by subclassing :class:`CoolingProcess` and implementing
``emissivity(rho, T, ambient)``. Combine multiple processes in a
:class:`Cooling` collection.
"""

from radbondi.cooling.base import Cooling, CoolingProcess
from radbondi.cooling.bremsstrahlung import RelativisticBremsstrahlung
from radbondi.cooling.pair_annihilation import PairAnnihilation

__all__ = [
    "Cooling",
    "CoolingProcess",
    "RelativisticBremsstrahlung",
    "PairAnnihilation",
]
