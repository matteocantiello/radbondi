"""Base classes for cooling processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from radbondi.ambient import AmbientMedium


class CoolingProcess(ABC):
    """Abstract base class for a microphysical cooling process.

    Subclasses must implement :meth:`emissivity` returning the volumetric
    emissivity (energy lost per unit volume per unit time) as a function of
    local density and temperature.
    """

    @abstractmethod
    def emissivity(self, rho, T, ambient: AmbientMedium) -> np.ndarray:
        """Volumetric emissivity [erg cm^-3 s^-1].

        Parameters
        ----------
        rho : array_like
            Mass density [g cm^-3].
        T : array_like
            Temperature [K].
        ambient : AmbientMedium
            Ambient medium (provides composition X, Y, etc.).
        """
        ...


class Cooling:
    """A collection of cooling processes whose emissivities sum.

    Use the alternate constructors :meth:`default` or :meth:`adiabatic`, or
    pass a list of :class:`CoolingProcess` instances to combine arbitrary
    processes.
    """

    def __init__(self, processes: list[CoolingProcess] | None = None):
        self.processes: list[CoolingProcess] = list(processes) if processes else []

    def total_emissivity(self, rho, T, ambient: AmbientMedium) -> np.ndarray:
        """Sum of emissivities from all processes [erg cm^-3 s^-1]."""
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if not self.processes:
            return np.zeros_like(rho)
        eps = np.zeros_like(rho)
        for proc in self.processes:
            eps = eps + proc.emissivity(rho, T, ambient)
        return eps

    def net_emissivity(
        self, rho, T, ambient: AmbientMedium, ambient_emissivity: float | None = None
    ) -> np.ndarray:
        """Emissivity above the ambient value (clipped at zero).

        The ambient floor prevents the solver from cooling gas below the
        ambient temperature; physically, the surrounding medium is assumed to
        radiate at the same rate, so only the *excess* removes net energy
        from the accretion flow.

        Parameters
        ----------
        ambient_emissivity : float, optional
            Pre-computed emissivity at (ambient.rho, ambient.T). If ``None``,
            it is computed on the fly. Pass a cached value for efficiency.
        """
        if ambient_emissivity is None:
            ambient_emissivity = self.ambient_emissivity(ambient)
        return np.maximum(self.total_emissivity(rho, T, ambient) - ambient_emissivity, 0.0)

    def ambient_emissivity(self, ambient: AmbientMedium) -> float:
        """Total emissivity evaluated at ambient (rho, T) — the floor value."""
        return float(self.total_emissivity(ambient.rho, ambient.T, ambient)[0])

    @classmethod
    def default(cls) -> Cooling:
        """Bremsstrahlung + electron-positron pair annihilation.

        Suitable for hot accretion flows onto compact objects in dense
        environments. Muon pair annihilation is included but typically
        negligible at the temperatures reached in stellar-interior settings.
        """
        from radbondi.cooling.bremsstrahlung import RelativisticBremsstrahlung
        from radbondi.cooling.pair_annihilation import PairAnnihilation

        return cls(
            [
                RelativisticBremsstrahlung(),
                PairAnnihilation(species="electron"),
                PairAnnihilation(species="muon"),
            ]
        )

    @classmethod
    def adiabatic(cls) -> Cooling:
        """No cooling — the flow evolves adiabatically."""
        return cls([])

    def __repr__(self) -> str:
        if not self.processes:
            return "Cooling(adiabatic)"
        names = ", ".join(type(p).__name__ for p in self.processes)
        return f"Cooling([{names}])"
