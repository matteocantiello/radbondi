"""Log-spaced spherical finite-volume grid in r/r_B coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Spherical finite-volume grid (logarithmic in radius).

    Faces are placed geometrically between ``x_min * r_B`` and
    ``x_max * r_B``; cell centers are the geometric mean of adjacent faces.

    Attributes
    ----------
    r_face : ndarray, shape (N+1,)
        Radial face positions [cm].
    r_cen : ndarray, shape (N,)
        Cell-center radii [cm].
    dr : ndarray, shape (N,)
        Cell widths ``r_face[i+1] - r_face[i]`` [cm].
    vol : ndarray, shape (N,)
        Cell volumes ``(r_face[i+1]^3 - r_face[i]^3) / 3`` [cm^3 / 4pi].
    area : ndarray, shape (N+1,)
        Face areas ``r_face^2`` [cm^2 / 4pi].
    N : int
        Number of cells.
    r_B : float
        Bondi radius used to set the domain [cm].
    x_min, x_max : float
        Inner / outer boundaries in units of r_B.
    """

    r_face: np.ndarray
    r_cen: np.ndarray
    dr: np.ndarray
    vol: np.ndarray
    area: np.ndarray
    N: int
    r_B: float
    x_min: float
    x_max: float

    @classmethod
    def log_spaced(cls, r_B: float, N: int = 800, x_min: float = 1e-3, x_max: float = 3.0) -> Grid:
        """Construct a logarithmically spaced grid.

        Parameters
        ----------
        r_B : float
            Bondi radius [cm].
        N : int
            Number of cells.
        x_min, x_max : float
            Inner and outer boundaries in units of ``r_B``.
        """
        r_min = x_min * r_B
        r_max = x_max * r_B
        r_face = np.geomspace(r_min, r_max, N + 1)
        r_cen = np.sqrt(r_face[:-1] * r_face[1:])
        dr = r_face[1:] - r_face[:-1]
        vol = (r_face[1:] ** 3 - r_face[:-1] ** 3) / 3.0
        area = r_face**2
        return cls(
            r_face=r_face, r_cen=r_cen, dr=dr, vol=vol, area=area,
            N=N, r_B=r_B, x_min=x_min, x_max=x_max,
        )

    @property
    def r_min(self) -> float:
        return float(self.r_face[0])

    @property
    def r_max(self) -> float:
        return float(self.r_face[-1])

    @property
    def x(self) -> np.ndarray:
        """Cell-center radii in units of r_B."""
        return self.r_cen / self.r_B
