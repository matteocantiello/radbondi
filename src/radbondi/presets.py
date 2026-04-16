"""Preset ambient media for common astrophysical scenarios."""

from radbondi.ambient import AmbientMedium


def solar_core() -> AmbientMedium:
    """Solar core conditions (current Sun, MESA model).

    T = 1.57e7 K, rho = 150 g/cm^3, mu = 0.85, X = 0.34, Y = 0.64.
    Hydrogen is depleted relative to the proto-solar composition because of
    nuclear burning over ~4.6 Gyr.
    """
    return AmbientMedium(T=1.57e7, rho=150.0, mu=0.85, X=0.34, Y=0.64)


def primordial_gas(T: float = 1e4, rho: float = 1e-24) -> AmbientMedium:
    """Primordial gas (H + He, no metals).

    Default: warm neutral medium conditions. Override T and rho for other
    environments.
    """
    return AmbientMedium(T=T, rho=rho, mu=1.22, X=0.75, Y=0.25)
