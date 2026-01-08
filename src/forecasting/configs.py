from __future__ import annotations

from dataclasses import dataclass


def _validate_horizon(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError("Horizon must be positive.")


@dataclass(frozen=True)
class NaiveForecastConfig:
    horizon: int = 5

    def __post_init__(self) -> None:
        _validate_horizon(self.horizon)


@dataclass(frozen=True)
class MovingAverageForecastConfig:
    horizon: int = 5
    window: int = 5

    def __post_init__(self) -> None:
        _validate_horizon(self.horizon)
        if self.window <= 0:
            raise ValueError("Window must be positive.")


@dataclass(frozen=True)
class ArimaForecastConfig:
    horizon: int = 5
    order: tuple[int, int, int] = (1, 1, 0)

    def __post_init__(self) -> None:
        _validate_horizon(self.horizon)
        if len(self.order) != 3:
            raise ValueError("ARIMA order must contain exactly three integers.")
        if any(part < 0 for part in self.order):
            raise ValueError("ARIMA order values must be non-negative.")


ForecastConfig = NaiveForecastConfig | MovingAverageForecastConfig | ArimaForecastConfig
