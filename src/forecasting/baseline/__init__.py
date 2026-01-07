from __future__ import annotations

from .configs import (
    ArimaForecastConfig,
    ForecastConfig,
    MovingAverageForecastConfig,
    NaiveForecastConfig,
)
from .metrics import _compute_mae
from .preprocessing import _build_horizon_index, normalize_history
from .strategies import (
    moving_average_forecast,
    naive_forecast,
    run_baseline_forecast,
)
from .types import ForecastResult, ForecastType

__all__ = [
    "ForecastType",
    "ForecastResult",
    "NaiveForecastConfig",
    "MovingAverageForecastConfig",
    "ArimaForecastConfig",
    "ForecastConfig",
    "normalize_history",
    "_build_horizon_index",
    "_compute_mae",
    "naive_forecast",
    "moving_average_forecast",
    "run_baseline_forecast",
]
