from __future__ import annotations

from functools import singledispatch

import pandas as pd

from .configs import (
    ArimaForecastConfig,
    ForecastConfig,
    MovingAverageForecastConfig,
    NaiveForecastConfig,
)
from .preprocessing import normalize_history
from .strategies import (
    run_arima_strategy,
    run_moving_average_strategy,
    run_naive_strategy,
)
from .types import ForecastResult


@singledispatch
def _run_with_config(config: ForecastConfig, history: pd.DataFrame) -> ForecastResult:
    raise TypeError(f"Unsupported forecast configuration: {config!r}")


@_run_with_config.register
def _(config: NaiveForecastConfig, history: pd.DataFrame) -> ForecastResult:
    return run_naive_strategy(history, config)


@_run_with_config.register
def _(config: MovingAverageForecastConfig, history: pd.DataFrame) -> ForecastResult:
    return run_moving_average_strategy(history, config)


@_run_with_config.register
def _(config: ArimaForecastConfig, history: pd.DataFrame) -> ForecastResult:
    return run_arima_strategy(history, config)


def run_forecast(history: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
    normalized = normalize_history(history)
    return _run_with_config(config, normalized)
