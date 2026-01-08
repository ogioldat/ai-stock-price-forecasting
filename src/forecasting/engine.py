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
from .types import ForecastResult, ForecastType


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


def naive_forecast(
    history: pd.DataFrame,
    horizon: int,
    *,
    config: NaiveForecastConfig | None = None,
) -> ForecastResult:
    resolved = config or NaiveForecastConfig(horizon=horizon)
    return _run_with_config(resolved, history)


def moving_average_forecast(
    history: pd.DataFrame,
    horizon: int,
    window: int = 5,
    *,
    config: MovingAverageForecastConfig | None = None,
) -> ForecastResult:
    resolved = config or MovingAverageForecastConfig(horizon=horizon, window=window)
    return _run_with_config(resolved, history)


def arima_forecast(
    history: pd.DataFrame,
    horizon: int,
    order: tuple[int, int, int],
    *,
    config: ArimaForecastConfig | None = None,
) -> ForecastResult:
    resolved = config or ArimaForecastConfig(horizon=horizon, order=order)
    return _run_with_config(resolved, history)


def _resolve_config(
    model_type: ForecastType,
    *,
    config: ForecastConfig | None,
    horizon: int,
    window: int,
    arima_order: tuple[int, int, int] | None,
) -> ForecastConfig:
    if config is not None:
        if model_type == "naive" and isinstance(config, NaiveForecastConfig):
            return config
        if model_type == "moving_average" and isinstance(
            config,
            MovingAverageForecastConfig,
        ):
            return config
        if model_type == "arima" and isinstance(config, ArimaForecastConfig):
            return config
        raise ValueError(
            f"Config {config.__class__.__name__} is incompatible with model_type='{model_type}'."
        )

    if model_type == "naive":
        return NaiveForecastConfig(horizon=horizon)
    if model_type == "moving_average":
        return MovingAverageForecastConfig(horizon=horizon, window=window)
    if model_type == "arima":
        return ArimaForecastConfig(horizon=horizon, order=arima_order or (1, 1, 0))

    raise ValueError(f"Unsupported model_type: {model_type}")


def run_baseline_forecast(
    history: pd.DataFrame,
    horizon: int = 5,
    model_type: ForecastType = "naive",
    window: int = 5,
    *,
    config: ForecastConfig | None = None,
    arima_order: tuple[int, int, int] | None = None,
) -> ForecastResult:
    history = normalize_history(history)
    resolved_config = _resolve_config(
        model_type,
        config=config,
        horizon=horizon,
        window=window,
        arima_order=arima_order,
    )
    return _run_with_config(resolved_config, history)
