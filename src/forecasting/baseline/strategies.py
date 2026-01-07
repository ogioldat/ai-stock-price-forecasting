from __future__ import annotations

import logging
from functools import singledispatch

import pandas as pd

from .configs import (
    ArimaForecastConfig,
    ForecastConfig,
    MovingAverageForecastConfig,
    NaiveForecastConfig,
)
from .metrics import _compute_mae
from .preprocessing import _build_horizon_index, normalize_history
from .types import ForecastResult, ForecastType


logger = logging.getLogger(__name__)


@singledispatch
def _run_with_config(config: ForecastConfig, history: pd.DataFrame) -> ForecastResult:
    raise TypeError(f"Unsupported forecast configuration: {config!r}")


@_run_with_config.register
def _(config: NaiveForecastConfig, history: pd.DataFrame) -> ForecastResult:
    cleaned = normalize_history(history)
    horizon = config.horizon

    logger.debug("Running naive forecast horizon=%s points=%s", horizon, len(cleaned))

    last_close = cleaned["Close"].iloc[-1]
    index = _build_horizon_index(cleaned, horizon)
    forecast = pd.Series(last_close, index=index, name="Close")

    closes = cleaned["Close"]
    y_true = closes.iloc[-horizon:] if len(closes) >= horizon else closes
    y_pred = pd.Series(last_close, index=y_true.index, name="Close")
    mae = _compute_mae(y_true, y_pred)

    return ForecastResult(
        history=cleaned,
        forecast=forecast,
        horizon=horizon,
        model_type="naive",
        mae=mae,
    )


@_run_with_config.register
def _(config: MovingAverageForecastConfig, history: pd.DataFrame) -> ForecastResult:
    cleaned = normalize_history(history)
    horizon = config.horizon
    window = min(config.window, len(cleaned))

    logger.debug(
        "Running moving-average forecast horizon=%s window=%s effective_window=%s",
        horizon,
        config.window,
        window,
    )

    closes = cleaned["Close"]
    last_ma = closes.rolling(window=window, min_periods=1).mean().iloc[-1]
    index = _build_horizon_index(cleaned, horizon)
    forecast = pd.Series(last_ma, index=index, name="Close")

    y_true = closes.iloc[-horizon:] if len(closes) >= horizon else closes
    rolling_ma = closes.rolling(window=window, min_periods=1).mean()
    y_pred = rolling_ma.iloc[-len(y_true) :]
    mae = _compute_mae(y_true, y_pred)

    return ForecastResult(
        history=cleaned,
        forecast=forecast,
        horizon=horizon,
        model_type="moving_average",
        mae=mae,
    )


@_run_with_config.register
def _(config: ArimaForecastConfig, history: pd.DataFrame) -> ForecastResult:
    from forecasting.classical import run_arima_forecast

    return run_arima_forecast(history, horizon=config.horizon, order=config.order)


def naive_forecast(
    history: pd.DataFrame,
    horizon: int,
    *,
    config: NaiveForecastConfig | None = None,
) -> ForecastResult:
    """Naive forecast: repeat the last observed Close price."""

    resolved = config or NaiveForecastConfig(horizon=horizon)
    return _run_with_config(resolved, history)


def moving_average_forecast(
    history: pd.DataFrame,
    horizon: int,
    window: int = 5,
    *,
    config: MovingAverageForecastConfig | None = None,
) -> ForecastResult:
    """Moving-average forecast based on the last ``window`` closes."""

    resolved = config or MovingAverageForecastConfig(horizon=horizon, window=window)
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
    resolved_config = _resolve_config(
        model_type,
        config=config,
        horizon=horizon,
        window=window,
        arima_order=arima_order,
    )
    return _run_with_config(resolved_config, history)
