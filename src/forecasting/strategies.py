from __future__ import annotations

import logging
from functools import singledispatch

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .configs import (
    ArimaForecastConfig,
    ForecastConfig,
    MovingAverageForecastConfig,
    NaiveForecastConfig,
)
from .metrics import compute_mae
from .preprocessing import build_horizon_index, normalize_history
from .types import ForecastResult, ForecastType


logger = logging.getLogger(__name__)


@singledispatch
def _run_with_config(config: ForecastConfig, history: pd.DataFrame) -> ForecastResult:
    raise TypeError(f"Unsupported forecast configuration: {config!r}")


@_run_with_config.register
def _(config: NaiveForecastConfig, history: pd.DataFrame) -> ForecastResult:
    horizon = config.horizon

    last_close = history["Close"].iloc[-1]
    index = build_horizon_index(history, horizon)
    forecast = pd.Series(last_close, index=index, name="Close")

    closes = history["Close"]
    y_true = closes.iloc[-horizon:] if len(closes) >= horizon else closes
    y_pred = pd.Series(last_close, index=y_true.index, name="Close")
    mae = compute_mae(y_true, y_pred)

    return ForecastResult(
        history=history,
        forecast=forecast,
        horizon=horizon,
        model_type="naive",
        mae=mae,
    )


@_run_with_config.register
def _(config: MovingAverageForecastConfig, history: pd.DataFrame) -> ForecastResult:
    horizon = config.horizon
    window = min(config.window, len(history))

    closes = history["Close"]
    last_ma = closes.rolling(window=window, min_periods=1).mean().iloc[-1]
    index = build_horizon_index(history, horizon)
    forecast = pd.Series(last_ma, index=index, name="Close")

    y_true = closes.iloc[-horizon:] if len(closes) >= horizon else closes
    rolling_ma = closes.rolling(window=window, min_periods=1).mean()
    y_pred = rolling_ma.iloc[-len(y_true) :]
    mae = compute_mae(y_true, y_pred)

    return ForecastResult(
        history=history,
        forecast=forecast,
        horizon=horizon,
        model_type="moving_average",
        mae=mae,
    )


@_run_with_config.register
def _(config: ArimaForecastConfig, history: pd.DataFrame) -> ForecastResult:
    return run_arima_forecast(history, horizon=config.horizon, order=config.order)


def run_arima_forecast(
    history: pd.DataFrame,
    horizon: int = 5,
    order: tuple[int, int, int] = (1, 1, 0),
) -> ForecastResult:
    if len(order) != 3:
        raise ValueError("ARIMA order must contain exactly three integers.")

    closes = history["Close"].astype("float64")

    logger.debug(
        "Running ARIMA forecast horizon=%s order=%s history_points=%s",
        horizon,
        order,
        len(history),
    )

    model = ARIMA(closes, order=order)
    fitted = model.fit()

    forecast_values = fitted.forecast(steps=horizon)
    forecast_index = build_horizon_index(history, horizon)
    forecast_series = pd.Series(
        forecast_values.values, index=forecast_index, name="Close"
    )

    mae = None
    if len(closes) > horizon:
        start = len(closes) - horizon
        end = len(closes) - 1
        y_true = closes.iloc[start : end + 1]
        y_pred = fitted.predict(start=start, end=end)
        y_pred = pd.Series(y_pred.values, index=y_true.index)
        mae = compute_mae(y_true, y_pred)

    return ForecastResult(
        history=history,
        forecast=forecast_series,
        horizon=horizon,
        model_type="arima",
        mae=mae,
    )


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
