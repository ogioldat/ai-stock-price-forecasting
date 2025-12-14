from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


ForecastType = Literal["naive", "moving_average", "arima"]


@dataclass
class ForecastResult:
    """Container for baseline forecast outputs and basic evaluation."""

    history: pd.DataFrame
    forecast: pd.Series
    horizon: int
    model_type: ForecastType
    mae: float | None


def _compute_mae(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return None
    return float(np.mean(np.abs(y_true.values - y_pred.values)))


def _build_horizon_index(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("History index must be a DatetimeIndex.")
    if len(history.index) < 2:
        # Fallback: repeat last timestamp with 1D frequency
        last = history.index[-1]
        return pd.date_range(last, periods=horizon + 1, freq="D")[1:]

    inferred = pd.infer_freq(history.index)
    if inferred is None:
        # Default to daily if we cannot infer frequency
        inferred = "D"

    last_timestamp = history.index[-1]
    return pd.date_range(start=last_timestamp, periods=horizon + 1, freq=inferred)[1:]


def naive_forecast(history: pd.DataFrame, horizon: int) -> ForecastResult:
    """Naive forecast: repeat the last observed Close price for all future steps."""

    if history.empty:
        raise ValueError("History is empty; cannot compute forecast.")

    last_close = history["Close"].iloc[-1]
    index = _build_horizon_index(history, horizon)
    forecast = pd.Series(last_close, index=index, name="Close")

    y_true = history["Close"].iloc[-horizon:] if len(history) >= horizon else history["Close"]
    y_pred = pd.Series(last_close, index=y_true.index, name="Close")
    mae = _compute_mae(y_true, y_pred)

    return ForecastResult(history=history, forecast=forecast, horizon=horizon, model_type="naive", mae=mae)


def moving_average_forecast(history: pd.DataFrame, horizon: int, window: int = 5) -> ForecastResult:
    """Moving-average forecast based on the last ``window`` closes."""

    if history.empty:
        raise ValueError("History is empty; cannot compute forecast.")

    if window <= 0:
        raise ValueError("Window must be positive.")

    closes = history["Close"]
    if len(closes) < window:
        window = len(closes)

    last_ma = closes.rolling(window=window).mean().iloc[-1]
    index = _build_horizon_index(history, horizon)
    forecast = pd.Series(last_ma, index=index, name="Close")

    eval_window = max(window, horizon)
    y_true = closes.iloc[-horizon:] if len(closes) >= horizon else closes

    rolling_ma = closes.rolling(window=window, min_periods=1).mean()
    y_pred = rolling_ma.iloc[-len(y_true):]

    mae = _compute_mae(y_true, y_pred)

    return ForecastResult(history=history, forecast=forecast, horizon=horizon, model_type="moving_average", mae=mae)


def run_baseline_forecast(
    history: pd.DataFrame,
    horizon: int = 5,
    model_type: ForecastType = "naive",
    window: int = 5,
) -> ForecastResult:
    """Convenience wrapper to run one of the baseline models.

    """

    if model_type == "naive":
        return naive_forecast(history, horizon)
    if model_type == "moving_average":
        return moving_average_forecast(history, horizon, window=window)

    raise ValueError(f"Unsupported model_type: {model_type}")
