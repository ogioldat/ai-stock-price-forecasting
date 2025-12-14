"""Classical time-series forecasting models (e.g. ARIMA)."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from forecasting.baseline import ForecastResult, _build_horizon_index, _compute_mae


def run_arima_forecast(
    history: pd.DataFrame,
    horizon: int = 5,
    order: Tuple[int, int, int] = (1, 1, 0),
) -> ForecastResult:
    """Fit an ARIMA model on the Close prices and forecast `horizon` steps ahead.

    Parameters
    ----------
    history: pd.DataFrame
        Historical OHLCV data with a DatetimeIndex and a `Close` column.
    horizon: int
        Number of future steps to forecast.
    order: tuple[int, int, int]
        ARIMA(p, d, q) order.
    """

    if history.empty:
        raise ValueError("History is empty; cannot compute ARIMA forecast.")

    closes = history["Close"].astype("float64")

    model = ARIMA(closes, order=order)
    fitted = model.fit()

    # Out-of-sample forecast for the next `horizon` steps
    forecast_values = fitted.forecast(steps=horizon)
    forecast_index = _build_horizon_index(history, horizon)
    forecast_series = pd.Series(forecast_values.values, index=forecast_index, name="Close")

    # Simple evaluation: use one-step-ahead predictions over the last `horizon` points where possible
    mae = None
    if len(closes) > horizon:
        start = len(closes) - horizon
        end = len(closes) - 1
        y_true = closes.iloc[start : end + 1]
        y_pred = fitted.predict(start=start, end=end)
        y_pred = pd.Series(y_pred.values, index=y_true.index)
        mae = _compute_mae(y_true, y_pred)

    return ForecastResult(
        history=history,
        forecast=forecast_series,
        horizon=horizon,
        model_type="naive",  # model_type is informational; value not used for logic here
        mae=mae,
    )
