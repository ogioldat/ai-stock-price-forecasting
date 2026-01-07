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
    """Fit an ARIMA model on the Close prices and forecast `horizon` steps ahead."""

    if history.empty:
        raise ValueError("History is empty; cannot compute ARIMA forecast.")

    closes = history["Close"].astype("float64")

    model = ARIMA(closes, order=order)
    fitted = model.fit()

    forecast_values = fitted.forecast(steps=horizon)
    forecast_index = _build_horizon_index(history, horizon)
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
        mae = _compute_mae(y_true, y_pred)

    return ForecastResult(
        history=history,
        forecast=forecast_series,
        horizon=horizon,
        model_type="naive",
        mae=mae,
    )
