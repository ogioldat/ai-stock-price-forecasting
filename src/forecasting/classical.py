from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from forecasting.baseline import (
    ForecastResult,
    _build_horizon_index,
    _compute_mae,
    normalize_history,
)


logger = logging.getLogger(__name__)


def run_arima_forecast(
    history: pd.DataFrame,
    horizon: int = 5,
    order: Tuple[int, int, int] = (1, 1, 0),
) -> ForecastResult:
    """Fit an ARIMA model on the Close prices and forecast `horizon` steps ahead."""

    if len(order) != 3:
        raise ValueError("ARIMA order must contain exactly three integers.")

    cleaned_history = normalize_history(history)
    closes = cleaned_history["Close"].astype("float64")

    logger.debug(
        "Running ARIMA forecast horizon=%s order=%s history_points=%s",
        horizon,
        order,
        len(cleaned_history),
    )

    model = ARIMA(closes, order=order)
    fitted = model.fit()

    forecast_values = fitted.forecast(steps=horizon)
    forecast_index = _build_horizon_index(cleaned_history, horizon)
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
        history=cleaned_history,
        forecast=forecast_series,
        horizon=horizon,
        model_type="arima",
        mae=mae,
    )
