from __future__ import annotations

import logging

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ..configs import ArimaForecastConfig
from ..metrics import compute_mae
from ..preprocessing import build_horizon_index
from ..types import ForecastResult


logger = logging.getLogger(__name__)


def run_arima_strategy(
    history: pd.DataFrame,
    config: ArimaForecastConfig,
) -> ForecastResult:
    if len(config.order) != 3:
        raise ValueError("ARIMA order must contain exactly three integers.")

    closes = history["Close"].astype("float64")

    logger.debug(
        "Running ARIMA forecast horizon=%s order=%s history_points=%s",
        config.horizon,
        config.order,
        len(history),
    )

    model = ARIMA(closes, order=config.order)
    fitted = model.fit()

    forecast_values = fitted.forecast(steps=config.horizon)
    forecast_index = build_horizon_index(history, config.horizon)
    forecast_series = pd.Series(
        forecast_values.values, index=forecast_index, name="Close"
    )

    mae = None
    if len(closes) > config.horizon:
        start = len(closes) - config.horizon
        end = len(closes) - 1
        y_true = closes.iloc[start : end + 1]
        y_pred = fitted.predict(start=start, end=end)
        y_pred = pd.Series(y_pred.values, index=y_true.index)
        mae = compute_mae(y_true, y_pred)

    return ForecastResult(
        history=history,
        forecast=forecast_series,
        horizon=config.horizon,
        model_type="arima",
        mae=mae,
    )
