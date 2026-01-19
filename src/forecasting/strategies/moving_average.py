from __future__ import annotations

import pandas as pd

from ..configs import MovingAverageForecastConfig
from ..metrics import compute_mae
from ..preprocessing import build_horizon_index
from ..types import ForecastResult


def run_moving_average_strategy(
    history: pd.DataFrame,
    config: MovingAverageForecastConfig,
) -> ForecastResult:
    """
    Forecast prices by extending the latest moving average across the horizon.

    Parameters
    ----------
    history:
        Cleaned dataframe with at least one `Close` observation.
    config:
        Moving-average configuration controlling window size and horizon.

    Returns
    -------
    ForecastResult
        Result bundle containing the constant forecast series and evaluation stats.
    """
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
