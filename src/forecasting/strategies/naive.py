from __future__ import annotations

import pandas as pd

from ..configs import NaiveForecastConfig
from ..metrics import compute_mae
from ..preprocessing import build_horizon_index
from ..types import ForecastResult


def run_naive_strategy(
    history: pd.DataFrame,
    config: NaiveForecastConfig,
) -> ForecastResult:
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
