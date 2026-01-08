from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


ForecastType = Literal["naive", "moving_average", "arima"]


@dataclass
class ForecastResult:
    """Container for forecast outputs and basic evaluation metrics."""

    history: pd.DataFrame
    forecast: pd.Series
    horizon: int
    model_type: ForecastType
    mae: float | None
