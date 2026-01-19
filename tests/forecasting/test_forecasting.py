from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from forecasting import (
    ArimaForecastConfig,
    ForecastResult,
    MovingAverageForecastConfig,
    NaiveForecastConfig,
    normalize_history,
    run_forecast,
)


def _build_history(values: list[float | None]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=index)


def test_normalize_history_sorts_and_fills_missing_values() -> None:
    raw_index = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-03"])
    history = pd.DataFrame(
        {"Close": [2.0, None, 1.5, None], "Volume": [10, 11, 12, 13]},
        index=raw_index,
    )

    normalized = normalize_history(history)

    assert normalized.index.is_monotonic_increasing
    assert list(normalized.index) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert normalized.iloc[0]["Close"] == pytest.approx(1.5)
    assert normalized.iloc[-1]["Close"] == pytest.approx(2.0)


def test_normalize_history_requires_close_column() -> None:
    history = pd.DataFrame(
        {"Open": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3)
    )
    with pytest.raises(ValueError):
        normalize_history(history)


def test_naive_forecast_uses_configured_horizon() -> None:
    history = _build_history([100, 101, 102])
    result = run_forecast(history, config=NaiveForecastConfig(horizon=2))

    assert isinstance(result, ForecastResult)
    assert len(result.forecast) == 2
    assert result.forecast.iloc[0] == pytest.approx(102)


def test_moving_average_forecast_adjusts_window() -> None:
    history = _build_history([10, 11, 12])
    result = run_forecast(
        history,
        config=MovingAverageForecastConfig(horizon=1, window=10),
    )

    expected_ma = np.mean([10, 11, 12])
    assert result.forecast.iloc[0] == pytest.approx(expected_ma)
    assert result.mae is not None


def test_arima_forecast_config_flow() -> None:
    values = list(np.linspace(50, 60, num=16))
    history = _build_history(values)

    config = ArimaForecastConfig(horizon=3, order=(0, 1, 0))
    result = run_forecast(history, config=config)

    assert len(result.forecast) == 3
    assert result.model_type == "arima"
    assert result.history.index.is_monotonic_increasing


def test_run_arima_forecast_handles_string_index() -> None:
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    history = pd.DataFrame({"Close": [1, 2, 3, 4]}, index=dates)

    result = run_forecast(history, config=NaiveForecastConfig(horizon=1))

    assert result.history.index.dtype == "datetime64[ns]"
    assert len(result.forecast) == 1


@dataclass
class _UnknownConfig:
    horizon: int = 1


def test_unregistered_config_raises_error() -> None:
    history = _build_history([1, 2, 3])

    with pytest.raises(TypeError):
        run_forecast(history, config=_UnknownConfig())
