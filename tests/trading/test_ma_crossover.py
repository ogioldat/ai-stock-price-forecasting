from __future__ import annotations

import pandas as pd
import pytest

from trading.ma_crossover import backtest_ma_crossover, ma_crossover_signals


def _build_price_series() -> tuple[pd.DatetimeIndex, pd.Series]:
    dates = pd.date_range("2020-01-01", periods=9, freq="D")
    closes = pd.Series(
        [12, 11, 10, 11, 12, 13, 11, 10, 9],
        index=dates,
        name="Close",
        dtype=float,
    )
    return dates, closes


def test_ma_crossover_signals_detects_bull_and_bear_crossings() -> None:
    dates, closes = _build_price_series()

    signals = ma_crossover_signals(closes, short_window=2, long_window=3)

    bull_dates = signals.index[signals["bull_cross"]].tolist()
    bear_dates = signals.index[signals["bear_cross"]].tolist()

    assert bull_dates == [dates[4]]
    assert bear_dates == [dates[7]]


def test_ma_crossover_signals_validates_window_order() -> None:
    _, closes = _build_price_series()
    with pytest.raises(ValueError):
        ma_crossover_signals(closes, short_window=3, long_window=2)


def test_backtest_ma_crossover_long_short_positions_follow_signals() -> None:
    dates, closes = _build_price_series()
    df = pd.DataFrame({"Close": closes})

    result = backtest_ma_crossover(
        df,
        short_window=2,
        long_window=3,
        long_only=False,
        fee_bps=0.0,
        initial_cash=1_000.0,
    )

    assert result.position.loc[dates[5]] == pytest.approx(1.0)
    assert result.position.loc[dates[8]] == pytest.approx(-1.0)

    equity = result.trades["equity"]
    expected_total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    assert result.metrics.total_return == pytest.approx(expected_total_return)
