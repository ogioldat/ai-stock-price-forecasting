from __future__ import annotations
from typing import Any

import pandas as pd
import pytest

from trading import tuner
from trading.ma_crossover import MACrossoverBacktestResult, MACrossoverMetrics


def test_tune_ma_crossover_prefers_highest_scoring_pair(monkeypatch: Any) -> None:
    df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]})

    def fake_backtest_ma_crossover(
        df: pd.DataFrame,
        short_window: int,
        long_window: int,
        **_: object,
    ) -> MACrossoverBacktestResult:
        score = float(long_window - short_window)
        metrics = MACrossoverMetrics(
            total_return=score,
            annualized_return=score,
            annualized_vol=1.0,
            sharpe=score,
            max_drawdown=0.0,
            win_rate_on_active_days=1.0,
        )
        empty_index = pd.RangeIndex(0)
        return MACrossoverBacktestResult(
            signals=pd.DataFrame(index=empty_index),
            position=pd.Series(dtype=float),
            trades=pd.DataFrame({"equity": [1.0]}, index=pd.RangeIndex(1)),
            metrics=metrics,
        )

    monkeypatch.setattr(tuner, "backtest_ma_crossover", fake_backtest_ma_crossover)

    result = tuner.tune_ma_crossover(
        df,
        short_window_range=(2, 3),
        long_window_range=(4, 6),
        population_size=4,
        iterations=5,
        evaporation=0.1,
        pheromone_deposit=1.0,
        metric="total_return",
        random_seed=42,
    )

    assert result.best_candidate.short_window == 2
    assert result.best_candidate.long_window == 6
    assert result.best_candidate.score == pytest.approx(4.0)
    assert len(result.evaluations) > 0
