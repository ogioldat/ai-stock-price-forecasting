from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MACrossoverMetrics:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float | None
    max_drawdown: float
    win_rate_on_active_days: float | None


@dataclass(frozen=True)
class MACrossoverBacktestResult:
    signals: pd.DataFrame
    position: pd.Series
    trades: pd.DataFrame
    metrics: MACrossoverMetrics


def ma_crossover_signals(
    close: pd.Series, short_window: int, long_window: int
) -> pd.DataFrame:
    """
    Compute fast/slow moving averages and label crossover events.

    Parameters
    ----------
    close:
        Series of closing prices indexed by timestamp.
    short_window:
        Lookback window for the fast moving average.
    long_window:
        Lookback window for the slow moving average.

    Returns
    -------
    pd.DataFrame
        Frame with the original prices, rolling averages, and boolean crossover flags.

    Raises
    ------
    ValueError
        If either window is non-positive or the short window is not less than the long.
    """
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Window sizes must be positive integers.")
    if short_window >= long_window:
        raise ValueError("Short window must be smaller than long window.")

    close = close.astype(float)
    ma_short = close.rolling(short_window, min_periods=short_window).mean()
    ma_long = close.rolling(long_window, min_periods=long_window).mean()

    bull_cross = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
    bear_cross = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

    return pd.DataFrame(
        {
            "close": close,
            "ma_short": ma_short,
            "ma_long": ma_long,
            "bull_cross": bull_cross.fillna(False),
            "bear_cross": bear_cross.fillna(False),
        },
        index=close.index,
    )


def _compute_metrics(
    strategy_returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> MACrossoverMetrics:
    """
    Summarize strategy performance into commonly used risk/return metrics.

    Parameters
    ----------
    strategy_returns:
        Net returns produced by the trading logic at each bar.
    equity_curve:
        Compounded value of the portfolio corresponding to the returns.
    periods_per_year:
        Number of sample periods in a trading year (used for annualization).

    Returns
    -------
    MACrossoverMetrics
        Dataclass aggregating total/annualized returns, volatility, Sharpe, drawdown, etc.
    """
    returns = strategy_returns.astype(float)
    equity = equity_curve.astype(float)

    if equity.empty:
        return MACrossoverMetrics(
            total_return=0.0,
            annualized_return=0.0,
            annualized_vol=0.0,
            sharpe=None,
            max_drawdown=0.0,
            win_rate_on_active_days=None,
        )

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    mean_ret = returns.mean()
    ann_return = float((1.0 + mean_ret) ** periods_per_year - 1.0)
    ann_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else None

    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    active = returns[returns != 0]
    win_rate = float((active > 0).mean()) if not active.empty else None

    return MACrossoverMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate_on_active_days=win_rate,
    )


def backtest_ma_crossover(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 30,
    *,
    long_only: bool = True,
    fee_bps: float = 10.0,
    initial_cash: float = 10_000.0,
) -> MACrossoverBacktestResult:
    """
    Backtest a moving-average crossover trading rule.

    Parameters
    ----------
    df:
        History dataframe with a numeric `Close` column.
    short_window:
        Number of periods for the fast moving average.
    long_window:
        Number of periods for the slow moving average (must be larger).
    long_only:
        Restrict exposure to {0, 1} if True, else permit {-1, 0, 1}.
    fee_bps:
        Transaction fee in basis points applied to turnover.
    initial_cash:
        Starting notional value used to scale the equity curve.

    Returns
    -------
    MACrossoverBacktestResult
        Struct containing signal diagnostics, realized trades, and summary metrics.
    """
    if "Close" not in df.columns:
        raise ValueError("Input dataframe must contain a 'Close' column.")
    if len(df) < max(short_window, long_window):
        raise ValueError("Not enough data to compute moving averages.")

    close = df["Close"].astype(float)
    signals = ma_crossover_signals(close, short_window, long_window)

    desired = pd.Series(0.0, index=signals.index, dtype=float)
    if long_only:
        desired.loc[signals["bull_cross"]] = 1.0
        desired.loc[signals["bear_cross"]] = 0.0
        desired = desired.replace(0.0, np.nan).ffill().fillna(0.0)
    else:
        desired.loc[signals["bull_cross"]] = 1.0
        desired.loc[signals["bear_cross"]] = -1.0
        desired = desired.replace(0.0, np.nan).ffill().fillna(0.0)

    position = desired.shift(1).fillna(0.0)

    ret = close.pct_change().fillna(0.0)
    strat_gross = position * ret

    fee = fee_bps / 10_000.0
    turnover = position.diff().abs().fillna(0.0)
    strat_net = strat_gross - turnover * fee

    equity = (1.0 + strat_net).cumprod() * float(initial_cash)

    trades = pd.DataFrame(
        {
            "position": position,
            "turnover": turnover,
            "return": ret,
            "strategy_gross": strat_gross,
            "strategy_net": strat_net,
            "equity": equity,
        },
        index=df.index,
    )

    metrics = _compute_metrics(trades["strategy_net"], trades["equity"])

    return MACrossoverBacktestResult(
        signals=signals,
        position=position,
        trades=trades,
        metrics=metrics,
    )
