from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data.models import Interval
from data.repositories.sqlite_stock_repository import SqliteStockRepository
from data.services.stock_data_service import StockDataService
from trading import MACrossoverBacktestResult, backtest_ma_crossover

st.set_page_config(page_title="Trading Strategies", layout="wide")

with st.sidebar:
    st.page_link("pages/dashboard_page.py", label="Dashboard", icon="🏠")
    st.page_link("pages/stocks_page.py", label="Stocks List", icon="📃")
    st.page_link("pages/forecast_page.py", label="Forecast", icon="📈")
    st.page_link("pages/strategies_page.py", label="Trading strategies", icon="🧠")

repo = SqliteStockRepository("stocks.db")
service = StockDataService(repo)


def _plot_ma_crossover(
    history: pd.DataFrame,
    result: MACrossoverBacktestResult,
    title: str,
) -> None:
    signals = result.signals
    trades = result.trades

    cumulative_strategy = trades["equity"] / trades["equity"].iloc[0] - 1.0
    benchmark = history["Close"] / history["Close"].iloc[0] - 1.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=(f"{title} price & signals", "Cumulative return"),
    )

    has_ohlc = {"Open", "High", "Low", "Close"}.issubset(history.columns)

    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=history.index,
                open=history["Open"],
                high=history["High"],
                low=history["Low"],
                close=history["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["Close"],
                mode="lines",
                name="Close",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals["ma_short"],
            mode="lines",
            name="MA short",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals["ma_long"],
            mode="lines",
            name="MA long",
        ),
        row=1,
        col=1,
    )

    buys = signals.index[signals["bull_cross"]]
    sells = signals.index[signals["bear_cross"]]

    fig.add_trace(
        go.Scatter(
            x=buys,
            y=history.loc[buys, "Low"] if "Low" in history.columns else history.loc[buys, "Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#16a34a"),
            name="Buy",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sells,
            y=history.loc[sells, "High"] if "High" in history.columns else history.loc[sells, "Close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#dc2626"),
            name="Sell",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=cumulative_strategy.index,
            y=cumulative_strategy.values,
            mode="lines",
            name="Strategy",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode="lines",
            name="Buy & hold",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_x=0,
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Return", tickformat=".0%", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _format_metric(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"

    return f"{value:.2%}"


st.title("Trading strategies")
st.markdown(
    "Explore rule-based trading ideas. Start with the classic moving-average crossover "
    "to understand when a strategy would buy, sell, and how it performed versus a simple buy-and-hold benchmark."
)

st.subheader("Moving-average crossover")
st.caption(
    "This strategy buys when a short moving average crosses above a long moving average "
    "and sells (or goes short) when the opposite happens."
)

available_tickers = service.get_list_of_stocks()

if not available_tickers:
    st.warning("No stocks found in the local database.")
else:
    with st.form("ma_crossover_form"):
        col_symbol, col_interval = st.columns(2)
        with col_symbol:
            ticker = st.selectbox("Ticker", options=available_tickers, index=0)
        with col_interval:
            interval = st.selectbox(
                "Interval",
                options=list(Interval),
                index=0,
                format_func=lambda option: option.display_name,
            )

        col_windows = st.columns(2)
        with col_windows[0]:
            short_window = int(
                st.number_input("Short window", min_value=3, max_value=200, value=10, step=1)
            )
        with col_windows[1]:
            long_window = int(
                st.number_input("Long window", min_value=5, max_value=250, value=30, step=1)
            )

        col_strategy = st.columns(3)
        with col_strategy[0]:
            long_only = st.toggle("Long-only", value=True)
        with col_strategy[1]:
            fee_bps = float(
                st.number_input("Fee (bps)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
            )
        with col_strategy[2]:
            initial_cash = float(
                st.number_input("Initial cash", min_value=1_000.0, max_value=1_000_000.0, value=10_000.0, step=1_000.0)
            )

        col_misc = st.columns(2)
        with col_misc[0]:
            lookback_days = int(
                st.number_input(
                    "Lookback window (most recent bars)",
                    min_value=60,
                    max_value=2000,
                    value=365,
                    step=30,
                )
            )
        with col_misc[1]:
            refresh = st.checkbox("Force data refresh", value=False)

        submitted = st.form_submit_button("Run backtest")

    if submitted:
        try:
            history = service.get_history(
                symbol=ticker,
                interval=interval,
                force_refresh=refresh,
                start="2024-12-04",
                end="2025-12-04",
            )
        

            if history.empty:
                st.warning("No price data available for this selection.")
            else:
                clipped = history.tail(lookback_days)
                
                print(len(history), long_window)

                if len(clipped) < long_window + 5:
                    st.warning(
                        "Not enough history for the selected window sizes. "
                        "Try reducing the long window or increasing the lookback period."
                    )
                else:
                    result = backtest_ma_crossover(
                        df=clipped,
                        short_window=short_window,
                        long_window=long_window,
                        long_only=long_only,
                        fee_bps=fee_bps,
                        initial_cash=initial_cash,
                    )

                    metrics = result.metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total return", f"{metrics.total_return:.2%}")
                    m2.metric("Annualized return", f"{metrics.annualized_return:.2%}")
                    m3.metric("Max drawdown", f"{metrics.max_drawdown:.2%}")

                    m4, m5, m6 = st.columns(3)
                    m4.metric(
                        "Annualized vol",
                        f"{metrics.annualized_vol:.2%}",
                    )
                    m5.metric(
                        "Sharpe",
                        "—" if metrics.sharpe is None or pd.isna(metrics.sharpe) else f"{metrics.sharpe:.2f}",
                    )
                    m6.metric(
                        "Win rate (active days)",
                        _format_metric(metrics.win_rate_on_active_days),
                    )

                    _plot_ma_crossover(
                        clipped,
                        result,
                        title=f"{ticker} ({interval.display_name})",
                    )

                    with st.expander("Show raw signals and trades"):
                        st.write("Signals")
                        st.dataframe(result.signals)

                        st.write("Trades & equity curve")
                        st.dataframe(result.trades)

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to run backtest: {exc}")
