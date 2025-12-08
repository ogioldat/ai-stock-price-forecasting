import sqlite3
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.services.stock_data_service import StockDataService
from data.repositories.sqlite_stock_repository import SqliteStockRepository


def plot_chart(df: pd.DataFrame, symbol: str, interval: str) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{symbol} Price ({interval})", "Volume"),
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Volume charts below the candlestick
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=750,
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def run() -> None:

    repo = SqliteStockRepository("stocks.db")
    service = StockDataService(repo)

    st.set_page_config(page_title="Stock Price Viewer", layout="wide")

    st.title("📈 Stock Candlestick Viewer")

    with st.form("ticker_form"):
        ticker = st.text_input("Enter ticker symbol", value="AAPL")
        submitted = st.form_submit_button("Fetch data")

    if not submitted:
        st.info("Enter a ticker and click **Fetch data**.")
        return

    ticker = ticker.strip().upper()

    try:
        service.get_history(ticker, "1d", force_refresh=False)
    except Exception as e:
        st.error(str(e))
        return

    # intervals = repo.get_available_intervals(ticker)

    # if not intervals:
    #     st.warning(f"No data found in database for ticker: {ticker}")
    #     return

    # if not ticker:
    #     st.error("Ticker cannot be empty.")
    #     return

    intervals = repo.get_available_intervals(ticker)

    if not intervals:
        st.warning(f"No data found in database for ticker: {ticker}")
        return

    interval = st.selectbox("Select interval", intervals)

    df = service.get_history(ticker, interval)

    if df.empty:
        st.warning("No data available for this selection.")
        return

    plot_chart(df, ticker, interval)

    with st.expander("Show raw data"):
        st.dataframe(df)

