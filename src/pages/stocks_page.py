import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from data.services.stock_data_service import StockDataService
from data.repositories.sqlite_stock_repository import SqliteStockRepository

st.set_page_config(page_title="Stocks List", layout="wide")

with st.sidebar:
    st.page_link("pages/dashboard_page.py", label="Dashboard", icon="🏠")
    st.page_link("pages/search_page.py", label="Stocks Search", icon="🔎")
    st.page_link("pages/stocks_page.py", label="Stocks List", icon="📃")

def plot_candlestick(df: pd.DataFrame, symbol: str, interval: str):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.7, 0.3], subplot_titles=(f"{symbol} Price ({interval})", "Volume")
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume"),
        row=2, col=1
    )
    fig.update_layout(height=750, xaxis_rangeslider_visible=False, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

repo = SqliteStockRepository("stocks.db")
service = StockDataService(repo)

with st.container():
    st.header("Stocks List")
    tickers = service.get_list_of_stocks()
    if not tickers:
        st.warning("No stocks available in the database.")
    else:
        selected = st.selectbox("Select a stock", tickers)
        interval = st.selectbox("Time interval", ["Day", "Week", "Month"])
        if st.button("Show Data"):
            try:
                df = service.get_history(symbol=selected, interval=interval, force_refresh=False)
                if df.empty:
                    st.warning("No data available for this stock and interval.")
                else:
                    tab1, tab2 = st.tabs(["Chart", "Data"])
                    with tab1:
                        plot_candlestick(df, selected, interval)
                    with tab2:
                        st.dataframe(df)
            except Exception as e:
                st.error(str(e))
