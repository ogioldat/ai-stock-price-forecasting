import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from data.services.stock_data_service import StockDataService
from data.repositories.sqlite_stock_repository import SqliteStockRepository

st.set_page_config(page_title="Stock Search", layout="wide")

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

st.header("Stock Search")

with st.form("search_form"):
    ticker = st.text_input("Ticker symbol", value="AAPL")
    interval = st.selectbox("Interval", ["Day", "Week", "Month"])
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date")
    with col2:
        end = st.date_input("End date")
    submitted = st.form_submit_button("Search")

if submitted:
    try:
        df = service.get_history(symbol=ticker, interval=interval, start=start, end=end, force_refresh=False)
        if df.empty:
            st.warning("No data available for this selection.")
        else:
            tab1, tab2 = st.tabs(["Chart", "Data"])
            with tab1:
                plot_candlestick(df, ticker, interval)
            with tab2:
                st.dataframe(df)
    except Exception as e:
        st.error(str(e))
else:
    st.info("Enter a ticker and click Search.")
