import streamlit as st

st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

with st.sidebar:
    st.page_link("pages/dashboard_page.py", label="Dashboard", icon="🏠")
    st.page_link("pages/stocks_page.py", label="Stocks List", icon="📃")
    st.page_link("pages/forecast_page.py", label="Forecast", icon="📈")
    st.page_link("pages/strategies_page.py", label="Trading strategies", icon="🧠")

st.title("AI Stock Toolkit")
st.caption("Browse prices, run forecasts, and experiment with trading rules in one place.")

col1, col2, col3 = st.columns(3)
col1.metric("Tracked top stocks", "AMZN, META, NVDA...", "")
col2.metric("Data source", "yfinance", help="Fetched via the CLI or UI pages")
col3.metric("Strategies", "MA crossover", "")

st.divider()

st.subheader("What you can do")
feat_cols = st.columns(3)
feat_cols[0].write("**Visualize**\n\nCandlestick charts with volume overlays, quick filtering, and data export.")
feat_cols[1].write("**Forecast**\n\nRun the forecasting notebook logic through the UI and compare scenarios.")
feat_cols[2].write("**Backtest**\n\nTry the moving-average crossover, then auto-tune it with the new genetic/ACO tool.")
