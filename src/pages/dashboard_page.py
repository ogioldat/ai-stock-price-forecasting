import streamlit as st

st.set_page_config(page_title="Stock Price Viewer", layout="wide")

with st.sidebar:
    st.page_link("pages/dashboard_page.py", label="Dashboard", icon="🏠")
    st.page_link("pages/search_page.py", label="Stocks Search", icon="🔎")
    st.page_link("pages/stocks_page.py", label="Stocks List", icon="📃")

with st.container():
    st.title("Stock Price Viewer")
    st.write(
        """
        This application allows you to visualize stock price data using candlestick charts.
        Select a stock symbol and time interval to view the corresponding price and volume data.
        """
    )

    st.markdown(
        """
        **Features:**
        - View candlestick charts for various stock symbols.
        - Analyze stock price movements over different time intervals.
        - Interactive charts with zoom and pan capabilities.
        """
    )
    st.markdown(
        """
        **Instructions:**
        1. Navigate to the "Stocks" page using the sidebar.
        2. Select a stock symbol from the dropdown menu.
        3. Choose a time interval (e.g., 1 day, 1 week, 1 month).
        4. View the generated candlestick chart and volume data.
        """
    )
    st.markdown(
        """
        **Note:** Ensure you have a stable internet connection to fetch the latest stock data.
        """
    )
    st.markdown("© 2024 Stock Price Viewer. All rights reserved.")
