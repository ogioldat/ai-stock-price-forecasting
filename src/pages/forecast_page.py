import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from data.services.stock_data_service import StockDataService
from data.repositories.sqlite_stock_repository import SqliteStockRepository
from forecasting.baseline import run_baseline_forecast
from forecasting.classical import run_arima_forecast


st.set_page_config(page_title="Forecast", layout="wide")

with st.sidebar:
    st.page_link("pages/dashboard_page.py", label="Dashboard", icon="🏠")
    st.page_link("pages/search_page.py", label="Stocks Search", icon="🔎")
    st.page_link("pages/stocks_page.py", label="Stocks List", icon="📃")
    st.page_link("pages/forecast_page.py", label="Forecast", icon="📈")


def plot_forecasts(history: pd.DataFrame, results: dict[str, object], symbol: str, interval: str):
    df_hist = history.copy()
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    fig.add_trace(
        go.Scatter(
            x=df_hist.index,
            y=df_hist["Close"],
            name="Historical Close",
            mode="lines",
        ),
        row=1,
        col=1,
    )

    for name, result in results.items():
        forecast = result.forecast
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.values,
                name=f"{name} forecast",
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=f"Forecast comparison for {symbol} ({interval})",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True,
        height=750,
    )

    st.plotly_chart(fig, use_container_width=True)


repo = SqliteStockRepository("stocks.db")
service = StockDataService(repo)


st.header("Forecast comparison")

with st.form("forecast_form"):
    ticker = st.text_input("Ticker symbol", value="AAPL")
    interval = st.selectbox("Interval", ["Day", "Week", "Month"], index=0)
    horizon = st.number_input("Forecast horizon (steps)", min_value=1, max_value=60, value=5, step=1)

    ma_window = st.number_input(
        "Moving average window (for MA model)",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
    )

    st.markdown("**ARIMA settings (p, d, q):**")
    col_p, col_d, col_q = st.columns(3)
    with col_p:
        ar_p = st.number_input("p", min_value=0, max_value=5, value=1, step=1)
    with col_d:
        ar_d = st.number_input("d", min_value=0, max_value=2, value=1, step=1)
    with col_q:
        ar_q = st.number_input("q", min_value=0, max_value=5, value=0, step=1)

    submitted = st.form_submit_button("Run forecast")


if submitted:
    try:
        history = service.get_history(symbol=ticker, interval=interval, force_refresh=False)
        if history.empty:
            st.warning("No data available for this selection.")
        else:
            # Run multiple models: naive, moving average, ARIMA
            results = {}

            naive_result = run_baseline_forecast(
                history=history,
                horizon=int(horizon),
                model_type="naive",
            )
            results["Naive"] = naive_result

            ma_result = run_baseline_forecast(
                history=history,
                horizon=int(horizon),
                model_type="moving_average",
                window=int(ma_window),
            )
            results["Moving average"] = ma_result

            try:
                arima_result = run_arima_forecast(
                    history=history,
                    horizon=int(horizon),
                    order=(int(ar_p), int(ar_d), int(ar_q)),
                )
                results["ARIMA"] = arima_result
            except Exception as arima_err:
                st.warning(f"ARIMA model failed: {arima_err}")

            # Metrics table
            metrics_rows = []
            for name, res in results.items():
                metrics_rows.append(
                    {
                        "Model": name,
                        "Horizon": res.horizon,
                        "MAE (last horizon)": None if res.mae is None else round(res.mae, 4),
                    }
                )

            metrics_df = pd.DataFrame(metrics_rows)

            st.subheader("Model comparison")
            st.dataframe(metrics_df, hide_index=True)

            plot_forecasts(history, results, ticker, interval)

            with st.expander("Show raw data"):
                st.subheader("Historical data")
                st.dataframe(history)

                for name, res in results.items():
                    st.subheader(f"{name} forecasted values")
                    st.dataframe(res.forecast)
    except Exception as e:
        st.error(str(e))
else:
    st.info("Configure the forecast parameters and click 'Run forecast'.")
