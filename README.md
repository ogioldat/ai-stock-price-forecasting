# ai-stock-price-forecasting

## How to run

1. Make sure you have [uv](https://docs.astral.sh/uv/guides/install-python/) installed on your system

2. Clone the project

```bash
git clone https://github.com/ogioldat/ai-stock-price-forecasting.git
```

3. Go to project root

```bash
cd ai-stock-price-forecasting
```

4. Run the app with uv

```bash
uv run streanlit run src/main.py
```

5. To run tests do the following

```bash
uv run pytest
```

## Forecasting module

The utilities in `src/forecasting` power both the Streamlit UI and CLI tools. The module now:

- normalizes any incoming price history (sorted, deduplicated, forward/back filled, and converted to a `DatetimeIndex`);
- exposes dataclass-based configs for naive, moving-average, and ARIMA forecasts, so inputs are validated before training;
- handles sparse or irregular data by inferring frequencies and falling back to daily stamps when necessary; and
- emits structured logging that records the selected model, parameters, and number of history points processed.

See `forecasting.baseline` and `forecasting.classical` for concrete usage examples and available configuration knobs.

