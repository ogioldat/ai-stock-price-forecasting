# AI Stock Forecasting & Trading Strategy

Explore stock forecasts and trading signals backed by local data storage.

## Features
- Streamlit UI (`src/main.py`) for interactive forecasts and trading views.
- Data layer (`src/data`) that fetches with yfinance, caches in SQLite, and serves typed requests.
- Forecasting package (`src/forecasting`) with configs, preprocessing, metrics, and strategy dispatch.
- Trading module (`src/trading`) delivering MA signals, backtests, and tuning used directly in the app.
- Pytest suite (`tests/`) covering preprocessing, strategies, and wiring.

## Prerequisites
- [uv](https://docs.astral.sh/uv/guides/install-python/) for managing Python versions and dependencies.
- Python 3.11+ (managed automatically when using `uv`).

## Setup
From the project root:
```bash
uv sync
```

The default SQLite database lives at `stocks.db`. You can start fresh by deleting the file; it will be recreated as symbols are fetched.

## Running the Streamlit App
```bash
uv run streamlit run src/main.py
```

The CLI helper `src/cli_fetch_stock.py` is also available if you prefer pulling quotes or warming caches from the terminal.

## Fetching Stock Data
Refresh caches from yfinance via:
```bash
python src/cli_fetch_stock.py WMT --start 2023-01-01 --end 2026-01-01 --force-refresh --rows 50
```

## Tests
```bash
uv run pytest
```

Pytest automatically discovers the suites in `tests/`. Target an individual module with `uv run pytest tests/forecasting/test_forecasting.py`.

## Project Layout
```
src/
  forecasting/
  data/
  trading/
  pages/
tests/
  forecasting/
stocks.db
```
