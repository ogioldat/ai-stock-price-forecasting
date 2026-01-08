# Technical Overview

Here are 6 excerpts from the code that demonstrate some more advanced Python techniques, along with explanations and practical use cases.

## 1. Generic LRU cache with controlled memory
**File:** `src/data/structures.py:11-54`

```python
class LRUCache(Generic[K, V]):
    __slots__ = ("_store", "_maxsize")

    def __init__(self, maxsize: int = 32):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive.")
        self._store: OrderedDict[K, V] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: K) -> V | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: K, value: V) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)
```

**Why it stands out**
- Uses `Generic` type parameters with `__slots__` for low-overhead caching without third-party deps.
- Leverages `OrderedDict` mutation (`move_to_end`, `popitem(last=False)`) to implement an SRAM-style eviction policy in ~10 lines.
- Provides familiar dictionary-like APIs (`__contains__`, `keys`, `items`, iteration) for drop-in use.

**Use case**
- `StockDataService` keeps both `yfinance.Ticker` objects and pandas histories in process-memory LRU caches to avoid repeated HTTP calls and recomputations.

## 2. Immutable request objects + Enum-driven parsing
**File:** `src/data/models.py:12-81`

```python
class Interval(StrEnum):
    DAY = "1d"
    WEEK = "1wk"
    MONTH = "1m"

    @classmethod
    def from_user_input(cls, raw: str | "Interval" | None) -> "Interval":
        if isinstance(raw, cls):
            return raw
        if raw is None:
            return cls.DAY

        normalized = raw.strip().lower()
        try:
            return _INTERVAL_ALIASES[normalized]
        except KeyError as e:
            raise ValueError(f"Unsupported interval '{raw}'.") from e


@dataclass(frozen=True, slots=True)
class HistoryRequest:
    symbol: str
    interval: Interval
    start: DateLike | None = None
    end: DateLike | None = None

    @classmethod
    def build(...):
        interval_obj = Interval.from_user_input(interval)
        normalized_symbol = symbol.strip().upper()
        return cls(symbol=normalized_symbol, interval=interval_obj, start=start, end=end)
```

**Why it stands out**
- `StrEnum` keeps wire-format values and rich enum semantics in sync, while `MappingProxyType` (not shown above) makes alias tables immutable.
- `HistoryRequest` is an immutable (`frozen=True`) and memory-efficient (`slots=True`) dataclass whose `build` factory does normalization and interval parsing.
- The combination makes it safe to use `HistoryRequest` instances as dictionary/LRU keys without hashing surprises.

**Use case**
- Every API or DB fetch uses `HistoryRequest.build(...)`, ensuring ticker symbols and interval strings from Streamlit forms are sanitized before caching or SQL queries.

## 3. Pandas-first data cleaning pipeline
**File:** `src/forecasting/preprocessing.py:11-47`

```python
def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in history.columns:
        raise ValueError("History frame must contain a 'Close' column.")

    normalized = history.copy()
    if not isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.to_datetime(normalized.index)

    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert(None)

    normalized.sort_index(inplace=True)
    if normalized.index.has_duplicates:
        logger.warning("History contains duplicate timestamps; keeping the last occurrence.")
        normalized = normalized[~normalized.index.duplicated(keep="last")]

    closes = pd.to_numeric(normalized["Close"], errors="coerce")
    if closes.isna().any():
        logger.info("Filling %s missing Close values via forward/backward fill.", int(closes.isna().sum()))
        closes = closes.ffill().bfill()
```

**Why it stands out**
- Defensive programming: strict dtype checks followed by canonicalization of indices, time zones, sorting, and deduplication.
- Uses vectorized pandas operations (e.g., `.duplicated`, `.ffill()/.bfill()`) with logging hooks so viewers can discuss both data engineering and observability.
- Coercing strings to numeric with `errors="coerce"` plus dropna ensures later models get clean floats.

**Use case**
- Every forecasting strategy calls `normalize_history` (via `run_forecast`) so even data scraped from SQLite or the network becomes model-ready with consistent timestamps.

## 4. singledispatch-powered forecasting router
**File:** `src/forecasting/core.py:22-44`

```python
@singledispatch
def _run_with_config(config: ForecastConfig, history: pd.DataFrame) -> ForecastResult:
    raise TypeError(f"Unsupported forecast configuration: {config!r}")

@_run_with_config.register
def _(config: NaiveForecastConfig, history: pd.DataFrame) -> ForecastResult:
    return run_naive_strategy(history, config)

@_run_with_config.register
def _(config: MovingAverageForecastConfig, history: pd.DataFrame) -> ForecastResult:
    return run_moving_average_strategy(history, config)

def run_forecast(history: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
    normalized = normalize_history(history)
    return _run_with_config(config, normalized)
```

**Why it stands out**
- `functools.singledispatch` turns the forecasting core into an extensible plug-in system: new config types register new strategy functions without touching control flow.
- Keeps validation (`normalize_history`) centralized before the dispatch, ensuring every strategy sees identical inputs.
- Demonstrates how to express polymorphism in a functional style instead of through big if/elif chains or class hierarchies.

**Use case**
- Streamlit page builds `NaiveForecastConfig`, `MovingAverageForecastConfig`, or `ArimaForecastConfig`; `run_forecast` automatically invokes the proper implementation.

## 5. Multi-tier caching + graceful fallbacks
**File:** `src/data/services/stock_data_service.py:19-130`

```python
def get_history(..., force_refresh: bool = False) -> pd.DataFrame:
    self._validate_symbol(symbol)
    request = HistoryRequest.build(...)

    if not force_refresh:
        cached = self._get_cached_history(request)
        if cached is not None:
            return cached

        data = self._get_data_from_db(request)
        if data is not None:
            return data

    data = self._get_data_from_api(request)
    self._save_history(request, data)
    return data
```

**Why it stands out**
- Composes two independent LRU caches (`yf.Ticker` objects and pandas histories) with a SQLite persistence layer to minimize expensive API calls.
- Uses immutable `HistoryRequest` instances as cache keys so `force_refresh` can deterministically bypass caches when needed.
- `_save_history` only writes unbounded requests back to SQLite, showing an awareness of storage blowups while still reusing historical data.

**Use case**
- Power your demo by clearing the cache and re-fetching a ticker to highlight the path: validation → cache lookup → DB query → yfinance call → persistence for the next user.

## 6. Statsmodels ARIMA integration with on-the-fly evaluation
**File:** `src/forecasting/strategies/arima.py:17-57`

```python
model = ARIMA(closes, order=config.order)
fitted = model.fit()

forecast_values = fitted.forecast(steps=config.horizon)
forecast_index = build_horizon_index(history, config.horizon)
forecast_series = pd.Series(forecast_values.values, index=forecast_index, name="Close")

mae = None
if len(closes) > config.horizon:
    start = len(closes) - config.horizon
    end = len(closes) - 1
    y_true = closes.iloc[start : end + 1]
    y_pred = fitted.predict(start=start, end=end)
    y_pred = pd.Series(y_pred.values, index=y_true.index)
    mae = compute_mae(y_true, y_pred)
```

**Why it stands out**
- Bridges third-party statistical modeling (`statsmodels.ARIMA`) with your Pandas-native data structures and forecasting abstractions.
- Builds a deterministic future index via `build_horizon_index`, decoupling the temporal resolution from the ARIMA internals.
- Computes MAE on the fly using the model’s own in-sample predictions, yielding immediate QA feedback without extra passes through the pipeline.

**Use case**
- Showcase how swapping ARIMA parameters (p, d, q) in the UI propagates through this pipeline, resulting in different residual diagnostics and MAE values for stakeholders.

