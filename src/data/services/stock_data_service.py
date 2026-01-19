from __future__ import annotations

import re
from typing import Final

import pandas as pd
import yfinance as yf

from data.exceptions import FetchError, InvalidTickerError
from data.models import DateLike, HistoryRequest, Interval
from data.repositories.sqlite_stock_repository import SqliteStockRepository
from data.structures import LRUCache


class StockDataService:
    _TICKER_REGEX = re.compile(r"^[A-Z]+(?:-[A-Z]+)*$")
    _TICKER_CACHE_SIZE: Final[int] = 32

    def __init__(
        self, repository: SqliteStockRepository, history_cache_size: int = 64
    ) -> None:
        self._ticker_cache: LRUCache[str, yf.Ticker] = LRUCache(
            maxsize=self._TICKER_CACHE_SIZE
        )
        self._history_cache: LRUCache[HistoryRequest, pd.DataFrame] = LRUCache(
            maxsize=history_cache_size
        )
        self._repo = repository

    def _validate_symbol(self, symbol: str) -> None:
        if not symbol.strip() or not isinstance(symbol, str):
            raise InvalidTickerError("Ticker symbol is missing or invalid.")

        symbol = symbol.strip().upper()

        if len(symbol) < 2:
            raise InvalidTickerError(f"Ticker {symbol} must be at least 2 characters.")

        if not self._TICKER_REGEX.fullmatch(symbol):
            raise InvalidTickerError(f"Ticker {symbol} is not a valid ticker symbol.")

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        symbol = symbol.strip().upper()

        cached = self._ticker_cache.get(symbol)
        if cached is not None:
            return cached

        ticker = yf.Ticker(symbol)
        self._ticker_cache.put(symbol, ticker)

        return ticker

    def _cache_history(self, request: HistoryRequest, history: pd.DataFrame) -> None:
        if history.empty:
            return

        self._history_cache.put(request, history.copy())

    def _get_cached_history(self, request: HistoryRequest) -> pd.DataFrame | None:
        cached = self._history_cache.get(request)
        if cached is None:
            return None

        return cached.copy()

    def _save_history(self, request: HistoryRequest, history: pd.DataFrame) -> None:
        if history.empty:
            return

        self._repo.save_history(request.symbol, request.interval.value, history)

    def _get_data_from_db(self, request: HistoryRequest) -> pd.DataFrame | None:
        history = self._repo.load_history(request.symbol, request.interval.value)

        if history is not None:
            self._cache_history(request, history)

        return history

    def _get_data_from_api(
        self,
        request: HistoryRequest,
    ) -> pd.DataFrame:
        """
        Fetch history data from the upstream API and cache the successful result.

        Parameters
        ----------
        request:
            Fully validated history request describing the ticker, interval,
            and optional start/end bounds.

        Returns
        -------
        pd.DataFrame
            Raw frame returned by yfinance, indexed by timestamp.

        Raises
        ------
        FetchError
            If the upstream provider fails or returns an empty dataset.
        """
        try:
            ticker = self._get_ticker(request.symbol)

            history_kwargs: dict[str, object] = {
                "interval": request.interval.value,
            }
            if request.start is None and request.end is None:
                history_kwargs["period"] = "max"
            else:
                if request.start is not None:
                    history_kwargs["start"] = request.start
                if request.end is not None:
                    history_kwargs["end"] = request.end

            history = ticker.history(**history_kwargs)

            if history.empty:
                raise FetchError(f"No data returned for ticker '{request.symbol}'.")

            self._cache_history(request, history)

            return history

        except Exception as exc:
            raise FetchError(
                f"Failed to fetch data for ticker '{request.symbol}'."
            ) from exc

    def get_history(
        self,
        symbol: str,
        interval: Interval | str = Interval.DAY,
        start: DateLike | None = None,
        end: DateLike | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Resolve a ticker's OHLCV history via cache, database, or API fetch.

        Parameters
        ----------
        symbol:
            Desired ticker symbol; validation and normalization are applied.
        interval:
            Interval enum or literal passed through to yfinance (defaults to daily).
        start:
            Optional inclusive start date for the history window.
        end:
            Optional exclusive end date for the window.
        force_refresh:
            When True, bypasses caches and always pulls fresh data from the API.

        Returns
        -------
        pd.DataFrame
            Historical price data indexed by timestamp.

        Raises
        ------
        InvalidTickerError
            If the ticker symbol cannot be normalized/validated.
        FetchError
            When every data source fails to provide history for the ticker.
        """
        self._validate_symbol(symbol)
        request = HistoryRequest.build(
            symbol=symbol, interval=interval, start=start, end=end
        )

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

    def get_list_of_stocks(self) -> list[str]:
        """
        Uses the repository to get a list of all stock tickers available in the database.
        """
        return self._repo.get_all_tickers()

    def clear_cache(self) -> None:
        self._ticker_cache.clear()
        self._history_cache.clear()
