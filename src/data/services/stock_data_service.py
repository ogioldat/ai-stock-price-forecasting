import yfinance as yf

from functools import lru_cache
from datetime import datetime, date
from typing import Optional, Union, Tuple, Dict
import pandas as pd
import re

from data.exceptions import InvalidTickerError, FetchError
from data.repositories.sqlite_stock_repository import SqliteStockRepository

DateLike = Union[str, date, datetime]


class StockDataService:
    """
    Service responsible for fetching and optionally caching stock data.
    """

    _TICKER_REGEX = re.compile(r"^[A-Z]+(?:-[A-Z]+)*$")

    def __init__(self, repository: SqliteStockRepository):
        self._ticker_cache: Dict[str, yf.Ticker] = {}
        self._history_cache: Dict[Tuple, pd.DataFrame] = {}
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

        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)

        return self._ticker_cache[symbol]

    def _save_history(self, symbol: str, interval: str, history: pd.DataFrame) -> None:
        if history.empty:
            return

        self._repo.save_history(symbol, interval, history)

    # TODO: Move interval mapping to a separate utility module if needed elsewhere
    @staticmethod
    def _map_interval(interval: str) -> str:
        interval = interval.strip().lower()

        interval_mapping = {
            'day': '1d',
            'week': '1wk',
            'month': '1m'
        }

        return interval_mapping.get(interval, '1d')

    def _check_for_data_in_cache(self, cache_key: tuple) -> bool:
        return cache_key in self._history_cache

    def _get_data_from_cache(self, cache_key: tuple) -> pd.DataFrame:
         return self._history_cache[cache_key]

    def _get_data_from_db(self, symbol: str, interval: str, cache_key: tuple) -> pd.DataFrame | None:
        history = self._repo.load_history(symbol, interval)

        if history is not None:
            self._history_cache[cache_key] = history

        return history

    def _get_data_from_api(self, symbol: str, interval: str, start: Optional[DateLike], end: Optional[DateLike], cache_key: tuple) -> pd.DataFrame:
        try:
            ticker = self._get_ticker(symbol)
            history = ticker.history(interval=interval, start=start, end=end)

            if history.empty:
                raise FetchError(f"No data returned for ticker '{symbol}'.")

            if start is None and end is None:
                self._repo.save_history(symbol, interval, history)

            self._history_cache[cache_key] = history

            return history

        except Exception:
            raise FetchError(f"Failed to fetch data for ticker '{symbol}'.")

    @lru_cache
    def get_history(
        self,
        symbol: str,
        interval: str = 'Day',
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch the historical data for a given symbol. Cached for repeated symbols.
        """

        self._validate_symbol(symbol)
        symbol = symbol.strip().upper()

        interval = self._map_interval(interval)

        cache_key = (symbol, interval, start, end)

        if not force_refresh:
            if self._check_for_data_in_cache(cache_key):
                return self._get_data_from_cache(cache_key)

            data = self._get_data_from_db(symbol, interval, cache_key)
            if data is not None:
                return data

        data = self._get_data_from_api(symbol, interval, start, end, cache_key)
        self._save_history(symbol, interval, data)

        return data

    def get_list_of_stocks(self) -> list[str]:
        """
        Uses the repository to get a list of all stock tickers available in the database.
        """
        return self._repo.get_all_tickers()

    def clear_cache(self) -> None:
        self._ticker_cache.clear()
        self._history_cache.clear()
