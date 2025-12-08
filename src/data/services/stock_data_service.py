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

    @lru_cache
    def get_history(
        self,
        symbol: str,
        interval: str = "1d",
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch the historical data for a given symbol. Cached for repeated symbols.
        """

        self._validate_symbol(symbol)
        symbol = symbol.strip().upper()

        cache_key = (symbol, interval, start, end)

        ## Try Cached
        if not force_refresh and cache_key in self._history_cache:
            return self._history_cache[cache_key]

        ## Try Databse
        if not force_refresh and start is not None and end is None:
            db_df = self._repo.load_history(symbol, interval)
            if db_df is not None:
                self._history_cache[cache_key] = db_df
                return db_df

        ## Fetch the data from API
        try:
            ticker = self._get_ticker(symbol)
            df = ticker.history(interval=interval, start=start, end=end)

            if df.empty:
                raise FetchError(f"No data returned for ticker '{symbol}'.")

            if start is None and end is None:
                self._repo.save_history(symbol, interval, df)

            self._history_cache[cache_key] = df

            return df

        except Exception:
            raise FetchError(f"Failed to fetch data for ticker '{symbol}'.")

    def clear_cache(self) -> None:
        self._ticker_cache.clear()
        self._history_cache.clear()
