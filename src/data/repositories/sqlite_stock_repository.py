import logging
import sqlite3
from typing import Optional

import pandas as pd


class SqliteStockRepository:
    """
        A class to represent the sqlite3 repository.
    """

    def __init__(self, db_path: str = "stocks.db") -> None:
        self.db_path = db_path
        self._ensure_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_prices (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timestamp, interval)
                );
                """
            )

            conn.commit()

    def save_history(self, symbol: str, interval: str, history_data: pd.DataFrame) -> None:
        if history_data.empty:
            return

        history_data.copy()
        history_data.reset_index(inplace=True)

        rows = [
            (
                symbol,
                row.iloc[0].isoformat(),
                interval,
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                row.get("Volume"),
            )
            for _, row in history_data.iterrows()
        ]

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stock_prices (
                    symbol, timestamp, interval, open, high, low, close, volume
                ) VALUES (
                    ? ,? ,? ,? ,? ,? ,? ,?
                );
                """,
                rows,
            )

            conn.commit()

    def load_history(self, symbol: str, interval: str) -> pd.DataFrame | None:
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM stock_prices
                WHERE symbol = ? AND interval = ?
                ORDER BY timestamp ASC;
                """,
                conn,
                params=(symbol, interval),
                parse_dates=["timestamp"],
            )

            if df.empty:
                return None

            df.set_index("timestamp", inplace=True)
            df.index.name = "Date"

            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )

            return df if not df.empty else None

    def get_available_intervals(self, symbol: str) -> pd.DataFrame:
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT interval
                FROM stock_prices
                WHERE symbol = ?
                ORDER BY interval;
                """,
                conn,
                params=(symbol.strip().upper(),)
            )

            return df["interval"].tolist()
    def get_all_tickers(self) -> list[str]:
        """
        Get a list of all stock tickers available in the database.
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT symbol
                FROM stock_prices
                ORDER BY symbol;
                """,
                conn,
            )
            logging.log(msg=f"{df["symbol"].tolist()}", level=10)
            return df["symbol"].tolist()
