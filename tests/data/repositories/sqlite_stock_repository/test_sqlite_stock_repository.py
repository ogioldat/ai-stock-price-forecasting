import sqlite3
import pandas as pd

from data.repositories.sqlite_stock_repository import SqliteStockRepository


def test_database_init(db_path: str, repository: SqliteStockRepository) -> None:
    # Assert
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='stock_prices';
            """
        )

        table = cursor.fetchone()

    assert table is not None


def test_database_persistance(
    db_path: str, repository: SqliteStockRepository, sample_df: pd.DataFrame
) -> None:
    # Arrange
    ticker_symbol = "AAPL"
    interval = "1d"

    # Act
    repository.save_history(ticker_symbol, interval, sample_df)

    # Assert
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM stock_prices;")
        count = cur.fetchone()[0]

    assert count == len(sample_df)

def test_loading_existing_data(
        repository: SqliteStockRepository, sample_df: pd.DataFrame
) -> None:

    # Arrange
    ticker_symbol = "AAPL"
    interval = "1d"

    repository.save_history(ticker_symbol, interval, sample_df)

    # Act
    df = repository.load_history(ticker_symbol, interval)

    # Assert
    assert df is not None
    assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

def test_load_history_for_unknown_ticker_returns_none(repository: SqliteStockRepository, sample_df: pd.DataFrame) -> None:

    # Arrange
    ticker_symbol_insert = 'AAPL'
    ticker_symbol_test = 'MSFT'

    interval_insert = '1d'
    interval_test = '1m'


    repository.save_history(ticker_symbol_insert, interval_insert, sample_df)

    # Act
    df = repository.load_history(ticker_symbol_test, interval_test)

    # Assert
    assert df is None
