import pytest
import pandas as pd
from pathlib import Path

from data.repositories.sqlite_stock_repository import SqliteStockRepository


@pytest.fixture(name="db_path")
def db_path_fixture(tmp_path: Path) -> str:
    return str(tmp_path / "test_stocks.db")


@pytest.fixture(name="repository")
def sqlite_repo_fixtue(db_path: str) -> SqliteStockRepository:
    return SqliteStockRepository(db_path)


@pytest.fixture(name="sample_df")
def sample_df_fixture() -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01", periods=3)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 101.0, 102.0],
            "High": [100.0, 101.0, 102.0],
            "Low": [100.0, 101.0, 102.0],
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1_000_000.0, 2_000_000.0, 3_000_000.0],
        },
    )

    df.set_index(['Date'], inplace=True)

    return df

