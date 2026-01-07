from __future__ import annotations

import argparse
from datetime import datetime, date
from pathlib import Path
from typing import Optional, cast

import pandas as pd

from data.exceptions import FetchError, InvalidTickerError
from data.repositories.sqlite_stock_repository import SqliteStockRepository
from data.services.stock_data_service import StockDataService

DEFAULT_DB_PATH = Path("./src/stocks.db")


def _parse_date(value: Optional[str]) -> Optional[date]:
    if value is None or value == "":
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD format."
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch historical stock data using the existing project services."
    )
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL or MSFT.")
    parser.add_argument(
        "--interval",
        choices=["day", "week", "month"],
        default="day",
        help="Sampling interval for the history (default: day).",
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="Optional start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="Optional end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database (default: stocks.db).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of recent rows to display (use 0 to show everything).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass the local cache and force API refresh.",
    )

    return parser


def _build_service(database: Path) -> StockDataService:
    repository = SqliteStockRepository(str(database))
    return StockDataService(repository)


def _format_dataframe(df: pd.DataFrame, rows: int) -> str:
    disp_df = df.reset_index()

    if rows > 0:
        disp_df = disp_df.tail(rows)

    formatted = cast(str, disp_df.to_string(index=False))
    return formatted


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    service = _build_service(args.database)

    try:
        df = service.get_history(
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
            end=args.end,
            force_refresh=args.force_refresh,
        )
    except (InvalidTickerError, FetchError) as exc:
        parser.error(str(exc))
        return 1

    if df.empty:
        print("No data returned for the requested parameters.")
        return 0

    print(
        f"Fetched {len(df)} rows for {args.symbol.upper()} "
        f"(interval: {args.interval.title()})."
    )
    if args.start or args.end:
        print(
            f"Date window: "
            f"{args.start.isoformat() if args.start else 'beginning'} -> "
            f"{args.end.isoformat() if args.end else 'latest'}"
        )

    print(_format_dataframe(df, args.rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
