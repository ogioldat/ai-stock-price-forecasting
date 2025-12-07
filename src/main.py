from data.repositories.sqlite_stock_repository import SqliteStockRepository
from data.services.stock_data_service import StockDataService

import logging


def main():
    logging.info("Starting execution")

    repo = SqliteStockRepository("stocks.db")
    service = StockDataService(repo)

    symbol = "AAPL"

    logging.info("Fetching the data ...")

    df = service.get_history(symbol)

    logging.info("Fetch succesfull!")

    print(df.tail())

    logging.info("Fetching again for cached version ...")

    df_cached = service.get_history(symbol)

    logging.info("Cached succesfull!")

    print(df_cached.tail())


if __name__ == "__main__":
    main()
