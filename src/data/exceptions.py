class StockServiceError(Exception):
    pass


class InvalidTickerError(StockServiceError):
    pass


class FetchError(StockServiceError):
    pass
