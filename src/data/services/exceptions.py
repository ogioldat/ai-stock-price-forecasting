class StockServiceError(Exception):
    """
        This is a base class for stock service errors
    """

class InvalidTickerError(StockServiceError):
    """
        Raised when ticker symbol is missing
    """
    pass

class FetchError(StockServiceError):
    """
        Raised when data fetching fails
    """
    pass
