from __future__ import annotations

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize a price history frame prior to modeling.

    Parameters
    ----------
    history:
        Raw OHLCV frame whose index may be unsorted/untyped and contain gaps.

    Returns
    -------
    pd.DataFrame
        Copy of the frame sorted by timestamp with a tz-naive `DatetimeIndex`
        and a fully populated numeric `Close` column.

    Raises
    ------
    TypeError
        When the input is not a dataframe.
    ValueError
        If required columns are missing or the cleaning step drops all rows.
    """
    if not isinstance(history, pd.DataFrame):
        raise TypeError("History must be a pandas DataFrame.")
    if "Close" not in history.columns:
        raise ValueError("History frame must contain a 'Close' column.")

    normalized = history.copy()
    if not isinstance(normalized.index, pd.DatetimeIndex):
        try:
            normalized.index = pd.to_datetime(normalized.index)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("History index must be convertible to datetimes.") from exc

    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert(None)

    normalized.sort_index(inplace=True)
    if normalized.index.has_duplicates:
        logger.warning(
            "History contains duplicate timestamps; keeping the last occurrence."
        )
        normalized = normalized[~normalized.index.duplicated(keep="last")]

    closes = pd.to_numeric(normalized["Close"], errors="coerce")
    if closes.isna().any():
        logger.info(
            "Filling %s missing Close values via forward/backward fill.",
            int(closes.isna().sum()),
        )
        closes = closes.ffill().bfill()

    normalized["Close"] = closes
    normalized.dropna(subset=["Close"], inplace=True)
    if normalized.empty:
        raise ValueError("History contains no usable Close values after cleaning.")

    return normalized


def build_horizon_index(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """
    Extend a history index to cover the desired forecast horizon.

    Parameters
    ----------
    history:
        Cleaned dataframe whose index defines the cadence of observations.
    horizon:
        Number of future observations that downstream forecasts will produce.

    Returns
    -------
    pd.DatetimeIndex
        Datetime index of length `horizon` that continues the last known cadence.

    Raises
    ------
    ValueError
        If the history index is not already a `DatetimeIndex`.
    """
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("History index must be a DatetimeIndex.")
    if len(history.index) < 2:
        last = history.index[-1]
        return pd.date_range(last, periods=horizon + 1, freq="D")[1:]

    inferred = pd.infer_freq(history.index)
    if inferred is None:
        inferred = "D"

    last_timestamp = history.index[-1]
    return pd.date_range(start=last_timestamp, periods=horizon + 1, freq=inferred)[1:]
