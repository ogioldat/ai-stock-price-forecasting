from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

DateLike = str | date | datetime


class Interval(StrEnum):
    DAY = "1d"
    WEEK = "1wk"
    MONTH = "1m"

    @property
    def display_name(self) -> str:
        return _INTERVAL_DISPLAY_NAMES[self]

    @classmethod
    def from_user_input(cls, raw: str | "Interval" | None) -> "Interval":
        if isinstance(raw, cls):
            return raw
        if raw is None:
            return cls.DAY

        normalized = raw.strip().lower()
        try:
            return _INTERVAL_ALIASES[normalized]
        except KeyError as e:
            raise ValueError(f"Unsupported interval '{raw}'.") from e


_INTERVAL_DISPLAY_NAMES: Mapping[Interval, str] = MappingProxyType(
    {
        Interval.DAY: "Day",
        Interval.WEEK: "Week",
        Interval.MONTH: "Month",
    }
)

_INTERVAL_ALIASES: Mapping[str, Interval] = MappingProxyType(
    {
        "day": Interval.DAY,
        "1d": Interval.DAY,
        "week": Interval.WEEK,
        "1wk": Interval.WEEK,
        "month": Interval.MONTH,
        "1m": Interval.MONTH,
    }
)


@dataclass(frozen=True, slots=True)
class HistoryRequest:
    symbol: str
    interval: Interval
    start: DateLike | None = None
    end: DateLike | None = None

    @classmethod
    def build(
        cls,
        symbol: str,
        interval: Interval | str | None,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> HistoryRequest:
        interval_obj = Interval.from_user_input(interval)
        normalized_symbol = symbol.strip().upper()
        return cls(
            symbol=normalized_symbol,
            interval=interval_obj,
            start=start,
            end=end,
        )

    @property
    def is_unbounded(self) -> bool:
        return self.start is None and self.end is None
