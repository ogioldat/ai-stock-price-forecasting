from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Iterable, Iterator, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    __slots__ = ("_store", "_maxsize")

    def __init__(self, maxsize: int = 32):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive.")
        self._store: OrderedDict[K, V] = OrderedDict()
        self._maxsize = maxsize

    def __contains__(self, key: K) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> Iterable[K]:
        return self._store.keys()

    def values(self) -> Iterable[V]:
        return self._store.values()

    def items(self) -> Iterable[tuple[K, V]]:
        return self._store.items()

    def clear(self) -> None:
        self._store.clear()

    def get(self, key: K) -> V | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: K, value: V) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def popitem(self) -> tuple[K, V]:
        return self._store.popitem()

    def __iter__(self) -> Iterator[K]:
        return iter(self._store)
