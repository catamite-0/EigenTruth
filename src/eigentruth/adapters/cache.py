"""Request-scoped adapter caches."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

from eigentruth.adapters.retrieval import RetrievalHit, RetrievalQuery, Retriever
from eigentruth.adapters.state import StateSource
from eigentruth.verify.cache import stable_cache_key


@dataclass(frozen=True)
class AdapterCacheStats:
    """Small JSON-serializable adapter cache summary."""

    size: int
    hits: int
    misses: int

    @property
    def requests(self) -> int:
        """Return total cache lookups."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        """Return cache hit rate, or None when there were no lookups."""
        if self.requests == 0:
            return None
        return self.hits / self.requests

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "requests": self.requests,
            "hit_rate": self.hit_rate,
        }


@dataclass
class CachedStateSource:
    """Cache one structured state load for a request or benchmark run."""

    source: StateSource
    copy_on_read: bool = True
    _loaded: bool = field(default=False, init=False, repr=False)
    _state: Mapping[str, Any] | None = field(default=None, init=False, repr=False)
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)

    @property
    def stats(self) -> AdapterCacheStats:
        """Return current cache statistics."""
        return AdapterCacheStats(size=1 if self._loaded else 0, hits=self._hits, misses=self._misses)

    def clear(self) -> None:
        """Clear cached state and counters."""
        self._loaded = False
        self._state = None
        self._hits = 0
        self._misses = 0

    def load_state(self) -> Mapping[str, Any]:
        """Load state once, then return cached state for repeated calls."""
        if self._loaded:
            self._hits += 1
        else:
            self._state = self.source.load_state()
            self._loaded = True
            self._misses += 1
        if self.copy_on_read:
            return copy.deepcopy(self._state)
        return self._state if self._state is not None else {}


@dataclass
class CachedRetriever:
    """Cache retrieval hits by query and limit."""

    retriever: Retriever
    max_size: int | None = 1024
    _cache: MutableMapping[str, tuple[RetrievalHit, ...]] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_size is not None and self.max_size < 1:
            raise ValueError("max_size must be >= 1 or None.")

    @property
    def stats(self) -> AdapterCacheStats:
        """Return current cache statistics."""
        return AdapterCacheStats(size=len(self._cache), hits=self._hits, misses=self._misses)

    def clear(self) -> None:
        """Clear cached retrieval results and counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def retrieve(self, query: RetrievalQuery, *, limit: int = 5) -> tuple[RetrievalHit, ...]:
        """Return cached retrieval hits when the query/limit repeats."""
        key = stable_cache_key({"query": query.to_dict(), "limit": int(limit)})
        hits = self._cache.get(key)
        if hits is not None:
            self._hits += 1
            if isinstance(self._cache, OrderedDict):
                self._cache.move_to_end(key)
            return hits

        hits = tuple(self.retriever.retrieve(query, limit=limit))
        self._misses += 1
        self._cache[key] = hits
        if isinstance(self._cache, OrderedDict):
            self._cache.move_to_end(key)
        if self.max_size is not None:
            while len(self._cache) > self.max_size:
                self._cache.pop(next(iter(self._cache)))
        return hits


def combine_cache_stats(*stats: Mapping[str, Any]) -> dict[str, Any]:
    """Combine JSON cache stat mappings into one aggregate summary."""
    hits = sum(int(item.get("hits", 0)) for item in stats)
    misses = sum(int(item.get("misses", 0)) for item in stats)
    size = sum(int(item.get("size", 0)) for item in stats)
    requests = hits + misses
    return {
        "size": size,
        "hits": hits,
        "misses": misses,
        "requests": requests,
        "hit_rate": None if requests == 0 else hits / requests,
    }
