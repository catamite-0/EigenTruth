"""Request-scoped verifier caching helpers."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, MutableMapping, Sequence

from eigentruth.verify.protocols import Claim, VerificationResult, Verifier


@dataclass(frozen=True)
class VerifierCacheStats:
    """Small JSON-serializable cache summary."""

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
class CachedVerifier:
    """Cache verifier results for repeated claim/context checks.

    The cache is intentionally request-local and in-memory. It preserves the
    wrapped verifier's result objects and exposes hit/miss counters separately
    instead of mutating result metadata.
    """

    verifier: Verifier
    max_size: int | None = 1024
    include_context: bool = True
    _cache: MutableMapping[str, VerificationResult] = field(default_factory=OrderedDict, init=False, repr=False)
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_size is not None and self.max_size < 1:
            raise ValueError("max_size must be >= 1 or None.")

    @property
    def stats(self) -> VerifierCacheStats:
        """Return current cache statistics."""
        return VerifierCacheStats(size=len(self._cache), hits=self._hits, misses=self._misses)

    def clear(self) -> None:
        """Clear cached results and counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim, returning a cached result when available."""
        key = verifier_cache_key(claim, context if self.include_context else None)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            if isinstance(self._cache, OrderedDict):
                self._cache.move_to_end(key)
            return result

        result = self.verifier.verify(claim, context=context)
        self._misses += 1
        self._cache[key] = result
        if isinstance(self._cache, OrderedDict):
            self._cache.move_to_end(key)
        if self.max_size is not None:
            while len(self._cache) > self.max_size:
                self._cache.pop(next(iter(self._cache)))
        return result

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims through the same cache."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def verifier_cache_key(claim: Claim, context: Mapping[str, Any] | None = None) -> str:
    """Return a stable JSON key for claim/context verifier inputs."""
    payload = {
        "claim": {
            "text": claim.text,
            "claim_id": claim.claim_id,
            "span": claim.span,
            "metadata": claim.metadata,
        },
        "context": context,
    }
    return stable_cache_key(payload)


def stable_cache_key(value: Any) -> str:
    """Return a deterministic string key for JSON-like values."""
    normalized = _normalize_cache_value(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_cache_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _normalize_cache_value(value.to_dict())
        except TypeError:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_cache_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize_cache_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_cache_value(item) for item in value]
    return repr(value)
