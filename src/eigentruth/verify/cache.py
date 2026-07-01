"""Request-scoped verifier caching helpers."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from eigentruth.verify.protocols import Claim, VerificationResult, Verifier
from eigentruth.verify.rules import normalize_claim_text

_CACHE_KEY_MODES = frozenset({"exact", "semantic"})
_DEFAULT_SEMANTIC_METADATA_KEYS = (
    "answer",
    "calculation",
    "citation",
    "citations",
    "features",
    "question",
    "qa_check",
    "question_answer_check",
    "references",
    "requires_triple_audit",
    "state_check",
    "state_transition",
    "triples",
    "claim_triples",
)


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


@dataclass(frozen=True)
class TraceCacheRecord:
    """One JSON-ready cached trace payload."""

    key: str
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable cache record."""
        return {
            "schema_version": int(self.schema_version),
            "key": self.key,
            "payload": _normalize_cache_value(self.payload),
            "metadata": _normalize_cache_value(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceCacheRecord":
        """Build a trace cache record from JSON-like data."""
        key = data.get("key")
        if key is None:
            raise ValueError("trace cache record must contain a key.")
        if "payload" not in data:
            raise ValueError("trace cache record must contain a payload.")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("trace cache record metadata must be a JSON object.")
        return cls(
            key=str(key),
            payload=data["payload"],
            metadata=dict(metadata),
            schema_version=int(data.get("schema_version", 1)),
        )


@dataclass(frozen=True)
class JsonTraceCache:
    """Small file-backed JSON trace cache for reproducible local workflows."""

    path: str | Path
    cache_type: str = "trace_cache"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not str(self.cache_type).strip():
            raise ValueError("cache_type must be non-empty.")

    def get(self, key: str) -> Any | None:
        """Return a cached payload for ``key``, if present."""
        record = self.get_record(key)
        return None if record is None else record.payload

    def get_record(self, key: str) -> TraceCacheRecord | None:
        """Return a cached record for ``key``, if present."""
        records = self._load_records()
        payload = records.get(str(key))
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise ValueError("trace cache record must be a JSON object.")
        record = TraceCacheRecord.from_dict(payload)
        if record.key != str(key):
            raise ValueError("trace cache record key mismatch.")
        return record

    def put(self, key: str, payload: Any, *, metadata: Mapping[str, Any] | None = None) -> TraceCacheRecord:
        """Persist a payload for ``key`` and return the stored record."""
        key_text = str(key)
        record = TraceCacheRecord(key=key_text, payload=payload, metadata=dict(metadata or {}))
        data = self._load()
        records = data.setdefault("records", {})
        if not isinstance(records, MutableMapping):
            raise ValueError("trace cache records must be a JSON object.")
        records[key_text] = record.to_dict()
        self._write(data)
        return record

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready cache summary."""
        records = self._load_records()
        return {
            "path": str(self.path),
            "cache_type": self.cache_type,
            "records": len(records),
        }

    def _load_records(self) -> dict[str, Any]:
        return dict(self._load().get("records", {}))

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": 1,
                "cache_type": self.cache_type,
                "records": {},
            }
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("trace cache must contain a JSON object.")
        if int(payload.get("schema_version", 1)) != 1:
            raise ValueError("unsupported trace cache schema_version.")
        if str(payload.get("cache_type", self.cache_type)) != self.cache_type:
            raise ValueError("trace cache_type mismatch.")
        records = payload.get("records", {})
        if not isinstance(records, Mapping):
            raise ValueError("trace cache records must be a JSON object.")
        return {
            "schema_version": 1,
            "cache_type": self.cache_type,
            "records": dict(records),
        }

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(_normalize_cache_value(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)


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
    cache_key_mode: str = "exact"
    semantic_metadata_keys: Sequence[str] = _DEFAULT_SEMANTIC_METADATA_KEYS
    _cache: MutableMapping[str, VerificationResult] = field(default_factory=OrderedDict, init=False, repr=False)
    _hits: int = field(default=0, init=False, repr=False)
    _misses: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_size is not None and self.max_size < 1:
            raise ValueError("max_size must be >= 1 or None.")
        cache_key_mode = str(self.cache_key_mode).strip().lower()
        if cache_key_mode not in _CACHE_KEY_MODES:
            choices = ", ".join(sorted(_CACHE_KEY_MODES))
            raise ValueError(f"cache_key_mode must be one of: {choices}.")
        if isinstance(self.semantic_metadata_keys, (str, bytes, bytearray)):
            raise ValueError("semantic_metadata_keys must be a sequence of metadata key names.")
        semantic_metadata_keys = tuple(str(key).strip() for key in self.semantic_metadata_keys)
        if any(not key for key in semantic_metadata_keys):
            raise ValueError("semantic_metadata_keys must not contain empty key names.")
        self.cache_key_mode = cache_key_mode
        self.semantic_metadata_keys = semantic_metadata_keys

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
        key = verifier_cache_key(
            claim,
            context if self.include_context else None,
            cache_key_mode=self.cache_key_mode,
            semantic_metadata_keys=self.semantic_metadata_keys,
        )
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


def verifier_cache_key(
    claim: Claim,
    context: Mapping[str, Any] | None = None,
    *,
    cache_key_mode: str = "exact",
    semantic_metadata_keys: Sequence[str] = _DEFAULT_SEMANTIC_METADATA_KEYS,
) -> str:
    """Return a stable JSON key for claim/context verifier inputs."""
    mode = str(cache_key_mode).strip().lower()
    if mode not in _CACHE_KEY_MODES:
        choices = ", ".join(sorted(_CACHE_KEY_MODES))
        raise ValueError(f"cache_key_mode must be one of: {choices}.")
    if mode == "semantic":
        payload = {
            "claim": {
                "text": normalize_claim_text(claim.text),
                "text_normalizer": "normalize_claim_text_v1",
                "metadata": _semantic_claim_metadata(
                    claim.metadata,
                    semantic_metadata_keys=semantic_metadata_keys,
                ),
            },
            "context": context,
        }
        return stable_cache_key(payload)
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


def _semantic_claim_metadata(
    metadata: Mapping[str, Any],
    *,
    semantic_metadata_keys: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    if isinstance(semantic_metadata_keys, (str, bytes, bytearray)):
        raise ValueError("semantic_metadata_keys must be a sequence of metadata key names.")
    selected = {}
    for key in semantic_metadata_keys:
        key_text = str(key).strip()
        if not key_text:
            raise ValueError("semantic_metadata_keys must not contain empty key names.")
        if key_text in metadata:
            selected[key_text] = metadata[key_text]
    return selected


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
