"""Shared JSON artifact loading cache for benchmark workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

from eigentruth.registry import (
    increment_json_cache_stat,
    json_cache_key,
    json_cache_summary,
    load_json_object,
    new_json_cache_stats,
)


def load_optional_json(
    path: Path,
    *,
    json_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache_stats: MutableMapping[str, int] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Load a JSON object, optionally using a path-signature cache."""
    return load_json_object(path, json_cache=json_cache, json_cache_stats=json_cache_stats)


__all__ = [
    "increment_json_cache_stat",
    "json_cache_key",
    "json_cache_summary",
    "load_optional_json",
    "new_json_cache_stats",
]
