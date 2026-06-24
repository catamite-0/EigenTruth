"""Shared JSON artifact loading cache for benchmark workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping


def load_optional_json(
    path: Path,
    *,
    json_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache_stats: MutableMapping[str, int] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Load a JSON object, optionally using a path-signature cache."""
    increment_json_cache_stat(json_cache_stats, "requests")
    cache_key = None if json_cache is None else json_cache_key(path)
    if cache_key is not None:
        cached = json_cache.get(cache_key)
        if cached is not None:
            error = cached.get("error")
            increment_json_cache_stat(json_cache_stats, "hits")
            if error is not None:
                increment_json_cache_stat(json_cache_stats, "errors")
            return _mapping(cached.get("payload")), error
    increment_json_cache_stat(json_cache_stats, "misses")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error = str(exc)
        increment_json_cache_stat(json_cache_stats, "errors")
        if cache_key is not None:
            json_cache[cache_key] = {"payload": {}, "error": error}
        return {}, error
    if not isinstance(payload, dict):
        error = f"{path} did not contain a JSON object"
        increment_json_cache_stat(json_cache_stats, "errors")
        if cache_key is not None:
            json_cache[cache_key] = {"payload": {}, "error": error}
        return {}, error
    if cache_key is not None:
        json_cache[cache_key] = {"payload": dict(payload), "error": None}
    return payload, None


def json_cache_key(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path.absolute()
    return f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}:{getattr(stat, 'st_ino', 0)}"


def new_json_cache_stats() -> dict[str, int]:
    return {
        "requests": 0,
        "hits": 0,
        "misses": 0,
        "errors": 0,
    }


def increment_json_cache_stat(stats: MutableMapping[str, int] | None, key: str) -> None:
    if stats is None:
        return
    stats[key] = int(stats.get(key, 0)) + 1


def json_cache_summary(
    json_cache: Mapping[str, Any],
    stats: Mapping[str, int],
) -> dict[str, Any]:
    requests = int(stats.get("requests", 0))
    hits = int(stats.get("hits", 0))
    return {
        "requests": requests,
        "hits": hits,
        "misses": int(stats.get("misses", 0)),
        "errors": int(stats.get("errors", 0)),
        "entries": len(json_cache),
        "hit_rate": 0.0 if requests <= 0 else float(hits) / float(requests),
    }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
