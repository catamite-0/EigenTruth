"""Shared parsing helpers for claim feature and metadata flags."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def flag_value_enabled(value: Any) -> bool:
    """Return whether a JSON-like feature/metadata flag is enabled.

    Strings use common boolean forms so external JSON configs do not silently
    fail open or misroute ``"false"`` as true. Ambiguous non-empty values are
    treated as enabled so verification/audit routing fails closed.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
        return True
    if isinstance(value, int | float):
        return value != 0
    return bool(value)


def enabled_feature_names(flags: Mapping[str, Any], feature_names: Sequence[str]) -> tuple[str, ...]:
    """Return configured feature names whose flag value is enabled."""
    return tuple(
        str(name)
        for name in feature_names
        if flag_value_enabled(flags.get(str(name)))
    )


def metadata_path_enabled(metadata: Mapping[str, Any], path: str) -> bool:
    """Return whether a dotted metadata path exists and is enabled."""
    current: Any = metadata
    for part in str(path).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return flag_value_enabled(current)


def normalized_feature_flags(features: Mapping[str, Any]) -> dict[str, bool]:
    """Return feature flags normalized to strict JSON-serializable booleans."""
    return {
        str(key): flag_value_enabled(value)
        for key, value in features.items()
    }
