"""Shared helpers for benchmark workflow configuration parsing."""

from __future__ import annotations

from typing import Any


def strict_bool(value: Any, *, name: str) -> bool:
    """Parse booleans without treating arbitrary non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or boolean string.")
