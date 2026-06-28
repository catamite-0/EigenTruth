"""Shared JSON normalization helpers."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any, Mapping, Sequence


def to_jsonable(value: Any) -> Any:
    """Return a deterministic value accepted by strict ``json.dumps``."""
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"encoding": "base64", "data": base64.b64encode(raw).decode("ascii")}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return to_jsonable(value.to_dict())
        except TypeError:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_jsonable(item) for item in value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [to_jsonable(item) for item in value]
        return tuple(sorted(normalized, key=_stable_sort_key))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    return repr(value)


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    """Dump JSON with normalization and ``allow_nan=False``."""
    return json.dumps(to_jsonable(value), allow_nan=False, **kwargs)


def _stable_sort_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
