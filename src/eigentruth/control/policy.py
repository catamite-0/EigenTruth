"""Shared decision types for factuality control workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class RiskLevel(str, Enum):
    """Coarse factuality risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ControlAction(str, Enum):
    """Action requested by a risk controller."""

    ACCEPT = "accept"
    RETRIEVE = "retrieve"
    REWRITE = "rewrite"
    STEER_REGENERATE = "steer_regenerate"
    ABSTAIN = "abstain"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class RiskDecision:
    """A controller decision with enough context for product traces."""

    action: ControlAction
    risk_level: RiskLevel
    confidence: float
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "action": self.action.value,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "diagnostics": _jsonable(self.diagnostics),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_jsonable(item) for item in value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
