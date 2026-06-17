"""Serializable traces for factuality-control workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from eigentruth.control.policy import ControlAction, RiskDecision
from eigentruth.verify.protocols import Claim, VerificationResult


@dataclass(frozen=True)
class TraceEvent:
    """One structured event in a product factuality trace."""

    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "event_type": self.event_type,
            "payload": _to_jsonable(self.payload),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceEvent":
        """Build a trace event from JSON-like data."""
        return cls(
            event_type=str(data["event_type"]),
            payload=dict(data.get("payload", {})),
            created_at=None if data.get("created_at") is None else str(data["created_at"]),
        )


@dataclass(frozen=True)
class ProductTrace:
    """Minimal JSON-ready trace for observe/calibrate/control/verify workflows."""

    request_id: Optional[str] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    claims: Sequence[Claim | Mapping[str, Any]] = ()
    verification_results: Sequence[VerificationResult | Mapping[str, Any]] = ()
    risk_decision: RiskDecision | Mapping[str, Any] | None = None
    actions: Sequence[ControlAction | str | Mapping[str, Any]] = ()
    events: Sequence[TraceEvent | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "request_id": self.request_id,
            "diagnostics": _to_jsonable(self.diagnostics),
            "claims": [_claim_to_dict(claim) for claim in self.claims],
            "verification_results": [
                _verification_result_to_dict(result) for result in self.verification_results
            ],
            "risk_decision": _risk_decision_to_dict(self.risk_decision),
            "actions": [_action_to_dict(action) for action in self.actions],
            "events": [_event_to_dict(event) for event in self.events],
            "metadata": _to_jsonable(self.metadata),
        }


def _claim_to_dict(claim: Claim | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(claim, Claim):
        return {
            "text": claim.text,
            "claim_id": claim.claim_id,
            "span": claim.span,
            "metadata": _to_jsonable(claim.metadata),
        }
    return dict(_to_jsonable(claim))


def _verification_result_to_dict(result: VerificationResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, VerificationResult):
        return {
            "status": result.status.value,
            "confidence": result.confidence,
            "evidence": tuple(result.evidence),
            "explanation": result.explanation,
            "metadata": _to_jsonable(result.metadata),
        }
    return dict(_to_jsonable(result))


def _risk_decision_to_dict(decision: RiskDecision | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    if isinstance(decision, RiskDecision):
        return decision.to_dict()
    return dict(_to_jsonable(decision))


def _action_to_dict(action: ControlAction | str | Mapping[str, Any]) -> Any:
    if isinstance(action, ControlAction):
        return action.value
    return _to_jsonable(action)


def _event_to_dict(event: TraceEvent | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(event, TraceEvent):
        return event.to_dict()
    return dict(_to_jsonable(event))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_jsonable(item) for item in value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if is_dataclass(value) and hasattr(value, "to_dict"):
        return value.to_dict()
    return value
