"""Serializable traces for factuality-control workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from eigentruth.control.actions import ActionRequest, ActionResult
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
class RuntimePhaseTiming:
    """Wall-clock timing for one phase in a product control trace."""

    name: str
    seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("runtime phase name must be non-empty")
        seconds = float(self.seconds)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("runtime phase seconds must be finite and non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "seconds": self.seconds,
            "metadata": _to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimePhaseTiming":
        """Build a phase timing from JSON-like data."""
        return cls(
            name=str(data["name"]),
            seconds=float(data["seconds"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RuntimeTrace:
    """Runtime profiling payload for one product factuality request."""

    phases: Sequence[RuntimePhaseTiming | Mapping[str, Any]] = ()
    total_seconds: float | None = None

    def __post_init__(self) -> None:
        phases = tuple(_runtime_phase_to_obj(phase) for phase in self.phases)
        total_seconds = (
            sum(phase.seconds for phase in phases)
            if self.total_seconds is None
            else float(self.total_seconds)
        )
        if not math.isfinite(total_seconds) or total_seconds < 0.0:
            raise ValueError("runtime trace total_seconds must be finite and non-negative")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "total_seconds", total_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "total_seconds": self.total_seconds,
            "phases": tuple(phase.to_dict() for phase in self.phases),
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, Any]:
        """Summarize phase timing counts and totals."""
        phase_seconds: dict[str, float] = {}
        phase_counts: dict[str, int] = {}
        slowest_phase: dict[str, Any] | None = None
        for phase in self.phases:
            phase_seconds[phase.name] = phase_seconds.get(phase.name, 0.0) + phase.seconds
            phase_counts[phase.name] = phase_counts.get(phase.name, 0) + 1
            if slowest_phase is None or phase.seconds > float(slowest_phase["seconds"]):
                slowest_phase = {"name": phase.name, "seconds": phase.seconds}
        accounted_seconds = sum(phase.seconds for phase in self.phases)
        return {
            "total_seconds": self.total_seconds,
            "accounted_seconds": accounted_seconds,
            "unaccounted_seconds": max(float(self.total_seconds) - accounted_seconds, 0.0),
            "measured_phases": len(self.phases),
            "phase_seconds": phase_seconds,
            "phase_counts": phase_counts,
            "slowest_phase": slowest_phase,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeTrace":
        """Build a runtime trace from JSON-like data."""
        return cls(
            phases=tuple(data.get("phases", ())),
            total_seconds=None if data.get("total_seconds") is None else float(data["total_seconds"]),
        )


@dataclass(frozen=True)
class ProductTrace:
    """Minimal JSON-ready trace for observe/calibrate/control/verify workflows."""

    request_id: Optional[str] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    claims: Sequence[Claim | Mapping[str, Any]] = ()
    verification_results: Sequence[VerificationResult | Mapping[str, Any]] = ()
    risk_decision: RiskDecision | Mapping[str, Any] | None = None
    actions: Sequence[ActionRequest | ControlAction | str | Mapping[str, Any]] = ()
    action_results: Sequence[ActionResult | Mapping[str, Any]] = ()
    events: Sequence[TraceEvent | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_trace: RuntimeTrace | Mapping[str, Any] | None = None

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
            "action_results": [_action_result_to_dict(result) for result in self.action_results],
            "events": [_event_to_dict(event) for event in self.events],
            "metadata": _to_jsonable(self.metadata),
            "runtime_trace": _runtime_trace_to_dict(self.runtime_trace),
        }

    def action_execution_summary(self) -> dict[str, Any]:
        """Summarize action execution results for trace/registry metadata."""
        results = [_action_result_to_dict(result) for result in self.action_results]
        counts_by_status: dict[str, int] = {}
        counts_by_action: dict[str, int] = {}
        side_effects = False
        for result in results:
            status = str(result.get("status", "unknown"))
            action = str(result.get("action", "unknown"))
            counts_by_status[status] = counts_by_status.get(status, 0) + 1
            counts_by_action[action] = counts_by_action.get(action, 0) + 1
            metadata = result.get("metadata", {})
            if isinstance(metadata, Mapping) and bool(metadata.get("side_effects", False)):
                side_effects = True
        return {
            "total": len(results),
            "counts_by_status": counts_by_status,
            "counts_by_action": counts_by_action,
            "side_effects": side_effects,
        }

    def verification_route_summary(self) -> dict[str, Any]:
        """Summarize verifier route choices recorded in result metadata."""
        results = [_verification_result_to_dict(result) for result in self.verification_results]
        counts_by_status: dict[str, int] = {}
        counts_by_selected_route: dict[str, int] = {}
        counts_by_selected_verifier: dict[str, int] = {}
        counts_by_matched_route: dict[str, int] = {}
        counts_by_skipped_route: dict[str, int] = {}
        skipped_routes = []
        routed_total = 0
        for result in results:
            status = str(result.get("status", "unknown"))
            counts_by_status[status] = counts_by_status.get(status, 0) + 1
            metadata = result.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            selected_route = metadata.get("selected_route")
            if selected_route is not None:
                routed_total += 1
                route_name = str(selected_route)
                counts_by_selected_route[route_name] = counts_by_selected_route.get(route_name, 0) + 1
            selected_verifier = metadata.get("selected_verifier")
            if selected_verifier is not None:
                verifier_name = str(selected_verifier)
                counts_by_selected_verifier[verifier_name] = counts_by_selected_verifier.get(verifier_name, 0) + 1
            for route in _as_sequence(metadata.get("matched_routes", ())):
                route_name = str(route)
                counts_by_matched_route[route_name] = counts_by_matched_route.get(route_name, 0) + 1
            for skipped in _as_sequence(metadata.get("skipped_routes", ())):
                if not isinstance(skipped, Mapping):
                    continue
                route_name = str(skipped.get("route", "unknown"))
                counts_by_skipped_route[route_name] = counts_by_skipped_route.get(route_name, 0) + 1
                skipped_routes.append(dict(_to_jsonable(skipped)))
        return {
            "total": len(results),
            "routed_total": routed_total,
            "unrouted_total": len(results) - routed_total,
            "counts_by_status": counts_by_status,
            "counts_by_selected_route": counts_by_selected_route,
            "counts_by_selected_verifier": counts_by_selected_verifier,
            "counts_by_matched_route": counts_by_matched_route,
            "counts_by_skipped_route": counts_by_skipped_route,
            "skipped_routes": skipped_routes,
        }

    def runtime_summary(self) -> dict[str, Any]:
        """Return a compact runtime profile summary for trace/registry metadata."""
        payload = _runtime_trace_to_dict(self.runtime_trace)
        if payload is None:
            return {
                "total_seconds": 0.0,
                "accounted_seconds": 0.0,
                "unaccounted_seconds": 0.0,
                "measured_phases": 0,
                "phase_seconds": {},
                "phase_counts": {},
                "slowest_phase": None,
            }
        summary = payload.get("summary", {})
        return dict(summary) if isinstance(summary, Mapping) else RuntimeTrace.from_dict(payload).summary()


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


def _action_to_dict(action: ActionRequest | ControlAction | str | Mapping[str, Any]) -> Any:
    if isinstance(action, ActionRequest):
        return action.to_dict()
    if isinstance(action, ControlAction):
        return action.value
    return _to_jsonable(action)


def _action_result_to_dict(result: ActionResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, ActionResult):
        return result.to_dict()
    return dict(_to_jsonable(result))


def _event_to_dict(event: TraceEvent | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(event, TraceEvent):
        return event.to_dict()
    return dict(_to_jsonable(event))


def _runtime_phase_to_obj(phase: RuntimePhaseTiming | Mapping[str, Any]) -> RuntimePhaseTiming:
    if isinstance(phase, RuntimePhaseTiming):
        return phase
    return RuntimePhaseTiming.from_dict(phase)


def _runtime_trace_to_dict(trace: RuntimeTrace | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if trace is None:
        return None
    if isinstance(trace, RuntimeTrace):
        return trace.to_dict()
    return RuntimeTrace.from_dict(trace).to_dict()


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


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)
