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
        phase_durations: dict[str, list[float]] = {}
        slowest_phase: dict[str, Any] | None = None
        for phase in self.phases:
            phase_seconds[phase.name] = phase_seconds.get(phase.name, 0.0) + phase.seconds
            phase_counts[phase.name] = phase_counts.get(phase.name, 0) + 1
            phase_durations.setdefault(phase.name, []).append(phase.seconds)
            if slowest_phase is None or phase.seconds > float(slowest_phase["seconds"]):
                slowest_phase = {"name": phase.name, "seconds": phase.seconds}
        accounted_seconds = sum(phase.seconds for phase in self.phases)
        phase_stats = {
            name: _phase_duration_stats(durations)
            for name, durations in phase_durations.items()
        }
        return {
            "total_seconds": self.total_seconds,
            "accounted_seconds": accounted_seconds,
            "unaccounted_seconds": max(float(self.total_seconds) - accounted_seconds, 0.0),
            "measured_phases": len(self.phases),
            "phase_seconds": phase_seconds,
            "phase_counts": phase_counts,
            "phase_stats": phase_stats,
            "phase_p95_seconds": {
                name: stats["p95_seconds"]
                for name, stats in phase_stats.items()
            },
            "phase_p99_seconds": {
                name: stats["p99_seconds"]
                for name, stats in phase_stats.items()
            },
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

    def verification_route_cost_summary(self) -> dict[str, Any]:
        """Summarize verifier route cost metadata from verification results."""
        results = [_verification_result_to_dict(result) for result in self.verification_results]
        records = [_route_cost_record(result) for result in results]
        by_route_records: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_route_records.setdefault(record["route"], []).append(record)
        summary = _route_cost_stats(records)
        summary["by_route"] = {
            route: _route_cost_stats(route_records)
            for route, route_records in by_route_records.items()
        }
        return summary

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

    def cache_summary(self) -> dict[str, Any]:
        """Return aggregate cache hit/miss statistics from trace metadata."""
        metadata = self.metadata if isinstance(self.metadata, Mapping) else {}
        caches = _cache_stats_from_metadata(metadata)
        aggregate = _combine_cache_stats(caches.values())
        return {
            "total_caches": len(caches),
            "aggregate": aggregate,
            "caches": caches,
        }

    def verification_stage_summary(self) -> dict[str, Any]:
        """Summarize staged-verification skip decisions from trace events."""
        events = [_event_to_dict(event) for event in self.events]
        metadata = self.metadata if isinstance(self.metadata, Mapping) else {}
        stage_event = _latest_event_payload(events, "verification_stage_decision")
        initial_event = _latest_event_payload(events, "initial_verification")
        skipped_event = _latest_event_payload(events, "initial_verification_skipped")
        claim_count = _first_non_negative_int(
            initial_event.get("n_claims"),
            len(self.claims),
        )
        result_count = _initial_verification_result_count(initial_event)
        if result_count is None:
            result_count = len(self.verification_results)
        run_verifier = _optional_bool(stage_event.get("run_verifier"))
        skipped = _optional_bool(initial_event.get("skipped"))
        if skipped is None and skipped_event:
            skipped = True
        if skipped is None and run_verifier is not None:
            skipped = not run_verifier
        if skipped is None:
            skipped = False
        saved_claim_count = claim_count if skipped is True and claim_count is not None else 0
        verified_claim_count = (
            result_count
            if result_count is not None
            else (0 if skipped is True else claim_count)
        )
        triggered_claim_ids = tuple(str(item) for item in _as_sequence(stage_event.get("triggered_claim_ids", ())))
        triggered_features = _string_sequence_mapping(stage_event.get("triggered_features"))
        triggered_metadata = _string_sequence_mapping(stage_event.get("triggered_metadata"))
        enabled = (
            bool(stage_event)
            or _truthy_flag(metadata.get("staged_verification_enabled"))
            or isinstance(metadata.get("staged_verification"), Mapping)
        )
        return {
            "enabled": enabled,
            "run_verifier": run_verifier,
            "skipped": bool(skipped),
            "reason": stage_event.get("reason", skipped_event.get("reason")),
            "claim_count": claim_count,
            "verification_result_count": result_count,
            "verified_claim_count": verified_claim_count,
            "saved_claim_count": saved_claim_count,
            "skip_rate": _safe_div(saved_claim_count, claim_count or 0),
            "triggered_claim_count": len(triggered_claim_ids),
            "triggered_claim_ids": triggered_claim_ids,
            "triggered_feature_counts": _count_nested_values(triggered_features),
            "triggered_metadata_counts": _count_nested_values(triggered_metadata),
            "triggered_features": triggered_features,
            "triggered_metadata": triggered_metadata,
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


def _latest_event_payload(events: Sequence[Mapping[str, Any]], event_type: str) -> dict[str, Any]:
    for event in reversed(tuple(events)):
        if str(event.get("event_type")) != event_type:
            continue
        payload = event.get("payload", {})
        return dict(payload) if isinstance(payload, Mapping) else {}
    return {}


def _initial_verification_result_count(initial_event: Mapping[str, Any]) -> int | None:
    results = initial_event.get("results")
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
        return len(results)
    return _non_negative_int(initial_event.get("verification_result_count"))


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _first_non_negative_int(*values: Any) -> int | None:
    for value in values:
        numeric = _non_negative_int(value)
        if numeric is not None:
            return numeric
    return None


def _string_sequence_mapping(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): tuple(str(item) for item in _as_sequence(items))
        for key, items in value.items()
    }


def _count_nested_values(values: Mapping[str, Sequence[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for items in values.values():
        for item in items:
            counts[str(item)] = counts.get(str(item), 0) + 1
    return counts


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


def _route_cost_record(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    selected_route = metadata.get("selected_route")
    route = "unrouted" if selected_route is None else str(selected_route)
    retrieval_hit_count = _retrieval_hit_count(metadata)
    return {
        "route": route,
        "routed": selected_route is not None,
        "total_duration_seconds": _finite_float(metadata.get("total_duration_seconds")),
        "selected_route_duration_seconds": _finite_float(
            metadata.get("selected_route_duration_seconds")
        ),
        "attempted_route_count": _attempted_route_count(metadata),
        "used_retrieval": _truthy_flag(metadata.get("used_retrieval")) or retrieval_hit_count > 0,
        "retrieval_hit_count": retrieval_hit_count,
    }


def _route_cost_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = len(records)
    routed_total = sum(1 for record in records if bool(record.get("routed")))
    if routed_total == 0:
        return _zero_route_cost_stats(selected)
    total_durations = [
        value
        for record in records
        if (value := _finite_float(record.get("total_duration_seconds"))) is not None
    ]
    selected_route_durations = [
        value
        for record in records
        if (value := _finite_float(record.get("selected_route_duration_seconds"))) is not None
    ]
    attempted_route_counts = [
        value
        for record in records
        if (value := _finite_float(record.get("attempted_route_count"))) is not None
    ]
    used_retrieval_count = sum(1 for record in records if bool(record.get("used_retrieval")))
    retrieval_hit_count = sum(_non_negative_int(record.get("retrieval_hit_count")) or 0 for record in records)
    total_duration = float(sum(total_durations)) if total_durations else None
    total_selected_route_duration = (
        float(sum(selected_route_durations))
        if selected_route_durations
        else None
    )
    total_attempted_route_count = (
        float(sum(attempted_route_counts))
        if attempted_route_counts
        else None
    )
    return {
        "total": selected,
        "routed_total": routed_total,
        "unrouted_total": selected - routed_total,
        "duration_observations": len(total_durations),
        "total_duration_seconds": total_duration,
        "mean_duration_seconds": _mean_or_none(total_durations),
        "p95_duration_seconds": _percentile_or_none(total_durations, 95.0),
        "p99_duration_seconds": _percentile_or_none(total_durations, 99.0),
        "max_duration_seconds": max(total_durations) if total_durations else None,
        "selected_route_duration_observations": len(selected_route_durations),
        "total_selected_route_duration_seconds": total_selected_route_duration,
        "mean_selected_route_duration_seconds": _mean_or_none(selected_route_durations),
        "p95_selected_route_duration_seconds": _percentile_or_none(
            selected_route_durations,
            95.0,
        ),
        "p99_selected_route_duration_seconds": _percentile_or_none(
            selected_route_durations,
            99.0,
        ),
        "attempted_route_count_observations": len(attempted_route_counts),
        "total_attempted_route_count": total_attempted_route_count,
        "mean_attempted_route_count": _mean_or_none(attempted_route_counts),
        "used_retrieval_count": used_retrieval_count,
        "retrieval_use_rate": _safe_div(used_retrieval_count, selected),
        "retrieval_hit_count": retrieval_hit_count,
        "mean_retrieval_hits": _safe_div(retrieval_hit_count, selected),
    }


def _zero_route_cost_stats(selected: int) -> dict[str, Any]:
    return {
        "total": selected,
        "routed_total": 0,
        "unrouted_total": selected,
        "duration_observations": 0,
        "total_duration_seconds": 0.0,
        "mean_duration_seconds": 0.0,
        "p95_duration_seconds": 0.0,
        "p99_duration_seconds": 0.0,
        "max_duration_seconds": 0.0,
        "selected_route_duration_observations": 0,
        "total_selected_route_duration_seconds": 0.0,
        "mean_selected_route_duration_seconds": 0.0,
        "p95_selected_route_duration_seconds": 0.0,
        "p99_selected_route_duration_seconds": 0.0,
        "attempted_route_count_observations": 0,
        "total_attempted_route_count": 0.0,
        "mean_attempted_route_count": 0.0,
        "used_retrieval_count": 0,
        "retrieval_use_rate": 0.0,
        "retrieval_hit_count": 0,
        "mean_retrieval_hits": 0.0,
    }


def _attempted_route_count(metadata: Mapping[str, Any]) -> float | None:
    explicit = _finite_float(metadata.get("attempted_route_count"))
    if explicit is not None and explicit >= 0.0:
        return explicit
    skipped_count = sum(1 for item in _as_sequence(metadata.get("skipped_routes", ())) if isinstance(item, Mapping))
    if metadata.get("selected_route") is not None:
        return float(skipped_count + 1)
    matched_count = len(_as_sequence(metadata.get("matched_routes", ())))
    if skipped_count:
        return float(skipped_count)
    if matched_count:
        return float(matched_count)
    return None


def _retrieval_hit_count(metadata: Mapping[str, Any]) -> int:
    explicit = _non_negative_int(metadata.get("retrieval_hit_count"))
    if explicit is not None:
        return explicit
    hits = metadata.get("retrieval_hits", ())
    if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes, bytearray)):
        return len(hits)
    return 0


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _phase_duration_stats(durations: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in durations]
    if not values:
        return {
            "count": 0,
            "total_seconds": None,
            "mean_seconds": None,
            "min_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
            "max_seconds": None,
        }
    total = float(sum(values))
    return {
        "count": len(values),
        "total_seconds": total,
        "mean_seconds": total / len(values),
        "min_seconds": min(values),
        "p95_seconds": _percentile_or_none(values, 95.0),
        "p99_seconds": _percentile_or_none(values, 99.0),
        "max_seconds": max(values),
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / len(values)


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _percentile_or_none(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be between 0 and 100.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * (rank - lower_index)


def _cache_stats_from_metadata(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    payload = metadata.get("cache", metadata.get("caches", {}))
    if not isinstance(payload, Mapping):
        return {}
    if _looks_like_cache_stats(payload):
        return {"default": _normalize_cache_stats(payload)}
    caches = {}
    for name, stats in payload.items():
        if not isinstance(stats, Mapping):
            continue
        caches[str(name)] = _normalize_cache_stats(stats)
    return caches


def _looks_like_cache_stats(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("hits", "misses", "requests", "hit_rate", "size"))


def _normalize_cache_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    hits = _non_negative_int(stats.get("hits"))
    misses = _non_negative_int(stats.get("misses"))
    requests = _non_negative_int(stats.get("requests"))
    if requests is None and hits is not None and misses is not None:
        requests = hits + misses
    hit_rate = _finite_float(stats.get("hit_rate"))
    if hit_rate is None and requests is not None and requests > 0 and hits is not None:
        hit_rate = hits / requests
    return {
        "size": _non_negative_int(stats.get("size")),
        "hits": hits,
        "misses": misses,
        "requests": requests,
        "hit_rate": hit_rate,
    }


def _combine_cache_stats(stats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_size = 0
    total_hits = 0
    total_misses = 0
    saw_size = False
    saw_counts = False
    for item in stats:
        size = _non_negative_int(item.get("size"))
        hits = _non_negative_int(item.get("hits"))
        misses = _non_negative_int(item.get("misses"))
        if size is not None:
            saw_size = True
            total_size += size
        if hits is not None and misses is not None:
            saw_counts = True
            total_hits += hits
            total_misses += misses
    requests = total_hits + total_misses if saw_counts else None
    return {
        "size": total_size if saw_size else None,
        "hits": total_hits if saw_counts else None,
        "misses": total_misses if saw_counts else None,
        "requests": requests,
        "hit_rate": None if not requests else total_hits / requests,
    }


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


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _non_negative_int(value: Any) -> int | None:
    numeric = _finite_float(value)
    if numeric is None or numeric < 0:
        return None
    return int(numeric)


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)
