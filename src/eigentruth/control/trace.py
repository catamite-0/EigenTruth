"""Serializable traces for factuality-control workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from eigentruth.control.action_audit import audit_action_requests
from eigentruth.control.actions import ActionRequest, ActionResult
from eigentruth.control.finalization import FinalAnswer
from eigentruth.control.metacognition import audit_metacognitive_alignment
from eigentruth.control.policy import ControlAction, RiskDecision
from eigentruth.control.provenance import audit_evidence_graph_consistency, audit_trace_provenance
from eigentruth.control.receipt_audit import audit_receipt_claim_support
from eigentruth.control.receipts import action_receipt_summary_from_results
from eigentruth.json_utils import to_jsonable
from eigentruth.verify.citations import extract_citation_references
from eigentruth.verify.localization import localize_claim_risk_spans
from eigentruth.verify.planning import ClaimVerificationPlan, estimate_verification_plan_cost
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
    verification_plan: ClaimVerificationPlan | Mapping[str, Any] | None = None
    verification_results: Sequence[VerificationResult | Mapping[str, Any]] = ()
    risk_decision: RiskDecision | Mapping[str, Any] | None = None
    actions: Sequence[ActionRequest | ControlAction | str | Mapping[str, Any]] = ()
    action_results: Sequence[ActionResult | Mapping[str, Any]] = ()
    events: Sequence[TraceEvent | Mapping[str, Any]] = ()
    final_answer: FinalAnswer | Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_trace: RuntimeTrace | Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "request_id": self.request_id,
            "diagnostics": _to_jsonable(self.diagnostics),
            "claims": [_claim_to_dict(claim) for claim in self.claims],
            "verification_plan": _verification_plan_to_dict(self.verification_plan),
            "verification_results": [
                _verification_result_to_dict(result) for result in self.verification_results
            ],
            "risk_decision": _risk_decision_to_dict(self.risk_decision),
            "actions": [_action_to_dict(action) for action in self.actions],
            "action_results": [_action_result_to_dict(result) for result in self.action_results],
            "events": [_event_to_dict(event) for event in self.events],
            "final_answer": _final_answer_to_dict(self.final_answer),
            "metadata": _to_jsonable(self.metadata),
            "runtime_trace": _runtime_trace_to_dict(self.runtime_trace),
        }

    def to_bounded_dict(
        self,
        *,
        max_diagnostics: int = 64,
        max_claims: int = 20,
        max_verification_results: int = 20,
        max_actions: int = 20,
        max_action_results: int = 20,
        max_events: int = 20,
        max_nested_items: int = 16,
        max_string_length: int = 500,
        include_runtime_trace: bool = False,
        metadata_keys: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return a bounded ProductTrace payload for product telemetry.

        The full ``to_dict()`` schema is preserved for offline reproduction. This
        method is for online traces where large evidence, action output, event,
        and metadata payloads should be summarized while keeping routing and
        budget diagnostics available.
        """
        prepared = _prepare_trace_payload(self)
        diagnostics = _bounded_mapping_payload(
            prepared.diagnostics,
            max_items=max_diagnostics,
            max_nested_items=max_nested_items,
            max_string_length=max_string_length,
        )
        claims = prepared.claims
        verification_results = [
            _bounded_verification_result(
                result,
                max_nested_items=max_nested_items,
                max_string_length=max_string_length,
            )
            for result in prepared.verification_results
        ]
        actions = [
            _bounded_jsonable(
                action,
                max_depth=4,
                max_items=max_nested_items,
                max_string_length=max_string_length,
            )
            for action in prepared.actions
        ]
        action_results = [
            _bounded_action_result(
                result,
                max_nested_items=max_nested_items,
                max_string_length=max_string_length,
            )
            for result in prepared.action_results
        ]
        events = [
            _bounded_event(
                event,
                max_nested_items=max_nested_items,
                max_string_length=max_string_length,
            )
            for event in prepared.events
        ]
        summaries = {
            "action_execution": _action_execution_summary_from_payload(
                prepared.actions,
                prepared.action_results,
            ),
            "action_receipts": action_receipt_summary_from_results(prepared.action_results),
            "evidence_quality": _evidence_quality_summary_from_action_results(
                prepared.action_results,
            ),
            "world_model_action_gate": _world_model_action_gate_summary_from_action_results(
                prepared.action_results,
            ),
            "metacognition": _metacognition_summary_from_payload(
                diagnostics=prepared.diagnostics,
                verification_results=prepared.verification_results,
                risk_decision=prepared.risk_decision,
                final_answer=prepared.final_answer,
            ),
            "receipt_claim_support": _receipt_claim_support_summary_from_payload(
                claims=prepared.claims,
                action_results=prepared.action_results,
                final_answer=prepared.final_answer,
            ),
            "action_audit": _action_audit_summary_from_payload(
                actions=prepared.actions,
                risk_decision=prepared.risk_decision,
                verification_plan=prepared.verification_plan,
            ),
            "trajectory_audit": _trajectory_audit_summary_from_payload(
                request_id=self.request_id,
                claims=prepared.claims,
                verification_plan=prepared.verification_plan,
                verification_results=prepared.verification_results,
                risk_decision=prepared.risk_decision,
                actions=prepared.actions,
                action_results=prepared.action_results,
                final_answer=prepared.final_answer,
            ),
            "provenance": _provenance_summary_from_payload(
                request_id=self.request_id,
                claims=prepared.claims,
                verification_results=prepared.verification_results,
                risk_decision=prepared.risk_decision,
                actions=prepared.actions,
                action_results=prepared.action_results,
                final_answer=prepared.final_answer,
            ),
            "evidence_graph_consistency": _evidence_graph_consistency_summary_from_payload(
                request_id=self.request_id,
                claims=prepared.claims,
                verification_results=prepared.verification_results,
                action_results=prepared.action_results,
            ),
            "verification_route": _verification_route_summary_from_results(
                prepared.verification_results,
            ),
            "verification_route_cost": _verification_route_cost_summary_from_results(
                prepared.verification_results,
            ),
            "world_model": _world_model_summary_from_results(
                prepared.verification_results,
            ),
            "context_sensitivity": _context_sensitivity_summary_from_results(
                prepared.verification_results,
            ),
            "evidence_alignment": _evidence_alignment_summary_from_results(
                prepared.verification_results,
            ),
            "counterfactual_robustness": _counterfactual_robustness_summary_from_results(
                prepared.verification_results,
            ),
            "citation_integrity": _citation_integrity_summary(
                prepared.claims,
                prepared.verification_results,
            ),
            "runtime": _runtime_summary_from_payload(prepared.runtime_trace),
            "cache": _cache_summary_from_metadata(prepared.metadata),
            "pre_generation_risk": _pre_generation_risk_summary_from_metadata(
                prepared.metadata,
            ),
            "verification_stage": _verification_stage_summary_from_payload(
                events=prepared.events,
                metadata=prepared.metadata,
                claim_count=len(prepared.claims),
                verification_result_count=len(prepared.verification_results),
            ),
            "verification_plan": _verification_plan_summary(prepared.verification_plan),
            "claim_risk_localization": _claim_risk_localization_summary(
                prepared.claims,
                prepared.verification_results,
                prepared.verification_plan,
            ),
            "triple_coverage": _triple_coverage_summary(
                prepared.claims,
                prepared.verification_results,
            ),
            "final_answer": _final_answer_summary(prepared.final_answer),
        }
        payload = {
            "schema_version": 1,
            "trace_format": "bounded_product_trace",
            "request_id": self.request_id,
            "diagnostics": diagnostics["items"],
            "risk_decision": prepared.risk_decision,
            "summaries": _bounded_summaries(
                summaries,
                max_nested_items=max_nested_items,
                max_string_length=max_string_length,
            ),
            "claims": _bounded_sequence(claims, max_claims),
            "verification_results": _bounded_sequence(
                verification_results,
                max_verification_results,
            ),
            "actions": _bounded_sequence(actions, max_actions),
            "action_results": _bounded_sequence(action_results, max_action_results),
            "events": _bounded_sequence(events, max_events),
            "final_answer": _bounded_final_answer(
                prepared.final_answer,
                max_nested_items=max_nested_items,
                max_string_length=max_string_length,
            ),
            "metadata": _bounded_metadata(
                self.metadata,
                metadata_keys=metadata_keys,
                max_nested_items=max_nested_items,
                max_string_length=max_string_length,
            ),
            "runtime_trace": prepared.runtime_trace if include_runtime_trace else None,
            "truncation": {
                "diagnostics": diagnostics["summary"],
                "claims": _truncation_summary(len(claims), max_claims),
                "verification_results": _truncation_summary(
                    len(verification_results),
                    max_verification_results,
                ),
                "actions": _truncation_summary(len(actions), max_actions),
                "action_results": _truncation_summary(len(action_results), max_action_results),
                "events": _truncation_summary(len(events), max_events),
                "runtime_trace_included": include_runtime_trace,
            },
        }
        return payload

    def action_execution_summary(self) -> dict[str, Any]:
        """Summarize action execution results for trace/registry metadata."""
        return _action_execution_summary_from_payload(
            tuple(_action_to_dict(action) for action in self.actions),
            tuple(_action_result_to_dict(result) for result in self.action_results),
        )

    def action_receipt_summary(self) -> dict[str, Any]:
        """Summarize receipt coverage for action results."""
        return action_receipt_summary_from_results(
            tuple(_action_result_to_dict(result) for result in self.action_results)
        )

    def evidence_quality_summary(self) -> dict[str, Any]:
        """Summarize retrieval evidence freshness/provenance checks."""
        return _evidence_quality_summary_from_action_results(
            tuple(_action_result_to_dict(result) for result in self.action_results)
        )

    def world_model_action_gate_summary(self) -> dict[str, Any]:
        """Summarize pre-action world-model gates attached to action results."""
        return _world_model_action_gate_summary_from_action_results(
            tuple(_action_result_to_dict(result) for result in self.action_results)
        )

    def metacognition_summary(self) -> dict[str, Any]:
        """Summarize alignment between expressed uncertainty and trace risk."""
        return _metacognition_summary_from_payload(
            diagnostics=_to_jsonable(self.diagnostics),
            verification_results=tuple(
                _verification_result_to_dict(result) for result in self.verification_results
            ),
            risk_decision=_risk_decision_to_dict(self.risk_decision),
            final_answer=_final_answer_to_dict(self.final_answer),
        )

    def receipt_claim_support_summary(self) -> dict[str, Any]:
        """Summarize explicit claim/final-answer references to action receipts."""
        return audit_receipt_claim_support(self).summary()

    def action_audit_summary(self) -> dict[str, Any]:
        """Summarize planned action/tool-selection audit results."""
        return _action_audit_summary_from_payload(
            actions=tuple(_action_to_dict(action) for action in self.actions),
            risk_decision=_risk_decision_to_dict(self.risk_decision),
            verification_plan=_verification_plan_to_dict(self.verification_plan),
        )

    def trajectory_audit_summary(self) -> dict[str, Any]:
        """Summarize trace-level hallucination taxonomy audit results."""
        from eigentruth.control.trajectory_audit import audit_product_trace_trajectory

        return audit_product_trace_trajectory(self).summary()

    def provenance_summary(self) -> dict[str, Any]:
        """Summarize trace evidence/execution provenance graph coverage."""
        return audit_trace_provenance(self).summary()

    def evidence_graph_consistency_summary(self) -> dict[str, Any]:
        """Summarize lightweight content consistency for supported-claim evidence."""
        return audit_evidence_graph_consistency(self).summary()

    def verification_route_summary(self) -> dict[str, Any]:
        """Summarize verifier route choices recorded in result metadata."""
        return _verification_route_summary_from_results(
            tuple(_verification_result_to_dict(result) for result in self.verification_results)
        )

    def verification_route_cost_summary(self) -> dict[str, Any]:
        """Summarize verifier route cost metadata from verification results."""
        return _verification_route_cost_summary_from_results(
            tuple(_verification_result_to_dict(result) for result in self.verification_results)
        )

    def world_model_summary(self) -> dict[str, Any]:
        """Summarize world-model evidence, conflicts, and traceability gaps."""
        return _world_model_summary_from_results(
            tuple(_verification_result_to_dict(result) for result in self.verification_results)
        )

    def context_sensitivity_summary(self) -> dict[str, Any]:
        """Summarize evidence-context sensitivity signals recorded on verifier results."""
        return _context_sensitivity_summary_from_results(
            tuple(_verification_result_to_dict(result) for result in self.verification_results)
        )

    def evidence_alignment_summary(self) -> dict[str, Any]:
        """Summarize claim/evidence alignment reports recorded on verifier results."""
        return _evidence_alignment_summary_from_results(
            tuple(_verification_result_to_dict(result) for result in self.verification_results)
        )

    def counterfactual_robustness_summary(self) -> dict[str, Any]:
        """Summarize counterfactual perturbation audit signals on verifier results."""
        return _counterfactual_robustness_summary_from_results(
            tuple(_verification_result_to_dict(result) for result in self.verification_results)
        )

    def citation_integrity_summary(self) -> dict[str, Any]:
        """Summarize citation-reference coverage and catalog-audit outcomes."""
        return _citation_integrity_summary(
            tuple(_claim_to_dict(claim) for claim in self.claims),
            tuple(_verification_result_to_dict(result) for result in self.verification_results),
        )

    def runtime_summary(self) -> dict[str, Any]:
        """Return a compact runtime profile summary for trace/registry metadata."""
        return _runtime_summary_from_payload(_runtime_trace_to_dict(self.runtime_trace))

    def cache_summary(self) -> dict[str, Any]:
        """Return aggregate cache hit/miss statistics from trace metadata."""
        metadata = self.metadata if isinstance(self.metadata, Mapping) else {}
        return _cache_summary_from_metadata(metadata)

    def pre_generation_risk_summary(self) -> dict[str, Any]:
        """Summarize pre-generation routing and learned-risk metadata."""
        metadata = self.metadata if isinstance(self.metadata, Mapping) else {}
        return _pre_generation_risk_summary_from_metadata(metadata)

    def verification_stage_summary(self) -> dict[str, Any]:
        """Summarize staged-verification skip decisions from trace events."""
        metadata = self.metadata if isinstance(self.metadata, Mapping) else {}
        return _verification_stage_summary_from_payload(
            events=tuple(_event_to_dict(event) for event in self.events),
            metadata=metadata,
            claim_count=len(self.claims),
            verification_result_count=len(self.verification_results),
        )

    def final_answer_summary(self) -> dict[str, Any]:
        """Summarize final answer status for trace/registry metadata."""
        return _final_answer_summary(_final_answer_to_dict(self.final_answer))

    def triple_coverage_summary(self) -> dict[str, Any]:
        """Summarize claim-triple and slot-audit coverage in this trace."""
        return _triple_coverage_summary(
            tuple(_claim_to_dict(claim) for claim in self.claims),
            tuple(_verification_result_to_dict(result) for result in self.verification_results),
        )

    def claim_risk_localization_summary(self) -> dict[str, Any]:
        """Summarize localized claim risk from claims and verifier outputs."""
        return _claim_risk_localization_summary(
            tuple(_claim_to_dict(claim) for claim in self.claims),
            tuple(_verification_result_to_dict(result) for result in self.verification_results),
            _verification_plan_to_dict(self.verification_plan),
        )


@dataclass(frozen=True)
class _PreparedTracePayload:
    diagnostics: Any
    claims: tuple[dict[str, Any], ...]
    verification_plan: dict[str, Any] | None
    verification_results: tuple[dict[str, Any], ...]
    risk_decision: dict[str, Any] | None
    actions: tuple[Any, ...]
    action_results: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    final_answer: dict[str, Any] | None
    metadata: Any
    runtime_trace: dict[str, Any] | None


def _prepare_trace_payload(trace: ProductTrace) -> _PreparedTracePayload:
    return _PreparedTracePayload(
        diagnostics=_to_jsonable(trace.diagnostics),
        claims=tuple(_claim_to_dict(claim) for claim in trace.claims),
        verification_plan=_verification_plan_to_dict(trace.verification_plan),
        verification_results=tuple(
            _verification_result_to_dict(result)
            for result in trace.verification_results
        ),
        risk_decision=_risk_decision_to_dict(trace.risk_decision),
        actions=tuple(_action_to_dict(action) for action in trace.actions),
        action_results=tuple(_action_result_to_dict(result) for result in trace.action_results),
        events=tuple(_event_to_dict(event) for event in trace.events),
        final_answer=_final_answer_to_dict(trace.final_answer),
        metadata=_to_jsonable(trace.metadata),
        runtime_trace=_runtime_trace_to_dict(trace.runtime_trace),
    )


def _action_execution_summary_from_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
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


def _action_execution_summary_from_payload(
    actions: Sequence[Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = _action_execution_summary_from_results(results)
    alignment = _action_result_alignment(actions, results)
    summary.update({
        "planned_action_count": len(actions),
        "result_count": len(results),
        "planned_counts_by_action": _planned_action_counts(actions),
        "alignment": alignment,
        "alignment_passed": alignment["passed"],
        "missing_result_count": alignment["missing_result_count"],
        "unexpected_result_count": alignment["unexpected_result_count"],
        "request_id_mismatch_count": alignment["request_id_mismatch_count"],
    })
    return summary


def _evidence_quality_summary_from_action_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result_count = len(results)
    checked_result_count = 0
    failed_result_count = 0
    document_count = 0
    applied_count = 0
    passed_count = 0
    failed_count = 0
    reason_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for result in results:
        summaries = _evidence_quality_summaries_from_action_result(result)
        if not summaries:
            continue
        checked_result_count += 1
        result_failed = False
        for summary in summaries:
            summary_document_count = _non_negative_int(summary.get("document_count")) or 0
            summary_applied_count = _non_negative_int(summary.get("applied_count")) or 0
            summary_passed_count = _non_negative_int(summary.get("passed_count")) or 0
            summary_failed_count = _non_negative_int(summary.get("failed_count")) or 0
            document_count += summary_document_count
            applied_count += summary_applied_count
            passed_count += summary_passed_count
            failed_count += summary_failed_count
            _merge_counts(reason_counts, _mapping(summary.get("reason_counts")))
            status = _evidence_quality_status(
                summary,
                document_count=summary_document_count,
                applied_count=summary_applied_count,
                failed_count=summary_failed_count,
            )
            _increment_count(status_counts, status)
            if summary_failed_count > 0 or status == "fail":
                result_failed = True
        if result_failed:
            failed_result_count += 1
    available = checked_result_count > 0
    return {
        "available": available,
        "status": _aggregate_evidence_quality_status(
            available=available,
            document_count=document_count,
            applied_count=applied_count,
            failed_count=failed_count,
        ),
        "result_count": result_count,
        "checked_result_count": checked_result_count,
        "coverage_rate": _safe_div(checked_result_count, result_count) or 0.0,
        "document_count": document_count,
        "applied_count": applied_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "failed_result_count": failed_result_count,
        "pass_rate": 1.0 if applied_count == 0 else passed_count / applied_count,
        "failure_rate": 0.0 if applied_count == 0 else failed_count / applied_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_source_count": reason_counts.get("missing_source", 0),
        "untrusted_source_count": reason_counts.get("untrusted_source", 0),
        "stale_evidence_count": reason_counts.get("stale_evidence", 0),
        "missing_timestamp_count": reason_counts.get("missing_timestamp", 0),
    }


def _evidence_quality_summaries_from_action_result(
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    output = _mapping(result.get("output"))
    top_level = _mapping(output.get("evidence_quality"))
    if top_level:
        return (top_level,)
    query_summaries: list[dict[str, Any]] = []
    for item in _as_sequence(output.get("hits_by_query")):
        quality = _mapping(_mapping(item).get("evidence_quality"))
        if quality:
            query_summaries.append(quality)
    return tuple(query_summaries)


def _world_model_action_gate_summary_from_action_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result_count = len(results)
    checked_result_count = 0
    passed_count = 0
    blocked_count = 0
    side_effect_block_violation_count = 0
    prediction_confidences: list[float] = []
    counts_by_status: dict[str, int] = {}
    counts_by_decision_rule: dict[str, int] = {}
    counts_by_code: dict[str, int] = {}
    counts_by_action: dict[str, int] = {}

    for result in results:
        gate = _world_model_gate_summary_from_action_result(result)
        if not gate:
            continue
        checked_result_count += 1
        _increment_count(counts_by_action, result.get("action"))
        status = gate.get("status")
        _increment_count(counts_by_status, status)
        _increment_count(counts_by_decision_rule, gate.get("decision_rule"))
        _merge_counts(counts_by_code, _mapping(gate.get("counts_by_code")))
        confidence = _finite_float(gate.get("prediction_confidence"))
        if confidence is not None:
            prediction_confidences.append(confidence)
        blocked = _optional_bool(gate.get("blocked"))
        if blocked is None:
            blocked = str(status) in {"blocked", "error"}
        passed = _optional_bool(gate.get("passed"))
        if passed is None:
            passed = str(status) == "passed"
        if blocked:
            blocked_count += 1
            metadata = _mapping(result.get("metadata"))
            if bool(metadata.get("side_effects", False)) or bool(metadata.get("possible_side_effects", False)):
                side_effect_block_violation_count += 1
        if passed:
            passed_count += 1

    return {
        "available": checked_result_count > 0,
        "result_count": result_count,
        "checked_result_count": checked_result_count,
        "coverage_rate": _safe_div(checked_result_count, result_count),
        "passed_count": passed_count,
        "blocked_count": blocked_count,
        "pass_rate": _safe_div(passed_count, checked_result_count),
        "blocked_rate": _safe_div(blocked_count, checked_result_count),
        "side_effect_block_violation_count": side_effect_block_violation_count,
        "prediction_confidence_mean": _mean_or_none(prediction_confidences),
        "prediction_confidence_min": min(prediction_confidences) if prediction_confidences else None,
        "low_prediction_confidence_count": counts_by_code.get("low_prediction_confidence", 0),
        "low_agreement_count": counts_by_code.get("low_agreement", 0),
        "no_rule_matched_count": counts_by_code.get("no_rule_matched", 0),
        "postcondition_refuted_count": counts_by_code.get("postcondition_refuted", 0),
        "postcondition_insufficient_evidence_count": counts_by_code.get(
            "postcondition_insufficient_evidence",
            0,
        ),
        "postcondition_error_count": counts_by_code.get("postcondition_error", 0),
        "counts_by_status": dict(sorted(counts_by_status.items())),
        "counts_by_decision_rule": dict(sorted(counts_by_decision_rule.items())),
        "counts_by_code": dict(sorted(counts_by_code.items())),
        "counts_by_action": dict(sorted(counts_by_action.items())),
    }


def _world_model_gate_summary_from_action_result(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata_gate = _mapping(_mapping(result.get("metadata")).get("world_model_gate"))
    output_gate = _mapping(_mapping(result.get("output")).get("world_model_gate"))
    output_summary = _mapping(output_gate.get("summary"))
    if output_summary:
        merged = dict(output_summary)
        merged.update(metadata_gate)
        return merged
    return metadata_gate


def _metacognition_summary_from_payload(
    *,
    diagnostics: Mapping[str, Any] | Any,
    verification_results: Sequence[Mapping[str, Any]],
    risk_decision: Mapping[str, Any] | None,
    final_answer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return audit_metacognitive_alignment(
        diagnostics=_mapping(diagnostics),
        verification_results=tuple(_mapping(result) for result in verification_results),
        risk_decision=_mapping(risk_decision),
        final_answer=_mapping(final_answer),
    ).summary()


def _evidence_quality_status(
    summary: Mapping[str, Any],
    *,
    document_count: int,
    applied_count: int,
    failed_count: int,
) -> str:
    raw_status = summary.get("status")
    if raw_status is not None:
        text = str(raw_status).strip()
        if text:
            return text
    return _aggregate_evidence_quality_status(
        available=True,
        document_count=document_count,
        applied_count=applied_count,
        failed_count=failed_count,
    )


def _aggregate_evidence_quality_status(
    *,
    available: bool,
    document_count: int,
    applied_count: int,
    failed_count: int,
) -> str:
    if not available:
        return "missing"
    if document_count == 0:
        return "empty"
    if applied_count == 0:
        return "not_applied"
    if failed_count > 0:
        return "fail"
    return "pass"


def _action_result_alignment(
    actions: Sequence[Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    planned_counts = _planned_action_counts(actions)
    result_counts = _result_action_counts(results)
    has_planned_actions = bool(planned_counts)
    missing_by_action = {
        action: planned_count - result_counts.get(action, 0)
        for action, planned_count in planned_counts.items()
        if planned_count > result_counts.get(action, 0)
    }
    unexpected_by_action = (
        {
            action: result_count - planned_counts.get(action, 0)
            for action, result_count in result_counts.items()
            if result_count > planned_counts.get(action, 0)
        }
        if has_planned_actions
        else {}
    )
    planned_request_ids = tuple(_action_request_ids(actions))
    result_request_ids = tuple(_result_request_ids(results))
    planned_request_id_set = set(planned_request_ids)
    result_request_id_set = set(result_request_ids)
    missing_request_ids = tuple(
        request_id
        for request_id in planned_request_ids
        if request_id not in result_request_id_set
    )
    unexpected_request_ids = (
        tuple(
            request_id
            for request_id in result_request_ids
            if request_id not in planned_request_id_set
        )
        if planned_request_ids
        else ()
    )
    missing_result_count = sum(missing_by_action.values())
    unexpected_result_count = sum(unexpected_by_action.values())
    request_id_mismatch_count = len(missing_request_ids) + len(unexpected_request_ids)
    return {
        "available": has_planned_actions,
        "passed": not has_planned_actions or (
            missing_result_count == 0
            and unexpected_result_count == 0
            and request_id_mismatch_count == 0
        ),
        "missing_result_count": missing_result_count,
        "unexpected_result_count": unexpected_result_count,
        "request_id_mismatch_count": request_id_mismatch_count,
        "missing_results_by_action": missing_by_action,
        "unexpected_results_by_action": unexpected_by_action,
        "planned_request_id_count": len(planned_request_ids),
        "result_request_id_count": len(result_request_ids),
        "missing_request_ids": missing_request_ids[:8],
        "unexpected_request_ids": unexpected_request_ids[:8],
    }


def _planned_action_counts(actions: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        action_name = _action_name_from_payload(action)
        if action_name is None:
            continue
        counts[action_name] = counts.get(action_name, 0) + 1
    return counts


def _result_action_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        action_name = _action_name_from_payload(result)
        if action_name is None:
            continue
        counts[action_name] = counts.get(action_name, 0) + 1
    return counts


def _action_request_ids(actions: Sequence[Any]) -> tuple[str, ...]:
    request_ids = []
    for action in actions:
        request_id = _request_id_from_payload(action)
        if request_id is not None:
            request_ids.append(request_id)
    return tuple(dict.fromkeys(request_ids))


def _result_request_ids(results: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    request_ids = []
    for result in results:
        request_id = _request_id_from_payload(result)
        if request_id is not None:
            request_ids.append(request_id)
    return tuple(dict.fromkeys(request_ids))


def _action_name_from_payload(payload: Any) -> str | None:
    if isinstance(payload, ControlAction):
        return payload.value
    if isinstance(payload, str):
        action_name = payload.strip()
        return action_name or None
    if isinstance(payload, Mapping):
        raw_action = payload.get("action")
        if raw_action is None:
            return None
        if isinstance(raw_action, ControlAction):
            return raw_action.value
        action_name = str(raw_action).strip()
        return action_name or None
    return None


def _request_id_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    raw_request_id = payload.get("request_id")
    if raw_request_id is None:
        return None
    request_id = str(raw_request_id).strip()
    return request_id or None


def _action_audit_summary_from_payload(
    *,
    actions: Sequence[Any],
    risk_decision: Mapping[str, Any] | None,
    verification_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = audit_action_requests(
        actions,
        decision=risk_decision,
        verification_plan=verification_plan,
    )
    return report.summary()


def _receipt_claim_support_summary_from_payload(
    *,
    claims: Sequence[Mapping[str, Any]],
    action_results: Sequence[Mapping[str, Any]],
    final_answer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = audit_receipt_claim_support({
        "claims": tuple(claims),
        "action_results": tuple(action_results),
        "final_answer": final_answer,
    })
    return report.summary()


def _trajectory_audit_summary_from_payload(
    *,
    request_id: str | None,
    claims: Sequence[Mapping[str, Any]],
    verification_plan: Mapping[str, Any] | None,
    verification_results: Sequence[Mapping[str, Any]],
    risk_decision: Mapping[str, Any] | None,
    actions: Sequence[Any],
    action_results: Sequence[Mapping[str, Any]],
    final_answer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from eigentruth.control.trajectory_audit import audit_product_trace_trajectory

    return audit_product_trace_trajectory({
        "request_id": request_id,
        "claims": tuple(claims),
        "verification_plan": verification_plan,
        "verification_results": tuple(verification_results),
        "risk_decision": risk_decision,
        "actions": tuple(actions),
        "action_results": tuple(action_results),
        "final_answer": final_answer,
    }).summary()


def _provenance_summary_from_payload(
    *,
    request_id: str | None,
    claims: Sequence[Mapping[str, Any]],
    verification_results: Sequence[Mapping[str, Any]],
    risk_decision: Mapping[str, Any] | None,
    actions: Sequence[Any],
    action_results: Sequence[Mapping[str, Any]],
    final_answer: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return audit_trace_provenance({
        "request_id": request_id,
        "claims": tuple(claims),
        "verification_results": tuple(verification_results),
        "risk_decision": risk_decision,
        "actions": tuple(actions),
        "action_results": tuple(action_results),
        "final_answer": final_answer,
    }).summary()


def _evidence_graph_consistency_summary_from_payload(
    *,
    request_id: str | None,
    claims: Sequence[Mapping[str, Any]],
    verification_results: Sequence[Mapping[str, Any]],
    action_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return audit_evidence_graph_consistency({
        "request_id": request_id,
        "claims": tuple(claims),
        "verification_results": tuple(verification_results),
        "action_results": tuple(action_results),
    }).summary()


def _verification_route_summary_from_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
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


def _verification_route_cost_summary_from_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
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


def _world_model_summary_from_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts_by_status: dict[str, int] = {}
    counts_by_adapter: dict[str, int] = {}
    counts_by_reference_id: dict[str, int] = {}
    counts_by_decision_rule: dict[str, int] = {}
    conflict_paths: dict[str, int] = {}
    prediction_confidences: list[float] = []
    agreement_rates: list[float] = []
    world_model_total = 0
    conflict_count = 0
    low_agreement_count = 0
    no_rule_matched_count = 0
    trace_gap_count = 0

    for result in results:
        metadata = _mapping(result.get("metadata"))
        if not _is_world_model_result(metadata):
            continue
        world_model_total += 1
        _increment_count(counts_by_status, result.get("status", "unknown"))
        prediction_metadata = _world_model_prediction_metadata(metadata)
        _increment_count(counts_by_adapter, _world_model_adapter_name(metadata))
        reference = _world_model_reference(metadata)
        view = _world_model_view(metadata)
        _increment_count(counts_by_reference_id, reference.get("reference_id"))
        _increment_count(counts_by_decision_rule, _world_model_decision_rule(metadata))

        confidence = _finite_float(metadata.get("prediction_confidence"))
        if confidence is not None:
            prediction_confidences.append(confidence)
        agreement_rate = _finite_float(metadata.get("agreement_rate"))
        if agreement_rate is None:
            agreement_rate = _finite_float(prediction_metadata.get("agreement_rate"))
        if agreement_rate is not None:
            agreement_rates.append(agreement_rate)

        conflict = _world_model_conflict(metadata)
        if conflict:
            conflict_count += 1
            _increment_count(conflict_paths, conflict.get("path"))
        if _world_model_low_agreement(metadata):
            low_agreement_count += 1
        if metadata.get("no_rule_matched") is True or prediction_metadata.get("no_rule_matched") is True:
            no_rule_matched_count += 1
        if not reference or not view:
            trace_gap_count += 1

    return {
        "total": len(results),
        "world_model_total": world_model_total,
        "coverage_rate": _safe_div(world_model_total, len(results)) or 0.0,
        "conflict_count": conflict_count,
        "conflict_rate": _safe_div(conflict_count, world_model_total) or 0.0,
        "low_agreement_count": low_agreement_count,
        "low_agreement_rate": _safe_div(low_agreement_count, world_model_total) or 0.0,
        "no_rule_matched_count": no_rule_matched_count,
        "trace_gap_count": trace_gap_count,
        "trace_gap_rate": _safe_div(trace_gap_count, world_model_total) or 0.0,
        "counts_by_status": counts_by_status,
        "counts_by_adapter": counts_by_adapter,
        "counts_by_reference_id": counts_by_reference_id,
        "counts_by_decision_rule": counts_by_decision_rule,
        "conflict_paths": conflict_paths,
        "prediction_confidence_min": min(prediction_confidences) if prediction_confidences else None,
        "prediction_confidence_mean": _mean_or_none(prediction_confidences),
        "agreement_rate_min": min(agreement_rates) if agreement_rates else None,
        "agreement_rate_mean": _mean_or_none(agreement_rates),
        "traceable": world_model_total > 0 and trace_gap_count == 0,
    }


def _is_world_model_result(metadata: Mapping[str, Any]) -> bool:
    verifier = metadata.get("verifier")
    if any(key in metadata for key in _WORLD_MODEL_TRACE_METADATA_KEYS):
        return True
    if verifier == "world_model_ensemble":
        return True
    prediction_metadata = _world_model_prediction_metadata(metadata)
    return any(key in prediction_metadata for key in _WORLD_MODEL_TRACE_METADATA_KEYS)


_WORLD_MODEL_TRACE_METADATA_KEYS = (
    "world_model",
    "world_model_reference",
    "world_model_view",
    "world_model_conflict",
)


def _world_model_adapter_name(metadata: Mapping[str, Any]) -> str | None:
    raw = metadata.get("world_model")
    if raw is not None:
        return str(raw)
    reference = _world_model_reference(metadata)
    raw = reference.get("adapter")
    if raw is not None:
        return str(raw)
    prediction_metadata = _world_model_prediction_metadata(metadata)
    raw = prediction_metadata.get("world_model")
    if raw is not None:
        return str(raw)
    if metadata.get("verifier") == "world_model_ensemble":
        return "EnsembleWorldModelAdapter"
    return None


def _world_model_prediction_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(metadata.get("prediction_metadata"))


def _world_model_reference(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = _mapping(metadata.get("world_model_reference"))
    if reference:
        return reference
    return _mapping(_world_model_prediction_metadata(metadata).get("world_model_reference"))


def _world_model_view(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    view = _mapping(metadata.get("world_model_view"))
    if view:
        return view
    return _mapping(_world_model_prediction_metadata(metadata).get("world_model_view"))


def _world_model_conflict(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    conflict = _mapping(metadata.get("world_model_conflict"))
    if conflict:
        return conflict
    return _mapping(_world_model_prediction_metadata(metadata).get("world_model_conflict"))


def _world_model_decision_rule(metadata: Mapping[str, Any]) -> Any:
    if metadata.get("decision_rule") is not None:
        return metadata.get("decision_rule")
    return _world_model_prediction_metadata(metadata).get("decision_rule")


def _world_model_low_agreement(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("below_min_agreement") is True:
        return True
    prediction_metadata = _world_model_prediction_metadata(metadata)
    if prediction_metadata.get("below_min_agreement") is True:
        return True
    decision_rule = str(metadata.get("decision_rule", ""))
    prediction_rule = str(prediction_metadata.get("decision_rule", ""))
    return "agreement_below_threshold" in decision_rule or "agreement_below_threshold" in prediction_rule


def _context_sensitivity_summary_from_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts_by_status: dict[str, int] = {}
    counts_by_source: dict[str, int] = {}
    flagged_rates: list[float] = []
    max_shifts: list[float] = []
    mean_shifts: list[float] = []
    max_ratios: list[float] = []
    context_sensitivity_total = 0
    flagged_result_count = 0
    trace_gap_count = 0

    for result in results:
        metadata = _mapping(result.get("metadata"))
        if not _is_context_sensitivity_result(metadata):
            continue
        context_sensitivity_total += 1
        _increment_count(counts_by_status, result.get("status", "unknown"))
        _increment_count(counts_by_source, _context_sensitivity_source(metadata))

        summary = _context_sensitivity_result_summary(metadata)
        if not summary:
            trace_gap_count += 1

        flagged_rate = _finite_float(
            _first_present(
                summary.get("flagged_rate"),
                metadata.get("context_sensitivity_flagged_rate"),
            )
        )
        max_shift = _finite_float(
            _first_present(
                summary.get("max_unsupported_context_shift"),
                summary.get("max_shift"),
                metadata.get("context_sensitivity_max_shift"),
            )
        )
        mean_shift = _finite_float(
            _first_present(
                summary.get("mean_unsupported_context_shift"),
                summary.get("mean_shift"),
                metadata.get("context_sensitivity_mean_shift"),
            )
        )
        max_ratio = _finite_float(
            _first_present(
                summary.get("max_context_sensitivity_ratio"),
                summary.get("max_ratio"),
                metadata.get("context_sensitivity_max_context_sensitivity_ratio"),
                metadata.get("context_sensitivity_max_ratio"),
            )
        )

        if flagged_rate is not None:
            flagged_rates.append(flagged_rate)
            if flagged_rate > 0.0:
                flagged_result_count += 1
        if max_shift is not None:
            max_shifts.append(max_shift)
        if mean_shift is not None:
            mean_shifts.append(mean_shift)
        if max_ratio is not None:
            max_ratios.append(max_ratio)

    return {
        "total": len(results),
        "context_sensitivity_total": context_sensitivity_total,
        "coverage_rate": _safe_div(context_sensitivity_total, len(results)) or 0.0,
        "flagged_result_count": flagged_result_count,
        "flagged_result_rate": _safe_div(
            flagged_result_count,
            context_sensitivity_total,
        ) or 0.0,
        "max_flagged_rate": max(flagged_rates) if flagged_rates else None,
        "mean_flagged_rate": _mean_or_none(flagged_rates),
        "max_unsupported_context_shift": max(max_shifts) if max_shifts else None,
        "mean_unsupported_context_shift": _mean_or_none(mean_shifts),
        "max_context_sensitivity_ratio": max(max_ratios) if max_ratios else None,
        "trace_gap_count": trace_gap_count,
        "trace_gap_rate": _safe_div(trace_gap_count, context_sensitivity_total) or 0.0,
        "counts_by_status": counts_by_status,
        "counts_by_source": counts_by_source,
        "traceable": context_sensitivity_total > 0 and trace_gap_count == 0,
    }


def _is_context_sensitivity_result(metadata: Mapping[str, Any]) -> bool:
    if any(key in metadata for key in _CONTEXT_SENSITIVITY_TRACE_METADATA_KEYS):
        return True
    context_sensitivity = _mapping(metadata.get("context_sensitivity"))
    return any(key in context_sensitivity for key in ("summary", "token_scores", "tokens"))


_CONTEXT_SENSITIVITY_TRACE_METADATA_KEYS = (
    "context_sensitivity",
    "context_sensitivity_summary",
    "context_sensitivity_flagged_rate",
    "context_sensitivity_max_shift",
    "context_sensitivity_mean_shift",
    "context_sensitivity_max_ratio",
    "context_sensitivity_max_context_sensitivity_ratio",
)


def _context_sensitivity_result_summary(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    context_sensitivity = _mapping(metadata.get("context_sensitivity"))
    summary = _mapping(context_sensitivity.get("summary"))
    if summary:
        return summary
    summary = _mapping(metadata.get("context_sensitivity_summary"))
    if summary:
        return summary
    flat = {
        "flagged_rate": metadata.get("context_sensitivity_flagged_rate"),
        "max_unsupported_context_shift": metadata.get("context_sensitivity_max_shift"),
        "mean_unsupported_context_shift": metadata.get("context_sensitivity_mean_shift"),
        "max_context_sensitivity_ratio": _first_present(
            metadata.get("context_sensitivity_max_context_sensitivity_ratio"),
            metadata.get("context_sensitivity_max_ratio"),
        ),
    }
    return {key: value for key, value in flat.items() if value is not None}


def _context_sensitivity_source(metadata: Mapping[str, Any]) -> str | None:
    context_sensitivity = _mapping(metadata.get("context_sensitivity"))
    context_metadata = _mapping(context_sensitivity.get("metadata"))
    paired_metadata = _mapping(context_metadata.get("paired_metadata"))
    raw = _first_present(
        context_metadata.get("adapter"),
        paired_metadata.get("adapter"),
        metadata.get("context_sensitivity_source"),
        metadata.get("selected_verifier"),
        metadata.get("verifier"),
    )
    if raw is None:
        return None
    return str(raw)


def _evidence_alignment_summary_from_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts_by_status: dict[str, int] = {}
    counts_by_alignment_status: dict[str, int] = {}
    counts_by_code: dict[str, int] = {}
    counts_by_source: dict[str, int] = {}
    keyword_overlaps = []
    number_recalls = []
    entity_recalls = []
    evidence_alignment_total = 0
    record_count = 0.0
    aligned_count = 0.0
    misaligned_count = 0.0
    insufficient_count = 0.0
    reference_count = 0.0
    matched_reference_count = 0.0
    cited_evidence_count = 0.0
    issue_count = 0.0
    trace_gap_count = 0

    for result in results:
        metadata = _mapping(result.get("metadata"))
        if not _is_evidence_alignment_result(metadata):
            continue
        evidence_alignment_total += 1
        _increment_count(counts_by_status, result.get("status", "unknown"))
        _increment_count(counts_by_source, _evidence_alignment_source(metadata))
        summary = _evidence_alignment_result_summary(metadata)
        if not summary:
            trace_gap_count += 1
            continue

        _merge_counts(counts_by_alignment_status, _mapping(summary.get("counts_by_status")))
        _merge_counts(counts_by_code, _mapping(summary.get("counts_by_code")))
        records = _first_non_negative_float(summary.get("record_count"))
        if records is None and any(
            summary.get(key) is not None
            for key in (
                "alignment_rate",
                "misalignment_rate",
                "insufficient_evidence_rate",
                "issue_count",
            )
        ):
            records = 1.0
        records = records or 0.0
        record_count += records

        aligned_count += _first_non_negative_float(
            summary.get("aligned_count"),
        ) or _count_from_rate(summary.get("alignment_rate"), records)
        misaligned_count += _first_non_negative_float(
            summary.get("misaligned_count"),
        ) or _count_from_rate(summary.get("misalignment_rate"), records)
        insufficient_count += _first_non_negative_float(
            summary.get("insufficient_evidence_count"),
        ) or _count_from_rate(summary.get("insufficient_evidence_rate"), records)
        reference_count += _first_non_negative_float(
            summary.get("citation_reference_count"),
        ) or 0.0
        matched_reference_count += _first_non_negative_float(
            summary.get("matched_citation_reference_count"),
        ) or _count_from_rate(
            summary.get("citation_reference_coverage_rate"),
            _first_non_negative_float(summary.get("citation_reference_count")),
        )
        cited_evidence_count += _first_non_negative_float(
            summary.get("cited_evidence_count"),
        ) or 0.0
        issue_count += _first_non_negative_float(summary.get("issue_count")) or 0.0
        for values, key in (
            (keyword_overlaps, "keyword_overlap_mean"),
            (number_recalls, "number_recall_mean"),
            (entity_recalls, "entity_recall_mean"),
        ):
            numeric = _finite_float(summary.get(key))
            if numeric is not None:
                values.append(numeric)

    return {
        "total": len(results),
        "available": evidence_alignment_total > 0,
        "evidence_alignment_total": evidence_alignment_total,
        "coverage_rate": _safe_div(evidence_alignment_total, len(results)) or 0.0,
        "record_count": record_count,
        "aligned_count": aligned_count,
        "misaligned_count": misaligned_count,
        "insufficient_evidence_count": insufficient_count,
        "alignment_rate": _safe_div(aligned_count, record_count) or 0.0,
        "misalignment_rate": _safe_div(misaligned_count, record_count) or 0.0,
        "insufficient_evidence_rate": _safe_div(insufficient_count, record_count) or 0.0,
        "keyword_overlap_mean": _mean_or_none(keyword_overlaps),
        "number_recall_mean": _mean_or_none(number_recalls),
        "entity_recall_mean": _mean_or_none(entity_recalls),
        "citation_reference_count": reference_count,
        "matched_citation_reference_count": matched_reference_count,
        "cited_evidence_count": cited_evidence_count,
        "citation_reference_coverage_rate": _safe_div(
            matched_reference_count,
            reference_count,
        ),
        "issue_count": issue_count,
        "issue_rate": _safe_div(issue_count, record_count) or 0.0,
        "trace_gap_count": trace_gap_count,
        "trace_gap_rate": _safe_div(trace_gap_count, evidence_alignment_total) or 0.0,
        "counts_by_status": counts_by_status,
        "counts_by_alignment_status": counts_by_alignment_status,
        "counts_by_code": counts_by_code,
        "counts_by_source": counts_by_source,
        "traceable": evidence_alignment_total > 0 and trace_gap_count == 0,
    }


def _is_evidence_alignment_result(metadata: Mapping[str, Any]) -> bool:
    if str(metadata.get("verifier", "")).strip() == "evidence_alignment":
        return True
    return any(key in metadata for key in _EVIDENCE_ALIGNMENT_TRACE_METADATA_KEYS)


_EVIDENCE_ALIGNMENT_TRACE_METADATA_KEYS = (
    "evidence_alignment",
    "evidence_alignment_summary",
    "evidence_alignment_alignment_rate",
    "evidence_alignment_misalignment_rate",
    "evidence_alignment_insufficient_evidence_rate",
    "evidence_alignment_issue_count",
)


def _evidence_alignment_result_summary(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    report = _mapping(metadata.get("evidence_alignment"))
    summary = _mapping(report.get("summary"))
    if summary:
        return summary
    summary = _mapping(metadata.get("evidence_alignment_summary"))
    if summary:
        return summary
    flat = {
        "record_count": metadata.get("evidence_alignment_record_count"),
        "aligned_count": metadata.get("evidence_alignment_aligned_count"),
        "misaligned_count": metadata.get("evidence_alignment_misaligned_count"),
        "insufficient_evidence_count": metadata.get(
            "evidence_alignment_insufficient_evidence_count"
        ),
        "alignment_rate": metadata.get("evidence_alignment_alignment_rate"),
        "misalignment_rate": metadata.get("evidence_alignment_misalignment_rate"),
        "insufficient_evidence_rate": metadata.get(
            "evidence_alignment_insufficient_evidence_rate"
        ),
        "keyword_overlap_mean": metadata.get("evidence_alignment_keyword_overlap_mean"),
        "number_recall_mean": metadata.get("evidence_alignment_number_recall_mean"),
        "entity_recall_mean": metadata.get("evidence_alignment_entity_recall_mean"),
        "citation_reference_count": metadata.get(
            "evidence_alignment_citation_reference_count"
        ),
        "matched_citation_reference_count": metadata.get(
            "evidence_alignment_matched_citation_reference_count"
        ),
        "citation_reference_coverage_rate": metadata.get(
            "evidence_alignment_citation_reference_coverage_rate"
        ),
        "issue_count": metadata.get("evidence_alignment_issue_count"),
    }
    return {key: value for key, value in flat.items() if value is not None}


def _evidence_alignment_source(metadata: Mapping[str, Any]) -> str | None:
    report = _mapping(metadata.get("evidence_alignment"))
    report_metadata = _mapping(report.get("metadata"))
    raw = _first_present(
        report_metadata.get("adapter"),
        report_metadata.get("source"),
        metadata.get("evidence_alignment_source"),
        metadata.get("selected_verifier"),
        metadata.get("verifier"),
    )
    if raw is None:
        return None
    return str(raw)


def _counterfactual_robustness_summary_from_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts_by_status: dict[str, int] = {}
    counts_by_source: dict[str, int] = {}
    counts_by_probe_type: dict[str, int] = {}
    counts_by_failure_reason: dict[str, int] = {}
    counts_by_entity_candidate: dict[str, int] = {}
    false_invariance_by_entity_candidate: dict[str, int] = {}
    counts_by_entity_source_kind: dict[str, int] = {}
    counterfactual_result_total = 0
    counterfactual_probe_total = 0.0
    entity_probe_count = 0.0
    passed_count = 0.0
    failed_count = 0.0
    expected_flip_count = 0.0
    flip_success_count = 0.0
    false_invariance_count = 0.0
    expected_stable_count = 0.0
    stable_success_count = 0.0
    unexpected_flip_count = 0.0
    trace_gap_count = 0

    for result in results:
        metadata = _mapping(result.get("metadata"))
        if not _is_counterfactual_robustness_result(metadata):
            continue
        counterfactual_result_total += 1
        _increment_count(counts_by_status, result.get("status", "unknown"))
        _increment_count(counts_by_source, _counterfactual_source(metadata))
        summary = _counterfactual_result_summary(metadata)
        if not summary:
            trace_gap_count += 1
            summary = _counterfactual_flat_summary(metadata)
            if not summary:
                continue

        record_count = _first_non_negative_float(
            summary.get("record_count"),
            summary.get("probe_count"),
            summary.get("counterfactual_probe_count"),
        )
        if record_count is None:
            record_count = 1.0
        counterfactual_probe_total += record_count
        passed_count += _first_non_negative_float(
            summary.get("passed_count"),
            summary.get("pass_count"),
        ) or _count_from_rate(summary.get("pass_rate"), record_count)
        failed_count += _first_non_negative_float(
            summary.get("failed_count"),
            summary.get("failure_count"),
        ) or _count_from_rate(summary.get("failure_rate"), record_count)
        expected_flip = _first_non_negative_float(
            summary.get("expected_flip_count"),
            summary.get("flip_expected_count"),
        )
        flip_success = _first_non_negative_float(
            summary.get("flip_success_count"),
            summary.get("status_changed_count"),
        )
        false_invariance = _first_non_negative_float(summary.get("false_invariance_count"))
        if false_invariance is None:
            false_invariance = _count_from_rate(summary.get("false_invariance_rate"), expected_flip)
        if expected_flip is None and flip_success is not None and false_invariance is not None:
            expected_flip = flip_success + false_invariance
        expected_flip_count += expected_flip or 0.0
        flip_success_count += flip_success or 0.0
        false_invariance_count += false_invariance or 0.0
        expected_stable = _first_non_negative_float(summary.get("expected_stable_count"))
        stable_success = _first_non_negative_float(summary.get("stable_success_count"))
        unexpected_flip = _first_non_negative_float(summary.get("unexpected_flip_count"))
        if unexpected_flip is None:
            unexpected_flip = _count_from_rate(summary.get("unexpected_flip_rate"), expected_stable)
        if expected_stable is None and stable_success is not None and unexpected_flip is not None:
            expected_stable = stable_success + unexpected_flip
        expected_stable_count += expected_stable or 0.0
        stable_success_count += stable_success or 0.0
        unexpected_flip_count += unexpected_flip or 0.0
        entity_probe = _first_non_negative_float(summary.get("entity_probe_count"))
        if entity_probe is None:
            entity_probe = _counterfactual_entity_probe_count(summary.get("by_entity_candidate"))
        entity_probe_count += entity_probe or 0.0
        _merge_counterfactual_group_counts(
            counts_by_probe_type,
            summary.get("by_probe_type"),
            count_key="record_count",
        )
        _merge_counts(counts_by_failure_reason, _mapping(summary.get("failure_reasons")))
        _merge_counts(counts_by_failure_reason, _mapping(summary.get("counts_by_failure_reason")))
        entity_source_kind_counts = _mapping(summary.get("counts_by_entity_source_kind"))
        _merge_counts(
            counts_by_entity_source_kind,
            entity_source_kind_counts,
        )
        _merge_counterfactual_entity_candidate_counts(
            counts_by_entity_candidate,
            false_invariance_by_entity_candidate,
            counts_by_entity_source_kind,
            summary.get("by_entity_candidate"),
            merge_source_kinds=not bool(entity_source_kind_counts),
        )
        failure_reason = metadata.get("counterfactual_failure_reason")
        if failure_reason is not None:
            _increment_count(counts_by_failure_reason, failure_reason)
        probe_type = metadata.get("counterfactual_probe_type")
        if probe_type is not None:
            _increment_count(counts_by_probe_type, probe_type)

    if failed_count == 0.0 and counterfactual_probe_total:
        failed_count = max(counterfactual_probe_total - passed_count, 0.0)
    return {
        "total": len(results),
        "counterfactual_result_total": counterfactual_result_total,
        "counterfactual_probe_total": counterfactual_probe_total,
        "entity_probe_count": entity_probe_count,
        "entity_candidate_count": len(counts_by_entity_candidate),
        "coverage_rate": _safe_div(counterfactual_result_total, len(results)) or 0.0,
        "pass_rate": _safe_div(passed_count, counterfactual_probe_total) or 0.0,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "expected_flip_count": expected_flip_count,
        "flip_success_count": flip_success_count,
        "flip_success_rate": _safe_div(flip_success_count, expected_flip_count) or 0.0,
        "false_invariance_count": false_invariance_count,
        "false_invariance_rate": _safe_div(
            false_invariance_count,
            expected_flip_count,
        ) or 0.0,
        "expected_stable_count": expected_stable_count,
        "stable_success_count": stable_success_count,
        "stable_success_rate": _safe_div(stable_success_count, expected_stable_count) or 0.0,
        "unexpected_flip_count": unexpected_flip_count,
        "unexpected_flip_rate": _safe_div(
            unexpected_flip_count,
            expected_stable_count,
        ) or 0.0,
        "trace_gap_count": trace_gap_count,
        "trace_gap_rate": _safe_div(trace_gap_count, counterfactual_result_total) or 0.0,
        "counts_by_status": counts_by_status,
        "counts_by_source": counts_by_source,
        "counts_by_probe_type": counts_by_probe_type,
        "counts_by_failure_reason": counts_by_failure_reason,
        "counts_by_entity_candidate": counts_by_entity_candidate,
        "false_invariance_by_entity_candidate": false_invariance_by_entity_candidate,
        "counts_by_entity_source_kind": counts_by_entity_source_kind,
        "traceable": counterfactual_result_total > 0 and trace_gap_count == 0,
    }


def _is_counterfactual_robustness_result(metadata: Mapping[str, Any]) -> bool:
    if any(key in metadata for key in _COUNTERFACTUAL_ROBUSTNESS_TRACE_METADATA_KEYS):
        return True
    counterfactual = _mapping(metadata.get("counterfactual_verification"))
    if counterfactual.get("workflow") == "counterfactual_verification_audit":
        return True
    if counterfactual.get("summary") is not None:
        return True
    return False


_COUNTERFACTUAL_ROBUSTNESS_TRACE_METADATA_KEYS = (
    "counterfactual_verification",
    "counterfactual_verification_summary",
    "counterfactual_probe",
    "counterfactual_probe_type",
    "counterfactual_status_changed",
    "counterfactual_passed",
    "counterfactual_false_invariance",
    "counterfactual_failure_reason",
)


def _counterfactual_result_summary(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    counterfactual = _mapping(metadata.get("counterfactual_verification"))
    summary = _mapping(counterfactual.get("summary"))
    if summary:
        return summary
    summary = _mapping(metadata.get("counterfactual_verification_summary"))
    if summary:
        return summary
    if counterfactual.get("workflow") == "counterfactual_verification_audit":
        return _mapping(counterfactual)
    return _counterfactual_flat_summary(metadata)


def _counterfactual_flat_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    status_changed = _optional_bool(metadata.get("counterfactual_status_changed"))
    expected_flip = _optional_bool(metadata.get("counterfactual_expected_flip"))
    passed = _optional_bool(metadata.get("counterfactual_passed"))
    false_invariance = _optional_bool(metadata.get("counterfactual_false_invariance"))
    unexpected_flip = _optional_bool(metadata.get("counterfactual_unexpected_flip"))
    if not any(
        value is not None
        for value in (status_changed, expected_flip, passed, false_invariance, unexpected_flip)
    ):
        return {}
    expected_flip = True if expected_flip is None else expected_flip
    false_invariance = (
        expected_flip and status_changed is False
        if false_invariance is None and status_changed is not None
        else false_invariance
    )
    unexpected_flip = (
        (not expected_flip) and status_changed is True
        if unexpected_flip is None and status_changed is not None
        else unexpected_flip
    )
    if passed is None and status_changed is not None:
        passed = status_changed is True if expected_flip else status_changed is False
    return {
        "record_count": 1,
        "passed_count": 1 if passed else 0,
        "failed_count": 0 if passed else 1,
        "expected_flip_count": 1 if expected_flip else 0,
        "flip_success_count": 1 if expected_flip and status_changed else 0,
        "false_invariance_count": 1 if false_invariance else 0,
        "expected_stable_count": 0 if expected_flip else 1,
        "stable_success_count": 1 if (not expected_flip) and status_changed is False else 0,
        "unexpected_flip_count": 1 if unexpected_flip else 0,
    }


def _counterfactual_source(metadata: Mapping[str, Any]) -> str | None:
    counterfactual = _mapping(metadata.get("counterfactual_verification"))
    counterfactual_metadata = _mapping(counterfactual.get("metadata"))
    raw = _first_present(
        counterfactual_metadata.get("adapter"),
        counterfactual_metadata.get("verifier"),
        metadata.get("counterfactual_source"),
        metadata.get("selected_verifier"),
        metadata.get("verifier"),
    )
    if raw is None:
        return None
    return str(raw)


def _merge_counterfactual_group_counts(
    target: dict[str, int],
    groups: Any,
    *,
    count_key: str,
) -> None:
    for raw_key, raw_value in _mapping(groups).items():
        key = str(raw_key).strip()
        if not key:
            continue
        if isinstance(raw_value, Mapping):
            count = _non_negative_int(raw_value.get(count_key))
        else:
            count = _non_negative_int(raw_value)
        if count is None:
            continue
        target[key] = target.get(key, 0) + count


def _merge_counterfactual_entity_candidate_counts(
    counts_by_entity_candidate: dict[str, int],
    false_invariance_by_entity_candidate: dict[str, int],
    counts_by_entity_source_kind: dict[str, int],
    groups: Any,
    *,
    merge_source_kinds: bool = True,
) -> None:
    for raw_entity, raw_value in _mapping(groups).items():
        entity = str(raw_entity).strip()
        if not entity:
            continue
        if isinstance(raw_value, Mapping):
            record_count = _non_negative_int(raw_value.get("record_count"))
            false_invariance_count = _non_negative_int(
                raw_value.get("false_invariance_count")
            )
            if merge_source_kinds:
                _merge_counts(
                    counts_by_entity_source_kind,
                    _mapping(raw_value.get("source_kinds")),
                )
        else:
            record_count = _non_negative_int(raw_value)
            false_invariance_count = None
        if record_count is not None:
            counts_by_entity_candidate[entity] = (
                counts_by_entity_candidate.get(entity, 0) + record_count
            )
        if false_invariance_count is not None:
            false_invariance_by_entity_candidate[entity] = (
                false_invariance_by_entity_candidate.get(entity, 0)
                + false_invariance_count
            )


def _counterfactual_entity_probe_count(groups: Any) -> float | None:
    total = 0
    observed = False
    for raw_value in _mapping(groups).values():
        if isinstance(raw_value, Mapping):
            count = _non_negative_int(raw_value.get("record_count"))
        else:
            count = _non_negative_int(raw_value)
        if count is None:
            continue
        observed = True
        total += count
    return float(total) if observed else None


def _citation_integrity_summary(
    claims: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    claim_reference_counts: dict[str, int] = {}
    counts_by_status: dict[str, int] = {}
    counts_by_decision_rule: dict[str, int] = {}
    counts_by_reference_source: dict[str, int] = {}
    mismatch_fields: dict[str, int] = {}
    matched_citation_ids: set[str] = set()
    catalog_sizes: list[float] = []

    for index, claim in enumerate(claims):
        claim_id = _payload_claim_id(claim, fallback=f"claim:{index}")
        references = _claim_citation_references(claim)
        if references:
            claim_reference_counts[claim_id] = len(references)
        for reference in references:
            _increment_count(counts_by_reference_source, reference.get("source"))

    citation_result_total = 0
    mismatch_count = 0
    unresolved_count = 0
    empty_catalog_count = 0
    no_reference_result_count = 0
    trace_gap_count = 0
    covered_claim_ids: set[str] = set()

    for index, result in enumerate(results):
        metadata = _mapping(result.get("metadata"))
        if not _is_citation_result(metadata):
            continue
        citation_result_total += 1
        _increment_count(counts_by_status, result.get("status", "unknown"))
        decision_rule = metadata.get("decision_rule")
        _increment_count(counts_by_decision_rule, decision_rule)
        claim_id = _payload_claim_id(metadata, fallback=f"result:{index}")
        if claim_id != f"result:{index}":
            covered_claim_ids.add(claim_id)

        catalog_size = _finite_float(metadata.get("catalog_size"))
        if catalog_size is not None:
            catalog_sizes.append(catalog_size)
        for citation_id in _as_sequence(metadata.get("matched_citation_ids", ())):
            text = str(citation_id).strip()
            if text:
                matched_citation_ids.add(text)

        references = tuple(
            item for item in _as_sequence(metadata.get("references", ())) if isinstance(item, Mapping)
        )
        audits = tuple(item for item in _as_sequence(metadata.get("audits", ())) if isinstance(item, Mapping))
        for reference in references:
            _increment_count(counts_by_reference_source, reference.get("source"))

        explicit_mismatch_count = _non_negative_int(metadata.get("mismatch_count"))
        explicit_unresolved_count = _non_negative_int(metadata.get("unresolved_count"))
        audit_mismatch_count = 0
        audit_unresolved_count = 0
        for audit in audits:
            if str(audit.get("status", "")).strip().lower() == "unresolved":
                audit_unresolved_count += 1
            mismatches = tuple(
                item for item in _as_sequence(audit.get("mismatches", ())) if isinstance(item, Mapping)
            )
            audit_mismatch_count += len(mismatches)
            for mismatch in mismatches:
                _increment_count(mismatch_fields, mismatch.get("field"))

        mismatch_count += (
            explicit_mismatch_count
            if explicit_mismatch_count is not None
            else audit_mismatch_count
        )
        unresolved_count += (
            explicit_unresolved_count
            if explicit_unresolved_count is not None
            else audit_unresolved_count
        )
        if decision_rule == "empty_catalog":
            empty_catalog_count += 1
        if decision_rule == "no_citation_reference":
            no_reference_result_count += 1
        if (
            decision_rule != "no_citation_reference"
            and not references
            and not audits
            and catalog_size is None
        ):
            trace_gap_count += 1

    cited_claim_count = len(claim_reference_counts)
    citation_reference_count = sum(claim_reference_counts.values())
    if covered_claim_ids:
        covered_cited_claim_count = len(covered_claim_ids & set(claim_reference_counts))
    else:
        covered_cited_claim_count = min(citation_result_total, cited_claim_count)
    issue_count = mismatch_count + unresolved_count + empty_catalog_count + trace_gap_count
    available = cited_claim_count > 0 or citation_result_total > 0
    return {
        "available": available,
        "passed": (issue_count == 0) if available else None,
        "claim_count": len(claims),
        "verification_result_count": len(results),
        "cited_claim_count": cited_claim_count,
        "citation_reference_count": citation_reference_count,
        "citation_result_total": citation_result_total,
        "coverage_rate": _safe_div(covered_cited_claim_count, cited_claim_count) or 0.0,
        "covered_cited_claim_count": covered_cited_claim_count,
        "mismatch_count": mismatch_count,
        "unresolved_count": unresolved_count,
        "empty_catalog_count": empty_catalog_count,
        "no_reference_result_count": no_reference_result_count,
        "issue_count": issue_count,
        "trace_gap_count": trace_gap_count,
        "trace_gap_rate": _safe_div(trace_gap_count, citation_result_total) or 0.0,
        "matched_citation_count": len(matched_citation_ids),
        "matched_citation_ids": tuple(sorted(matched_citation_ids))[:16],
        "catalog_size_min": min(catalog_sizes) if catalog_sizes else None,
        "catalog_size_mean": _mean_or_none(catalog_sizes),
        "counts_by_status": counts_by_status,
        "counts_by_decision_rule": counts_by_decision_rule,
        "counts_by_reference_source": counts_by_reference_source,
        "mismatch_fields": mismatch_fields,
        "claim_reference_counts": dict(sorted(claim_reference_counts.items())),
        "traceable": citation_result_total > 0 and trace_gap_count == 0,
    }


def _claim_citation_references(claim: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    try:
        payload = Claim(
            text=str(claim.get("text", "")),
            claim_id=None if claim.get("claim_id") is None else str(claim.get("claim_id")),
            span=None,
            metadata=_mapping(claim.get("metadata")),
        )
        return extract_citation_references(payload)
    except (TypeError, ValueError):
        return ()


def _is_citation_result(metadata: Mapping[str, Any]) -> bool:
    verifier = metadata.get("verifier")
    selected_route = metadata.get("selected_route")
    selected_verifier = metadata.get("selected_verifier")
    decision_rule = str(metadata.get("decision_rule", "")).strip()
    if verifier == "citation" or selected_route == "citation":
        return True
    if selected_verifier is not None and "citation" in str(selected_verifier).lower():
        return True
    if decision_rule.startswith("citation_") or decision_rule in {
        "empty_catalog",
        "no_citation_reference",
    }:
        return True
    return any(
        key in metadata
        for key in (
            "references",
            "audits",
            "matched_citation_ids",
            "mismatch_count",
            "unresolved_count",
            "catalog_size",
        )
    )


def _first_non_negative_float(*values: Any) -> float | None:
    for value in values:
        numeric = _finite_float(value)
        if numeric is not None and numeric >= 0.0:
            return numeric
    return None


def _count_from_rate(rate: Any, denominator: float | None) -> float:
    numeric_rate = _finite_float(rate)
    if numeric_rate is None or denominator is None:
        return 0.0
    return max(numeric_rate, 0.0) * max(float(denominator), 0.0)


def _runtime_summary_from_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
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


def _cache_summary_from_metadata(metadata: Any) -> dict[str, Any]:
    metadata_payload = metadata if isinstance(metadata, Mapping) else {}
    caches = _cache_stats_from_metadata(metadata_payload)
    aggregate = _combine_cache_stats(caches.values())
    return {
        "total_caches": len(caches),
        "aggregate": aggregate,
        "caches": caches,
    }


def _pre_generation_risk_summary_from_metadata(metadata: Any) -> dict[str, Any]:
    metadata_payload = metadata if isinstance(metadata, Mapping) else {}
    assessment = _mapping(metadata_payload.get("pre_generation_risk_assessment"))
    policy = _mapping(metadata_payload.get("pre_generation_risk_policy"))
    soft_risk = _mapping(assessment.get("soft_risk"))
    learned_risk = _mapping(assessment.get("learned_risk"))
    runtime_profile_source = metadata_payload.get("runtime_profile_source")
    reason = assessment.get("reason")
    used_for_runtime_profile = runtime_profile_source == "pre_generation"
    soft_reason = "soft pre-generation risk estimate" in str(reason or "")
    learned_reason = "learned pre-generation risk estimate" in str(reason or "")
    soft_config = _mapping(policy.get("soft_risk_config"))
    return {
        "available": bool(assessment),
        "requested": metadata_payload.get("pre_generation_profile_requested"),
        "selected_profile": assessment.get("selected_profile"),
        "risk_level": assessment.get("risk_level"),
        "reason": reason,
        "runtime_profile_source": runtime_profile_source,
        "used_for_runtime_profile": used_for_runtime_profile,
        "triggered_feature_count": len(_as_sequence(assessment.get("triggered_features", ()))),
        "triggered_metadata_count": len(_as_sequence(assessment.get("triggered_metadata", ()))),
        "soft_risk_available": bool(soft_risk),
        "soft_risk_score": _finite_float(soft_risk.get("score")),
        "soft_risk_probability": _finite_float(soft_risk.get("probability")),
        "soft_risk_level": soft_risk.get("risk_level"),
        "soft_risk_routed": bool(used_for_runtime_profile and soft_reason),
        "route_on_soft_risk": _optional_bool(soft_config.get("route_on_soft_risk")),
        "learned_risk_available": bool(learned_risk),
        "learned_risk_score": _finite_float(learned_risk.get("score")),
        "learned_risk_probability": _finite_float(learned_risk.get("probability")),
        "learned_risk_level": learned_risk.get("risk_level"),
        "learned_risk_source": learned_risk.get("source"),
        "learned_risk_layer_idx": learned_risk.get("layer_idx"),
        "learned_risk_routed": bool(used_for_runtime_profile and learned_reason),
        "route_on_learned_risk": _optional_bool(policy.get("route_on_learned_risk")),
        "learned_attention_max_weight": _finite_float(
            _mapping(learned_risk.get("attention_summary")).get("max_weight")
        ),
    }


def _verification_stage_summary_from_payload(
    *,
    events: Sequence[Mapping[str, Any]],
    metadata: Any,
    claim_count: int,
    verification_result_count: int,
) -> dict[str, Any]:
    metadata_payload = metadata if isinstance(metadata, Mapping) else {}
    stage_event = _latest_event_payload(events, "verification_stage_decision")
    initial_event = _latest_event_payload(events, "initial_verification")
    skipped_event = _latest_event_payload(events, "initial_verification_skipped")
    total_claims = _first_non_negative_int(
        initial_event.get("n_claims"),
        claim_count,
    )
    result_count = _initial_verification_result_count(initial_event)
    if result_count is None:
        result_count = verification_result_count
    verified_claim_ids = tuple(str(item) for item in _as_sequence(initial_event.get("verified_claim_ids", ())))
    skipped_claim_ids = tuple(str(item) for item in _as_sequence(initial_event.get("skipped_claim_ids", ())))
    verification_scope = str(
        initial_event.get("verification_scope")
        or stage_event.get("verification_scope")
        or ""
    ).strip().lower()
    if not verification_scope:
        verification_scope = "none" if _optional_bool(stage_event.get("run_verifier")) is False else "all"
    run_verifier = _optional_bool(stage_event.get("run_verifier"))
    skipped = _optional_bool(initial_event.get("skipped"))
    if skipped is None and skipped_event:
        skipped = True
    if skipped is None and run_verifier is not None:
        skipped = not run_verifier
    if skipped is None:
        skipped = False
    verified_claim_count = len(verified_claim_ids) if verified_claim_ids else result_count
    if verified_claim_count is None:
        verified_claim_count = 0 if skipped is True else total_claims
    if skipped is True and total_claims is not None:
        saved_claim_count = total_claims
    elif skipped_claim_ids:
        saved_claim_count = len(skipped_claim_ids)
    elif verification_scope == "triggered" and total_claims is not None and verified_claim_count is not None:
        saved_claim_count = max(total_claims - verified_claim_count, 0)
    else:
        saved_claim_count = 0
    triggered_claim_ids = tuple(str(item) for item in _as_sequence(stage_event.get("triggered_claim_ids", ())))
    triggered_features = _string_sequence_mapping(stage_event.get("triggered_features"))
    triggered_metadata = _string_sequence_mapping(stage_event.get("triggered_metadata"))
    enabled = (
        bool(stage_event)
        or _truthy_flag(metadata_payload.get("staged_verification_enabled"))
        or isinstance(metadata_payload.get("staged_verification"), Mapping)
    )
    return {
        "enabled": enabled,
        "run_verifier": run_verifier,
        "skipped": bool(skipped),
        "verification_scope": verification_scope,
        "reason": stage_event.get("reason", skipped_event.get("reason")),
        "claim_count": total_claims,
        "verification_result_count": result_count,
        "verified_claim_count": verified_claim_count,
        "saved_claim_count": saved_claim_count,
        "verified_claim_ids": verified_claim_ids,
        "skipped_claim_ids": skipped_claim_ids,
        "skipped_claim_count": len(skipped_claim_ids),
        "skip_rate": _safe_div(saved_claim_count, total_claims or 0),
        "triggered_claim_count": len(triggered_claim_ids),
        "triggered_claim_ids": triggered_claim_ids,
        "triggered_feature_counts": _count_nested_values(triggered_features),
        "triggered_metadata_counts": _count_nested_values(triggered_metadata),
        "triggered_features": triggered_features,
        "triggered_metadata": triggered_metadata,
    }


def _verification_plan_summary(plan: Mapping[str, Any] | None) -> dict[str, Any]:
    if plan is None:
        return {
            "available": False,
            "run_verifier": None,
            "verification_scope": None,
            "claim_count": 0,
            "verify_claim_count": 0,
            "skipped_claim_count": 0,
            "triggered_claim_count": 0,
            "route_counts": {},
            "tool_payload_counts": {},
            "dependency_count": 0,
            "cost_estimate": None,
            "budget": {},
            "hidden_evidence": {
                "available": False,
                "selected_count": 0,
                "claim_count": 0,
                "evidence_ref_count": 0,
                "score_counts": {},
                "layer_counts": {},
                "max_anomaly_score": None,
            },
        }
    route_counts: dict[str, int] = {}
    for hint in _as_sequence(plan.get("route_hints", ())):
        if not isinstance(hint, Mapping):
            continue
        for route in _as_sequence(hint.get("routes", ())):
            route_name = str(route)
            route_counts[route_name] = route_counts.get(route_name, 0) + 1
    cost_estimate = estimate_verification_plan_cost(plan).to_dict()
    budget = plan.get("budget")
    budget_summary = dict(budget) if isinstance(budget, Mapping) else {}
    if budget_summary:
        selected_claim_count = budget_summary.get(
            "selected_claim_count",
            len(_as_sequence(budget_summary.get("selected_claim_ids", ()))),
        )
        dropped_claim_count = budget_summary.get(
            "dropped_claim_count",
            len(_as_sequence(budget_summary.get("dropped_claim_ids", ()))),
        )
        budget_summary = {
            "enabled": budget_summary.get("enabled"),
            "selected_claim_count": selected_claim_count,
            "dropped_claim_count": dropped_claim_count,
            **{key: value for key, value in budget_summary.items() if key != "enabled"},
        }
    return {
        "available": True,
        "run_verifier": _optional_bool(plan.get("run_verifier")),
        "verification_scope": plan.get("verification_scope"),
        "reason": plan.get("reason"),
        "claim_count": len(_as_sequence(plan.get("claims", ()))),
        "verify_claim_count": len(_as_sequence(plan.get("verify_claim_ids", ()))),
        "skipped_claim_count": len(_as_sequence(plan.get("skipped_claim_ids", ()))),
        "triggered_claim_count": len(_as_sequence(plan.get("triggered_claim_ids", ()))),
        "route_counts": route_counts,
        "tool_payload_counts": {
            "retrieval_queries": len(_as_sequence(plan.get("retrieval_queries", ()))),
            "calculation_checks": len(_as_sequence(plan.get("calculation_checks", ()))),
            "state_checks": len(_as_sequence(plan.get("state_checks", ()))),
            "world_model_checks": len(_as_sequence(plan.get("world_model_checks", ()))),
        },
        "dependency_count": len(_as_sequence(plan.get("dependencies", ()))),
        "cost_estimate": cost_estimate,
        "budget": budget_summary,
        "hidden_evidence": _verification_plan_hidden_evidence_summary(plan),
    }


def _verification_plan_hidden_evidence_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    selected_count = 0
    claim_ids: list[str] = []
    evidence_refs: list[str] = []
    score_counts: dict[str, int] = {}
    layer_counts: dict[str, int] = {}
    max_anomaly_score: float | None = None
    for hint in _as_sequence(plan.get("route_hints", ())):
        if not isinstance(hint, Mapping):
            continue
        metadata = hint.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        hidden = metadata.get("hidden_evidence")
        if not isinstance(hidden, Mapping):
            continue
        claim_id = str(hint.get("claim_id", "")).strip()
        if claim_id:
            claim_ids.append(claim_id)
        items = tuple(item for item in _as_sequence(hidden.get("selected", ())) if isinstance(item, Mapping))
        count = _non_negative_int(hidden.get("selected_count"))
        if count is None:
            count = len(items) or len(_as_sequence(hidden.get("evidence_refs", ())))
        selected_count += count
        if items:
            for item in items:
                ref = item.get("evidence_ref")
                if ref is not None and str(ref).strip():
                    evidence_refs.append(str(ref).strip())
                score_name = item.get("score_name")
                if score_name is not None and str(score_name).strip():
                    text = str(score_name).strip()
                    score_counts[text] = score_counts.get(text, 0) + 1
                layer = "primary" if item.get("layer") is None else str(item.get("layer")).strip()
                if layer:
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1
                anomaly = _finite_float(item.get("anomaly_score"))
                if anomaly is not None:
                    max_anomaly_score = anomaly if max_anomaly_score is None else max(max_anomaly_score, anomaly)
        else:
            for ref in _as_sequence(hidden.get("evidence_refs", ())):
                text = str(ref).strip()
                if text:
                    evidence_refs.append(text)
            for score_name in _as_sequence(hidden.get("score_names", ())):
                text = str(score_name).strip()
                if text:
                    score_counts[text] = score_counts.get(text, 0) + 1
            for layer in _as_sequence(hidden.get("layers", ())):
                text = str(layer).strip()
                if text:
                    layer_counts[text] = layer_counts.get(text, 0) + 1
            hidden_max = _finite_float(hidden.get("max_anomaly_score"))
            if hidden_max is not None:
                max_anomaly_score = hidden_max if max_anomaly_score is None else max(max_anomaly_score, hidden_max)
    unique_refs = tuple(dict.fromkeys(evidence_refs))
    unique_claim_ids = tuple(dict.fromkeys(claim_ids))
    return {
        "available": bool(selected_count or unique_refs),
        "selected_count": selected_count,
        "claim_count": len(unique_claim_ids),
        "claim_ids": unique_claim_ids,
        "evidence_ref_count": len(unique_refs),
        "evidence_refs": unique_refs[:12],
        "score_counts": dict(sorted(score_counts.items())),
        "layer_counts": dict(sorted(layer_counts.items())),
        "max_anomaly_score": max_anomaly_score,
    }


def _claim_risk_localization_summary(
    claims: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = localize_claim_risk_spans(
        claims,
        verification_results=results,
        verification_plan=plan,
    )
    summary = report.summary()
    top_spans = sorted(
        report.spans,
        key=lambda span: (span.risk_score, span.claim_id),
        reverse=True,
    )[:5]
    return {
        **summary,
        "top_risk_spans": tuple(span.to_dict() for span in top_spans),
    }


def _triple_coverage_summary(
    claims: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    claim_predicate_counts: dict[str, int] = {}
    audit_predicate_counts: dict[str, int] = {}
    structured_fact_predicate_counts: dict[str, int] = {}
    structured_fact_status_counts: dict[str, int] = {}
    covered_slot_counts: dict[str, int] = {}
    missing_slot_counts: dict[str, int] = {}
    slot_coverage_totals: dict[str, float] = {}
    slot_coverage_counts: dict[str, int] = {}

    claims_with_triples = 0
    claim_triple_count = 0
    triple_claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        metadata = _mapping(claim.get("metadata"))
        triples = _triple_payloads(metadata.get("claim_triples", metadata.get("triples")))
        if triples:
            claims_with_triples += 1
            triple_claim_ids.add(_payload_claim_id(claim, fallback=f"claim:{index}"))
        claim_triple_count += len(triples)
        for triple in triples:
            _increment_count(claim_predicate_counts, _triple_predicate(triple))

    audit_report_count = 0
    audit_triple_count = 0
    audit_passed_count = 0
    audit_failed_count = 0
    covered_slot_count = 0
    missing_slot_count = 0
    structured_fact_result_count = 0
    audit_claim_ids: set[str] = set()
    for index, result in enumerate(results):
        metadata = _mapping(result.get("metadata"))
        audit_report = _mapping(metadata.get("audit_report"))
        if audit_report:
            audit_report_count += 1
            audit_claim_ids.add(
                _payload_claim_id(
                    audit_report,
                    fallback=_payload_claim_id(metadata, fallback=f"claim:{index}"),
                )
            )
            audits = _as_sequence(audit_report.get("audits", ()))
            if audits:
                for audit in audits:
                    if not isinstance(audit, Mapping):
                        continue
                    audit_triple_count += 1
                    if bool(audit.get("passed")):
                        audit_passed_count += 1
                    else:
                        audit_failed_count += 1
                    _increment_count(audit_predicate_counts, _triple_predicate(_mapping(audit.get("triple"))))
                    for slot in _as_sequence(audit.get("covered_slots", ())):
                        slot_name = str(slot)
                        covered_slot_count += 1
                        covered_slot_counts[slot_name] = covered_slot_counts.get(slot_name, 0) + 1
                    for slot in _as_sequence(audit.get("missing_slots", ())):
                        slot_name = str(slot)
                        missing_slot_count += 1
                        missing_slot_counts[slot_name] = missing_slot_counts.get(slot_name, 0) + 1
                    for slot, value in _mapping(audit.get("slot_coverage")).items():
                        numeric = _finite_float(value)
                        if numeric is None:
                            continue
                        slot_name = str(slot)
                        slot_coverage_totals[slot_name] = slot_coverage_totals.get(slot_name, 0.0) + numeric
                        slot_coverage_counts[slot_name] = slot_coverage_counts.get(slot_name, 0) + 1
            else:
                audit_triple_count += _non_negative_int(audit_report.get("triple_count")) or 0
                audit_passed_count += _non_negative_int(audit_report.get("passed_count")) or 0
                audit_failed_count += _non_negative_int(audit_report.get("failed_count")) or 0
                covered_slot_count += _non_negative_int(audit_report.get("covered_slot_count")) or 0
                missing_slot_count += _non_negative_int(audit_report.get("missing_slot_count")) or 0

        for triple_result in _as_sequence(metadata.get("all_triple_results", ())):
            if not isinstance(triple_result, Mapping):
                continue
            structured_fact_result_count += 1
            status = triple_result.get("status", result.get("status"))
            _increment_count(structured_fact_status_counts, status)
            triple_metadata = _mapping(triple_result.get("metadata"))
            _increment_count(
                structured_fact_predicate_counts,
                _triple_predicate(_mapping(triple_metadata.get("triple"))),
            )

    total_audit_slots = covered_slot_count + missing_slot_count
    audit_claim_covered_count = len(triple_claim_ids & audit_claim_ids)
    return {
        "claim_count": len(claims),
        "verification_result_count": len(results),
        "claims_with_triples": claims_with_triples,
        "claim_triple_count": claim_triple_count,
        "claim_triple_coverage_rate": _safe_div(claims_with_triples, len(claims)),
        "claim_predicate_counts": claim_predicate_counts,
        "audit_available": audit_report_count > 0,
        "audit_report_count": audit_report_count,
        "audit_claim_covered_count": audit_claim_covered_count,
        "audit_claim_coverage_rate": _safe_div(audit_claim_covered_count, claims_with_triples),
        "audit_triple_count": audit_triple_count,
        "audit_passed_count": audit_passed_count,
        "audit_failed_count": audit_failed_count,
        "audit_pass_rate": _safe_div(audit_passed_count, audit_triple_count),
        "audit_predicate_counts": audit_predicate_counts,
        "covered_slot_count": covered_slot_count,
        "missing_slot_count": missing_slot_count,
        "slot_coverage_rate": _safe_div(covered_slot_count, total_audit_slots),
        "covered_slot_counts": covered_slot_counts,
        "missing_slot_counts": missing_slot_counts,
        "slot_mean_coverage": {
            slot: slot_coverage_totals[slot] / slot_coverage_counts[slot]
            for slot in sorted(slot_coverage_totals)
            if slot_coverage_counts.get(slot)
        },
        "structured_fact_result_count": structured_fact_result_count,
        "structured_fact_status_counts": structured_fact_status_counts,
        "structured_fact_predicate_counts": structured_fact_predicate_counts,
    }


def _final_answer_summary(final_answer: Mapping[str, Any] | None) -> dict[str, Any]:
    if final_answer is None:
        return {
            "available": False,
            "status": None,
            "answerable": None,
            "action": None,
            "risk_level": None,
            "confidence": None,
            "evidence_count": 0,
            "total_claims": 0,
            "blocked_claim_count": 0,
            "supported_claim_count": 0,
            "refuted_claim_count": 0,
            "unsupported_claim_count": 0,
            "requires_followup": None,
        }
    claim_summary = final_answer.get("claim_summary", {})
    if not isinstance(claim_summary, Mapping):
        claim_summary = {}
    status_counts = claim_summary.get("status_counts", {})
    if not isinstance(status_counts, Mapping):
        status_counts = {}
    followup = final_answer.get("followup", {})
    if not isinstance(followup, Mapping):
        followup = {}
    return {
        "available": True,
        "status": final_answer.get("status"),
        "answerable": _optional_bool(final_answer.get("answerable")),
        "action": final_answer.get("action"),
        "risk_level": final_answer.get("risk_level"),
        "confidence": _finite_float(final_answer.get("confidence")),
        "evidence_count": len(_as_sequence(final_answer.get("evidence", ()))),
        "total_claims": _non_negative_int(claim_summary.get("total_claims")) or 0,
        "blocked_claim_count": len(_as_sequence(claim_summary.get("blocked_claims", ()))),
        "supported_claim_count": _non_negative_int(status_counts.get("supported")) or 0,
        "refuted_claim_count": _non_negative_int(status_counts.get("refuted")) or 0,
        "unsupported_claim_count": _non_negative_int(status_counts.get("insufficient_evidence")) or 0,
        "requires_followup": _optional_bool(followup.get("requires_followup")),
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


def _verification_plan_to_dict(plan: ClaimVerificationPlan | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    if isinstance(plan, ClaimVerificationPlan):
        return plan.to_dict()
    if isinstance(plan, Mapping):
        return dict(_to_jsonable(plan))
    raise ValueError("verification_plan must be a ClaimVerificationPlan, mapping, or None.")


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


_FRONTIER_RELEASE_CITATION_BATCH_BOUNDED_METADATA_KEYS = tuple(
    f"promotion_contract_frontier_release_evidence_{field_name}"
    for field_name in (
        "citation_batch_provenance_present_count",
        "citation_batch_provenance_passed_count",
        "citation_batch_provenance_failed_count",
        "citation_batch_provenance_status_counts",
        "citation_batch_evidence_class_counts",
        "citation_batch_query_sweep_present_count",
        "citation_batch_query_sweep_no_passing_strategy_count",
        "citation_batch_query_sweep_best_strategy_counts",
        "citation_batch_query_sweep_best_passing_strategy_counts",
        "citation_batch_query_sweep_best_passing_blind_refuted_count_sum",
        "citation_batch_query_sweep_best_passing_blind_refuted_count_max",
        "citation_batch_comparison_present_count",
        "citation_batch_comparison_passed_count",
        "citation_batch_comparison_failed_count",
        "citation_batch_comparison_status_counts",
    )
)


DEFAULT_BOUNDED_TRACE_METADATA_KEYS = (
    "artifact_model_id",
    "artifact_source",
    "artifact_target_layer",
    "artifact_scores",
    "source",
    "promotion_contract_source",
    "promotion_contract_budget_enabled",
    "promotion_contract_model_id",
    "promotion_contract_source_workflow",
    "promotion_contract_source_status",
    "promotion_contract_manifest",
    "promotion_contract_manifest_verification",
    "promotion_contract_registry",
    "promotion_contract_registry_key",
    "promotion_contract_promotion_summary",
    "promotion_contract_runtime",
    "promotion_contract_verifier_route",
    "promotion_contract_recommended_route_covered_fact_property_count",
    "promotion_contract_recommended_route_covered_fact_properties",
    "promotion_contract_required_route_baseline_covered_fact_property_counts",
    "promotion_contract_required_route_baseline_covered_fact_properties",
    "promotion_contract_structured_fact_robustness_property_counts",
    "promotion_contract_structured_fact_robustness_properties",
    "promotion_contract_control_policy_config",
    "promotion_contract_control_defaults",
    "promotion_contract_product_trace_replay_workflow",
    "promotion_contract_selfcheck_signal_fusion_workflow",
    "promotion_contract_world_model_signal_workflow",
    "promotion_contract_pathway_intervention_workflow",
    "promotion_contract_feedback_policy_workflow",
    "promotion_contract_external_evidence_baseline_comparison",
    "promotion_contract_frontier_release_evidence",
    "promotion_contract_triple_extraction_fixture_matrix",
    "promotion_contract_release_efficiency",
    "promotion_contract_frontier_release_evidence_status",
    "promotion_contract_frontier_release_evidence_report",
    "promotion_contract_frontier_release_evidence_manifest",
    "promotion_contract_frontier_release_evidence_source",
    "promotion_contract_frontier_release_evidence_registry",
    "promotion_contract_frontier_release_evidence_record",
    "promotion_contract_frontier_release_evidence_workflow",
    "promotion_contract_frontier_release_evidence_report_status",
    "promotion_contract_frontier_release_evidence_decision_status",
    "promotion_contract_frontier_release_evidence_verifier_track_status",
    "promotion_contract_frontier_release_evidence_abstention_track_status",
    "promotion_contract_frontier_release_evidence_multiple_testing_track_status",
    "promotion_contract_frontier_release_evidence_citation_batch_track_status",
    "promotion_contract_frontier_release_evidence_citation_batch_rollup_count",
    "promotion_contract_frontier_release_evidence_citation_batch_expected_batch_count",
    "promotion_contract_frontier_release_evidence_citation_batch_observed_batch_count",
    "promotion_contract_frontier_release_evidence_citation_batch_missing_expected_batch_count",
    "promotion_contract_frontier_release_evidence_citation_batch_duplicate_batch_count",
    "promotion_contract_frontier_release_evidence_citation_batch_unexpected_batch_count",
    "promotion_contract_frontier_release_evidence_citation_batch_adapter_gate_present_count",
    "promotion_contract_frontier_release_evidence_citation_batch_adapter_gate_passed_count",
    "promotion_contract_frontier_release_evidence_citation_batch_adapter_gate_failed_count",
    "promotion_contract_frontier_release_evidence_citation_batch_adapter_gate_status_counts",
    *_FRONTIER_RELEASE_CITATION_BATCH_BOUNDED_METADATA_KEYS,
    "promotion_contract_frontier_release_evidence_run_count",
    "promotion_contract_frontier_release_evidence_run_names",
    "external_evidence_baseline_comparison_report",
    "external_evidence_baseline_comparison_source",
    "external_evidence_baseline_comparison_registry",
    "external_evidence_baseline_comparison_registry_key",
    "external_evidence_baseline_comparison_status",
    "external_evidence_baseline_comparison_decision_status",
    "external_evidence_baseline_comparison_recommended_route",
    "external_evidence_baseline_comparison_route_passed",
    "external_evidence_baseline_comparison_text_redline_passed",
    "triple_extraction_fixture_matrix_report",
    "triple_extraction_fixture_matrix_manifest",
    "triple_extraction_fixture_matrix_source",
    "triple_extraction_fixture_matrix_manifest_verification",
    "triple_extraction_fixture_matrix_registry",
    "triple_extraction_fixture_matrix_registry_key",
    "triple_extraction_fixture_matrix_status",
    "triple_extraction_fixture_matrix_n_corpora",
    "triple_extraction_fixture_matrix_promoted_corpora",
    "triple_extraction_fixture_matrix_distinct_predicate_count",
    "triple_extraction_fixture_matrix_mean_best_f1",
    "triple_extraction_fixture_matrix_mean_f1_lift",
    "runtime_profile",
    "runtime_profile_requested",
    "runtime_profile_selection",
    "runtime_profile_selector_policy",
    "runtime_profile_source",
    "pre_generation_profile_requested",
    "pre_generation_risk_assessment",
    "pre_generation_risk_policy",
    "pre_generation_metadata",
    "staged_verification_enabled",
    "effective_control_policy_config",
    "control_policy_source",
    "verifier_type",
    "calculator_enabled",
    "action_executor_type",
    "registered_actions",
    "runtime_budget",
    "cache_summary",
    "route_cost_summary",
    "verification_stage_summary",
    "final_answer_summary",
)


def _bounded_sequence(items: Sequence[Any], max_items: int) -> list[Any]:
    limit = _non_negative_limit(max_items)
    return list(items[:limit])


def _bounded_mapping_payload(
    value: Mapping[str, Any],
    *,
    max_items: int,
    max_nested_items: int,
    max_string_length: int,
) -> dict[str, Any]:
    normalized = _to_jsonable(value)
    if not isinstance(normalized, Mapping):
        normalized = {}
    limit = _non_negative_limit(max_items)
    items = list(normalized.items())
    return {
        "items": {
            str(key): _bounded_jsonable(
                item,
                max_depth=3,
                max_items=max_nested_items,
                max_string_length=max_string_length,
            )
            for key, item in items[:limit]
        },
        "summary": _truncation_summary(len(items), limit),
    }


def _bounded_metadata(
    metadata: Mapping[str, Any],
    *,
    metadata_keys: Sequence[str] | None,
    max_nested_items: int,
    max_string_length: int,
) -> dict[str, Any]:
    normalized = _to_jsonable(metadata)
    if not isinstance(normalized, Mapping):
        normalized = {}
    selected_keys = tuple(metadata_keys or DEFAULT_BOUNDED_TRACE_METADATA_KEYS)
    return {
        key: _bounded_jsonable(
            normalized[key],
            max_depth=4,
            max_items=max_nested_items,
            max_string_length=max_string_length,
        )
        for key in selected_keys
        if key in normalized
    }


def _bounded_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    max_nested_items: int,
    max_string_length: int,
) -> dict[str, Any]:
    return {
        str(name): {
            str(key): _bounded_jsonable(
                value,
                max_depth=4,
                max_items=max_nested_items,
                max_string_length=max_string_length,
            )
            for key, value in summary.items()
        }
        for name, summary in summaries.items()
    }


def _bounded_verification_result(
    result: Mapping[str, Any],
    *,
    max_nested_items: int,
    max_string_length: int,
) -> dict[str, Any]:
    evidence = _as_sequence(result.get("evidence", ()))
    metadata = result.get("metadata", {})
    return {
        "status": result.get("status"),
        "confidence": result.get("confidence"),
        "evidence_count": len(evidence),
        "evidence": _bounded_sequence(
            [
                _bounded_jsonable(
                    item,
                    max_depth=2,
                    max_items=max_nested_items,
                    max_string_length=max_string_length,
                )
                for item in evidence
            ],
            min(max_nested_items, 3),
        ),
        "explanation": _truncate_string(result.get("explanation"), max_string_length=max_string_length),
        "metadata": _bounded_jsonable(
            metadata if isinstance(metadata, Mapping) else {},
            max_depth=4,
            max_items=max_nested_items,
            max_string_length=max_string_length,
        ),
    }


def _bounded_action_result(
    result: Mapping[str, Any],
    *,
    max_nested_items: int,
    max_string_length: int,
) -> dict[str, Any]:
    output = result.get("output", {})
    metadata = result.get("metadata", {})
    return {
        "action": result.get("action"),
        "status": result.get("status"),
        "request_id": result.get("request_id"),
        "error": _truncate_string(result.get("error"), max_string_length=max_string_length),
        "output_summary": _mapping_summary(output),
        "output": _bounded_jsonable(
            output if isinstance(output, Mapping) else {},
            max_depth=3,
            max_items=max_nested_items,
            max_string_length=max_string_length,
        ),
        "metadata": _bounded_jsonable(
            metadata if isinstance(metadata, Mapping) else {},
            max_depth=4,
            max_items=max_nested_items,
            max_string_length=max_string_length,
        ),
    }


def _bounded_event(
    event: Mapping[str, Any],
    *,
    max_nested_items: int,
    max_string_length: int,
) -> dict[str, Any]:
    payload = event.get("payload", {})
    return {
        "event_type": event.get("event_type"),
        "created_at": event.get("created_at"),
        "payload": _bounded_jsonable(
            payload if isinstance(payload, Mapping) else {},
            max_depth=4,
            max_items=max_nested_items,
            max_string_length=max_string_length,
        ),
    }


def _bounded_final_answer(
    final_answer: Mapping[str, Any] | None,
    *,
    max_nested_items: int,
    max_string_length: int,
) -> dict[str, Any] | None:
    if final_answer is None:
        return None
    return {
        "status": final_answer.get("status"),
        "answerable": final_answer.get("answerable"),
        "action": final_answer.get("action"),
        "risk_level": final_answer.get("risk_level"),
        "confidence": final_answer.get("confidence"),
        "reason": _truncate_string(final_answer.get("reason"), max_string_length=max_string_length),
        "text": _truncate_string(final_answer.get("text"), max_string_length=max_string_length),
        "claim_summary": _bounded_jsonable(
            final_answer.get("claim_summary", {}),
            max_depth=3,
            max_items=max_nested_items,
            max_string_length=max_string_length,
        ),
        "evidence": _bounded_sequence(
            [
                _bounded_jsonable(
                    item,
                    max_depth=2,
                    max_items=max_nested_items,
                    max_string_length=max_string_length,
                )
                for item in _as_sequence(final_answer.get("evidence", ()))
            ],
            min(max_nested_items, 3),
        ),
        "followup": _bounded_jsonable(
            final_answer.get("followup", {}),
            max_depth=3,
            max_items=max_nested_items,
            max_string_length=max_string_length,
        ),
    }


def _bounded_jsonable(
    value: Any,
    *,
    max_depth: int,
    max_items: int,
    max_string_length: int,
) -> Any:
    normalized = _to_jsonable(value)
    if max_depth <= 0:
        return _summarize_leaf(normalized)
    if isinstance(normalized, Mapping):
        limit = _non_negative_limit(max_items)
        items = list(normalized.items())
        bounded = {
            str(key): _bounded_jsonable(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            for key, item in items[:limit]
        }
        if len(items) > limit:
            bounded["_truncated"] = True
            bounded["_omitted_keys"] = len(items) - limit
        return bounded
    if isinstance(normalized, Sequence) and not isinstance(normalized, (str, bytes, bytearray)):
        limit = _non_negative_limit(max_items)
        values = list(normalized)
        bounded_items = [
            _bounded_jsonable(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            for item in values[:limit]
        ]
        if len(values) > limit:
            bounded_items.append({"_truncated": True, "_omitted_items": len(values) - limit})
        return bounded_items
    if isinstance(normalized, str):
        return _truncate_string(normalized, max_string_length=max_string_length)
    return normalized


def _mapping_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        keys = tuple(str(key) for key in value.keys())
        return {
            "kind": "mapping",
            "key_count": len(keys),
            "keys": keys[:16],
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"kind": "sequence", "item_count": len(value)}
    return {"kind": type(value).__name__}


def _truncation_summary(total: int, max_items: int) -> dict[str, int]:
    limit = _non_negative_limit(max_items)
    included = min(total, limit)
    return {
        "total": total,
        "included": included,
        "omitted": max(total - included, 0),
    }


def _non_negative_limit(value: int) -> int:
    return max(int(value), 0)


def _truncate_string(value: Any, *, max_string_length: int) -> Any:
    if not isinstance(value, str):
        return value
    limit = _non_negative_limit(max_string_length)
    if len(value) <= limit:
        return value
    if limit == 0:
        return ""
    suffix = f"...[truncated {len(value) - limit} chars]"
    if len(suffix) >= limit:
        return value[:limit]
    return value[: limit - len(suffix)] + suffix


def _summarize_leaf(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {"kind": "mapping", "key_count": len(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"kind": "sequence", "item_count": len(value)}
    return value


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


def _final_answer_to_dict(answer: FinalAnswer | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if answer is None:
        return None
    if isinstance(answer, FinalAnswer):
        return answer.to_dict()
    if isinstance(answer, Mapping):
        return dict(_to_jsonable(answer))
    raise ValueError("final_answer must be a FinalAnswer, mapping, or None.")


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
    route_budget_limit = _non_negative_int(metadata.get("route_budget_limit"))
    route_budget_exhausted = _truthy_flag(metadata.get("route_budget_exhausted"))
    unattempted_route_count = len(_as_sequence(metadata.get("unattempted_routes", ())))
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
        "route_budget_limit": route_budget_limit,
        "route_budget_exhausted": route_budget_exhausted,
        "unattempted_route_count": unattempted_route_count,
        "selected_route_was_fallthrough": _truthy_flag(
            metadata.get("selected_route_was_fallthrough")
        ),
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
    route_budget_limit_count = sum(
        1
        for record in records
        if _non_negative_int(record.get("route_budget_limit")) is not None
    )
    route_budget_exhausted_count = sum(
        1
        for record in records
        if bool(record.get("route_budget_exhausted"))
    )
    selected_fallthrough_budget_stop_count = sum(
        1
        for record in records
        if bool(record.get("route_budget_exhausted"))
        and bool(record.get("selected_route_was_fallthrough"))
    )
    total_unattempted_route_count = sum(
        _non_negative_int(record.get("unattempted_route_count")) or 0
        for record in records
    )
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
        "route_budget_limit_observations": route_budget_limit_count,
        "route_budget_exhausted_count": route_budget_exhausted_count,
        "route_budget_exhaustion_rate": _safe_div(
            route_budget_exhausted_count,
            route_budget_limit_count,
        ),
        "selected_fallthrough_budget_stop_count": selected_fallthrough_budget_stop_count,
        "unattempted_route_count": total_unattempted_route_count,
        "mean_unattempted_route_count": _safe_div(
            total_unattempted_route_count,
            selected,
        ),
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
        "route_budget_limit_observations": 0,
        "route_budget_exhausted_count": 0,
        "route_budget_exhaustion_rate": 0.0,
        "selected_fallthrough_budget_stop_count": 0,
        "unattempted_route_count": 0,
        "mean_unattempted_route_count": 0.0,
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


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _triple_payloads(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if _looks_like_triple_payload(value):
            return (dict(value),)
        raw = value.get("triples", value.get("claim_triples"))
        return _triple_payloads(raw)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        triples = []
        for item in value:
            if isinstance(item, Mapping) and _looks_like_triple_payload(item):
                triples.append(dict(item))
        return tuple(triples)
    return ()


def _looks_like_triple_payload(value: Mapping[str, Any]) -> bool:
    return (
        value.get("subject") is not None
        and value.get("predicate") is not None
        and (value.get("object") is not None or value.get("object_text") is not None)
    )


def _triple_predicate(value: Mapping[str, Any]) -> str | None:
    predicate = value.get("predicate")
    if predicate is None:
        return None
    text = str(predicate).strip()
    return text or None


def _payload_claim_id(value: Mapping[str, Any], *, fallback: str) -> str:
    claim_id = value.get("claim_id")
    if claim_id is None:
        claim_id = value.get("id")
    if claim_id is None:
        return fallback
    text = str(claim_id).strip()
    return text or fallback


def _increment_count(counts: dict[str, int], value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    counts[text] = counts.get(text, 0) + 1


def _merge_counts(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for raw_key, raw_value in source.items():
        key = str(raw_key).strip()
        if not key:
            continue
        count = _non_negative_int(raw_value)
        if count is None:
            continue
        target[key] = target.get(key, 0) + count


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
    return to_jsonable(value)


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
