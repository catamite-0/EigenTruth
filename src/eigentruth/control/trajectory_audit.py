"""Trace-level hallucination audit for product control workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from eigentruth.control.action_audit import (
    ActionAuditSeverity,
    audit_action_requests,
)
from eigentruth.control.policy import ControlAction
from eigentruth.json_utils import to_jsonable


class TrajectoryHallucinationType(str, Enum):
    """Coarse hallucination types for multi-step product traces."""

    FACTUAL = "factual"
    REFERENTIAL = "referential"
    LOGICAL = "logical"
    PROCEDURAL = "procedural"
    SCOPE = "scope"


@dataclass(frozen=True)
class TrajectoryAuditIssue:
    """One trace-level hallucination or control-consistency finding."""

    code: str
    hallucination_type: TrajectoryHallucinationType | str
    severity: ActionAuditSeverity | str
    message: str
    location: str | None = None
    claim_ids: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        if not code:
            raise ValueError("trajectory audit issue code must be non-empty.")
        message = str(self.message).strip()
        if not message:
            raise ValueError("trajectory audit issue message must be non-empty.")
        hallucination_type = (
            self.hallucination_type
            if isinstance(self.hallucination_type, TrajectoryHallucinationType)
            else TrajectoryHallucinationType(str(self.hallucination_type))
        )
        severity = (
            self.severity
            if isinstance(self.severity, ActionAuditSeverity)
            else ActionAuditSeverity(str(self.severity))
        )
        location = None if self.location is None else str(self.location).strip() or None
        claim_ids = tuple(str(item).strip() for item in self.claim_ids if str(item).strip())
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "hallucination_type", hallucination_type)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "claim_ids", claim_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "code": self.code,
            "hallucination_type": self.hallucination_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "claim_ids": tuple(self.claim_ids),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrajectoryAuditIssue":
        """Build an issue from JSON-like data."""
        return cls(
            code=str(data["code"]),
            hallucination_type=str(data.get("hallucination_type", data.get("type"))),
            severity=str(data["severity"]),
            message=str(data["message"]),
            location=None if data.get("location") is None else str(data["location"]),
            claim_ids=tuple(_sequence(data.get("claim_ids", ()))),
            metadata=dict(_mapping(data.get("metadata"))),
        )


@dataclass(frozen=True)
class TrajectoryAuditReport:
    """JSON-ready report for trace-level hallucination taxonomy checks."""

    trace_id: str | None = None
    issues: Sequence[TrajectoryAuditIssue | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_id = None if self.trace_id is None else str(self.trace_id).strip() or None
        issues = tuple(_coerce_issue(issue) for issue in self.issues)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        """Return whether the report has no error-level trace issues."""
        return not any(issue.severity is ActionAuditSeverity.ERROR for issue in self.issues)

    def summary(self) -> dict[str, Any]:
        """Return compact telemetry counts for bounded traces and baselines."""
        counts_by_type: dict[str, int] = {}
        counts_by_severity: dict[str, int] = {}
        counts_by_code: dict[str, int] = {}
        cascade_count = 0
        for issue in self.issues:
            hallucination_type = issue.hallucination_type.value
            severity = issue.severity.value
            counts_by_type[hallucination_type] = counts_by_type.get(hallucination_type, 0) + 1
            counts_by_severity[severity] = counts_by_severity.get(severity, 0) + 1
            counts_by_code[issue.code] = counts_by_code.get(issue.code, 0) + 1
            if issue.metadata.get("cascade") is True:
                cascade_count += 1
        hallucination_types = tuple(
            item.value for item in TrajectoryHallucinationType if counts_by_type.get(item.value, 0) > 0
        )
        return {
            "available": True,
            "passed": self.passed,
            "trace_id": self.trace_id,
            "issue_count": len(self.issues),
            "error_count": counts_by_severity.get(ActionAuditSeverity.ERROR.value, 0),
            "warning_count": counts_by_severity.get(ActionAuditSeverity.WARNING.value, 0),
            "info_count": counts_by_severity.get(ActionAuditSeverity.INFO.value, 0),
            "cascade_count": cascade_count,
            "hallucination_types": hallucination_types,
            "counts_by_type": counts_by_type,
            "counts_by_severity": counts_by_severity,
            "counts_by_code": counts_by_code,
            "top_issues": tuple(issue.to_dict() for issue in self.issues[:8]),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "trace_id": self.trace_id,
            "issues": tuple(issue.to_dict() for issue in self.issues),
            "metadata": to_jsonable(dict(self.metadata)),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrajectoryAuditReport":
        """Build a report from JSON-like data."""
        return cls(
            trace_id=None if data.get("trace_id") is None else str(data["trace_id"]),
            issues=tuple(_sequence(data.get("issues", ()))),
            metadata=dict(_mapping(data.get("metadata"))),
        )


def audit_product_trace_trajectory(trace: Any) -> TrajectoryAuditReport:
    """Audit a ProductTrace-like payload for trajectory-level hallucinations.

    The audit is dependency-free and monitor-first. It checks structural
    consistency between claims, verifier outputs, risk decisions, action
    requests/results, and final answers, then labels findings with the five
    coarse trajectory hallucination categories used for agentic workflows.
    """
    payload = _trace_payload(trace)
    summaries = _mapping(payload.get("summaries"))
    issues: list[TrajectoryAuditIssue] = []
    issues.extend(_issues_from_action_audit(payload, summaries))
    issues.extend(_issues_from_action_execution(payload, summaries))
    issues.extend(_issues_from_verification_decision(payload))
    issues.extend(_issues_from_cascading_evidence(payload))
    return TrajectoryAuditReport(
        trace_id=_optional_string(payload.get("request_id")),
        issues=tuple(issues),
        metadata={
            "audit_version": 1,
            "claim_count": len(_sequence(payload.get("claims", ()))),
            "verification_result_count": len(_sequence(payload.get("verification_results", ()))),
            "action_count": len(_sequence(payload.get("actions", ()))),
            "action_result_count": len(_sequence(payload.get("action_results", ()))),
            "has_bounded_summaries": bool(summaries),
        },
    )


def _issues_from_action_audit(
    payload: Mapping[str, Any],
    summaries: Mapping[str, Any],
) -> tuple[TrajectoryAuditIssue, ...]:
    action_summary = _mapping(summaries.get("action_audit"))
    raw_issues = tuple(_sequence(action_summary.get("top_issues", ())))
    if not raw_issues:
        report = audit_action_requests(
            tuple(_sequence(payload.get("actions", ()))),
            decision=_optional_mapping(payload.get("risk_decision")),
            verification_plan=_optional_mapping(payload.get("verification_plan")),
        )
        raw_issues = tuple(issue.to_dict() for issue in report.issues)

    issues = []
    for raw_issue in raw_issues:
        issue = _mapping(raw_issue)
        if not issue:
            continue
        code = str(issue.get("code", "")).strip()
        if not code:
            continue
        hallucination_type = _ACTION_AUDIT_TYPE_BY_CODE.get(
            code,
            TrajectoryHallucinationType.PROCEDURAL,
        )
        action_index = issue.get("action_index")
        action = issue.get("action")
        issues.append(TrajectoryAuditIssue(
            code=code,
            hallucination_type=hallucination_type,
            severity=str(issue.get("severity", ActionAuditSeverity.WARNING.value)),
            message=str(issue.get("message", "action audit issue")),
            location=_action_location(action_index, action),
            claim_ids=tuple(_sequence(issue.get("claim_ids", ()))),
            metadata={
                "source": "action_audit",
                "source_issue": issue,
            },
        ))
    return tuple(issues)


def _issues_from_action_execution(
    payload: Mapping[str, Any],
    summaries: Mapping[str, Any],
) -> tuple[TrajectoryAuditIssue, ...]:
    summary = _mapping(summaries.get("action_execution"))
    if not summary:
        summary = _action_execution_summary(
            tuple(_sequence(payload.get("actions", ()))),
            tuple(_mapping(item) for item in _sequence(payload.get("action_results", ())) if isinstance(item, Mapping)),
        )
    issues = []
    missing_result_count = _non_negative_int(summary.get("missing_result_count")) or 0
    unexpected_result_count = _non_negative_int(summary.get("unexpected_result_count")) or 0
    request_id_mismatch_count = _non_negative_int(summary.get("request_id_mismatch_count")) or 0
    if missing_result_count > 0:
        issues.append(TrajectoryAuditIssue(
            code="missing_action_result",
            hallucination_type=TrajectoryHallucinationType.PROCEDURAL,
            severity=ActionAuditSeverity.WARNING,
            message="planned actions did not all produce action results",
            location="action_results",
            metadata={
                "source": "action_execution",
                "missing_result_count": missing_result_count,
                "missing_results_by_action": _mapping(
                    _mapping(summary.get("alignment")).get("missing_results_by_action")
                )
                or _mapping(summary.get("missing_results_by_action")),
            },
        ))
    if unexpected_result_count > 0:
        issues.append(TrajectoryAuditIssue(
            code="unexpected_action_result",
            hallucination_type=TrajectoryHallucinationType.SCOPE,
            severity=ActionAuditSeverity.WARNING,
            message="action results included actions that were not planned",
            location="action_results",
            metadata={
                "source": "action_execution",
                "unexpected_result_count": unexpected_result_count,
                "unexpected_results_by_action": _mapping(
                    _mapping(summary.get("alignment")).get("unexpected_results_by_action")
                )
                or _mapping(summary.get("unexpected_results_by_action")),
            },
        ))
    if request_id_mismatch_count > 0:
        issues.append(TrajectoryAuditIssue(
            code="action_result_request_id_mismatch",
            hallucination_type=TrajectoryHallucinationType.REFERENTIAL,
            severity=ActionAuditSeverity.WARNING,
            message="action result request ids did not match planned action request ids",
            location="action_results",
            metadata={
                "source": "action_execution",
                "request_id_mismatch_count": request_id_mismatch_count,
                "alignment": _mapping(summary.get("alignment")),
            },
        ))

    for index, result in enumerate(
        _mapping(item) for item in _sequence(payload.get("action_results", ())) if isinstance(item, Mapping)
    ):
        status = str(result.get("status", "")).strip()
        if status in {"failed", "timed_out"}:
            issues.append(TrajectoryAuditIssue(
                code=f"action_execution_{status}",
                hallucination_type=TrajectoryHallucinationType.PROCEDURAL,
                severity=ActionAuditSeverity.ERROR,
                message=f"action execution ended with status {status!r}",
                location=f"action_results[{index}]",
                metadata={
                    "source": "action_execution",
                    "action": result.get("action"),
                    "request_id": result.get("request_id"),
                    "error": result.get("error"),
                },
            ))
    return tuple(issues)


def _issues_from_cascading_evidence(payload: Mapping[str, Any]) -> tuple[TrajectoryAuditIssue, ...]:
    """Detect upstream action evidence failures that propagate downstream."""
    action_results = tuple(
        _mapping(item) for item in _sequence(payload.get("action_results", ())) if isinstance(item, Mapping)
    )
    if not action_results:
        return ()
    claims = tuple(_mapping(item) for item in _sequence(payload.get("claims", ())) if isinstance(item, Mapping))
    results = tuple(
        _mapping(item) for item in _sequence(payload.get("verification_results", ())) if isinstance(item, Mapping)
    )
    claim_ids = _claim_ids(claims)
    decision = _mapping(payload.get("risk_decision"))
    final_answer = _mapping(payload.get("final_answer"))
    answered_or_accepted = _answered_or_accepted(decision, final_answer)
    status_claim_ids = _verification_status_claim_ids(results, claim_ids)
    unsupported_claim_ids = tuple(
        tuple(status_claim_ids.get("insufficient_evidence", ()))
        + tuple(status_claim_ids.get("error", ()))
    )
    verified_claim_ids = set(claim_id for ids in status_claim_ids.values() for claim_id in ids)
    missing_claim_ids = tuple(claim_id for claim_id in claim_ids if claim_id not in verified_claim_ids)

    failed_by_request_id: dict[str, dict[str, Any]] = {}
    failed_results: list[dict[str, Any]] = []
    empty_retrieval_by_request_id: dict[str, dict[str, Any]] = {}
    empty_retrieval_results: list[dict[str, Any]] = []
    for index, result in enumerate(action_results):
        action = _action_name(result.get("action"))
        status = str(result.get("status", "")).strip()
        request_id = _optional_string(result.get("request_id"))
        indexed = {
            "index": index,
            "action": action,
            "status": status,
            "request_id": request_id,
        }
        if status in _FAILED_ACTION_STATUSES and action in _EVIDENCE_ACTIONS:
            failed_results.append(indexed)
            if request_id is not None:
                failed_by_request_id[request_id] = indexed
        if action == ControlAction.RETRIEVE.value and status in _COMPLETED_RETRIEVAL_STATUSES:
            hit_count = _retrieval_hit_count(result)
            if hit_count == 0:
                indexed = {
                    **indexed,
                    "hit_count": hit_count,
                }
                empty_retrieval_results.append(indexed)
                if request_id is not None:
                    empty_retrieval_by_request_id[request_id] = indexed

    issues: list[TrajectoryAuditIssue] = []
    downstream_claim_ids = tuple(dict.fromkeys(unsupported_claim_ids + missing_claim_ids))
    if answered_or_accepted and failed_results:
        issues.append(TrajectoryAuditIssue(
            code="accepted_after_failed_upstream_action",
            hallucination_type=TrajectoryHallucinationType.PROCEDURAL,
            severity=ActionAuditSeverity.ERROR,
            message="the trace answered after an evidence-bearing upstream action failed",
            location="action_results",
            claim_ids=downstream_claim_ids,
            metadata={
                "source": "cascading_evidence",
                "cascade": True,
                "failed_actions": tuple(failed_results[:8]),
                "unsupported_claim_ids": unsupported_claim_ids,
                "missing_claim_ids": missing_claim_ids,
            },
        ))
    if answered_or_accepted and empty_retrieval_results and (unsupported_claim_ids or missing_claim_ids):
        issues.append(TrajectoryAuditIssue(
            code="accepted_after_empty_retrieval",
            hallucination_type=TrajectoryHallucinationType.REFERENTIAL,
            severity=ActionAuditSeverity.ERROR,
            message="the trace answered after retrieval returned no evidence for unresolved claims",
            location="action_results",
            claim_ids=downstream_claim_ids,
            metadata={
                "source": "cascading_evidence",
                "cascade": True,
                "empty_retrieval_results": tuple(empty_retrieval_results[:8]),
                "unsupported_claim_ids": unsupported_claim_ids,
                "missing_claim_ids": missing_claim_ids,
            },
        ))

    for index, result in enumerate(results):
        status = str(result.get("status", "")).strip()
        if status != "supported":
            continue
        claim_id = _verification_result_claim_id(result, index=index, claim_ids=claim_ids)
        referenced_request_ids = tuple(
            request_id
            for request_id in _referenced_request_ids(result)
            if request_id in failed_by_request_id or request_id in empty_retrieval_by_request_id
        )
        for request_id in referenced_request_ids:
            if request_id in failed_by_request_id:
                issues.append(TrajectoryAuditIssue(
                    code="supported_claim_from_failed_action",
                    hallucination_type=TrajectoryHallucinationType.REFERENTIAL,
                    severity=ActionAuditSeverity.ERROR,
                    message="a supported claim referenced a failed evidence-bearing action",
                    location=f"verification_results[{index}]",
                    claim_ids=(claim_id,),
                    metadata={
                        "source": "cascading_evidence",
                        "cascade": True,
                        "request_id": request_id,
                        "upstream_action": failed_by_request_id[request_id],
                    },
                ))
            elif request_id in empty_retrieval_by_request_id:
                issues.append(TrajectoryAuditIssue(
                    code="supported_claim_from_empty_retrieval",
                    hallucination_type=TrajectoryHallucinationType.REFERENTIAL,
                    severity=ActionAuditSeverity.ERROR,
                    message="a supported claim referenced a retrieval action with no evidence hits",
                    location=f"verification_results[{index}]",
                    claim_ids=(claim_id,),
                    metadata={
                        "source": "cascading_evidence",
                        "cascade": True,
                        "request_id": request_id,
                        "upstream_action": empty_retrieval_by_request_id[request_id],
                    },
                ))
    return tuple(issues)


def _answered_or_accepted(
    decision: Mapping[str, Any],
    final_answer: Mapping[str, Any],
) -> bool:
    decision_action = _action_name(decision.get("action"))
    final_action = _action_name(final_answer.get("action"))
    final_status = str(final_answer.get("status", "")).strip()
    return (
        decision_action == ControlAction.ACCEPT.value
        or final_action == ControlAction.ACCEPT.value
        or final_status == "answered"
        or final_answer.get("answerable") is True
    )


def _verification_status_claim_ids(
    results: Sequence[Mapping[str, Any]],
    claim_ids: Sequence[str],
) -> dict[str, list[str]]:
    status_claim_ids: dict[str, list[str]] = {}
    for index, result in enumerate(results):
        status = str(result.get("status", "")).strip()
        if not status:
            continue
        claim_id = _verification_result_claim_id(result, index=index, claim_ids=claim_ids)
        status_claim_ids.setdefault(status, []).append(claim_id)
    return status_claim_ids


def _retrieval_hit_count(result: Mapping[str, Any]) -> int:
    output = _mapping(result.get("output"))
    direct_hits = _sequence(output.get("hits", ()))
    hits_by_query = _sequence(output.get("hits_by_query", ()))
    if direct_hits:
        return sum(1 for item in direct_hits if isinstance(item, Mapping) or item)
    total = 0
    for item in hits_by_query:
        query_result = _mapping(item)
        if not query_result:
            continue
        hits = _sequence(query_result.get("hits", ()))
        total += sum(1 for hit in hits if isinstance(hit, Mapping) or hit)
    return total


def _referenced_request_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    request_ids: list[str] = []
    for value in _request_id_values(payload):
        request_id = _optional_string(value)
        if request_id is not None:
            request_ids.append(request_id)
    return tuple(dict.fromkeys(request_ids))


def _request_id_values(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        values: list[Any] = []
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in _REQUEST_ID_KEYS or (
                key_text.endswith("_request_id") and "fingerprint" not in key_text
            ):
                values.append(item)
            if key_text in {"metadata", "evidence", "source", "sources", "references", "trace"}:
                values.extend(_request_id_values(item))
        return tuple(values)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        for item in value:
            values.extend(_request_id_values(item))
        return tuple(values)
    return ()


def _issues_from_verification_decision(payload: Mapping[str, Any]) -> tuple[TrajectoryAuditIssue, ...]:
    claims = tuple(_mapping(item) for item in _sequence(payload.get("claims", ())) if isinstance(item, Mapping))
    results = tuple(
        _mapping(item) for item in _sequence(payload.get("verification_results", ())) if isinstance(item, Mapping)
    )
    decision = _mapping(payload.get("risk_decision"))
    final_answer = _mapping(payload.get("final_answer"))
    claim_ids = _claim_ids(claims)
    known_claim_ids = set(claim_ids)
    issues: list[TrajectoryAuditIssue] = []
    status_claim_ids: dict[str, list[str]] = {}
    for index, result in enumerate(results):
        status = str(result.get("status", "")).strip()
        if not status:
            continue
        claim_id = _verification_result_claim_id(result, index=index, claim_ids=claim_ids)
        status_claim_ids.setdefault(status, []).append(claim_id)
        if known_claim_ids:
            explicit_claim_id = _explicit_verification_result_claim_id(result)
            if explicit_claim_id is not None and explicit_claim_id not in known_claim_ids:
                issues.append(TrajectoryAuditIssue(
                    code="verification_result_unknown_claim_id",
                    hallucination_type=TrajectoryHallucinationType.SCOPE,
                    severity=ActionAuditSeverity.WARNING,
                    message="verification result referenced a claim id outside the trace claims",
                    location=f"verification_results[{index}]",
                    claim_ids=(explicit_claim_id,),
                    metadata={
                        "known_claim_count": len(known_claim_ids),
                        "status": status,
                    },
                ))

    refuted_claim_ids = tuple(status_claim_ids.get("refuted", ()))
    unsupported_claim_ids = tuple(
        tuple(status_claim_ids.get("insufficient_evidence", ()))
        + tuple(status_claim_ids.get("error", ()))
    )
    verified_claim_ids = tuple(
        claim_id
        for ids in status_claim_ids.values()
        for claim_id in ids
    )
    missing_claim_ids = tuple(
        claim_id for claim_id in claim_ids if claim_id not in set(verified_claim_ids)
    )
    decision_action = _action_name(decision.get("action"))
    final_action = _action_name(final_answer.get("action"))
    final_status = str(final_answer.get("status", "")).strip()
    answered_or_accepted = (
        decision_action == ControlAction.ACCEPT.value
        or final_action == ControlAction.ACCEPT.value
        or final_status == "answered"
        or final_answer.get("answerable") is True
    )
    if refuted_claim_ids and answered_or_accepted:
        issues.append(TrajectoryAuditIssue(
            code="accepted_refuted_claim",
            hallucination_type=TrajectoryHallucinationType.FACTUAL,
            severity=ActionAuditSeverity.ERROR,
            message="a refuted claim was allowed into an accepted or answered trace",
            location="risk_decision",
            claim_ids=refuted_claim_ids,
            metadata={
                "decision_action": decision_action,
                "final_action": final_action,
                "final_status": final_status,
            },
        ))
    if refuted_claim_ids and decision_action == ControlAction.ACCEPT.value:
        issues.append(TrajectoryAuditIssue(
            code="decision_conflicts_with_refutation",
            hallucination_type=TrajectoryHallucinationType.LOGICAL,
            severity=ActionAuditSeverity.ERROR,
            message="risk decision accepted claims that verifier output refuted",
            location="risk_decision",
            claim_ids=refuted_claim_ids,
            metadata={"decision_action": decision_action},
        ))
    if unsupported_claim_ids and answered_or_accepted:
        issues.append(TrajectoryAuditIssue(
            code="accepted_unsupported_claim",
            hallucination_type=TrajectoryHallucinationType.FACTUAL,
            severity=ActionAuditSeverity.ERROR,
            message="an unsupported or errored claim was allowed into an accepted or answered trace",
            location="risk_decision",
            claim_ids=unsupported_claim_ids,
            metadata={
                "decision_action": decision_action,
                "final_action": final_action,
                "final_status": final_status,
            },
        ))
    if missing_claim_ids and answered_or_accepted:
        issues.append(TrajectoryAuditIssue(
            code="accepted_unverified_claims",
            hallucination_type=TrajectoryHallucinationType.FACTUAL,
            severity=ActionAuditSeverity.WARNING,
            message="the trace was accepted or answered with claims lacking verifier results",
            location="verification_results",
            claim_ids=missing_claim_ids,
            metadata={
                "claim_count": len(claim_ids),
                "verification_result_count": len(results),
            },
        ))
    if final_status == "answered" and decision_action in _NON_ANSWERING_ACTIONS:
        issues.append(TrajectoryAuditIssue(
            code="final_answer_conflicts_with_decision",
            hallucination_type=TrajectoryHallucinationType.LOGICAL,
            severity=ActionAuditSeverity.ERROR,
            message="final answer is answered while risk decision requested a non-answer action",
            location="final_answer",
            metadata={
                "decision_action": decision_action,
                "final_status": final_status,
            },
        ))
    return tuple(issues)


def _trace_payload(trace: Any) -> dict[str, Any]:
    if hasattr(trace, "to_dict"):
        raw = trace.to_dict()
    elif isinstance(trace, Mapping):
        raw = trace
    else:
        raise ValueError("trace must be a ProductTrace-like object or JSON mapping.")
    payload = to_jsonable(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("trace payload must serialize to a JSON object.")
    return dict(payload)


def _action_execution_summary(
    actions: Sequence[Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    planned_counts = _action_counts(actions)
    result_counts = _action_counts(results)
    missing_by_action = {
        action: planned_count - result_counts.get(action, 0)
        for action, planned_count in planned_counts.items()
        if planned_count > result_counts.get(action, 0)
    }
    unexpected_by_action = {
        action: result_count - planned_counts.get(action, 0)
        for action, result_count in result_counts.items()
        if result_count > planned_counts.get(action, 0)
    }
    planned_request_ids = tuple(dict.fromkeys(_request_ids(actions)))
    result_request_ids = tuple(dict.fromkeys(_request_ids(results)))
    result_request_id_set = set(result_request_ids)
    planned_request_id_set = set(planned_request_ids)
    missing_request_ids = tuple(
        request_id for request_id in planned_request_ids if request_id not in result_request_id_set
    )
    unexpected_request_ids = tuple(
        request_id for request_id in result_request_ids if request_id not in planned_request_id_set
    )
    missing_result_count = sum(missing_by_action.values())
    unexpected_result_count = sum(unexpected_by_action.values())
    request_id_mismatch_count = len(missing_request_ids) + len(unexpected_request_ids)
    return {
        "missing_result_count": missing_result_count,
        "unexpected_result_count": unexpected_result_count,
        "request_id_mismatch_count": request_id_mismatch_count,
        "alignment": {
            "missing_results_by_action": missing_by_action,
            "unexpected_results_by_action": unexpected_by_action,
            "missing_request_ids": missing_request_ids[:8],
            "unexpected_request_ids": unexpected_request_ids[:8],
        },
    }


def _action_counts(items: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        action = _action_name(item.get("action")) if isinstance(item, Mapping) else _action_name(item)
        if action is not None:
            counts[action] = counts.get(action, 0) + 1
    return counts


def _request_ids(items: Sequence[Any]) -> tuple[str, ...]:
    request_ids = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        request_id = _optional_string(item.get("request_id"))
        if request_id is not None:
            request_ids.append(request_id)
    return tuple(request_ids)


def _claim_ids(claims: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    claim_ids = []
    for index, claim in enumerate(claims):
        claim_id = _optional_string(claim.get("claim_id", claim.get("id")))
        claim_ids.append(claim_id or f"c{index + 1}")
    return tuple(claim_ids)


def _verification_result_claim_id(
    result: Mapping[str, Any],
    *,
    index: int,
    claim_ids: Sequence[str],
) -> str:
    explicit = _explicit_verification_result_claim_id(result)
    if explicit is not None:
        return explicit
    if index < len(claim_ids):
        return claim_ids[index]
    return f"r{index + 1}"


def _explicit_verification_result_claim_id(result: Mapping[str, Any]) -> str | None:
    direct = _optional_string(result.get("claim_id", result.get("id")))
    if direct is not None:
        return direct
    metadata = _mapping(result.get("metadata"))
    claim_id = _optional_string(metadata.get("claim_id"))
    if claim_id is not None:
        return claim_id
    claim_ids = tuple(_sequence(metadata.get("claim_ids", ())))
    if len(claim_ids) == 1:
        return _optional_string(claim_ids[0])
    return None


def _action_location(action_index: Any, action: Any) -> str | None:
    if action_index is not None:
        try:
            return f"actions[{int(action_index)}]"
        except (TypeError, ValueError):
            return "actions"
    action_name = _action_name(action)
    if action_name is not None:
        return f"actions.{action_name}"
    return None


def _coerce_issue(issue: TrajectoryAuditIssue | Mapping[str, Any]) -> TrajectoryAuditIssue:
    if isinstance(issue, TrajectoryAuditIssue):
        return issue
    return TrajectoryAuditIssue.from_dict(issue)


def _action_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, ControlAction):
        return value.value
    text = str(value).strip()
    return text or None


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _non_negative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


_ACTION_AUDIT_TYPE_BY_CODE = {
    "unexpected_action_for_decision": TrajectoryHallucinationType.SCOPE,
    "unknown_claim_id": TrajectoryHallucinationType.SCOPE,
    "missing_plan_retrieval_query": TrajectoryHallucinationType.REFERENTIAL,
}

_NON_ANSWERING_ACTIONS = {
    ControlAction.RETRIEVE.value,
    ControlAction.REWRITE.value,
    ControlAction.STEER_REGENERATE.value,
    ControlAction.EXECUTE_TOOL.value,
    ControlAction.ABSTAIN.value,
    ControlAction.CLARIFY.value,
}

_EVIDENCE_ACTIONS = {
    ControlAction.RETRIEVE.value,
    ControlAction.EXECUTE_TOOL.value,
}

_FAILED_ACTION_STATUSES = {"failed", "timed_out"}

_COMPLETED_RETRIEVAL_STATUSES = {"succeeded", "dry_run"}

_REQUEST_ID_KEYS = {
    "request_id",
    "action_request_id",
    "source_request_id",
    "retrieval_request_id",
    "tool_request_id",
    "evidence_request_id",
}
