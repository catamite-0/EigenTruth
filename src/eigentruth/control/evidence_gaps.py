"""Plan next evidence work from blocked release and runtime reports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import to_jsonable

_MISSING_METRICS_RE = re.compile(r"metrics are incomplete:\s*(?P<metrics>.+)$")


@dataclass(frozen=True)
class EvidenceGapAction:
    """One recommended next action for closing one or more evidence gaps."""

    action_id: str
    title: str
    action_type: str
    priority: int
    rationale: str
    evidence_routes: tuple[str, ...] = ()
    suggested_commands: tuple[str, ...] = ()
    source_gap_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", str(self.action_id))
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "action_type", str(self.action_type))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "rationale", str(self.rationale))
        object.__setattr__(
            self,
            "evidence_routes",
            tuple(str(item) for item in self.evidence_routes if str(item)),
        )
        object.__setattr__(
            self,
            "suggested_commands",
            tuple(str(item) for item in self.suggested_commands if str(item)),
        )
        object.__setattr__(
            self,
            "source_gap_ids",
            tuple(str(item) for item in self.source_gap_ids if str(item)),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready action record."""
        return {
            "action_id": self.action_id,
            "title": self.title,
            "action_type": self.action_type,
            "priority": self.priority,
            "rationale": self.rationale,
            "evidence_routes": self.evidence_routes,
            "suggested_commands": self.suggested_commands,
            "source_gap_ids": self.source_gap_ids,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceGapAction":
        """Build an action from JSON-like data."""
        return cls(
            action_id=str(data["action_id"]),
            title=str(data["title"]),
            action_type=str(data["action_type"]),
            priority=int(data.get("priority", 0)),
            rationale=str(data.get("rationale", "")),
            evidence_routes=_string_tuple(data.get("evidence_routes", ())),
            suggested_commands=_string_tuple(data.get("suggested_commands", ())),
            source_gap_ids=_string_tuple(data.get("source_gap_ids", ())),
            metadata=_mapping(data.get("metadata")),
        )


@dataclass(frozen=True)
class EvidenceGap:
    """One blocker or missing evidence item surfaced by an offline report."""

    gap_id: str
    gate: str
    status: str
    reason: str
    root_cause: str
    recommended_action_ids: tuple[str, ...]
    missing_metrics: tuple[str, ...] = ()
    source: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gap_id", str(self.gap_id))
        object.__setattr__(self, "gate", str(self.gate))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "root_cause", str(self.root_cause))
        object.__setattr__(
            self,
            "recommended_action_ids",
            tuple(str(item) for item in self.recommended_action_ids if str(item)),
        )
        object.__setattr__(
            self,
            "missing_metrics",
            tuple(str(item) for item in self.missing_metrics if str(item)),
        )
        object.__setattr__(self, "source", dict(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready gap record."""
        return {
            "gap_id": self.gap_id,
            "gate": self.gate,
            "status": self.status,
            "reason": self.reason,
            "root_cause": self.root_cause,
            "recommended_action_ids": self.recommended_action_ids,
            "missing_metrics": self.missing_metrics,
            "source": to_jsonable(self.source),
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceGap":
        """Build a gap from JSON-like data."""
        return cls(
            gap_id=str(data["gap_id"]),
            gate=str(data["gate"]),
            status=str(data.get("status", "blocked")),
            reason=str(data.get("reason", "")),
            root_cause=str(data.get("root_cause", "unknown")),
            recommended_action_ids=_string_tuple(data.get("recommended_action_ids", ())),
            missing_metrics=_string_tuple(data.get("missing_metrics", ())),
            source=_mapping(data.get("source")),
            metadata=_mapping(data.get("metadata")),
        )


@dataclass(frozen=True)
class EvidenceGapPlan:
    """Structured next-work plan derived from failed release evidence."""

    status: str
    gaps: tuple[EvidenceGap, ...] = ()
    actions: tuple[EvidenceGapAction, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    source_workflow: str | None = None
    source_status: str | None = None
    source_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    workflow: str = "evidence_gap_plan"
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "gaps", tuple(self.gaps))
        object.__setattr__(
            self,
            "actions",
            tuple(sorted(self.actions, key=lambda item: (-item.priority, item.action_id))),
        )
        object.__setattr__(self, "summary", dict(self.summary))
        object.__setattr__(
            self,
            "source_workflow",
            None if self.source_workflow is None else str(self.source_workflow),
        )
        object.__setattr__(
            self,
            "source_status",
            None if self.source_status is None else str(self.source_status),
        )
        object.__setattr__(
            self,
            "source_path",
            None if self.source_path is None else str(self.source_path),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "workflow", str(self.workflow))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready plan payload."""
        return {
            "schema_version": self.schema_version,
            "workflow": self.workflow,
            "status": self.status,
            "source_workflow": self.source_workflow,
            "source_status": self.source_status,
            "source_path": self.source_path,
            "summary": to_jsonable(self.summary),
            "gaps": tuple(gap.to_dict() for gap in self.gaps),
            "actions": tuple(action.to_dict() for action in self.actions),
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceGapPlan":
        """Build a plan from JSON-like data."""
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            workflow=str(data.get("workflow", "evidence_gap_plan")),
            status=str(data.get("status", "needs_evidence")),
            source_workflow=_optional_str(data.get("source_workflow")),
            source_status=_optional_str(data.get("source_status")),
            source_path=_optional_str(data.get("source_path")),
            summary=_mapping(data.get("summary")),
            gaps=tuple(EvidenceGap.from_dict(item) for item in _mapping_sequence(data.get("gaps", ()))),
            actions=tuple(
                EvidenceGapAction.from_dict(item)
                for item in _mapping_sequence(data.get("actions", ()))
            ),
            metadata=_mapping(data.get("metadata")),
        )


def plan_evidence_gaps_from_release_candidate(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceGapPlan:
    """Convert a release-candidate comparison or workflow report into next evidence work.

    The output is a planning artifact only. It does not prove that evidence exists,
    promote a route, or modify runtime control defaults.
    """
    comparison = _release_candidate_comparison(payload)
    decision = _mapping(comparison.get("decision"))
    source_workflow = _optional_str(payload.get("workflow") or comparison.get("workflow"))
    source_status = _optional_str(decision.get("status") or comparison.get("status"))
    if comparison.get("workflow") == "frontier_release_evidence_comparison":
        blocker_records = _frontier_release_evidence_blocking_records(comparison, decision)
    else:
        blocker_records = _blocking_records(decision)

    gaps: list[EvidenceGap] = []
    action_sources: dict[str, set[str]] = {}
    action_templates: dict[str, EvidenceGapAction] = {}
    for gap_index, blocker in enumerate(blocker_records, start=1):
        gate = blocker["gate"]
        gate_status = blocker["status"]
        blocker_metadata = _mapping(blocker.get("metadata"))
        for reason_index, reason in enumerate(blocker["reasons"], start=1):
            missing_metrics = _extract_missing_metrics(reason)
            kind = _classify_gap(gate, reason, missing_metrics=missing_metrics)
            action = _action_template(kind, gate=gate, reason=reason)
            gap_id = f"gap-{gap_index:03d}-{reason_index:02d}"
            action_sources.setdefault(action.action_id, set()).add(gap_id)
            action_templates[action.action_id] = action
            gaps.append(
                EvidenceGap(
                    gap_id=gap_id,
                    gate=gate,
                    status=gate_status,
                    reason=reason,
                    root_cause=kind["root_cause"],
                    recommended_action_ids=(action.action_id,),
                    missing_metrics=missing_metrics,
                    source={
                        "decision_status": source_status,
                        "blocker_index": gap_index,
                        "reason_index": reason_index,
                    },
                    metadata={
                        "evidence_kind": kind["evidence_kind"],
                        "research_axis": kind["research_axis"],
                        **blocker_metadata,
                    },
                )
            )

    actions = tuple(
        _replace_action_sources(template, source_ids=tuple(sorted(action_sources[action_id])))
        for action_id, template in action_templates.items()
    )
    summary = _summary(gaps, actions, decision=decision)
    return EvidenceGapPlan(
        status="ready" if not gaps else "needs_evidence",
        source_workflow=source_workflow,
        source_status=source_status,
        source_path=None if source_path is None else str(source_path),
        gaps=tuple(gaps),
        actions=actions,
        summary=summary,
        metadata={} if metadata is None else dict(metadata),
    )


def _release_candidate_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("workflow") == "release_candidate_registry_workflow":
        comparison = payload.get("release_candidate_comparison")
        if not isinstance(comparison, Mapping):
            raise ValueError("registry workflow payload is missing release_candidate_comparison.")
        return dict(comparison)
    return dict(payload)


def _blocking_records(decision: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = decision.get("blocking_reasons", ())
    records: list[dict[str, Any]] = []
    if isinstance(raw, str):
        return ({"gate": "release_candidate", "status": "blocked", "reasons": (raw,)},)
    if not isinstance(raw, Sequence):
        return ()
    for item in raw:
        if isinstance(item, Mapping):
            reasons = _string_tuple(item.get("reasons", ()))
            if not reasons and item.get("reason") is not None:
                reasons = (str(item["reason"]),)
            reasons = _drop_redundant_status_reasons(reasons)
            records.append({
                "gate": str(item.get("gate") or "release_candidate"),
                "status": str(item.get("status") or "blocked"),
                "reasons": reasons,
            })
        elif item is not None:
            records.append({
                "gate": "release_candidate",
                "status": "blocked",
                "reasons": (str(item),),
            })
    return tuple(record for record in records if record["reasons"])


def _frontier_release_evidence_blocking_records(
    payload: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw_reasons = list(_string_tuple(decision.get("blocking_reasons", ())))
    if not raw_reasons:
        raw_reasons = list(_nested_frontier_blocking_reasons(payload))
    raw_reasons = list(_drop_redundant_status_reasons(raw_reasons))
    abstention_metadata = _frontier_abstention_metadata(payload)
    multiple_testing_metadata = _frontier_multiple_testing_metadata(payload)
    citation_batch_metadata = _frontier_citation_batch_metadata(payload)
    frontier_rerun_rollup_metadata = _frontier_rerun_rollup_metadata(payload)
    records: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    track_specs = (
        (
            "verifier_stability",
            "verifier_track_status",
            ("verifier_stability",),
        ),
        (
            "abstention_stability",
            "abstention_track_status",
            ("abstention_stability", "participation-gate", "participation gate"),
        ),
        (
            "detectability_taxonomy",
            "detectability_track_status",
            ("detectability_taxonomy", "entrenched_false_rate"),
        ),
        (
            "frontier_multiple_testing",
            "multiple_testing_track_status",
            (
                "multiple_testing",
                "multiple-testing",
                "multiple testing",
                "family-wise",
                "familywise",
                "truthfulqa_frontier_workflow",
            ),
        ),
        (
            "citation_batch_evidence",
            "citation_batch_track_status",
            (
                "citation_batch",
                "citation batch",
                "citation_search_batch_evidence_rollup",
                "batch_coverage",
                "unresolved evidence batch",
            ),
        ),
        (
            "frontier_rerun_rollup_evidence",
            "frontier_rerun_rollup_track_status",
            (
                "frontier_rerun_rollup",
                "frontier rerun rollup",
                "frontier-rerun-rollup",
                "frontier_stability_evidence_rerun_rollup",
                "frontier_abstention_evidence_rerun_rollup",
                "frontier_detectability_evidence_rerun_rollup",
                "frontier_multiple_testing_rerun_rollup",
                "rerun rollup",
            ),
        ),
    )
    for gate, status_key, patterns in track_specs:
        status = _optional_str(decision.get(status_key))
        if status != "blocked":
            continue
        reasons = []
        for index, reason in enumerate(raw_reasons):
            reason_lower = reason.lower()
            if any(pattern in reason_lower for pattern in patterns):
                used_indices.add(index)
                reasons.append(reason)
        if not reasons:
            reasons.append(f"frontier release evidence {status_key} is blocked")
        record = {
            "gate": gate,
            "status": "blocked",
            "reasons": tuple(reasons),
        }
        if gate == "abstention_stability" and abstention_metadata:
            record["metadata"] = abstention_metadata
        if gate == "frontier_multiple_testing" and multiple_testing_metadata:
            record["metadata"] = multiple_testing_metadata
        if gate == "citation_batch_evidence" and citation_batch_metadata:
            record["metadata"] = citation_batch_metadata
        if gate == "frontier_rerun_rollup_evidence" and frontier_rerun_rollup_metadata:
            record["metadata"] = frontier_rerun_rollup_metadata
        records.append(record)
    remaining_reasons = tuple(
        reason for index, reason in enumerate(raw_reasons) if index not in used_indices
    )
    if remaining_reasons:
        records.append({
            "gate": "frontier_release_evidence",
            "status": _optional_str(decision.get("status")) or "blocked",
            "reasons": remaining_reasons,
        })
    if not records and _optional_str(decision.get("status")) not in {None, "promote"}:
        records.append({
            "gate": "frontier_release_evidence",
            "status": _optional_str(decision.get("status")) or "blocked",
            "reasons": ("frontier release evidence decision is blocked",),
        })
    return tuple(records)


def _drop_redundant_status_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    """Remove summary-only blocked-status reasons when detailed blockers exist."""
    normalized = tuple(str(reason) for reason in reasons if str(reason))
    if len(normalized) <= 1:
        return normalized
    detailed = tuple(reason for reason in normalized if not _is_summary_status_reason(reason))
    return detailed or normalized


def _is_summary_status_reason(reason: str) -> bool:
    text = reason.lower()
    if "_track_status" in text or " track status " in text:
        return False
    if "evidence blocked" in text:
        return False
    return (
        "decision status is 'blocked'" in text
        or "decision status is blocked" in text
        or "decision is blocked" in text
        or re.match(r"^[a-z0-9 _-]+ blocked \d+ metric\(s\)", text) is not None
        or re.match(r"^[a-z0-9 _-]+ status is '?blocked'?", text) is not None
    )


def _nested_frontier_blocking_reasons(payload: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    for key in (
        "run_decisions",
        "multiple_testing_decisions",
        "citation_batch_decisions",
        "frontier_rerun_rollup_decisions",
    ):
        for item in _mapping_sequence(payload.get(key, ())):
            reasons.extend(_string_tuple(item.get("blocking_reasons", ())))
            for nested_key in (
                "verifier_decision",
                "abstention_decision",
                "detectability_decision",
            ):
                nested = item.get(nested_key)
                if isinstance(nested, Mapping):
                    reasons.extend(_string_tuple(nested.get("blocking_reasons", ())))
    return tuple(reason for reason in reasons if reason)


def _frontier_abstention_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    blocked_runs = []
    for decision in _mapping_sequence(payload.get("run_decisions", ())):
        abstention = _mapping(decision.get("abstention_decision"))
        if abstention.get("status") != "blocked":
            continue
        metrics = _mapping(abstention.get("metrics"))
        blocked_runs.append({
            "run": _optional_str(decision.get("name")) or _optional_str(abstention.get("name")),
            "conditional_correctness_lower_bound_mean": metrics.get(
                "conditional_correctness_lower_bound_mean"
            ),
            "empirical_abstention_rate_mean": metrics.get("empirical_abstention_rate_mean"),
            "release_gate_pass_seed_rate": metrics.get("release_gate_pass_seed_rate"),
            "stable_recommended_score_name": metrics.get("stable_recommended_score_name"),
            "supervised_feasibility_target_passed": metrics.get(
                "supervised_feasibility_target_passed"
            ),
        })
    if not blocked_runs:
        return {}
    return {"abstention_blocked_runs": tuple(blocked_runs)}


def _frontier_rerun_rollup_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    evidence_summary = _mapping(payload.get("evidence_summary"))
    blocked_rollups: list[dict[str, Any]] = []
    for decision in _mapping_sequence(payload.get("frontier_rerun_rollup_decisions", ())):
        if decision.get("status") == "promote":
            continue
        metrics = _mapping(decision.get("metrics"))
        blocked_rollups.append(
            _frontier_rerun_rollup_decision_metadata(decision, metrics=metrics)
        )
    metadata = {
        "frontier_rerun_rollup_names": _string_tuple(
            evidence_summary.get("frontier_rerun_rollup_names", ())
        ),
        "frontier_rerun_rollup_blocked_names": _string_tuple(
            evidence_summary.get("frontier_rerun_rollup_blocked_names", ())
        ),
        "frontier_rerun_rollup_workflows": _string_tuple(
            evidence_summary.get("frontier_rerun_rollup_workflows", ())
        ),
        "frontier_rerun_rollup_tracks": _string_tuple(
            evidence_summary.get("frontier_rerun_rollup_tracks", ())
        ),
        "frontier_rerun_rollup_candidate_count": evidence_summary.get(
            "frontier_rerun_rollup_candidate_count"
        ),
        "frontier_rerun_rollup_observed_report_count": evidence_summary.get(
            "frontier_rerun_rollup_observed_report_count"
        ),
        "frontier_rerun_rollup_missing_report_count": evidence_summary.get(
            "frontier_rerun_rollup_missing_report_count"
        ),
        "frontier_rerun_rollup_invalid_report_count": evidence_summary.get(
            "frontier_rerun_rollup_invalid_report_count"
        ),
        "frontier_rerun_rollup_blocked_candidate_count": evidence_summary.get(
            "frontier_rerun_rollup_blocked_candidate_count"
        ),
        "frontier_rerun_rollup_promotion_ready_count": evidence_summary.get(
            "frontier_rerun_rollup_promotion_ready_count"
        ),
        "frontier_rerun_rollup_blocked_rollups": tuple(blocked_rollups),
    }
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != () and value != []
    }


def _frontier_rerun_rollup_decision_metadata(
    decision: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": _optional_str(decision.get("name")) or "",
        "status": _optional_str(decision.get("status")) or "blocked",
        "workflow": _optional_str(metrics.get("workflow")),
        "track": _optional_str(metrics.get("track")),
        "candidate_count": metrics.get("candidate_count"),
        "observed_report_count": metrics.get("observed_report_count"),
        "missing_report_count": metrics.get("missing_report_count"),
        "invalid_report_count": metrics.get("invalid_report_count"),
        "blocked_candidate_count": metrics.get("blocked_candidate_count"),
        "promotion_ready_count": metrics.get("promotion_ready_count"),
        "blocking_reasons": _string_tuple(decision.get("blocking_reasons", ())),
    }


def _frontier_citation_batch_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    evidence_summary = _mapping(payload.get("evidence_summary"))
    missing_batches = [
        _citation_batch_metadata(item)
        for item in _mapping_sequence(evidence_summary.get("citation_batch_missing_expected_batches", ()))
    ]
    duplicate_batches = [
        _citation_batch_metadata(item)
        for item in _mapping_sequence(evidence_summary.get("citation_batch_duplicate_batches", ()))
    ]
    unexpected_batches = [
        _citation_batch_metadata(item)
        for item in _mapping_sequence(evidence_summary.get("citation_batch_unexpected_batches", ()))
    ]
    for decision in _mapping_sequence(payload.get("citation_batch_decisions", ())):
        rollup = _optional_str(decision.get("name"))
        metrics = _mapping(decision.get("metrics"))
        for batch_id in _string_tuple(metrics.get("missing_expected_batch_ids", ())):
            missing_batches.append({"rollup": rollup, "batch_id": batch_id})
        for batch_id in _string_tuple(metrics.get("duplicate_batch_ids", ())):
            duplicate_batches.append({"rollup": rollup, "batch_id": batch_id})
        for batch_id in _string_tuple(metrics.get("unexpected_batch_ids", ())):
            unexpected_batches.append({"rollup": rollup, "batch_id": batch_id})
    metadata = {
        "citation_batch_rollup_names": _string_tuple(
            evidence_summary.get("citation_batch_rollup_names", ())
        ),
        "citation_batch_blocked_rollups": _string_tuple(
            evidence_summary.get("citation_batch_blocked_rollups", ())
        ),
        "citation_batch_expected_batch_ids": _string_tuple(
            evidence_summary.get("citation_batch_expected_batch_ids", ())
        ),
        "citation_batch_observed_batch_ids": _string_tuple(
            evidence_summary.get("citation_batch_observed_batch_ids", ())
        ),
        "citation_batch_missing_expected_batches": tuple(
            _unique_citation_batch_rows(missing_batches)
        ),
        "citation_batch_duplicate_batches": tuple(
            _unique_citation_batch_rows(duplicate_batches)
        ),
        "citation_batch_unexpected_batches": tuple(
            _unique_citation_batch_rows(unexpected_batches)
        ),
        "citation_batch_missing_expected_batch_count": evidence_summary.get(
            "citation_batch_missing_expected_batch_count"
        ),
        "citation_batch_duplicate_batch_count": evidence_summary.get(
            "citation_batch_duplicate_batch_count"
        ),
        "citation_batch_unexpected_batch_count": evidence_summary.get(
            "citation_batch_unexpected_batch_count"
        ),
        "citation_batch_child_manifest_failed_count": evidence_summary.get(
            "citation_batch_child_manifest_failed_count"
        ),
        "citation_batch_blocked_child_report_count": evidence_summary.get(
            "citation_batch_blocked_child_report_count"
        ),
    }
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != () and value != []
    }


def _citation_batch_metadata(item: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "rollup": _optional_str(item.get("rollup")),
        "batch_id": _optional_str(item.get("batch_id")) or "",
    }


def _unique_citation_batch_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str | None]]:
    seen: set[tuple[str | None, str]] = set()
    unique: list[dict[str, str | None]] = []
    for row in rows:
        rollup = _optional_str(row.get("rollup"))
        batch_id = _optional_str(row.get("batch_id")) or ""
        key = (rollup, batch_id)
        if key in seen or not batch_id:
            continue
        seen.add(key)
        unique.append({"rollup": rollup, "batch_id": batch_id})
    return unique


def _frontier_multiple_testing_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    failed_cells: list[dict[str, Any]] = []
    unknown_cells: list[dict[str, Any]] = []
    evidence_summary = _mapping(payload.get("evidence_summary"))
    for key, target in (
        ("multiple_testing_failed_cells", failed_cells),
        ("multiple_testing_unknown_cells", unknown_cells),
    ):
        for item in _mapping_sequence(evidence_summary.get(key, ())):
            target.append(_multiple_testing_cell_metadata(item))
    for decision in _mapping_sequence(payload.get("multiple_testing_decisions", ())):
        run_name = _optional_str(decision.get("name"))
        metrics = _mapping(decision.get("metrics"))
        for item in _mapping_sequence(metrics.get("failed_cells", ())):
            failed_cells.append(_multiple_testing_cell_metadata(item, run_name=run_name))
        for item in _mapping_sequence(metrics.get("unknown_cells", ())):
            unknown_cells.append(_multiple_testing_cell_metadata(item, run_name=run_name))
    failed_cells = _unique_multiple_testing_cells(failed_cells)
    unknown_cells = _unique_multiple_testing_cells(unknown_cells)
    blocked_cells = failed_cells + unknown_cells
    if not blocked_cells:
        return {}
    return {
        "multiple_testing_failed_cells": tuple(failed_cells),
        "multiple_testing_unknown_cells": tuple(unknown_cells),
        "multiple_testing_blocked_cells": tuple(blocked_cells),
    }


def _multiple_testing_cell_metadata(
    item: Mapping[str, Any],
    *,
    run_name: str | None = None,
) -> dict[str, Any]:
    run = _optional_str(item.get("run")) or run_name
    return {
        "run": run,
        "cell": _optional_str(item.get("cell")) or "",
        "status": _optional_str(item.get("status")) or "unknown",
        "false_alarm": item.get("false_alarm"),
        "detection": item.get("detection"),
        "report": item.get("report"),
        "calibration": item.get("calibration"),
    }


def _unique_multiple_testing_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str | None, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for cell in cells:
        key = (
            _optional_str(cell.get("run")),
            _optional_str(cell.get("cell")) or "",
            _optional_str(cell.get("status")) or "unknown",
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(cell))
    return unique


def _classify_gap(
    gate: str,
    reason: str,
    *,
    missing_metrics: Sequence[str],
) -> dict[str, str]:
    text = f"{gate} {reason}".lower()
    if _is_product_runtime_claim_factuality_evidence(gate, text, missing_metrics):
        return _kind("product_runtime_claim_factuality_evidence", "claim_factuality", "runtime_drift")
    if _is_product_runtime_claim_risk_localization_evidence(gate, text, missing_metrics):
        return _kind(
            "product_runtime_claim_risk_localization_evidence",
            "claim_risk_localization",
            "runtime_drift",
        )
    if _is_product_runtime_world_model_evidence(gate, text, missing_metrics):
        return _kind("product_runtime_world_model_evidence", "world_model", "runtime_drift")
    if _is_product_runtime_context_sensitivity_evidence(gate, text, missing_metrics):
        return _kind(
            "product_runtime_context_sensitivity_evidence",
            "context_sensitivity",
            "runtime_drift",
        )
    if _is_product_runtime_counterfactual_robustness_evidence(gate, text, missing_metrics):
        return _kind(
            "product_runtime_counterfactual_robustness_evidence",
            "counterfactual_robustness",
            "runtime_drift",
        )
    if _is_product_runtime_trajectory_audit_evidence(gate, text, missing_metrics):
        return _kind(
            "product_runtime_trajectory_audit_evidence",
            "trajectory_audit",
            "runtime_drift",
        )
    if _is_product_runtime_evidence_handoff_evidence(gate, text, missing_metrics):
        return _kind(
            "product_runtime_evidence_handoff_evidence",
            "product_handoff",
            "runtime_drift",
        )
    if _is_product_runtime_frontier_release_evidence(gate, text, missing_metrics):
        return _kind(
            "product_runtime_frontier_release_evidence",
            "product_handoff",
            "runtime_drift",
        )
    if (
        gate == "frontier_rerun_rollup_evidence"
        or "frontier_rerun_rollup" in text
        or "frontier rerun rollup" in text
        or "frontier-rerun-rollup" in text
        or "frontier_stability_evidence_rerun_rollup" in text
        or "frontier_abstention_evidence_rerun_rollup" in text
        or "frontier_detectability_evidence_rerun_rollup" in text
        or "frontier_multiple_testing_rerun_rollup" in text
    ):
        return _kind(
            "frontier_rerun_rollup_evidence",
            "evidence_coverage",
            "frontier_rerun_validation",
        )
    if (
        "multiple_testing" in text
        or "multiple-testing" in text
        or "multiple testing" in text
        or "family-wise" in text
        or "familywise" in text
    ):
        return _kind("frontier_multiple_testing", "model", "multi_signal_calibration")
    if (
        "citation_batch" in text
        or "citation batch" in text
        or "citation_search_batch_evidence_rollup" in text
        or "batch_coverage" in text
        or "unresolved evidence batch" in text
    ):
        return _kind("citation_batch_evidence", "evidence_coverage", "external_citation")
    if "pre-generation" in text or "pre_generation" in text:
        return _kind("pre_generation_probe", "model", "internal_state")
    if (
        "abstention_stability" in text
        or "abstention_track_status" in text
        or "abstention track" in text
        or "participation-gate" in text
        or "participation gate" in text
    ):
        return _kind("abstention_stability", "model", "participation_calibration")
    if "verifier_stability" in text:
        return _kind("verifier_stability", "evidence_coverage", "external_verification")
    if "detectability_taxonomy" in text or "entrenched_false_rate" in text:
        return _kind("detectability_taxonomy", "model", "blind_spot_taxonomy")
    if "counterfactual" in text:
        return _kind("counterfactual_verification", "context", "counterfactual")
    if "triple audit" in text or "triple_audit" in text or "triple_coverage" in text:
        return _kind("triple_audit", "evidence_coverage", "fact_level")
    if "covered-fact" in text or "covered_fact" in text:
        return _kind("covered_fact_property", "evidence_coverage", "structured_facts")
    if "action-gate" in text or "action_audit" in text or "action_execution" in text:
        return _kind("action_gate", "action_execution", "tool_receipts")
    if "promotion evidence" in text or "promotion_contract" in text:
        return _kind("promotion_contract", "product_handoff", "release_contract")
    if gate == "readiness_baseline":
        return _kind("readiness_baseline", "model", "internal_state")
    if gate == "performance_baseline":
        return _kind("performance_baseline", "runtime", "release_gate")
    if "required_route" in gate or "route_baseline" in gate:
        return _kind("route_baseline", "evidence_coverage", "retrieval_verification")
    if "adapter_family" in gate:
        return _kind("adapter_family", "evidence_coverage", "tool_routes")
    if "external_evidence" in gate:
        return _kind("external_evidence", "evidence_coverage", "external_grounding")
    if "world_model" in text or "mechanism" in text:
        return _kind("world_model", "world_model", "state_transition")
    if missing_metrics:
        return _kind("missing_metrics", "evidence_coverage", "release_gate")
    return _kind("manual_triage", "unknown", "release_gate")


def _is_product_runtime_world_model_evidence(
    gate: str,
    text: str,
    missing_metrics: Sequence[str],
) -> bool:
    return _is_product_runtime_evidence_kind(
        gate,
        text,
        missing_metrics,
        patterns=(
            "world-model evidence",
            "world model evidence",
            "world_model_evidence",
            "world_model.",
        ),
    )


def _is_product_runtime_claim_factuality_evidence(
    gate: str,
    text: str,
    missing_metrics: Sequence[str],
) -> bool:
    return _is_product_runtime_evidence_kind(
        gate,
        text,
        missing_metrics,
        patterns=(
            "claim factuality evidence",
            "claim-factuality evidence",
            "claim_factuality_evidence",
            "claim_factuality_probe_comparison.",
        ),
    )


def _is_product_runtime_claim_risk_localization_evidence(
    gate: str,
    text: str,
    missing_metrics: Sequence[str],
) -> bool:
    return _is_product_runtime_evidence_kind(
        gate,
        text,
        missing_metrics,
        patterns=(
            "claim-risk localization evidence",
            "claim risk localization evidence",
            "claim_risk_localization_evidence",
            "claim_risk_localization.",
        ),
    )


def _is_product_runtime_context_sensitivity_evidence(
    gate: str,
    text: str,
    missing_metrics: Sequence[str],
) -> bool:
    return _is_product_runtime_evidence_kind(
        gate,
        text,
        missing_metrics,
        patterns=(
            "context-sensitivity evidence",
            "context sensitivity evidence",
            "context_sensitivity_evidence",
            "context_sensitivity.",
        ),
    )


def _is_product_runtime_counterfactual_robustness_evidence(
    gate: str,
    text: str,
    missing_metrics: Sequence[str],
) -> bool:
    return _is_product_runtime_evidence_kind(
        gate,
        text,
        missing_metrics,
        patterns=(
            "counterfactual-robustness evidence",
            "counterfactual robustness evidence",
            "counterfactual_robustness_evidence",
            "counterfactual_robustness.",
        ),
    )


def _is_product_runtime_trajectory_audit_evidence(
    gate: str,
    text: str,
    missing_metrics: Sequence[str],
) -> bool:
    return _is_product_runtime_evidence_kind(
        gate,
        text,
        missing_metrics,
        patterns=(
            "trajectory-audit evidence",
            "trajectory audit evidence",
            "trajectory_audit_evidence",
            "trajectory_audit.",
            "product_trace_trajectory_audit",
        ),
    )


def _is_product_runtime_evidence_handoff_evidence(
    gate: str,
    text: str,
    missing_metrics: Sequence[str],
) -> bool:
    return _is_product_runtime_evidence_kind(
        gate,
        text,
        missing_metrics,
        patterns=(
            "evidence-handoff evidence",
            "evidence handoff evidence",
            "evidence_handoff_evidence",
            "promotion_contract.evidence_handoff.",
            "evidence_handoff.",
        ),
    )


def _is_product_runtime_frontier_release_evidence(
    gate: str,
    text: str,
    missing_metrics: Sequence[str],
) -> bool:
    return _is_product_runtime_evidence_kind(
        gate,
        text,
        missing_metrics,
        patterns=(
            "frontier release evidence blocked",
            "frontier release evidence metrics",
            "frontier_release_evidence blocked",
            "frontier_release_evidence metrics",
            "frontier_release_evidence.",
            "promotion_contract.frontier_release_evidence.",
        ),
    )


def _is_product_runtime_evidence_kind(
    gate: str,
    text: str,
    missing_metrics: Sequence[str],
    *,
    patterns: Sequence[str],
) -> bool:
    gate_text = gate.lower()
    metric_text = " ".join(str(metric).lower() for metric in missing_metrics)
    runtime_drift_context = (
        gate_text == "product_runtime_drift"
        or "product runtime drift" in text
        or "product_runtime_drift" in text
        or "product_runtime_drift" in metric_text
    )
    if not runtime_drift_context:
        return False
    return any(pattern in text or pattern in metric_text for pattern in patterns)


def _kind(evidence_kind: str, root_cause: str, research_axis: str) -> dict[str, str]:
    return {
        "evidence_kind": evidence_kind,
        "root_cause": root_cause,
        "research_axis": research_axis,
    }


def _action_template(kind: Mapping[str, str], *, gate: str, reason: str) -> EvidenceGapAction:
    evidence_kind = kind["evidence_kind"]
    if evidence_kind == "readiness_baseline":
        return EvidenceGapAction(
            action_id="refresh_readiness_baseline",
            title="Refresh stronger adapter readiness evidence",
            action_type="benchmark",
            priority=100,
            rationale=(
                "Frontier gates need stronger model/layer diagnostic quality and "
                "runtime-cost readiness evidence before a product contract can be trusted."
            ),
            evidence_routes=("readiness_baseline", "calibrated_observability"),
            suggested_commands=(
                "benchmarks/run_adapter_readiness_registry_workflow.py",
                "benchmarks/run_calibrated_observability_workflow.py",
            ),
        )
    if evidence_kind == "performance_baseline":
        return EvidenceGapAction(
            action_id="refresh_performance_baseline",
            title="Refresh matching performance baseline",
            action_type="benchmark",
            priority=95,
            rationale=(
                "The selected runtime must have a matching performance baseline before "
                "release comparison can measure cost drift."
            ),
            evidence_routes=("performance_baseline", "runtime_budget"),
            suggested_commands=("benchmarks/run_performance_baseline_workflow.py",),
        )
    if evidence_kind == "pre_generation_probe":
        return EvidenceGapAction(
            action_id="run_pre_generation_probe_comparison",
            title="Add pre-generation probe redline evidence",
            action_type="experiment",
            priority=90,
            rationale=(
                "Current frontier work treats hallucination risk before decoding as a "
                "probability; release drift needs multi-run probe evidence and redline margins."
            ),
            evidence_routes=(
                "pre_generation_probe_workflow",
                "pre_generation_text_redline",
                "pre_generation_probe_comparison",
                "product_promotion_contract",
                "product_runtime_drift",
            ),
            suggested_commands=(
                "benchmarks/run_pre_generation_probe_workflow.py "
                "--output-dir ... --json ... --artifact-manifest ...",
                "benchmarks/eval_pre_generation_text_baselines.py "
                "--records ... --json ... --artifact-manifest ...",
                "benchmarks/compare_pre_generation_probe_workflows.py "
                "--workflow-report MODEL=... --redline-report MODEL=... "
                "--json ... --artifact-manifest ...",
                "benchmarks/export_product_promotion_contract.py "
                "--source ... --output ... --artifact-manifest ...",
                "benchmarks/run_product_runtime_baseline.py "
                "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
                "benchmarks/compare_product_runtime_baselines.py "
                "--current ... --baseline ... "
                "--min-pre-generation-probe-comparison-coverage ... "
                "--min-pre-generation-probe-comparison-manifest-verified-rate ... "
                "--min-pre-generation-probe-comparison-redline-pass-rate ... "
                "--json ... --artifact-manifest ...",
            ),
            metadata={
                "workflow_script": "benchmarks/run_pre_generation_probe_workflow.py",
                "redline_script": "benchmarks/eval_pre_generation_text_baselines.py",
                "comparison_script": "benchmarks/compare_pre_generation_probe_workflows.py",
                "promotion_contract_script": "benchmarks/export_product_promotion_contract.py",
                "runtime_baseline_script": "benchmarks/run_product_runtime_baseline.py",
                "runtime_drift_script": "benchmarks/compare_product_runtime_baselines.py",
                "workflow": "pre_generation_probe_workflow",
                "redline_workflow": "pre_generation_text_baseline_eval",
                "comparison_workflow": "pre_generation_probe_workflow_comparison",
                "handoff_artifact_kind": "product_promotion_contract",
                "runtime_baseline_workflow": "product_runtime_baseline",
                "runtime_drift_workflow": "product_runtime_drift_comparison",
                "risk_control_method": "pre_generation_hidden_state_probe",
                "redline_required": True,
                "required_inputs": (
                    "pre_generation_hidden_state_records_or_truthfulqa_export",
                    "pre_generation_probe_workflow_reports",
                    "pre_generation_text_redline_reports",
                    "release_candidate_or_product_contract_source",
                    "product_trace_corpus",
                ),
                "closure_outputs": (
                    "pre_generation_probe_workflow_comparison",
                    "product_promotion_contract",
                    "product_runtime_baseline",
                    "product_runtime_drift_comparison",
                ),
            },
        )
    if evidence_kind == "product_runtime_claim_factuality_evidence":
        return EvidenceGapAction(
            action_id="rerun_claim_factuality_probe_comparison",
            title="Add claim factuality probe comparison evidence",
            action_type="experiment",
            priority=89,
            rationale=(
                "Runtime drift gates need multi-run, manifest-backed claim factuality "
                "probe evidence before claim-level detector behavior is treated as "
                "stable enough for release."
            ),
            evidence_routes=("claim_factuality_probe_comparison", "product_runtime_drift"),
            suggested_commands=(
                "benchmarks/run_claim_factuality_probe_workflow.py",
                "benchmarks/compare_claim_factuality_probe_workflows.py",
                "benchmarks/compare_product_runtime_baselines.py",
            ),
        )
    if evidence_kind == "product_runtime_claim_risk_localization_evidence":
        return EvidenceGapAction(
            action_id="rerun_product_trace_claim_risk_localization_evidence",
            title="Replay product traces with claim-risk localization evidence",
            action_type="workflow",
            priority=88,
            rationale=(
                "Fine-grained hallucination control needs trace-level span, claim, "
                "and entity-risk localization coverage so release checks can catch "
                "concentrated high-risk claim regressions instead of only aggregate "
                "verifier drift."
            ),
            evidence_routes=(
                "product_trace_replay",
                "claim_risk_localization",
                "product_runtime_baseline",
                "product_runtime_drift",
                "span_entity_risk_evidence",
            ),
            suggested_commands=(
                "benchmarks/run_product_trace_replay_workflow.py "
                "--trace-glob ... --promotion-contract ... "
                "--min-runtime-drift-claim-risk-localization-coverage-rate ... "
                "--max-runtime-drift-claim-risk-localization-high-risk-claim-count-increase ... "
                "--max-runtime-drift-claim-risk-localization-medium-or-high-risk-claim-count-increase ... "
                "--max-runtime-drift-claim-risk-localization-entity-candidate-observation-count-increase ... "
                "--max-runtime-drift-claim-risk-localization-unique-entity-candidate-count-increase ... "
                "--max-runtime-drift-claim-risk-localization-high-risk-entity-candidate-count-increase ... "
                "--max-runtime-drift-claim-risk-localization-medium-or-high-entity-candidate-count-increase ...",
                "benchmarks/run_product_runtime_baseline.py "
                "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
                "benchmarks/compare_product_runtime_baselines.py "
                "--current ... --baseline ... "
                "--min-claim-risk-localization-coverage-rate ... "
                "--max-claim-risk-localization-high-risk-claim-count-increase ... "
                "--max-claim-risk-localization-medium-or-high-risk-claim-count-increase ... "
                "--max-claim-risk-localization-entity-candidate-observation-count-increase ... "
                "--max-claim-risk-localization-unique-entity-candidate-count-increase ... "
                "--max-claim-risk-localization-high-risk-entity-candidate-count-increase ... "
                "--max-claim-risk-localization-medium-or-high-entity-candidate-count-increase ... "
                "--json ... --artifact-manifest ...",
            ),
            metadata={
                "claim_risk_localization_api": (
                    "eigentruth.verify.localize_claim_risk_spans"
                ),
                "trace_summary_api": (
                    "eigentruth.control.ProductTrace.claim_risk_localization_summary"
                ),
                "trace_replay_script": "benchmarks/run_product_trace_replay_workflow.py",
                "runtime_baseline_script": "benchmarks/run_product_runtime_baseline.py",
                "runtime_drift_script": "benchmarks/compare_product_runtime_baselines.py",
                "trace_replay_workflow": "product_trace_replay_workflow",
                "runtime_baseline_workflow": "product_runtime_baseline",
                "runtime_drift_workflow": "product_runtime_drift_comparison",
                "risk_control_method": "span_entity_claim_risk_localization",
                "localization_granularity": (
                    "span",
                    "claim",
                    "entity_candidate",
                ),
                "required_trace_metrics": (
                    "claim_risk_localization.coverage_rate",
                    "claim_risk_localization.high_risk_claim_count",
                    "claim_risk_localization.medium_or_high_risk_claim_count",
                    "claim_risk_localization.entity_candidate_observation_count",
                    "claim_risk_localization.unique_entity_candidate_count",
                    "claim_risk_localization.high_risk_entity_candidate_count",
                    "claim_risk_localization.medium_or_high_entity_candidate_count",
                ),
                "default_gate_thresholds": {
                    "min_claim_risk_localization_coverage_rate": 1.0,
                    "max_claim_risk_localization_high_risk_claim_count_increase": 0.0,
                    "max_claim_risk_localization_medium_or_high_risk_claim_count_increase": (
                        0.0
                    ),
                    "max_claim_risk_localization_entity_candidate_observation_count_increase": (
                        0.0
                    ),
                    "max_claim_risk_localization_unique_entity_candidate_count_increase": (
                        0.0
                    ),
                    "max_claim_risk_localization_high_risk_entity_candidate_count_increase": (
                        0.0
                    ),
                    "max_claim_risk_localization_medium_or_high_entity_candidate_count_increase": (
                        0.0
                    ),
                },
                "required_inputs": (
                    "full_product_trace_corpus",
                    "promotion_contract_or_release_candidate",
                    "baseline_product_runtime_report",
                ),
                "closure_outputs": (
                    "product_trace_replay_workflow",
                    "product_runtime_baseline",
                    "product_runtime_drift_comparison",
                ),
            },
        )
    if evidence_kind == "frontier_multiple_testing":
        return EvidenceGapAction(
            action_id="rerun_frontier_multiple_testing_gate",
            title="Rerun frontier workflow multi-signal conformal gate",
            action_type="experiment",
            priority=89,
            rationale=(
                "The release is blocked by the family-wise multi-signal hallucination gate; "
                "rerun or inspect the frontier workflow cells to identify failing signal/layer "
                "families before promoting runtime defaults."
            ),
            evidence_routes=(
                "truthfulqa_frontier_workflow",
                "multiple_testing_gate",
                "frontier_release_evidence",
            ),
            suggested_commands=(
                "benchmarks/plan_frontier_multiple_testing_reruns.py --source ... --json ...",
                "benchmarks/run_truthfulqa_frontier_workflow.py --multiple-testing-signals ...",
                "benchmarks/rollup_frontier_multiple_testing_reruns.py --queue ... --json ...",
                "benchmarks/compare_frontier_release_evidence.py --frontier-rerun-rollup-report ...",
            ),
            metadata={
                "planner_script": "benchmarks/plan_frontier_multiple_testing_reruns.py",
                "child_workflow_script": "benchmarks/run_truthfulqa_frontier_workflow.py",
                "rollup_script": "benchmarks/rollup_frontier_multiple_testing_reruns.py",
                "release_gate_script": "benchmarks/compare_frontier_release_evidence.py",
                "rerun_queue_workflow": "frontier_multiple_testing_rerun_queue",
                "child_workflow": "truthfulqa_frontier_workflow",
                "rollup_workflow": "frontier_multiple_testing_rerun_rollup",
                "derived_artifact_key": "frontier_multiple_testing_rerun_queue",
                "derived_artifact_kind": "frontier_multiple_testing_rerun_queue",
                "rollup_track": "multiple_testing",
                "release_gate_track": "frontier_rerun_rollup",
                "queue_entry_report_kind": "truthfulqa_frontier_workflow",
                "risk_control_method": "multiple_testing_conformal",
                "required_inputs": (
                    "frontier_release_report_or_evidence_gap_plan",
                    "frontier_workflow_report_with_multiple_testing_config",
                ),
                "closure_outputs": (
                    "frontier_multiple_testing_rerun_queue",
                    "frontier_multiple_testing_rerun_rollup",
                    "frontier_release_evidence_comparison",
                ),
            },
        )
    if evidence_kind == "citation_batch_evidence":
        return EvidenceGapAction(
            action_id="complete_citation_batch_evidence_rollup",
            title="Complete citation batch evidence rollup",
            action_type="workflow",
            priority=88,
            rationale=(
                "The frontier release is blocked because one or more unresolved citation "
                "or source-family evidence batches did not produce promotion-ready, "
                "manifest-backed child evidence."
            ),
            evidence_routes=(
                "unresolved_evidence_queue",
                "citation_search_evidence",
                "source_family_citation",
                "frontier_release_evidence",
            ),
            suggested_commands=(
                "benchmarks/plan_citation_batch_evidence_reruns.py --source ... --json ...",
                "benchmarks/run_external_citation_search_adapter_workflow.py "
                "--queue ... --batch-id ... --workflow-report ...",
                "benchmarks/run_source_family_citation_search_workflow.py "
                "--queue ... --batch-id ... --workflow-report ...",
                "benchmarks/rollup_citation_search_batch_evidence.py --queue ... --batch-report ... --json ...",
                "benchmarks/compare_frontier_release_evidence.py --citation-batch-rollup-report ...",
            ),
            metadata={
                "planner_script": "benchmarks/plan_citation_batch_evidence_reruns.py",
                "external_workflow_script": "benchmarks/run_external_citation_search_adapter_workflow.py",
                "source_family_workflow_script": "benchmarks/run_source_family_citation_search_workflow.py",
                "rollup_script": "benchmarks/rollup_citation_search_batch_evidence.py",
                "release_gate_script": "benchmarks/compare_frontier_release_evidence.py",
                "rerun_queue_workflow": "citation_batch_evidence_rerun_queue",
                "external_workflow": "external_citation_search_adapter_workflow",
                "source_family_workflow": "source_family_citation_search_workflow",
                "rollup_workflow": "citation_search_batch_evidence_rollup",
                "derived_artifact_key": "citation_batch_evidence_rerun_queue",
                "derived_artifact_kind": "citation_batch_evidence_rerun_queue",
                "rollup_track": "citation_batch",
                "release_gate_track": "citation_batch",
                "queue_entry_report_kinds": (
                    "external_citation_search_adapter_workflow",
                    "source_family_citation_search_workflow",
                ),
                "risk_control_method": "citation_traceability",
                "required_inputs": (
                    "frontier_release_report_or_evidence_gap_plan",
                    "unresolved_evidence_queue",
                    "score_dump",
                    "blind_spot_rows",
                    "source_catalog_or_search_command",
                ),
                "closure_outputs": (
                    "citation_batch_evidence_rerun_queue",
                    "citation_search_batch_evidence_rollup",
                    "frontier_release_evidence_comparison",
                ),
            },
        )
    if evidence_kind == "frontier_rerun_rollup_evidence":
        return EvidenceGapAction(
            action_id="complete_frontier_rerun_rollup_evidence",
            title="Complete frontier rerun-rollup evidence",
            action_type="workflow",
            priority=90,
            rationale=(
                "The frontier release has targeted rerun evidence, but the rerun rollup "
                "is missing, invalid, or still blocked; complete the per-track reruns and "
                "feed the promotion-ready rollup back into the frontier release comparator."
            ),
            evidence_routes=(
                "frontier_rerun_rollup",
                "frontier_release_evidence",
                "verifier_stability",
                "abstention_stability",
                "detectability_taxonomy",
                "multiple_testing_gate",
            ),
            suggested_commands=(
                "benchmarks/plan_frontier_stability_evidence_reruns.py",
                "benchmarks/plan_frontier_abstention_evidence_reruns.py",
                "benchmarks/plan_frontier_detectability_evidence_reruns.py",
                "benchmarks/plan_frontier_multiple_testing_reruns.py",
                "benchmarks/rollup_frontier_*_reruns.py --queue ... --report-json ...",
                "benchmarks/compare_frontier_release_evidence.py --frontier-rerun-rollup-report ...",
            ),
        )
    if evidence_kind == "abstention_stability":
        return EvidenceGapAction(
            action_id="improve_abstention_participation_gate",
            title="Improve abstention participation-gate stability",
            action_type="experiment",
            priority=89,
            rationale=(
                "The frontier release is blocked because retained-answer quality or "
                "abstention cost is not stable enough across held-out splits; compare "
                "candidate participation signals before changing runtime defaults."
            ),
            evidence_routes=(
                "abstention_stability",
                "participation_gate",
                "frontier_release_evidence",
            ),
            suggested_commands=(
                "benchmarks/plan_frontier_abstention_evidence_reruns.py --source ... --json ...",
                "benchmarks/eval_abstention_stability.py --json ...",
                "benchmarks/rollup_frontier_abstention_evidence_reruns.py --queue ... --json ...",
                "benchmarks/compare_frontier_release_evidence.py --frontier-rerun-rollup-report ...",
            ),
            metadata={
                "planner_script": "benchmarks/plan_frontier_abstention_evidence_reruns.py",
                "child_benchmark_script": "benchmarks/eval_abstention_stability.py",
                "rollup_script": "benchmarks/rollup_frontier_abstention_evidence_reruns.py",
                "release_gate_script": "benchmarks/compare_frontier_release_evidence.py",
                "rerun_queue_workflow": "frontier_abstention_evidence_rerun_queue",
                "rollup_workflow": "frontier_abstention_evidence_rerun_rollup",
                "derived_artifact_key": "abstention_rerun_queue",
                "derived_artifact_kind": "frontier_abstention_rerun_queue",
                "rollup_track": "abstention",
                "release_gate_track": "frontier_rerun_rollup",
                "queue_entry_report_kind": "abstention_stability",
                "required_inputs": (
                    "frontier_release_report_or_evidence_gap_plan",
                    "abstention_score_dump_paths",
                    "abstention_signal_groups",
                ),
                "closure_outputs": (
                    "abstention_rerun_queue",
                    "abstention_rerun_rollup",
                    "frontier_release_evidence_comparison",
                ),
            },
        )
    if evidence_kind == "verifier_stability":
        return EvidenceGapAction(
            action_id="rerun_verifier_stability_replay",
            title="Rerun staged verifier-stability replay",
            action_type="experiment",
            priority=86,
            rationale=(
                "The frontier release needs stable verifier false-alarm and detection "
                "improvement evidence before verifier routing can be promoted."
            ),
            evidence_routes=("verifier_stability", "frontier_release_evidence"),
            suggested_commands=("benchmarks/eval_verifier_stability.py",),
        )
    if evidence_kind == "detectability_taxonomy":
        return EvidenceGapAction(
            action_id="audit_detectability_blind_spots",
            title="Audit detectability-taxonomy blind spots",
            action_type="analysis",
            priority=85,
            rationale=(
                "High-confidence/high-consistency false records need explicit blind-spot "
                "analysis before output-level uncertainty signals are trusted."
            ),
            evidence_routes=(
                "detectability_taxonomy",
                "blind_spot_audit",
                "frontier_release_evidence",
            ),
            suggested_commands=(
                "benchmarks/plan_frontier_detectability_evidence_reruns.py --source ... --json ...",
                "benchmarks/analyze_detectability_blind_spots.py --taxonomy-report ... --json ...",
                "benchmarks/eval_detectability_taxonomy.py --scores ... --json ...",
                "benchmarks/rollup_frontier_detectability_evidence_reruns.py --queue ... --json ...",
                "benchmarks/compare_frontier_release_evidence.py --frontier-rerun-rollup-report ...",
            ),
            metadata={
                "planner_script": "benchmarks/plan_frontier_detectability_evidence_reruns.py",
                "blind_spot_analysis_script": "benchmarks/analyze_detectability_blind_spots.py",
                "taxonomy_rerun_script": "benchmarks/eval_detectability_taxonomy.py",
                "rollup_script": "benchmarks/rollup_frontier_detectability_evidence_reruns.py",
                "release_gate_script": "benchmarks/compare_frontier_release_evidence.py",
                "rerun_queue_workflow": "frontier_detectability_evidence_rerun_queue",
                "blind_spot_workflow": "detectability_blind_spot_analysis",
                "taxonomy_workflow": "detectability_taxonomy",
                "rollup_workflow": "frontier_detectability_evidence_rerun_rollup",
                "derived_artifact_key": "frontier_detectability_evidence_rerun_queue",
                "derived_artifact_kind": "frontier_detectability_evidence_rerun_queue",
                "rollup_track": "detectability",
                "release_gate_track": "frontier_rerun_rollup",
                "queue_entry_report_kinds": (
                    "detectability_blind_spot_analysis",
                    "detectability_taxonomy",
                ),
                "risk_control_method": "detectability_taxonomy",
                "default_blind_spot_cell": "entrenched",
                "required_inputs": (
                    "frontier_release_report_or_evidence_gap_plan",
                    "detectability_taxonomy_report_or_score_dump",
                    "consistency_signal",
                    "confidence_signal",
                ),
                "closure_outputs": (
                    "frontier_detectability_evidence_rerun_queue",
                    "frontier_detectability_evidence_rerun_rollup",
                    "frontier_release_evidence_comparison",
                ),
            },
        )
    if evidence_kind == "counterfactual_verification":
        return EvidenceGapAction(
            action_id="run_counterfactual_verifier_audit",
            title="Add counterfactual verifier sensitivity evidence",
            action_type="experiment",
            priority=88,
            rationale=(
                "Verifier routes should flip on plausible false variants instead of "
                "remaining invariant to entity, numeric, temporal, or negation changes."
            ),
            evidence_routes=("counterfactual_verification", "product_runtime_drift"),
            suggested_commands=("benchmarks/eval_counterfactual_verification.py",),
        )
    if evidence_kind == "action_gate":
        return EvidenceGapAction(
            action_id="rerun_product_trace_action_gates",
            title="Replay product traces with action audit and execution gates",
            action_type="workflow",
            priority=87,
            rationale=(
                "Tool-use hallucinations include fabricated, missing, malformed, and "
                "misaligned actions; release evidence should tie replayed traces to "
                "action-audit, execution-alignment, and runtime-drift gates."
            ),
            evidence_routes=(
                "product_trace_replay",
                "action_audit",
                "action_execution",
                "product_runtime_baseline",
                "product_runtime_drift",
                "tool_use_evidence",
            ),
            suggested_commands=(
                "benchmarks/run_product_trace_replay_workflow.py "
                "--trace-glob ... --promotion-contract ... "
                "--max-action-audit-error-rate ... "
                "--max-action-audit-missing-retrieval-rate ... "
                "--max-action-audit-missing-plan-retrieval-query-rate ... "
                "--max-action-audit-malformed-payload-rate ... "
                "--max-action-audit-unexpected-action-rate ... "
                "--max-action-audit-unknown-claim-id-rate ... "
                "--max-action-execution-missing-result-rate ... "
                "--max-action-execution-unexpected-result-rate ... "
                "--max-action-execution-request-id-mismatch-rate ... "
                "--max-runtime-drift-product-trace-action-audit-error-rate-increase ... "
                "--max-runtime-drift-product-trace-action-audit-missing-retrieval-action-rate-increase ... "
                "--max-runtime-drift-product-trace-action-audit-missing-plan-retrieval-query-rate-increase ... "
                "--max-runtime-drift-product-trace-action-audit-malformed-payload-rate-increase ... "
                "--max-runtime-drift-product-trace-action-audit-unexpected-action-rate-increase ... "
                "--max-runtime-drift-product-trace-action-audit-unknown-claim-id-rate-increase ... "
                "--max-runtime-drift-product-trace-action-execution-alignment-failed-trace-rate-increase ... "
                "--max-runtime-drift-product-trace-action-execution-missing-result-rate-increase ... "
                "--max-runtime-drift-product-trace-action-execution-unexpected-result-rate-increase ... "
                "--max-runtime-drift-product-trace-action-execution-request-id-mismatch-rate-increase ...",
                "benchmarks/run_product_runtime_baseline.py "
                "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
                "benchmarks/compare_product_runtime_baselines.py "
                "--current ... --baseline ... "
                "--max-product-trace-action-audit-error-rate-increase ... "
                "--max-product-trace-action-audit-missing-retrieval-action-rate-increase ... "
                "--max-product-trace-action-audit-missing-plan-retrieval-query-rate-increase ... "
                "--max-product-trace-action-audit-malformed-payload-rate-increase ... "
                "--max-product-trace-action-audit-unexpected-action-rate-increase ... "
                "--max-product-trace-action-audit-unknown-claim-id-rate-increase ... "
                "--max-product-trace-action-execution-alignment-failed-trace-rate-increase ... "
                "--max-product-trace-action-execution-missing-result-rate-increase ... "
                "--max-product-trace-action-execution-unexpected-result-rate-increase ... "
                "--max-product-trace-action-execution-request-id-mismatch-rate-increase ... "
                "--json ... --artifact-manifest ...",
            ),
            metadata={
                "action_audit_api": "eigentruth.control.audit_action_requests",
                "action_execution_summary_api": (
                    "eigentruth.control.ProductTrace.action_execution_summary"
                ),
                "trace_replay_script": "benchmarks/run_product_trace_replay_workflow.py",
                "runtime_baseline_script": "benchmarks/run_product_runtime_baseline.py",
                "runtime_drift_script": "benchmarks/compare_product_runtime_baselines.py",
                "trace_replay_workflow": "product_trace_replay_workflow",
                "action_audit_gate_workflow": "product_trace_action_audit_gate",
                "action_execution_gate_workflow": "product_trace_action_execution_gate",
                "runtime_baseline_workflow": "product_runtime_baseline",
                "runtime_drift_workflow": "product_runtime_drift_comparison",
                "risk_control_method": "tool_use_action_audit",
                "tool_use_failure_modes": (
                    "fabricated_action",
                    "missing_required_action",
                    "malformed_payload",
                    "unexpected_action",
                    "unknown_claim_reference",
                    "missing_action_result",
                    "unexpected_action_result",
                    "request_id_mismatch",
                    "execution_alignment_failure",
                ),
                "required_trace_metrics": (
                    "action_audit.error_rate",
                    "action_audit.missing_retrieval_action_rate",
                    "action_audit.missing_plan_retrieval_query_rate",
                    "action_audit.malformed_payload_rate",
                    "action_audit.unexpected_action_rate",
                    "action_audit.unknown_claim_id_rate",
                    "action_execution.alignment_failed_trace_rate",
                    "action_execution.missing_result_rate",
                    "action_execution.unexpected_result_rate",
                    "action_execution.request_id_mismatch_rate",
                ),
                "default_gate_thresholds": {
                    "max_action_audit_error_rate": 0.0,
                    "max_action_audit_missing_retrieval_rate": 0.0,
                    "max_action_audit_missing_plan_retrieval_query_rate": 0.0,
                    "max_action_audit_malformed_payload_rate": 0.0,
                    "max_action_audit_unexpected_action_rate": 0.0,
                    "max_action_audit_unknown_claim_id_rate": 0.0,
                    "max_action_execution_missing_result_rate": 0.0,
                    "max_action_execution_unexpected_result_rate": 0.0,
                    "max_action_execution_request_id_mismatch_rate": 0.0,
                    "max_product_trace_action_audit_error_rate_increase": 0.0,
                    "max_product_trace_action_audit_missing_retrieval_action_rate_increase": (
                        0.0
                    ),
                    "max_product_trace_action_audit_missing_plan_retrieval_query_rate_increase": (
                        0.0
                    ),
                    "max_product_trace_action_audit_malformed_payload_rate_increase": 0.0,
                    "max_product_trace_action_audit_unexpected_action_rate_increase": 0.0,
                    "max_product_trace_action_audit_unknown_claim_id_rate_increase": 0.0,
                    "max_product_trace_action_execution_alignment_failed_trace_rate_increase": (
                        0.0
                    ),
                    "max_product_trace_action_execution_missing_result_rate_increase": 0.0,
                    "max_product_trace_action_execution_unexpected_result_rate_increase": 0.0,
                    "max_product_trace_action_execution_request_id_mismatch_rate_increase": (
                        0.0
                    ),
                },
                "required_inputs": (
                    "full_product_trace_corpus",
                    "promotion_contract_or_release_candidate",
                    "baseline_product_runtime_report",
                ),
                "closure_outputs": (
                    "product_trace_replay_workflow",
                    "product_trace_action_audit_gate",
                    "product_trace_action_execution_gate",
                    "product_runtime_baseline",
                    "product_runtime_drift_comparison",
                ),
            },
        )
    if evidence_kind == "product_runtime_world_model_evidence":
        return EvidenceGapAction(
            action_id="rerun_product_trace_world_model_evidence",
            title="Replay product traces with world-model evidence summaries",
            action_type="workflow",
            priority=86,
            rationale=(
                "Frontier runtime drift gates need trace-level world-model participation, "
                "coverage, conflict, low-agreement, and trace-gap evidence before the "
                "release can trust model-state correction signals."
            ),
            evidence_routes=(
                "world_model_signal_calibration",
                "product_trace_runtime_evidence",
                "product_trace_replay",
                "product_runtime_baseline",
                "product_runtime_drift",
                "world_model_evidence",
            ),
            suggested_commands=(
                "benchmarks/run_world_model_signal_calibration_workflow.py "
                "--output-dir ... --registry ... --registry-name ... --registry-version ...",
                "benchmarks/enrich_product_trace_runtime_evidence.py "
                "--trace-glob ... --output-dir ... --report ... --artifact-manifest ... "
                "--min-world-model-participating-trace-rate ... "
                "--min-world-model-coverage-rate ... --max-world-model-trace-gap-rate ...",
                "benchmarks/run_product_trace_replay_workflow.py "
                "--trace-glob ... --promotion-contract ... "
                "--min-runtime-drift-world-model-participating-trace-rate ... "
                "--min-runtime-drift-world-model-coverage-rate ... "
                "--max-runtime-drift-world-model-trace-gap-rate-increase ...",
                "benchmarks/run_product_runtime_baseline.py "
                "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
                "benchmarks/compare_product_runtime_baselines.py "
                "--current ... --baseline ... "
                "--min-world-model-participating-trace-rate ... "
                "--min-world-model-coverage-rate ... "
                "--max-world-model-trace-gap-rate-increase ... "
                "--json ... --artifact-manifest ...",
            ),
            metadata={
                "signal_workflow_script": (
                    "benchmarks/run_world_model_signal_calibration_workflow.py"
                ),
                "trace_enrichment_script": "benchmarks/enrich_product_trace_runtime_evidence.py",
                "trace_replay_script": "benchmarks/run_product_trace_replay_workflow.py",
                "runtime_baseline_script": "benchmarks/run_product_runtime_baseline.py",
                "runtime_drift_script": "benchmarks/compare_product_runtime_baselines.py",
                "signal_workflow": "world_model_signal_calibration_workflow",
                "trace_enrichment_workflow": "product_trace_runtime_evidence_enrichment",
                "trace_replay_workflow": "product_trace_replay_workflow",
                "runtime_baseline_workflow": "product_runtime_baseline",
                "runtime_drift_workflow": "product_runtime_drift_comparison",
                "risk_control_method": "state_transition_world_model",
                "required_trace_metrics": (
                    "world_model.participating_trace_rate",
                    "world_model.coverage_rate",
                    "world_model.conflict_rate",
                    "world_model.low_agreement_rate",
                    "world_model.trace_gap_rate",
                ),
                "default_gate_thresholds": {
                    "min_world_model_participating_trace_rate": 1.0,
                    "min_world_model_coverage_rate": 1.0,
                    "max_world_model_trace_gap_rate_increase": 0.0,
                },
                "required_inputs": (
                    "world_model_rules_or_state_transition_fixture",
                    "product_trace_corpus",
                    "promotion_contract_or_release_candidate",
                    "baseline_product_runtime_report",
                ),
                "closure_outputs": (
                    "world_model_signal_calibration_workflow",
                    "product_trace_runtime_evidence_enrichment",
                    "product_trace_replay_workflow",
                    "product_runtime_baseline",
                    "product_runtime_drift_comparison",
                ),
            },
        )
    if evidence_kind == "product_runtime_context_sensitivity_evidence":
        return EvidenceGapAction(
            action_id="rerun_product_trace_context_sensitivity_evidence",
            title="Replay product traces with context-sensitivity evidence summaries",
            action_type="workflow",
            priority=86,
            rationale=(
                "Frontier runtime drift gates need trace-level context-sensitivity "
                "participation, coverage, flagged-result, trace-gap, and ratio evidence "
                "before the release can trust evidence-conditioned correction signals."
            ),
            evidence_routes=(
                "context_sensitivity_workflow",
                "product_trace_runtime_evidence",
                "product_trace_replay",
                "product_runtime_baseline",
                "product_runtime_drift",
                "context_sensitivity_evidence",
            ),
            suggested_commands=(
                "benchmarks/run_context_sensitivity_workflow.py "
                "--scores ... --verified-records-jsonl ... --model-id ... "
                "--output-dir ... --registry-path ... --registry-name ... --registry-version ...",
                "benchmarks/enrich_product_trace_runtime_evidence.py "
                "--trace-glob ... --output-dir ... --report ... --artifact-manifest ... "
                "--min-context-sensitivity-participating-trace-rate ... "
                "--min-context-sensitivity-coverage-rate ... "
                "--max-context-sensitivity-trace-gap-rate ...",
                "benchmarks/run_product_trace_replay_workflow.py "
                "--trace-glob ... --promotion-contract ... "
                "--min-runtime-drift-context-sensitivity-participating-trace-rate ... "
                "--min-runtime-drift-context-sensitivity-coverage-rate ... "
                "--max-runtime-drift-context-sensitivity-trace-gap-rate-increase ... "
                "--max-runtime-drift-context-sensitivity-max-ratio-increase ...",
                "benchmarks/run_product_runtime_baseline.py "
                "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
                "benchmarks/compare_product_runtime_baselines.py "
                "--current ... --baseline ... "
                "--min-context-sensitivity-participating-trace-rate ... "
                "--min-context-sensitivity-coverage-rate ... "
                "--max-context-sensitivity-trace-gap-rate-increase ... "
                "--max-context-sensitivity-max-ratio-increase ... "
                "--json ... --artifact-manifest ...",
            ),
            metadata={
                "context_workflow_script": "benchmarks/run_context_sensitivity_workflow.py",
                "trace_enrichment_script": "benchmarks/enrich_product_trace_runtime_evidence.py",
                "trace_replay_script": "benchmarks/run_product_trace_replay_workflow.py",
                "runtime_baseline_script": "benchmarks/run_product_runtime_baseline.py",
                "runtime_drift_script": "benchmarks/compare_product_runtime_baselines.py",
                "context_workflow": "context_sensitivity_workflow",
                "paired_logprob_workflow": "context_sensitivity_paired_logprob_extraction",
                "trace_enrichment_workflow": "product_trace_runtime_evidence_enrichment",
                "trace_replay_workflow": "product_trace_replay_workflow",
                "runtime_baseline_workflow": "product_runtime_baseline",
                "runtime_drift_workflow": "product_runtime_drift_comparison",
                "risk_control_method": "evidence_conditioned_context_sensitivity",
                "required_trace_metrics": (
                    "context_sensitivity.participating_trace_rate",
                    "context_sensitivity.coverage_rate",
                    "context_sensitivity.flagged_result_rate",
                    "context_sensitivity.trace_gap_rate",
                    "context_sensitivity.max_flagged_rate",
                    "context_sensitivity.max_context_sensitivity_ratio",
                ),
                "default_gate_thresholds": {
                    "min_context_sensitivity_participating_trace_rate": 1.0,
                    "min_context_sensitivity_coverage_rate": 1.0,
                    "max_context_sensitivity_trace_gap_rate_increase": 0.0,
                },
                "required_inputs": (
                    "score_dump",
                    "verified_records_jsonl_with_evidence_context",
                    "context_logprob_model",
                    "product_trace_corpus",
                    "promotion_contract_or_release_candidate",
                    "baseline_product_runtime_report",
                ),
                "closure_outputs": (
                    "context_sensitivity_workflow",
                    "product_trace_runtime_evidence_enrichment",
                    "product_trace_replay_workflow",
                    "product_runtime_baseline",
                    "product_runtime_drift_comparison",
                ),
            },
        )
    if evidence_kind == "product_runtime_counterfactual_robustness_evidence":
        return EvidenceGapAction(
            action_id="rerun_product_trace_counterfactual_robustness_evidence",
            title="Replay product traces with counterfactual-robustness evidence summaries",
            action_type="workflow",
            priority=86,
            rationale=(
                "Frontier runtime drift gates need trace-level counterfactual robustness "
                "participation, coverage, pass, flip-success, false-invariance, and "
                "trace-gap evidence before verifier behavior is treated as stable."
            ),
            evidence_routes=(
                "counterfactual_verification_eval",
                "product_trace_runtime_evidence",
                "product_trace_replay",
                "product_runtime_baseline",
                "product_runtime_drift",
                "counterfactual_robustness_evidence",
            ),
            suggested_commands=(
                "benchmarks/eval_counterfactual_verification.py "
                "--verified-records ... --verifier ... --fact-corpus ... "
                "--json ... --artifact-manifest ... --registry ... "
                "--register-name ... --register-version ...",
                "benchmarks/enrich_product_trace_runtime_evidence.py "
                "--trace-glob ... --output-dir ... --report ... --artifact-manifest ... "
                "--min-counterfactual-robustness-participating-trace-rate ... "
                "--min-counterfactual-robustness-coverage-rate ... "
                "--min-counterfactual-robustness-pass-rate ... "
                "--min-counterfactual-robustness-flip-success-rate ... "
                "--max-counterfactual-robustness-false-invariance-rate ... "
                "--max-counterfactual-robustness-trace-gap-rate ...",
                "benchmarks/run_product_trace_replay_workflow.py "
                "--trace-glob ... --promotion-contract ... "
                "--min-runtime-drift-counterfactual-robustness-participating-trace-rate ... "
                "--min-runtime-drift-counterfactual-robustness-coverage-rate ... "
                "--min-runtime-drift-counterfactual-robustness-pass-rate ... "
                "--min-runtime-drift-counterfactual-robustness-flip-success-rate ... "
                "--max-runtime-drift-counterfactual-robustness-false-invariance-rate-increase ... "
                "--max-runtime-drift-counterfactual-robustness-trace-gap-rate-increase ...",
                "benchmarks/run_product_runtime_baseline.py "
                "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
                "benchmarks/compare_product_runtime_baselines.py "
                "--current ... --baseline ... "
                "--min-counterfactual-robustness-participating-trace-rate ... "
                "--min-counterfactual-robustness-coverage-rate ... "
                "--min-counterfactual-robustness-pass-rate ... "
                "--min-counterfactual-robustness-flip-success-rate ... "
                "--max-counterfactual-robustness-false-invariance-rate-increase ... "
                "--max-counterfactual-robustness-trace-gap-rate-increase ... "
                "--json ... --artifact-manifest ...",
            ),
            metadata={
                "counterfactual_eval_script": "benchmarks/eval_counterfactual_verification.py",
                "trace_enrichment_script": "benchmarks/enrich_product_trace_runtime_evidence.py",
                "trace_replay_script": "benchmarks/run_product_trace_replay_workflow.py",
                "runtime_baseline_script": "benchmarks/run_product_runtime_baseline.py",
                "runtime_drift_script": "benchmarks/compare_product_runtime_baselines.py",
                "counterfactual_eval_workflow": "counterfactual_verification_eval",
                "trace_enrichment_workflow": "product_trace_runtime_evidence_enrichment",
                "trace_replay_workflow": "product_trace_replay_workflow",
                "runtime_baseline_workflow": "product_runtime_baseline",
                "runtime_drift_workflow": "product_runtime_drift_comparison",
                "risk_control_method": "counterfactual_probe_robustness",
                "required_trace_metrics": (
                    "counterfactual_robustness.participating_trace_rate",
                    "counterfactual_robustness.coverage_rate",
                    "counterfactual_robustness.pass_rate",
                    "counterfactual_robustness.flip_success_rate",
                    "counterfactual_robustness.false_invariance_rate",
                    "counterfactual_robustness.trace_gap_rate",
                ),
                "default_gate_thresholds": {
                    "min_counterfactual_robustness_participating_trace_rate": 1.0,
                    "min_counterfactual_robustness_coverage_rate": 1.0,
                    "min_counterfactual_robustness_pass_rate": 1.0,
                    "min_counterfactual_robustness_flip_success_rate": 1.0,
                    "max_counterfactual_robustness_false_invariance_rate_increase": 0.0,
                    "max_counterfactual_robustness_trace_gap_rate_increase": 0.0,
                },
                "required_inputs": (
                    "verified_records_jsonl_or_counterfactual_probe_records",
                    "counterfactual_verifier_or_fact_corpus",
                    "product_trace_corpus",
                    "promotion_contract_or_release_candidate",
                    "baseline_product_runtime_report",
                ),
                "closure_outputs": (
                    "counterfactual_verification_eval",
                    "product_trace_runtime_evidence_enrichment",
                    "product_trace_replay_workflow",
                    "product_runtime_baseline",
                    "product_runtime_drift_comparison",
                ),
            },
        )
    if evidence_kind == "product_runtime_trajectory_audit_evidence":
        return EvidenceGapAction(
            action_id="rerun_product_trace_trajectory_audit_evidence",
            title="Replay product traces with trajectory-audit evidence",
            action_type="workflow",
            priority=86,
            rationale=(
                "Runtime drift gates need trajectory-level factual, referential, "
                "logical, procedural, and scope audit rates before agent behavior is "
                "treated as stable across releases."
            ),
            evidence_routes=(
                "product_trace_replay",
                "trajectory_audit",
                "product_runtime_baseline",
                "product_runtime_drift",
                "trajectory_audit_evidence",
            ),
            suggested_commands=(
                "benchmarks/run_product_trace_replay_workflow.py "
                "--trace-glob ... --promotion-contract ... "
                "--max-runtime-drift-product-trace-trajectory-audit-failed-trace-rate-increase ... "
                "--max-runtime-drift-product-trace-trajectory-audit-error-rate-increase ... "
                "--max-runtime-drift-product-trace-trajectory-audit-factual-rate-increase ... "
                "--max-runtime-drift-product-trace-trajectory-audit-referential-rate-increase ... "
                "--max-runtime-drift-product-trace-trajectory-audit-logical-rate-increase ... "
                "--max-runtime-drift-product-trace-trajectory-audit-procedural-rate-increase ... "
                "--max-runtime-drift-product-trace-trajectory-audit-scope-rate-increase ...",
                "benchmarks/run_product_runtime_baseline.py "
                "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
                "benchmarks/compare_product_runtime_baselines.py "
                "--current ... --baseline ... "
                "--max-product-trace-trajectory-audit-failed-trace-rate-increase ... "
                "--max-product-trace-trajectory-audit-error-rate-increase ... "
                "--max-product-trace-trajectory-audit-factual-rate-increase ... "
                "--max-product-trace-trajectory-audit-referential-rate-increase ... "
                "--max-product-trace-trajectory-audit-logical-rate-increase ... "
                "--max-product-trace-trajectory-audit-procedural-rate-increase ... "
                "--max-product-trace-trajectory-audit-scope-rate-increase ... "
                "--json ... --artifact-manifest ...",
            ),
            metadata={
                "trajectory_audit_api": "eigentruth.control.audit_product_trace_trajectory",
                "trace_replay_script": "benchmarks/run_product_trace_replay_workflow.py",
                "runtime_baseline_script": "benchmarks/run_product_runtime_baseline.py",
                "runtime_drift_script": "benchmarks/compare_product_runtime_baselines.py",
                "trace_replay_workflow": "product_trace_replay_workflow",
                "runtime_baseline_workflow": "product_runtime_baseline",
                "runtime_drift_workflow": "product_runtime_drift_comparison",
                "risk_control_method": "trajectory_level_hallucination_audit",
                "hallucination_taxonomy": (
                    "factual",
                    "referential",
                    "logical",
                    "procedural",
                    "scope",
                ),
                "required_trace_metrics": (
                    "trajectory_audit.failed_trace_rate",
                    "trajectory_audit.error_rate",
                    "trajectory_audit.factual_rate",
                    "trajectory_audit.referential_rate",
                    "trajectory_audit.logical_rate",
                    "trajectory_audit.procedural_rate",
                    "trajectory_audit.scope_rate",
                ),
                "default_gate_thresholds": {
                    "max_product_trace_trajectory_audit_failed_trace_rate_increase": 0.0,
                    "max_product_trace_trajectory_audit_error_rate_increase": 0.0,
                    "max_product_trace_trajectory_audit_factual_rate_increase": 0.0,
                    "max_product_trace_trajectory_audit_referential_rate_increase": 0.0,
                    "max_product_trace_trajectory_audit_logical_rate_increase": 0.0,
                    "max_product_trace_trajectory_audit_procedural_rate_increase": 0.0,
                    "max_product_trace_trajectory_audit_scope_rate_increase": 0.0,
                },
                "required_inputs": (
                    "full_product_trace_corpus",
                    "promotion_contract_or_release_candidate",
                    "baseline_product_runtime_report",
                ),
                "closure_outputs": (
                    "product_trace_replay_workflow",
                    "product_runtime_baseline",
                    "product_runtime_drift_comparison",
                ),
            },
        )
    if evidence_kind == "product_runtime_evidence_handoff_evidence":
        return EvidenceGapAction(
            action_id="refresh_product_promotion_evidence_handoff",
            title="Refresh product-promotion evidence handoff",
            action_type="workflow",
            priority=85,
            rationale=(
                "Runtime drift gates need the promotion contract to carry a verified "
                "evidence-handoff audit, including missing-metric and promoted-group "
                "rates, before downstream release evidence can be trusted."
            ),
            evidence_routes=(
                "product_promotion_contract",
                "evidence_handoff",
                "product_runtime_drift",
            ),
            suggested_commands=(
                "benchmarks/export_product_promotion_contract_evidence_handoff.py",
                "benchmarks/run_product_runtime_baseline.py",
                "benchmarks/compare_product_runtime_baselines.py",
            ),
        )
    if evidence_kind == "product_runtime_frontier_release_evidence":
        return EvidenceGapAction(
            action_id="refresh_frontier_release_evidence_promotion_metrics",
            title="Refresh frontier release-evidence promotion metrics",
            action_type="workflow",
            priority=85,
            rationale=(
                "Runtime drift gates need the promotion contract to carry frontier "
                "release-evidence coverage, track status, rerun-rollup, and citation "
                "batch metrics before top-level release checks can consume the evidence "
                "without re-running the frontier workflows."
            ),
            evidence_routes=(
                "frontier_release_evidence",
                "product_promotion_contract",
                "product_runtime_drift",
            ),
            suggested_commands=(
                "benchmarks/export_product_promotion_contract_evidence_handoff.py --frontier-release-evidence ...",
                "benchmarks/run_product_runtime_baseline.py",
                "benchmarks/compare_product_runtime_baselines.py",
            ),
        )
    if evidence_kind == "triple_audit":
        return EvidenceGapAction(
            action_id="add_trace_level_triple_audit",
            title="Add trace-level triple and slot-audit evidence",
            action_type="workflow",
            priority=84,
            rationale=(
                "Fact-level verification needs extracted triples, claim coverage, slot "
                "coverage, and pass rates rather than only sentence-level groundedness."
            ),
            evidence_routes=(
                "triple_extraction_fixture_matrix",
                "product_trace_triple_audit_enrichment",
                "product_promotion_contract",
                "product_trace_replay",
                "product_runtime_baseline",
                "product_runtime_drift",
                "triple_audit_evidence",
            ),
            suggested_commands=(
                "benchmarks/run_triple_extraction_fixture_matrix.py "
                "--corpus NAME=... --output-dir ... --artifact-manifest ...",
                "benchmarks/enrich_product_trace_triple_audit.py "
                "--trace-glob ... --evidence-corpus ... --output-dir ... "
                "--registry ... --name ... --version ... "
                "--min-audit-claim-coverage ... --min-audit-pass-rate ... "
                "--min-slot-coverage-rate ...",
                "benchmarks/export_product_promotion_contract_evidence_handoff.py "
                "--contract ... --json ... --audit-json ... "
                "--triple-extraction-fixture-matrix ... "
                "--triple-audit-enrichment ... --artifact-manifest ... "
                "--registry ... --name ... --version ...",
                "benchmarks/run_product_trace_replay_workflow.py "
                "--trace-glob ... --promotion-contract ... "
                "--min-runtime-drift-triple-claim-coverage ... "
                "--min-runtime-drift-triple-audit-claim-coverage ... "
                "--min-runtime-drift-triple-audit-pass-rate ... "
                "--min-runtime-drift-triple-slot-coverage ...",
                "benchmarks/run_product_runtime_baseline.py "
                "--trace ... --promotion-contract ... --json ... --artifact-manifest ...",
                "benchmarks/compare_product_runtime_baselines.py "
                "--current ... --baseline ... "
                "--min-triple-claim-coverage ... "
                "--min-triple-audit-claim-coverage ... "
                "--min-triple-audit-pass-rate ... "
                "--min-triple-slot-coverage ... "
                "--json ... --artifact-manifest ...",
            ),
            metadata={
                "triple_extraction_matrix_script": (
                    "benchmarks/run_triple_extraction_fixture_matrix.py"
                ),
                "trace_enrichment_script": (
                    "benchmarks/enrich_product_trace_triple_audit.py"
                ),
                "evidence_handoff_script": (
                    "benchmarks/export_product_promotion_contract_evidence_handoff.py"
                ),
                "trace_replay_script": "benchmarks/run_product_trace_replay_workflow.py",
                "runtime_baseline_script": "benchmarks/run_product_runtime_baseline.py",
                "runtime_drift_script": "benchmarks/compare_product_runtime_baselines.py",
                "claim_triple_extraction_api": "eigentruth.verify.extract_claim_triples",
                "claim_triple_audit_api": "eigentruth.verify.audit_claim_triples",
                "trace_summary_api": (
                    "eigentruth.control.ProductTrace.triple_coverage_summary"
                ),
                "triple_extraction_matrix_workflow": "triple_extraction_fixture_matrix",
                "trace_enrichment_workflow": "product_trace_triple_audit_enrichment",
                "evidence_handoff_workflow": "product_promotion_evidence_handoff_export",
                "trace_replay_workflow": "product_trace_replay_workflow",
                "runtime_baseline_workflow": "product_runtime_baseline",
                "runtime_drift_workflow": "product_runtime_drift_comparison",
                "risk_control_method": "fact_level_triple_audit",
                "fact_granularity": ("claim_triple", "slot", "predicate"),
                "required_trace_metrics": (
                    "triple_coverage.claim_triple_coverage_rate",
                    "triple_coverage.audit_claim_coverage_rate",
                    "triple_coverage.audit_pass_rate",
                    "triple_coverage.slot_coverage_rate",
                ),
                "default_gate_thresholds": {
                    "min_triple_claim_coverage": 1.0,
                    "min_triple_audit_claim_coverage": 1.0,
                    "min_triple_audit_pass_rate": 1.0,
                    "min_triple_slot_coverage": 1.0,
                },
                "required_inputs": (
                    "structured_fact_corpora",
                    "full_product_trace_corpus",
                    "local_evidence_corpus",
                    "promotion_contract_or_release_candidate",
                    "baseline_product_runtime_report",
                ),
                "closure_outputs": (
                    "triple_extraction_fixture_matrix",
                    "product_trace_triple_audit_enrichment",
                    "product_promotion_evidence_handoff_export",
                    "product_trace_replay_workflow",
                    "product_runtime_baseline",
                    "product_runtime_drift_comparison",
                ),
            },
        )
    if evidence_kind == "covered_fact_property":
        return EvidenceGapAction(
            action_id="refresh_covered_fact_property_routes",
            title="Refresh covered-fact property robustness evidence",
            action_type="workflow",
            priority=82,
            rationale=(
                "Structured fact routes need per-property record/source/quality metrics "
                "so the release can show exactly which predicates are gated."
            ),
            evidence_routes=("structured_fact", "covered_fact_property"),
            suggested_commands=("benchmarks/run_wikidata_structured_qa_route_workflow.py",),
        )
    if evidence_kind == "promotion_contract":
        return EvidenceGapAction(
            action_id="export_promotion_contract_runtime_evidence",
            title="Export promotion contract and runtime evidence",
            action_type="workflow",
            priority=80,
            rationale=(
                "Runtime drift gates require a product promotion contract to be present "
                "inside trace metrics before they can compare handoff coverage."
            ),
            evidence_routes=("product_promotion_contract", "product_runtime_baseline"),
            suggested_commands=(
                "benchmarks/export_product_promotion_contract.py",
                "benchmarks/run_product_runtime_baseline.py",
            ),
        )
    if evidence_kind == "route_baseline":
        return EvidenceGapAction(
            action_id="refresh_required_route_baseline",
            title="Refresh required verifier/retrieval route baseline",
            action_type="benchmark",
            priority=78,
            rationale="Required routes must promote under their own quality, provenance, and stress gates.",
            evidence_routes=("route_baseline",),
            suggested_commands=("benchmarks/compare_route_baselines.py",),
        )
    if evidence_kind == "adapter_family":
        return EvidenceGapAction(
            action_id="refresh_adapter_family_matrix",
            title="Refresh adapter-family matrix evidence",
            action_type="workflow",
            priority=76,
            rationale=(
                "Strict frontier releases need promoted structured-state, state-transition, "
                "triple-evidence, and world-model-backed adapter routes."
            ),
            evidence_routes=("adapter_family_matrix",),
            suggested_commands=("benchmarks/run_adapter_family_matrix.py",),
        )
    if evidence_kind == "external_evidence":
        return EvidenceGapAction(
            action_id="refresh_external_evidence_handoff",
            title="Refresh external evidence handoff",
            action_type="workflow",
            priority=74,
            rationale="External grounding should be provenance-audited before release gates trust it.",
            evidence_routes=("external_evidence_baseline_comparison",),
            suggested_commands=("benchmarks/run_covered_facts_external_evidence_workflow.py",),
        )
    if evidence_kind == "world_model":
        return EvidenceGapAction(
            action_id="refresh_world_model_mechanism_handoff",
            title="Refresh world-model mechanism handoff",
            action_type="workflow",
            priority=72,
            rationale=(
                "World-model correction evidence should expose state references, view functions, "
                "conflicts, and source-backed rule inputs."
            ),
            evidence_routes=("world_model", "mechanism_handoff"),
            suggested_commands=("benchmarks/build_mechanism_handoff_evidence_bundle.py",),
        )
    return EvidenceGapAction(
        action_id=f"inspect_{_slug(gate) or 'release'}_blocker",
        title="Inspect release blocker",
        action_type="manual_triage",
        priority=10,
        rationale=f"No specific planner mapping exists yet for this blocker: {reason}",
        evidence_routes=(gate,),
    )


def _replace_action_sources(
    action: EvidenceGapAction,
    *,
    source_ids: Sequence[str],
) -> EvidenceGapAction:
    return EvidenceGapAction(
        action_id=action.action_id,
        title=action.title,
        action_type=action.action_type,
        priority=action.priority,
        rationale=action.rationale,
        evidence_routes=action.evidence_routes,
        suggested_commands=action.suggested_commands,
        source_gap_ids=tuple(source_ids),
        metadata=action.metadata,
    )


def _extract_missing_metrics(reason: str) -> tuple[str, ...]:
    match = _MISSING_METRICS_RE.search(reason)
    if not match:
        return ()
    return tuple(
        item.strip().strip(".")
        for item in match.group("metrics").split(",")
        if item.strip()
    )


def _summary(
    gaps: Sequence[EvidenceGap],
    actions: Sequence[EvidenceGapAction],
    *,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    gates: dict[str, int] = {}
    root_causes: dict[str, int] = {}
    research_axes: dict[str, int] = {}
    missing_metric_count = 0
    for gap in gaps:
        gates[gap.gate] = gates.get(gap.gate, 0) + 1
        root_causes[gap.root_cause] = root_causes.get(gap.root_cause, 0) + 1
        axis = str(gap.metadata.get("research_axis") or "unknown")
        research_axes[axis] = research_axes.get(axis, 0) + 1
        missing_metric_count += len(gap.missing_metrics)
    return {
        "source_decision_status": decision.get("status"),
        "gap_count": len(gaps),
        "action_count": len(actions),
        "missing_metric_count": missing_metric_count,
        "gates": dict(sorted(gates.items())),
        "root_causes": dict(sorted(root_causes.items())),
        "research_axes": dict(sorted(research_axes.items())),
        "top_action_ids": tuple(
            action.action_id
            for action in sorted(
                actions,
                key=lambda item: (-item.priority, item.action_id),
            )[:5]
        ),
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
