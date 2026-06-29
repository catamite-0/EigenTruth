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
    blocker_records = _blocking_records(decision)

    gaps: list[EvidenceGap] = []
    action_sources: dict[str, set[str]] = {}
    action_templates: dict[str, EvidenceGapAction] = {}
    for gap_index, blocker in enumerate(blocker_records, start=1):
        gate = blocker["gate"]
        gate_status = blocker["status"]
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


def _classify_gap(
    gate: str,
    reason: str,
    *,
    missing_metrics: Sequence[str],
) -> dict[str, str]:
    text = f"{gate} {reason}".lower()
    if "pre-generation" in text or "pre_generation" in text:
        return _kind("pre_generation_probe", "model", "internal_state")
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
            evidence_routes=("pre_generation_probe_comparison", "product_runtime_drift"),
            suggested_commands=("benchmarks/compare_pre_generation_probe_workflows.py",),
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
                "Agent-style hallucinations include fabricated, missing, or malformed tool "
                "actions; release evidence should carry action-audit and execution alignment rates."
            ),
            evidence_routes=("product_trace_replay", "action_audit", "action_execution"),
            suggested_commands=("benchmarks/run_product_trace_replay_workflow.py",),
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
            evidence_routes=("triple_evidence", "product_runtime_drift"),
            suggested_commands=(
                "benchmarks/run_triple_extraction_fixture_matrix.py",
                "benchmarks/run_product_runtime_baseline.py",
            ),
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
