"""Evidence handoff audits for promotion contracts.

The release gate already knows how to fail closed when product-runtime drift
evidence is incomplete. This module makes the missing handoff explicit before a
drift comparison is run: a promotion contract can be audited for the exact
frontier evidence metrics that runtime-drift gates expect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

_ALL_GROUPS = (
    "promotion",
    "pre_generation",
    "counterfactual",
    "triple_audit",
    "covered_fact_property",
    "action_gate",
    "frontier_release_evidence",
)

_ACTION_IDS = {
    "promotion": "export_promotion_contract_runtime_evidence",
    "pre_generation": "run_pre_generation_probe_comparison",
    "counterfactual": "run_counterfactual_verifier_audit",
    "triple_audit": "add_trace_level_triple_audit",
    "covered_fact_property": "refresh_covered_fact_property_routes",
    "action_gate": "rerun_product_trace_action_gates",
    "frontier_release_evidence": "run_frontier_release_evidence_comparison",
}


@dataclass(frozen=True)
class ProductPromotionEvidenceMetric:
    """One runtime-drift evidence metric expected from a promotion contract."""

    group: str
    metric: str
    evidence_key: str
    value: Any = None
    source_path: tuple[str, ...] = ()
    status: str = "missing"

    @property
    def present(self) -> bool:
        """Return whether this metric has usable evidence."""
        return self.status == "present"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "group": self.group,
            "metric": self.metric,
            "evidence_key": self.evidence_key,
            "value": self.value,
            "source_path": self.source_path,
            "status": self.status,
            "present": self.present,
        }


@dataclass(frozen=True)
class ProductPromotionEvidenceGroup:
    """Audit result for one runtime-drift evidence group."""

    group: str
    status: str
    required: bool
    metrics: tuple[ProductPromotionEvidenceMetric, ...] = ()
    recommended_action_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expected_metric_count(self) -> int:
        """Return the number of expected metrics in this group."""
        return len(self.metrics)

    @property
    def present_metric_count(self) -> int:
        """Return the number of present metrics in this group."""
        return sum(1 for metric in self.metrics if metric.present)

    @property
    def missing_metrics(self) -> tuple[str, ...]:
        """Return missing or invalid metric names."""
        return tuple(metric.metric for metric in self.metrics if not metric.present)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "group": self.group,
            "status": self.status,
            "required": self.required,
            "expected_metric_count": self.expected_metric_count,
            "present_metric_count": self.present_metric_count,
            "missing_metric_count": len(self.missing_metrics),
            "missing_metrics": self.missing_metrics,
            "recommended_action_id": self.recommended_action_id,
            "metrics": tuple(metric.to_dict() for metric in self.metrics),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProductPromotionEvidenceAudit:
    """Promotion-contract evidence handoff audit."""

    status: str
    source_workflow: str | None = None
    source_status: str | None = None
    model_id: str | None = None
    required_groups: tuple[str, ...] = _ALL_GROUPS
    groups: tuple[ProductPromotionEvidenceGroup, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @property
    def missing_metrics(self) -> tuple[str, ...]:
        """Return all missing required metric names."""
        required = set(self.required_groups)
        return tuple(metric for group in self.groups if group.group in required for metric in group.missing_metrics)

    @property
    def recommended_action_ids(self) -> tuple[str, ...]:
        """Return de-duplicated actions for missing required groups."""
        action_ids: list[str] = []
        required = set(self.required_groups)
        for group in self.groups:
            if group.group not in required or not group.missing_metrics:
                continue
            if group.recommended_action_id and group.recommended_action_id not in action_ids:
                action_ids.append(group.recommended_action_id)
        return tuple(action_ids)

    @property
    def summary(self) -> dict[str, Any]:
        """Return compact counts for registry metadata and release notes."""
        return {
            "group_count": len(self.groups),
            "required_group_count": len(self.required_groups),
            "expected_metric_count": sum(group.expected_metric_count for group in self.groups),
            "present_metric_count": sum(group.present_metric_count for group in self.groups),
            "missing_metric_count": len(self.missing_metrics),
            "blocked_group_count": sum(
                1 for group in self.groups if group.group in set(self.required_groups) and group.status == "blocked"
            ),
            "groups": {group.group: group.status for group in self.groups},
            "recommended_action_ids": self.recommended_action_ids,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "workflow": "product_promotion_evidence_handoff_audit",
            "status": self.status,
            "source_workflow": self.source_workflow,
            "source_status": self.source_status,
            "model_id": self.model_id,
            "required_groups": self.required_groups,
            "groups": tuple(group.to_dict() for group in self.groups),
            "missing_metrics": self.missing_metrics,
            "recommended_action_ids": self.recommended_action_ids,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProductPromotionEvidenceExport:
    """Result of enriching a promotion contract with runtime evidence handoff fields."""

    contract: Mapping[str, Any]
    before_audit: ProductPromotionEvidenceAudit
    after_audit: ProductPromotionEvidenceAudit
    filled_groups: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @property
    def summary(self) -> dict[str, Any]:
        """Return compact before/after evidence counts."""
        before_summary = self.before_audit.summary
        after_summary = self.after_audit.summary
        return {
            "before_missing_metric_count": before_summary["missing_metric_count"],
            "after_missing_metric_count": after_summary["missing_metric_count"],
            "resolved_missing_metric_count": (
                before_summary["missing_metric_count"] - after_summary["missing_metric_count"]
            ),
            "filled_groups": self.filled_groups,
            "status": self.after_audit.status,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "workflow": "product_promotion_evidence_handoff_export",
            "status": self.after_audit.status,
            "contract": dict(self.contract),
            "before_audit": self.before_audit.to_dict(),
            "after_audit": self.after_audit.to_dict(),
            "filled_groups": self.filled_groups,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


def audit_product_promotion_contract_evidence(
    contract: Mapping[str, Any] | Any,
    *,
    required_groups: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProductPromotionEvidenceAudit:
    """Audit whether a promotion contract carries runtime-drift evidence fields."""
    payload = _payload_mapping(contract)
    required = _required_groups(required_groups)
    groups = tuple(_audit_group(group, payload=payload, required=group in set(required)) for group in _ALL_GROUPS)
    blocked = any(group.status == "blocked" for group in groups if group.group in set(required))
    return ProductPromotionEvidenceAudit(
        status="blocked" if blocked else "promote",
        source_workflow=_optional_str(payload.get("source_workflow") or payload.get("workflow")),
        source_status=_optional_str(payload.get("source_status")),
        model_id=_optional_str(payload.get("model_id")),
        required_groups=required,
        groups=groups,
        metadata=dict(metadata or {}),
    )


def enrich_product_promotion_contract_evidence(
    contract: Mapping[str, Any] | Any,
    *,
    pre_generation_probe_comparison: Mapping[str, Any] | None = None,
    pre_generation_probe_comparison_path: str | None = None,
    triple_extraction_fixture_matrix: Mapping[str, Any] | None = None,
    triple_extraction_fixture_matrix_path: str | None = None,
    counterfactual_verification: Mapping[str, Any] | None = None,
    counterfactual_verification_path: str | None = None,
    product_trace_replay_workflow: Mapping[str, Any] | None = None,
    product_trace_replay_workflow_path: str | None = None,
    frontier_release_evidence: Mapping[str, Any] | None = None,
    frontier_release_evidence_path: str | None = None,
    runtime_baseline: Mapping[str, Any] | None = None,
    runtime_baseline_path: str | None = None,
    covered_fact_property_metrics: Mapping[str, Any] | None = None,
    required_groups: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProductPromotionEvidenceExport:
    """Enrich a promotion contract with auditable runtime-evidence handoff fields.

    Only fields present in the supplied reports are copied into the contract.
    Missing evidence stays missing, so downstream drift gates can still fail
    closed on incomplete handoff coverage.
    """
    payload = deepcopy(dict(_payload_mapping(contract)))
    before = audit_product_promotion_contract_evidence(
        payload,
        required_groups=required_groups,
        metadata=metadata,
    )
    filled_groups: list[str] = []
    export_metadata: dict[str, Any] = {"sources": {}, **dict(metadata or {})}

    matrix = _matrix_handoff_from_report(
        triple_extraction_fixture_matrix,
        path=triple_extraction_fixture_matrix_path,
    )
    if matrix:
        _merge_nested(payload, "triple_extraction_fixture_matrix", matrix)
        _merge_metadata(payload, _triple_matrix_flat_metadata(matrix))
        filled_groups.append("promotion")
        export_metadata["sources"]["triple_extraction_fixture_matrix"] = triple_extraction_fixture_matrix_path

    pre_generation = _pre_generation_handoff_from_report(
        pre_generation_probe_comparison,
        path=pre_generation_probe_comparison_path,
    )
    if pre_generation:
        _merge_nested(payload, "pre_generation_probe_comparison", pre_generation)
        _merge_metadata(payload, _pre_generation_flat_metadata(pre_generation))
        filled_groups.append("pre_generation")
        export_metadata["sources"]["pre_generation_probe_comparison"] = pre_generation_probe_comparison_path

    counterfactual = _counterfactual_handoff_from_report(
        counterfactual_verification,
        path=counterfactual_verification_path,
    )
    if counterfactual:
        _merge_nested(payload, "counterfactual_verification", counterfactual)
        _merge_metadata(payload, _counterfactual_flat_metadata(counterfactual))
        filled_groups.append("counterfactual")
        export_metadata["sources"]["counterfactual_verification"] = counterfactual_verification_path

    product_trace = _product_trace_replay_handoff_from_report(
        product_trace_replay_workflow,
        path=product_trace_replay_workflow_path,
    )
    if product_trace:
        _merge_nested(payload, "product_trace_replay_workflow", product_trace)
        _merge_metadata(payload, _action_gate_flat_metadata(product_trace))
        filled_groups.append("action_gate")
        export_metadata["sources"]["product_trace_replay_workflow"] = product_trace_replay_workflow_path

    frontier_evidence = _frontier_release_evidence_handoff_from_report(
        frontier_release_evidence,
        path=frontier_release_evidence_path,
    )
    if frontier_evidence:
        _merge_nested(payload, "frontier_release_evidence", frontier_evidence)
        _merge_metadata(payload, _frontier_release_evidence_flat_metadata(frontier_evidence))
        filled_groups.append("frontier_release_evidence")
        export_metadata["sources"]["frontier_release_evidence"] = frontier_release_evidence_path

    triple_audit = _triple_audit_flat_metadata_from_runtime_baseline(runtime_baseline)
    if triple_audit:
        _merge_metadata(payload, triple_audit)
        filled_groups.append("triple_audit")
        export_metadata["sources"]["runtime_baseline"] = runtime_baseline_path

    covered_fact = _covered_fact_property_rollup(covered_fact_property_metrics)
    if covered_fact:
        _merge_metadata(
            payload,
            {"recommended_route_covered_fact_property_metrics": covered_fact},
        )
        filled_groups.append("covered_fact_property")
        export_metadata["sources"]["covered_fact_property_metrics"] = "provided"

    filled = tuple(dict.fromkeys(filled_groups))
    export_metadata["filled_groups"] = filled
    after = audit_product_promotion_contract_evidence(
        payload,
        required_groups=required_groups,
        metadata=metadata,
    )
    return ProductPromotionEvidenceExport(
        contract=payload,
        before_audit=before,
        after_audit=after,
        filled_groups=filled,
        metadata=export_metadata,
    )


def _audit_group(
    group: str,
    *,
    payload: Mapping[str, Any],
    required: bool,
) -> ProductPromotionEvidenceGroup:
    metrics = tuple(builder(payload) for builder in _GROUP_BUILDERS[group])
    missing = tuple(metric.metric for metric in metrics if not metric.present)
    status = "blocked" if required and missing else "promote" if not missing else "observed"
    return ProductPromotionEvidenceGroup(
        group=group,
        status=status,
        required=required,
        metrics=metrics,
        recommended_action_id=_ACTION_IDS.get(group),
    )


def _payload_mapping(contract: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(contract, Mapping):
        return contract
    to_dict = getattr(contract, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError("contract must be a mapping or expose to_dict().")


def _required_groups(groups: Sequence[str] | None) -> tuple[str, ...]:
    if groups is None:
        return _ALL_GROUPS
    parsed: list[str] = []
    for group in groups:
        name = str(group).strip()
        if not name:
            continue
        if name not in _ALL_GROUPS:
            raise ValueError(f"unknown evidence group: {name!r}")
        if name not in parsed:
            parsed.append(name)
    return tuple(parsed)


def _metric(
    *,
    group: str,
    metric: str,
    evidence_key: str,
    value: Any,
    source_path: Sequence[str],
) -> ProductPromotionEvidenceMetric:
    status = _value_status(value)
    return ProductPromotionEvidenceMetric(
        group=group,
        metric=metric,
        evidence_key=evidence_key,
        value=value,
        source_path=tuple(source_path),
        status=status,
    )


def _value_status(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "present"
    if isinstance(value, int | float):
        return "present" if math.isfinite(float(value)) else "nonfinite"
    if isinstance(value, str):
        return "present" if value.strip() else "missing"
    if isinstance(value, Mapping | Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "present" if bool(value) else "missing"
    return "present"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("metadata"))


def _first_present(*items: tuple[Any, Sequence[str]]) -> tuple[Any, tuple[str, ...]]:
    for value, path in items:
        if _value_status(value) == "present":
            return value, tuple(path)
    return None, ()


def _nested(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _merge_nested(payload: dict[str, Any], key: str, values: Mapping[str, Any]) -> None:
    existing = _mapping(payload.get(key))
    merged = dict(existing)
    merged.update(_drop_none_values(values))
    payload[key] = merged


def _merge_metadata(payload: dict[str, Any], values: Mapping[str, Any]) -> None:
    metadata = dict(_metadata(payload))
    metadata.update(_drop_none_values(values))
    payload["metadata"] = metadata


def _drop_none_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in values.items() if value is not None}


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_or_original(value: Any) -> Any:
    numeric = _float_or_none(value)
    return numeric if numeric is not None else value


def _manifest_path_from_report(report: Mapping[str, Any], *, path: str | None) -> str | None:
    manifest = _nested(report, "paths", "artifact_manifest")
    if manifest is not None:
        return str(manifest)
    if path is None:
        return None
    return None


def _manifest_verified_from_report(report: Mapping[str, Any]) -> bool | None:
    summary = _mapping(report.get("artifact_manifest_summary"))
    missing = _float_or_none(summary.get("missing_count"))
    if missing is None:
        return None
    return missing == 0.0


def _pre_generation_handoff_from_report(
    report: Mapping[str, Any] | None,
    *,
    path: str | None,
) -> dict[str, Any]:
    if not report:
        return {}
    promotion_gate = _mapping(report.get("promotion_gate"))
    leaderboard = tuple(_mapping(item) for item in report.get("leaderboard") or () if isinstance(item, Mapping))
    best_run = leaderboard[0] if leaderboard else {}
    run_count = sum(1 for run in report.get("runs") or () if isinstance(run, Mapping))
    gate_failures = tuple(promotion_gate.get("failures") or ())
    model_count = _float_or_none(promotion_gate.get("model_count"))
    status = "promote"
    if (
        report.get("workflow") != "pre_generation_probe_workflow_comparison"
        or report.get("status") != "ready"
        or gate_failures
        or model_count is None
        or model_count < 2
        or promotion_gate.get("redline_passed") is not True
        or not leaderboard
    ):
        status = "blocked"
    redline_best_auroc = _float_or_none(best_run.get("redline_best_auroc"))
    redline_margin = _float_or_none(best_run.get("redline_margin"))
    test_label_auroc = _float_or_none(best_run.get("test_label_auroc"))
    return _drop_none_values(
        {
            "report_path": path,
            "manifest_path": _manifest_path_from_report(report, path=path),
            "source": "file" if path is not None else None,
            "workflow": report.get("workflow"),
            "report_status": report.get("status"),
            "status": status,
            "run_count": run_count,
            "model_count": model_count,
            "models": tuple(promotion_gate.get("models") or ()),
            "redline_passed": promotion_gate.get("redline_passed"),
            "redline_run_count": _float_or_none(promotion_gate.get("redline_run_count")),
            "manifest_verified": _manifest_verified_from_report(report),
            "best_run": {
                "name": best_run.get("name"),
                "model": best_run.get("effective_model") or best_run.get("model"),
                "recommended_layer": best_run.get("recommended_layer"),
                "test_label_auroc": test_label_auroc,
                "redline_best_signal": best_run.get("redline_best_signal"),
                "redline_best_auroc": redline_best_auroc,
                "redline_margin": redline_margin,
            },
            "best_test_label_auroc": test_label_auroc,
            "best_redline_auroc": redline_best_auroc,
            "best_redline_margin": redline_margin,
            "blocking_reasons": gate_failures,
        }
    )


def _pre_generation_flat_metadata(comparison: Mapping[str, Any]) -> dict[str, Any]:
    best_run = _mapping(comparison.get("best_run"))
    return _drop_none_values(
        {
            "pre_generation_probe_comparison_report": comparison.get("report_path"),
            "pre_generation_probe_comparison_manifest": comparison.get("manifest_path"),
            "pre_generation_probe_comparison_source": comparison.get("source"),
            "pre_generation_probe_comparison_status": comparison.get("status"),
            "pre_generation_probe_comparison_model_count": comparison.get("model_count"),
            "pre_generation_probe_comparison_run_count": comparison.get("run_count"),
            "pre_generation_probe_comparison_redline_passed": comparison.get("redline_passed"),
            "pre_generation_probe_comparison_redline_run_count": (comparison.get("redline_run_count")),
            "pre_generation_probe_comparison_best_run": best_run.get("name"),
            "pre_generation_probe_comparison_best_model": best_run.get("model"),
            "pre_generation_probe_comparison_best_layer": best_run.get("recommended_layer"),
            "pre_generation_probe_comparison_best_test_label_auroc": (best_run.get("test_label_auroc")),
            "pre_generation_probe_comparison_best_redline_signal": (best_run.get("redline_best_signal")),
            "pre_generation_probe_comparison_best_redline_auroc": (best_run.get("redline_best_auroc")),
            "pre_generation_probe_comparison_best_redline_margin": best_run.get("redline_margin"),
            "pre_generation_probe_comparison_manifest_verified": (comparison.get("manifest_verified")),
        }
    )


def _matrix_handoff_from_report(
    report: Mapping[str, Any] | None,
    *,
    path: str | None,
) -> dict[str, Any]:
    if not report:
        return {}
    status = (
        "promote"
        if report.get("workflow") == "triple_extraction_fixture_matrix" and report.get("status") == "promote"
        else "blocked"
    )
    return _drop_none_values(
        {
            "report_path": path,
            "manifest_path": _manifest_path_from_report(report, path=path),
            "source": "file" if path is not None else None,
            "workflow": report.get("workflow"),
            "report_status": report.get("status"),
            "status": status,
            "n_corpora": _finite_or_original(report.get("n_corpora")),
            "promoted_corpora": _finite_or_original(report.get("promoted_corpora")),
            "distinct_predicate_count": _finite_or_original(report.get("distinct_predicate_count")),
            "distinct_predicates": tuple(report.get("distinct_predicates") or ()),
            "mean_baseline_f1": _float_or_none(report.get("mean_baseline_f1")),
            "mean_best_f1": _float_or_none(report.get("mean_best_f1")),
            "mean_f1_lift": _float_or_none(report.get("mean_f1_lift")),
            "external_prediction_count": _float_or_none(report.get("external_prediction_count")),
            "external_prediction_corpora": tuple(report.get("external_prediction_corpora") or ()),
            "mean_best_external_f1": _float_or_none(report.get("mean_best_external_f1")),
        }
    )


def _triple_matrix_flat_metadata(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _drop_none_values(
        {
            "triple_extraction_fixture_matrix_report": matrix.get("report_path"),
            "triple_extraction_fixture_matrix_manifest": matrix.get("manifest_path"),
            "triple_extraction_fixture_matrix_source": matrix.get("source"),
            "triple_extraction_fixture_matrix_status": matrix.get("status"),
            "triple_extraction_fixture_matrix_n_corpora": matrix.get("n_corpora"),
            "triple_extraction_fixture_matrix_promoted_corpora": matrix.get("promoted_corpora"),
            "triple_extraction_fixture_matrix_distinct_predicate_count": matrix.get("distinct_predicate_count"),
            "triple_extraction_fixture_matrix_distinct_predicates": matrix.get("distinct_predicates"),
            "triple_extraction_fixture_matrix_mean_baseline_f1": matrix.get("mean_baseline_f1"),
            "triple_extraction_fixture_matrix_mean_best_f1": matrix.get("mean_best_f1"),
            "triple_extraction_fixture_matrix_mean_f1_lift": matrix.get("mean_f1_lift"),
        }
    )


def _counterfactual_handoff_from_report(
    report: Mapping[str, Any] | None,
    *,
    path: str | None,
) -> dict[str, Any]:
    if not report:
        return {}
    summary = _mapping(_mapping(report.get("report")).get("summary")) or _mapping(report.get("summary"))
    status = "promote"
    if report.get("workflow") not in {
        "counterfactual_verification_eval",
        "counterfactual_verification_audit",
    }:
        status = "blocked"
    pass_rate = _float_or_none(summary.get("pass_rate"))
    false_invariance_rate = _float_or_none(summary.get("false_invariance_rate"))
    if pass_rate is not None and pass_rate < 1.0:
        status = "blocked"
    if false_invariance_rate is not None and false_invariance_rate > 0.0:
        status = "blocked"
    return _drop_none_values(
        {
            "report_path": path,
            "manifest_path": _manifest_path_from_report(report, path=path),
            "source": "file" if path is not None else None,
            "workflow": report.get("workflow"),
            "status": status,
            "record_count": _float_or_none(summary.get("record_count")),
            "pass_rate": pass_rate,
            "false_invariance_rate": false_invariance_rate,
            "flip_success_count": _float_or_none(summary.get("flip_success_count")),
            "manifest_verified": _manifest_verified_from_report(report),
        }
    )


def _counterfactual_flat_metadata(audit: Mapping[str, Any]) -> dict[str, Any]:
    return _drop_none_values(
        {
            "counterfactual_verification_report": audit.get("report_path"),
            "counterfactual_verification_manifest": audit.get("manifest_path"),
            "counterfactual_verification_source": audit.get("source"),
            "counterfactual_verification_status": audit.get("status"),
            "counterfactual_verification_workflow": audit.get("workflow"),
            "counterfactual_verification_record_count": audit.get("record_count"),
            "counterfactual_verification_pass_rate": audit.get("pass_rate"),
            "counterfactual_verification_false_invariance_rate": (audit.get("false_invariance_rate")),
            "counterfactual_verification_flip_success_count": (audit.get("flip_success_count")),
            "counterfactual_verification_manifest_verified": audit.get("manifest_verified"),
        }
    )


def _product_trace_replay_handoff_from_report(
    report: Mapping[str, Any] | None,
    *,
    path: str | None,
) -> dict[str, Any]:
    if not report:
        return {}
    action_audit = _mapping(report.get("action_audit_gate"))
    action_execution = _mapping(report.get("action_execution_gate"))
    if not action_audit:
        action_audit = _mapping(_nested(report, "runtime_baseline", "summary", "action_audit"))
    if not action_execution:
        action_execution = _mapping(_nested(report, "runtime_baseline", "summary", "action_execution"))
    paths = _mapping(report.get("paths"))
    return _drop_none_values(
        {
            "report_path": path or paths.get("report"),
            "manifest_path": _manifest_path_from_report(report, path=path),
            "source": "file" if path is not None else None,
            "workflow": report.get("workflow"),
            "status": report.get("status"),
            "report_status": report.get("status"),
            "selector_replay_report_path": paths.get("selector_replay_report"),
            "product_runtime_drift_report_path": paths.get("runtime_drift_report"),
            "action_audit_gate": dict(action_audit),
            "action_execution_gate": dict(action_execution),
        }
    )


def _action_gate_flat_metadata(workflow: Mapping[str, Any]) -> dict[str, Any]:
    action_audit = _mapping(workflow.get("action_audit_gate"))
    action_execution = _mapping(workflow.get("action_execution_gate"))
    return _drop_none_values(
        {
            "product_trace_replay_workflow_report": workflow.get("report_path"),
            "product_trace_replay_workflow_manifest": workflow.get("manifest_path"),
            "product_trace_replay_workflow_status": workflow.get("status"),
            "product_trace_replay_workflow_report_status": workflow.get("report_status"),
            "product_trace_replay_workflow_selector_replay_report": (workflow.get("selector_replay_report_path")),
            "product_trace_replay_workflow_runtime_drift_report": (workflow.get("product_runtime_drift_report_path")),
            "product_trace_action_audit_error_rate": _float_or_none(action_audit.get("error_rate")),
            "product_trace_action_audit_missing_retrieval_action_rate": _float_or_none(
                action_audit.get("missing_retrieval_action_rate")
            ),
            "product_trace_action_audit_missing_plan_retrieval_query_rate": _float_or_none(
                action_audit.get("missing_plan_retrieval_query_rate")
            ),
            "product_trace_action_audit_malformed_payload_rate": _float_or_none(
                action_audit.get("malformed_payload_rate")
            ),
            "product_trace_action_audit_unexpected_action_rate": _float_or_none(
                action_audit.get("unexpected_action_rate")
            ),
            "product_trace_action_audit_unknown_claim_id_rate": _float_or_none(
                action_audit.get("unknown_claim_id_rate")
            ),
            "product_trace_action_execution_alignment_failed_trace_rate": _float_or_none(
                action_execution.get("alignment_failed_trace_rate")
            ),
            "product_trace_action_execution_missing_result_rate": _float_or_none(
                action_execution.get("missing_result_rate")
            ),
            "product_trace_action_execution_unexpected_result_rate": _float_or_none(
                action_execution.get("unexpected_result_rate")
            ),
            "product_trace_action_execution_request_id_mismatch_rate": _float_or_none(
                action_execution.get("request_id_mismatch_rate")
            ),
        }
    )


def _frontier_release_evidence_handoff_from_report(
    report: Mapping[str, Any] | None,
    *,
    path: str | None,
) -> dict[str, Any]:
    if not report:
        return {}
    decision = _mapping(report.get("decision"))
    summary = _mapping(report.get("evidence_summary"))
    paths = _mapping(report.get("paths"))
    blocking_reasons = tuple(decision.get("blocking_reasons") or ())
    status = "promote"
    if (
        report.get("workflow") != "frontier_release_evidence_comparison"
        or report.get("status") != "complete"
        or decision.get("status") != "promote"
        or decision.get("verifier_track_status") != "promote"
        or decision.get("abstention_track_status") != "promote"
        or decision.get("multiple_testing_track_status") not in {None, "promote", "not_required"}
        or blocking_reasons
    ):
        status = "blocked"
    return _drop_none_values(
        {
            "report_path": path or paths.get("report") or report.get("report_path"),
            "manifest_path": _manifest_path_from_report(report, path=path),
            "source": "file" if path is not None else None,
            "workflow": report.get("workflow"),
            "status": status,
            "report_status": report.get("status"),
            "decision_status": decision.get("status"),
            "verifier_track_status": decision.get("verifier_track_status"),
            "abstention_track_status": decision.get("abstention_track_status"),
            "multiple_testing_track_status": decision.get("multiple_testing_track_status"),
            "run_names": tuple(summary.get("run_names") or ()),
            "blocking_reasons": blocking_reasons,
        }
    )


def _frontier_release_evidence_flat_metadata(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return _drop_none_values(
        {
            "frontier_release_evidence_report": evidence.get("report_path"),
            "frontier_release_evidence_manifest": evidence.get("manifest_path"),
            "frontier_release_evidence_source": evidence.get("source"),
            "frontier_release_evidence_status": evidence.get("status"),
            "frontier_release_evidence_workflow": evidence.get("workflow"),
            "frontier_release_evidence_report_status": evidence.get("report_status"),
            "frontier_release_evidence_decision_status": evidence.get("decision_status"),
            "frontier_release_evidence_verifier_track_status": (evidence.get("verifier_track_status")),
            "frontier_release_evidence_abstention_track_status": (evidence.get("abstention_track_status")),
            "frontier_release_evidence_multiple_testing_track_status": (
                evidence.get("multiple_testing_track_status")
            ),
            "frontier_release_evidence_run_names": evidence.get("run_names"),
            "frontier_release_evidence_blocking_reasons": evidence.get("blocking_reasons"),
        }
    )


def _triple_audit_flat_metadata_from_runtime_baseline(
    runtime_baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not runtime_baseline:
        return {}
    triple_coverage = _mapping(_nested(runtime_baseline, "summary", "triple_coverage"))
    if not triple_coverage:
        return {}
    return _drop_none_values(
        {
            "triple_claim_coverage_rate": _float_or_none(triple_coverage.get("claim_triple_coverage_rate")),
            "triple_audit_claim_coverage_rate": _float_or_none(triple_coverage.get("audit_claim_coverage_rate")),
            "triple_audit_pass_rate": _float_or_none(triple_coverage.get("audit_pass_rate")),
            "triple_slot_coverage_rate": _float_or_none(triple_coverage.get("slot_coverage_rate")),
        }
    )


def _covered_fact_property_rollup(
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not metrics:
        return {}
    if any(
        key in metrics
        for key in (
            "property_metric_count",
            "min_records",
            "min_source_documents",
            "min_decision_accuracy",
            "max_false_supported_rate",
            "min_false_refuted_rate",
        )
    ):
        return _drop_none_values(metrics)

    property_metrics = _covered_fact_property_metric_items(metrics)
    records: list[float] = []
    source_documents: list[float] = []
    decision_accuracy: list[float] = []
    false_supported: list[float] = []
    false_refuted: list[float] = []
    for item in property_metrics.values():
        if not isinstance(item, Mapping):
            continue
        record_count = _float_or_none(_first_not_none(item.get("n_records"), item.get("records"), item.get("selected")))
        source_count = _float_or_none(
            _first_not_none(
                item.get("n_source_documents"),
                item.get("source_documents"),
                item.get("source_document_count"),
            )
        )
        accuracy = _float_or_none(item.get("decision_accuracy"))
        false_supported_rate = _float_or_none(item.get("false_supported_rate"))
        false_refuted_rate = _float_or_none(item.get("false_refuted_rate"))
        if record_count is not None:
            records.append(record_count)
        if source_count is not None:
            source_documents.append(source_count)
        if accuracy is not None:
            decision_accuracy.append(accuracy)
        if false_supported_rate is not None:
            false_supported.append(false_supported_rate)
        if false_refuted_rate is not None:
            false_refuted.append(false_refuted_rate)
    return _drop_none_values(
        {
            "property_metric_count": len(property_metrics),
            "min_records": min(records) if records else None,
            "min_source_documents": min(source_documents) if source_documents else None,
            "min_decision_accuracy": min(decision_accuracy) if decision_accuracy else None,
            "max_false_supported_rate": max(false_supported) if false_supported else None,
            "min_false_refuted_rate": min(false_refuted) if false_refuted else None,
        }
    )


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _covered_fact_property_metric_items(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in (
        "covered_fact_property_metrics",
        "recommended_route_covered_fact_property_metrics",
        "fact_group_metrics",
        "property_metrics",
    ):
        nested = _mapping(metrics.get(key))
        if nested:
            return _merge_covered_fact_source_counts(
                nested,
                _mapping(_nested(metrics, "score_dump_summary", "by_fact_group"))
                or _mapping(_nested(metrics, "summary", "by_fact_group")),
            )
    return metrics


def _merge_covered_fact_source_counts(
    property_metrics: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if not source_metrics:
        return dict(property_metrics)
    merged: dict[str, Any] = {}
    for key, value in property_metrics.items():
        if not isinstance(value, Mapping):
            merged[key] = value
            continue
        item = dict(value)
        source_item = _mapping(source_metrics.get(key))
        source_documents = _float_or_none(source_item.get("n_source_documents"))
        if source_documents is not None:
            current = _float_or_none(item.get("n_source_documents"))
            if current is None or current <= 0:
                item["n_source_documents"] = source_documents
        merged[key] = item
    return merged


def _coverage_from_group(payload: Mapping[str, Any], group_name: str) -> tuple[Any, tuple[str, ...]]:
    group = _mapping(payload.get(group_name))
    metadata = _metadata(payload)
    available = bool(group)
    source = group.get("source") or group.get("report_path") or group.get("record_key")
    status = group.get("status")
    if available or _value_status(source) == "present" or _value_status(status) == "present":
        return 1.0, (group_name,)
    flat_source = metadata.get(f"{group_name}_source")
    flat_status = metadata.get(f"{group_name}_status")
    flat_report = metadata.get(f"{group_name}_report")
    if any(_value_status(value) == "present" for value in (flat_source, flat_status, flat_report)):
        return 1.0, ("metadata",)
    return None, ()


def _manifest_verified_rate(
    payload: Mapping[str, Any],
    group_name: str,
) -> tuple[Any, tuple[str, ...]]:
    group = _mapping(payload.get(group_name))
    metadata = _metadata(payload)
    value, path = _first_present(
        (group.get("manifest_verified"), (group_name, "manifest_verified")),
        (_nested(group, "manifest_verification", "passed"), (group_name, "manifest_verification", "passed")),
        (metadata.get(f"{group_name}_manifest_verified"), ("metadata", f"{group_name}_manifest_verified")),
    )
    if isinstance(value, bool):
        return (1.0 if value else 0.0), path
    return value, path


def _group_or_metadata_value(
    payload: Mapping[str, Any],
    group_name: str,
    key: str,
    *,
    metadata_key: str | None = None,
) -> tuple[Any, tuple[str, ...]]:
    group = _mapping(payload.get(group_name))
    metadata = _metadata(payload)
    flat_key = metadata_key or f"{group_name}_{key}"
    return _first_present(
        (group.get(key), (group_name, key)),
        (metadata.get(flat_key), ("metadata", flat_key)),
    )


def _metadata_value(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[Any, tuple[str, ...]]:
    metadata = _metadata(payload)
    return _first_present((metadata.get(key), ("metadata", key)))


def _contract_coverage(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
    metadata = _metadata(payload)
    value = 1.0 if payload.get("workflow") == "product_promotion_contract" else None
    source_path: tuple[str, ...] = ("workflow",) if value is not None else ()
    if value is None and (
        _value_status(payload.get("source_status")) == "present"
        or _value_status(metadata.get("promotion_contract_source_status")) == "present"
    ):
        value = 1.0
        source_path = ("source_status",)
    return _metric(
        group="promotion",
        metric="promotion_contract.coverage_rate",
        evidence_key="promotion_contract_coverage_rate",
        value=value,
        source_path=source_path,
    )


def _triple_matrix_coverage(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
    value, path = _coverage_from_group(payload, "triple_extraction_fixture_matrix")
    return _metric(
        group="promotion",
        metric="promotion_contract.triple_extraction_fixture_matrix.coverage_rate",
        evidence_key="triple_extraction_fixture_matrix_coverage_rate",
        value=value,
        source_path=path,
    )


def _triple_matrix_metric(metric_suffix: str, key: str):
    def build(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
        value, path = _group_or_metadata_value(payload, "triple_extraction_fixture_matrix", key)
        return _metric(
            group="promotion",
            metric=f"promotion_contract.triple_extraction_fixture_matrix.{metric_suffix}",
            evidence_key=f"triple_extraction_fixture_matrix_{key}",
            value=value,
            source_path=path,
        )

    return build


def _pre_generation_coverage(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
    value, path = _coverage_from_group(payload, "pre_generation_probe_comparison")
    return _metric(
        group="pre_generation",
        metric="promotion_contract.pre_generation_probe_comparison.coverage_rate",
        evidence_key="pre_generation_probe_comparison_coverage_rate",
        value=value,
        source_path=path,
    )


def _pre_generation_manifest(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
    value, path = _manifest_verified_rate(payload, "pre_generation_probe_comparison")
    return _metric(
        group="pre_generation",
        metric="promotion_contract.pre_generation_probe_comparison.manifest_verified_rate",
        evidence_key="pre_generation_probe_comparison_manifest_verified_rate",
        value=value,
        source_path=path,
    )


def _pre_generation_metric(metric_suffix: str, key: str, *, bool_rate: bool = False):
    def build(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
        value, path = _group_or_metadata_value(payload, "pre_generation_probe_comparison", key)
        if bool_rate and isinstance(value, bool):
            value = 1.0 if value else 0.0
        return _metric(
            group="pre_generation",
            metric=f"promotion_contract.pre_generation_probe_comparison.{metric_suffix}",
            evidence_key=f"pre_generation_probe_comparison_{key}",
            value=value,
            source_path=path,
        )

    return build


def _counterfactual_coverage(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
    value, path = _coverage_from_group(payload, "counterfactual_verification")
    return _metric(
        group="counterfactual",
        metric="promotion_contract.counterfactual_verification.coverage_rate",
        evidence_key="counterfactual_verification_coverage_rate",
        value=value,
        source_path=path,
    )


def _counterfactual_manifest(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
    value, path = _manifest_verified_rate(payload, "counterfactual_verification")
    return _metric(
        group="counterfactual",
        metric="promotion_contract.counterfactual_verification.manifest_verified_rate",
        evidence_key="counterfactual_verification_manifest_verified_rate",
        value=value,
        source_path=path,
    )


def _counterfactual_metric(metric_suffix: str, key: str):
    def build(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
        value, path = _group_or_metadata_value(payload, "counterfactual_verification", key)
        return _metric(
            group="counterfactual",
            metric=f"promotion_contract.counterfactual_verification.{metric_suffix}",
            evidence_key=f"counterfactual_verification_{key}",
            value=value,
            source_path=path,
        )

    return build


def _triple_audit_metric(metric_suffix: str, key: str):
    def build(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
        value, path = _metadata_value(payload, key)
        return _metric(
            group="triple_audit",
            metric=f"triple_coverage.{metric_suffix}",
            evidence_key=key,
            value=value,
            source_path=path,
        )

    return build


def _covered_fact_metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _metadata(payload)
    route = _mapping(payload.get("verifier_route"))
    return _mapping(
        _first_present(
            (
                metadata.get("recommended_route_covered_fact_property_metrics"),
                ("metadata", "recommended_route_covered_fact_property_metrics"),
            ),
            (
                route.get("covered_fact_property_metrics"),
                ("verifier_route", "covered_fact_property_metrics"),
            ),
        )[0]
    )


def _covered_fact_metric(metric_suffix: str, key: str):
    def build(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
        metrics = _covered_fact_metrics(payload)
        source_path = ("metadata", "recommended_route_covered_fact_property_metrics", key)
        value = metrics.get(key)
        if key == "property_metric_count" and value is None:
            value, source_path = _metadata_value(
                payload,
                "recommended_route_covered_fact_property_count",
            )
        return _metric(
            group="covered_fact_property",
            metric=(f"promotion_contract.covered_fact_properties.recommended_route_property_metrics.{metric_suffix}"),
            evidence_key=f"covered_fact_recommended_route_{key}",
            value=value,
            source_path=source_path if _value_status(value) == "present" else (),
        )

    return build


def _action_gate_metric(metric_suffix: str, key: str):
    def build(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
        value, path = _metadata_value(payload, key)
        return _metric(
            group="action_gate",
            metric=f"promotion_contract.product_trace_replay.{metric_suffix}",
            evidence_key=key,
            value=value,
            source_path=path,
        )

    return build


def _frontier_release_evidence_coverage(
    payload: Mapping[str, Any],
) -> ProductPromotionEvidenceMetric:
    value, path = _coverage_from_group(payload, "frontier_release_evidence")
    if value is None:
        metadata = _metadata(payload)
        if any(
            _value_status(metadata.get(key)) == "present"
            for key in (
                "promotion_contract_frontier_release_evidence_report",
                "promotion_contract_frontier_release_evidence_status",
                "promotion_contract_frontier_release_evidence_decision_status",
            )
        ):
            value = 1.0
            path = ("metadata",)
    return _metric(
        group="frontier_release_evidence",
        metric="promotion_contract.frontier_release_evidence.coverage_rate",
        evidence_key="frontier_release_evidence_coverage_rate",
        value=value,
        source_path=path,
    )


def _frontier_release_evidence_value(
    payload: Mapping[str, Any],
    key: str,
    *metadata_keys: str,
) -> tuple[Any, tuple[str, ...]]:
    group = _mapping(payload.get("frontier_release_evidence"))
    metadata = _metadata(payload)
    candidates: list[tuple[Any, Sequence[str]]] = [
        (group.get(key), ("frontier_release_evidence", key)),
    ]
    for metadata_key in metadata_keys:
        candidates.append((metadata.get(metadata_key), ("metadata", metadata_key)))
    candidates.append(
        (
            metadata.get(f"frontier_release_evidence_{key}"),
            ("metadata", f"frontier_release_evidence_{key}"),
        )
    )
    candidates.append(
        (
            metadata.get(f"promotion_contract_frontier_release_evidence_{key}"),
            ("metadata", f"promotion_contract_frontier_release_evidence_{key}"),
        )
    )
    return _first_present(*candidates)


def _frontier_release_evidence_metric(metric_suffix: str, key: str, *metadata_keys: str):
    def build(payload: Mapping[str, Any]) -> ProductPromotionEvidenceMetric:
        value, path = _frontier_release_evidence_value(
            payload,
            key,
            *metadata_keys,
        )
        return _metric(
            group="frontier_release_evidence",
            metric=f"promotion_contract.frontier_release_evidence.{metric_suffix}",
            evidence_key=f"frontier_release_evidence_{key}",
            value=value,
            source_path=path,
        )

    return build


_GROUP_BUILDERS = {
    "promotion": (
        _contract_coverage,
        _triple_matrix_coverage,
        _triple_matrix_metric("mean_best_f1.mean", "mean_best_f1"),
        _triple_matrix_metric("mean_f1_lift.mean", "mean_f1_lift"),
    ),
    "pre_generation": (
        _pre_generation_coverage,
        _pre_generation_manifest,
        _pre_generation_metric("model_count.mean", "model_count"),
        _pre_generation_metric("run_count.mean", "run_count"),
        _pre_generation_metric("redline_pass_rate", "redline_passed", bool_rate=True),
        _pre_generation_metric("best_test_label_auroc.mean", "best_test_label_auroc"),
        _pre_generation_metric("best_redline_auroc.mean", "best_redline_auroc"),
        _pre_generation_metric("best_redline_margin.mean", "best_redline_margin"),
    ),
    "counterfactual": (
        _counterfactual_coverage,
        _counterfactual_manifest,
        _counterfactual_metric("record_count.mean", "record_count"),
        _counterfactual_metric("pass_rate.mean", "pass_rate"),
        _counterfactual_metric("false_invariance_rate.mean", "false_invariance_rate"),
        _counterfactual_metric("flip_success_count.mean", "flip_success_count"),
    ),
    "triple_audit": (
        _triple_audit_metric("claim_triple_coverage_rate", "triple_claim_coverage_rate"),
        _triple_audit_metric("audit_claim_coverage_rate", "triple_audit_claim_coverage_rate"),
        _triple_audit_metric("audit_pass_rate", "triple_audit_pass_rate"),
        _triple_audit_metric("slot_coverage_rate", "triple_slot_coverage_rate"),
    ),
    "covered_fact_property": (
        _covered_fact_metric("property_metric_count.mean", "property_metric_count"),
        _covered_fact_metric("min_records.mean", "min_records"),
        _covered_fact_metric("min_source_documents.mean", "min_source_documents"),
        _covered_fact_metric("min_decision_accuracy.mean", "min_decision_accuracy"),
        _covered_fact_metric("max_false_supported_rate.mean", "max_false_supported_rate"),
        _covered_fact_metric("min_false_refuted_rate.mean", "min_false_refuted_rate"),
    ),
    "action_gate": (
        _action_gate_metric("action_audit_gate.error_rate.mean", "product_trace_action_audit_error_rate"),
        _action_gate_metric(
            "action_audit_gate.missing_retrieval_action_rate.mean",
            "product_trace_action_audit_missing_retrieval_action_rate",
        ),
        _action_gate_metric(
            "action_audit_gate.missing_plan_retrieval_query_rate.mean",
            "product_trace_action_audit_missing_plan_retrieval_query_rate",
        ),
        _action_gate_metric(
            "action_audit_gate.malformed_payload_rate.mean",
            "product_trace_action_audit_malformed_payload_rate",
        ),
        _action_gate_metric(
            "action_audit_gate.unexpected_action_rate.mean",
            "product_trace_action_audit_unexpected_action_rate",
        ),
        _action_gate_metric(
            "action_audit_gate.unknown_claim_id_rate.mean",
            "product_trace_action_audit_unknown_claim_id_rate",
        ),
        _action_gate_metric(
            "action_execution_gate.alignment_failed_trace_rate.mean",
            "product_trace_action_execution_alignment_failed_trace_rate",
        ),
        _action_gate_metric(
            "action_execution_gate.missing_result_rate.mean",
            "product_trace_action_execution_missing_result_rate",
        ),
        _action_gate_metric(
            "action_execution_gate.unexpected_result_rate.mean",
            "product_trace_action_execution_unexpected_result_rate",
        ),
        _action_gate_metric(
            "action_execution_gate.request_id_mismatch_rate.mean",
            "product_trace_action_execution_request_id_mismatch_rate",
        ),
    ),
    "frontier_release_evidence": (
        _frontier_release_evidence_coverage,
        _frontier_release_evidence_metric(
            "report_path",
            "report_path",
            "frontier_release_evidence_report",
            "promotion_contract_frontier_release_evidence_report",
        ),
        _frontier_release_evidence_metric(
            "manifest_path",
            "manifest_path",
            "frontier_release_evidence_manifest",
            "promotion_contract_frontier_release_evidence_manifest",
        ),
        _frontier_release_evidence_metric("status", "status"),
        _frontier_release_evidence_metric("decision_status", "decision_status"),
        _frontier_release_evidence_metric(
            "verifier_track_status",
            "verifier_track_status",
        ),
        _frontier_release_evidence_metric(
            "abstention_track_status",
            "abstention_track_status",
        ),
        _frontier_release_evidence_metric(
            "multiple_testing_track_status",
            "multiple_testing_track_status",
        ),
        _frontier_release_evidence_metric("run_names", "run_names"),
    ),
}
