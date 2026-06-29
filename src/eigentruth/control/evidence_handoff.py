"""Evidence handoff audits for promotion contracts.

The release gate already knows how to fail closed when product-runtime drift
evidence is incomplete. This module makes the missing handoff explicit before a
drift comparison is run: a promotion contract can be audited for the exact
frontier evidence metrics that runtime-drift gates expect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_ALL_GROUPS = (
    "promotion",
    "pre_generation",
    "counterfactual",
    "triple_audit",
    "covered_fact_property",
    "action_gate",
)

_ACTION_IDS = {
    "promotion": "export_promotion_contract_runtime_evidence",
    "pre_generation": "run_pre_generation_probe_comparison",
    "counterfactual": "run_counterfactual_verifier_audit",
    "triple_audit": "add_trace_level_triple_audit",
    "covered_fact_property": "refresh_covered_fact_property_routes",
    "action_gate": "rerun_product_trace_action_gates",
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
        return tuple(
            metric
            for group in self.groups
            if group.group in required
            for metric in group.missing_metrics
        )

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
                1
                for group in self.groups
                if group.group in set(self.required_groups) and group.status == "blocked"
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


def audit_product_promotion_contract_evidence(
    contract: Mapping[str, Any] | Any,
    *,
    required_groups: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProductPromotionEvidenceAudit:
    """Audit whether a promotion contract carries runtime-drift evidence fields."""
    payload = _payload_mapping(contract)
    required = _required_groups(required_groups)
    groups = tuple(
        _audit_group(group, payload=payload, required=group in set(required))
        for group in _ALL_GROUPS
    )
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
            metric=(
                "promotion_contract.covered_fact_properties."
                f"recommended_route_property_metrics.{metric_suffix}"
            ),
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
}
