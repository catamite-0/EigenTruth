"""Build a ProductTrace runtime baseline report.

This workflow aggregates already-emitted product traces. It does not run a
model, verifier, retriever, or external service. The purpose is to make the
control-plane runtime budget auditable across a sample of real or demo requests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TRACE_RECORD_CACHE_SCHEMA_VERSION = 5
_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "promotion_contract_coverage_rate",
    "triple_extraction_fixture_matrix_coverage_rate",
    "triple_extraction_fixture_matrix_mean_best_f1",
    "triple_extraction_fixture_matrix_mean_f1_lift",
)
_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "triple_claim_coverage_rate",
    "triple_audit_claim_coverage_rate",
    "triple_audit_pass_rate",
    "triple_slot_coverage_rate",
)
_PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "covered_fact_recommended_route_property_metric_count",
    "covered_fact_recommended_route_min_records",
    "covered_fact_recommended_route_min_source_documents",
    "covered_fact_recommended_route_min_decision_accuracy",
    "covered_fact_recommended_route_max_false_supported_rate",
    "covered_fact_recommended_route_min_false_refuted_rate",
)
_PROMOTION_CONTRACT_PRODUCT_RUNTIME_DRIFT_FIELDS: tuple[str, ...] = (
    "promotion_contract_product_runtime_drift_available",
    "promotion_contract_product_runtime_drift_status",
    "promotion_contract_product_runtime_drift_report",
    "promotion_contract_product_runtime_drift_manifest",
    "promotion_contract_product_runtime_drift_baseline_path",
    "promotion_contract_product_runtime_drift_current_path",
    "promotion_contract_product_runtime_drift_gate_enabled",
    "promotion_contract_product_runtime_drift_promotion_evidence_required",
    "promotion_contract_product_runtime_drift_triple_audit_evidence_required",
    "promotion_contract_product_runtime_drift_covered_fact_property_evidence_required",
    "promotion_contract_product_runtime_drift_compared_metric_count",
    "promotion_contract_product_runtime_drift_blocked_metric_count",
    "promotion_contract_product_runtime_drift_promotion_evidence_metric_count",
    "promotion_contract_product_runtime_drift_promotion_evidence_blocked_metric_count",
    "promotion_contract_product_runtime_drift_triple_audit_evidence_metric_count",
    "promotion_contract_product_runtime_drift_triple_audit_evidence_blocked_metric_count",
    "promotion_contract_product_runtime_drift_covered_fact_property_evidence_metric_count",
    "promotion_contract_product_runtime_drift_covered_fact_property_evidence_blocked_metric_count",
)
_PROMOTION_CONTRACT_COVERED_FACT_ROLLUP_FIELDS: tuple[str, ...] = (
    "promotion_contract_recommended_route_covered_fact_property_metric_count",
    "promotion_contract_recommended_route_covered_fact_min_records",
    "promotion_contract_recommended_route_covered_fact_min_source_documents",
    "promotion_contract_recommended_route_covered_fact_min_decision_accuracy",
    "promotion_contract_recommended_route_covered_fact_max_false_supported_rate",
    "promotion_contract_recommended_route_covered_fact_min_false_refuted_rate",
    "promotion_contract_required_route_baseline_covered_fact_property_metric_count",
    "promotion_contract_required_route_baseline_covered_fact_min_records",
    "promotion_contract_required_route_baseline_covered_fact_min_source_documents",
    "promotion_contract_required_route_baseline_covered_fact_min_decision_accuracy",
    "promotion_contract_required_route_baseline_covered_fact_max_false_supported_rate",
    "promotion_contract_required_route_baseline_covered_fact_min_false_refuted_rate",
    "promotion_contract_structured_fact_robustness_property_metric_count",
    "promotion_contract_structured_fact_robustness_min_records",
    "promotion_contract_structured_fact_robustness_min_source_documents",
    "promotion_contract_structured_fact_robustness_min_decision_accuracy",
    "promotion_contract_structured_fact_robustness_max_false_supported_rate",
    "promotion_contract_structured_fact_robustness_min_false_refuted_rate",
)

from benchmarks.config_utils import (  # noqa: E402
    planned_artifact_manifest_summary,
    reject_bounded_product_trace,
    strict_bool,
    strict_positive_int,
)
from eigentruth.control import (  # noqa: E402
    ProductPromotionContract,
    ProductRuntimeBudgetPolicy,
    evaluate_product_runtime_budget,
    product_runtime_metrics,
)
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest, fingerprint_path  # noqa: E402


@dataclass(frozen=True)
class ProductRuntimeBaselineConfig:
    """Configuration for a ProductTrace runtime baseline report."""

    trace_paths: Sequence[str | Path]
    report_path: str | Path
    policy: ProductRuntimeBudgetPolicy | Mapping[str, Any] | None = None
    policy_path: str | Path | None = None
    promotion_contract_path: str | Path | None = None
    trace_records_path: str | Path | None = None
    trace_records_cache_path: str | Path | None = None
    refresh_trace_records_cache: bool = False
    trace_scan_workers: int = 1
    recommended_policy_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    compact_json: bool = False

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        if not trace_paths:
            raise ValueError("at least one ProductTrace path is required.")
        if self.policy is not None and (self.policy_path is not None or self.promotion_contract_path is not None):
            raise ValueError("policy object is mutually exclusive with policy_path and promotion_contract_path.")
        if self.policy_path is not None and self.promotion_contract_path is not None:
            raise ValueError("policy_path and promotion_contract_path are mutually exclusive.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "report_path", Path(self.report_path))
        if self.policy_path is not None:
            object.__setattr__(self, "policy_path", Path(self.policy_path))
        if self.promotion_contract_path is not None:
            object.__setattr__(self, "promotion_contract_path", Path(self.promotion_contract_path))
        if self.trace_records_path is not None:
            object.__setattr__(self, "trace_records_path", Path(self.trace_records_path))
        if self.trace_records_cache_path is not None:
            object.__setattr__(self, "trace_records_cache_path", Path(self.trace_records_cache_path))
        if self.recommended_policy_path is not None:
            object.__setattr__(self, "recommended_policy_path", Path(self.recommended_policy_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))
        object.__setattr__(
            self,
            "refresh_trace_records_cache",
            strict_bool(
                self.refresh_trace_records_cache,
                name="refresh_trace_records_cache",
            ),
        )
        object.__setattr__(
            self,
            "trace_scan_workers",
            strict_positive_int(self.trace_scan_workers, name="trace_scan_workers"),
        )

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the output artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.report_path).with_name("product-runtime-baseline-artifact-manifest.json")


def build_product_runtime_baseline(config: ProductRuntimeBaselineConfig) -> dict[str, Any]:
    """Aggregate ProductTrace runtime metrics and optional budget results."""
    policy, policy_source = _load_policy(config)
    records, summary_records, trace_record_cache = _build_trace_records(config, policy=policy)
    trace_record_count = len(summary_records)
    budget_summary = _budget_summary(summary_records, policy=policy)
    summary = _aggregate_records(summary_records)
    status = _status_from_budget(budget_summary)
    report = {
        "schema_version": 1,
        "workflow": "product_runtime_baseline",
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": _blocking_reasons(budget_summary),
        },
        "summary": summary,
        "optimization": _optimization_report(
            summary,
            budget=budget_summary,
            trace_record_cache=trace_record_cache,
        ),
        "budget": budget_summary,
        "traces": list(records),
        "trace_records": {
            "storage": "jsonl_sidecar" if config.trace_records_path is not None else "inline",
            "count": trace_record_count,
            "path": None if config.trace_records_path is None else str(config.trace_records_path),
        },
        "paths": {
            "report": str(config.report_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "trace_records_jsonl": None if config.trace_records_path is None else str(config.trace_records_path),
            "trace_records_cache": (
                None if config.trace_records_cache_path is None else str(config.trace_records_cache_path)
            ),
            "recommended_policy": (
                None if config.recommended_policy_path is None else str(config.recommended_policy_path)
            ),
            "policy": None if config.policy_path is None else str(config.policy_path),
            "promotion_contract": (
                None if config.promotion_contract_path is None else str(config.promotion_contract_path)
            ),
            "traces": [str(path) for path in config.trace_paths],
        },
        "config": {
            "trace_count": trace_record_count,
            "policy_source": policy_source,
            "trace_records_sidecar": config.trace_records_path is not None,
            "trace_record_cache": trace_record_cache,
            "trace_scan_workers": config.trace_scan_workers,
            "recommended_policy": {
                "enabled": config.recommended_policy_path is not None,
                "path": None if config.recommended_policy_path is None else str(config.recommended_policy_path),
            },
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
    }
    _write_recommended_policy(config, report)
    _write_report_and_manifest(config, report)
    _record_registry(config, report)
    return report


def _build_trace_records(
    config: ProductRuntimeBaselineConfig,
    *,
    policy: ProductRuntimeBudgetPolicy | None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    cache_path = config.trace_records_cache_path
    invalidation_reason = None
    if cache_path is not None and cache_path.exists() and not config.refresh_trace_records_cache:
        cached = _load_trace_records_cache(cache_path, trace_paths=config.trace_paths, policy=policy)
        if cached is not None:
            cached_records, payload = cached
            records, summary_records = _emit_trace_records(config, cached_records)
            return records, summary_records, {
                "enabled": True,
                "source": "trace_record_cache",
                "path": str(cache_path),
                "cache_hit": True,
                "cache_written": False,
                "trace_count": len(cached_records),
                "source_count": len(_sequence(payload.get("sources"))),
                "refresh": False,
                "invalidation_reason": None,
                "trace_scan_workers": 0,
            }
        invalidation_reason = "fingerprint_policy_or_schema_mismatch"

    scanned_records = _scan_trace_records(
        config.trace_paths,
        policy=policy,
        max_workers=config.trace_scan_workers,
    )
    if cache_path is not None:
        payload = _trace_records_cache_payload(config, scanned_records, policy=policy)
        _write_report(cache_path, payload, compact=config.compact_json)
    records, summary_records = _emit_trace_records(config, scanned_records)
    return records, summary_records, {
        "enabled": cache_path is not None,
        "source": "trace_scan",
        "path": None if cache_path is None else str(cache_path),
        "cache_hit": False,
        "cache_written": cache_path is not None,
        "trace_count": len(scanned_records),
        "source_count": len(config.trace_paths),
        "refresh": config.refresh_trace_records_cache,
        "invalidation_reason": invalidation_reason,
        "trace_scan_workers": _effective_worker_count(
            config.trace_scan_workers,
            item_count=len(config.trace_paths),
        ),
    }


def _scan_trace_records(
    trace_paths: Sequence[Path],
    *,
    policy: ProductRuntimeBudgetPolicy | None,
    max_workers: int,
) -> tuple[dict[str, Any], ...]:
    if max_workers <= 1 or len(trace_paths) <= 1:
        return tuple(
            _trace_record(path, _load_trace(path), policy=policy)
            for path in trace_paths
        )
    worker_count = _effective_worker_count(max_workers, item_count=len(trace_paths))

    def scan(path: Path) -> dict[str, Any]:
        return _trace_record(path, _load_trace(path), policy=policy)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return tuple(executor.map(scan, trace_paths))


def _effective_worker_count(max_workers: int, *, item_count: int) -> int:
    return max(1, min(max_workers, max(1, item_count)))


def _emit_trace_records(
    config: ProductRuntimeBaselineConfig,
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    inline_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    if config.trace_records_path is None:
        for record in records:
            inline_records.append(dict(record))
        return tuple(inline_records), tuple(inline_records)

    output_path = Path(config.trace_records_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(_jsonl_text(record))
            summary_records.append({
                "context": dict(_mapping(record.get("context"))),
                "metrics": record["metrics"],
                "budget": record["budget"],
            })
    return (), tuple(summary_records)


def _write_report_and_manifest(
    config: ProductRuntimeBaselineConfig,
    report: dict[str, Any],
) -> dict[str, Any]:
    artifacts = _artifact_paths(config)
    report["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.report_path,),
    )
    _write_report(config.report_path, report, compact=config.compact_json)
    return _write_artifact_manifest(config, report, artifacts=artifacts)


def _trace_record(
    path: Path,
    trace: Mapping[str, Any],
    *,
    policy: ProductRuntimeBudgetPolicy | None,
) -> dict[str, Any]:
    metrics = product_runtime_metrics(trace)
    budget = None if policy is None else evaluate_product_runtime_budget(trace, policy)
    return {
        "path": str(path),
        "request_id": trace.get("request_id"),
        "context": _trace_context(trace),
        "metrics": _compact_metrics(metrics),
        "budget": budget,
    }


def _trace_context(trace: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(trace.get("metadata"))
    risk_decision = _mapping(trace.get("risk_decision"))
    return {
        "runtime_profile": _optional_string(metadata.get("runtime_profile")),
        "runtime_profile_source": _optional_string(metadata.get("runtime_profile_source")),
        "pre_generation_profile_requested": _optional_string(
            metadata.get("pre_generation_profile_requested")
        ),
        "max_verifier_route_attempts": _finite_float(metadata.get("max_verifier_route_attempts")),
        "staged_verification_enabled": metadata.get("staged_verification_enabled"),
        "risk_level": _optional_string(risk_decision.get("risk_level")),
        "action": _optional_string(risk_decision.get("action")),
    }


def _load_trace_records_cache(
    path: str | Path,
    *,
    trace_paths: Sequence[Path],
    policy: ProductRuntimeBudgetPolicy | None,
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, Any]] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != _TRACE_RECORD_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("workflow") != "product_runtime_baseline_trace_records":
        return None
    if _mapping(payload.get("policy")).get("signature") != _policy_signature(policy):
        return None
    sources = _sequence(payload.get("sources"))
    records = _sequence(payload.get("records"))
    if len(sources) != len(trace_paths) or len(records) != len(trace_paths):
        return None
    for trace_path, source in zip(trace_paths, sources, strict=True):
        if not isinstance(source, Mapping):
            return None
        if str(source.get("path")) != str(trace_path):
            return None
        expected = _mapping(source.get("fingerprint"))
        if not expected:
            return None
        actual = fingerprint_path(trace_path).to_dict()
        if not _fingerprint_matches(expected, actual):
            return None
    try:
        parsed_records = tuple(_trace_record_from_cache(record) for record in records)
    except (TypeError, ValueError):
        return None
    if tuple(str(record.get("path")) for record in parsed_records) != tuple(str(path) for path in trace_paths):
        return None
    return parsed_records, payload


def _trace_records_cache_payload(
    config: ProductRuntimeBaselineConfig,
    records: Sequence[Mapping[str, Any]],
    *,
    policy: ProductRuntimeBudgetPolicy | None,
) -> dict[str, Any]:
    return {
        "schema_version": _TRACE_RECORD_CACHE_SCHEMA_VERSION,
        "workflow": "product_runtime_baseline_trace_records",
        "paths": {
            "trace_records_cache": (
                None if config.trace_records_cache_path is None else str(config.trace_records_cache_path)
            ),
            "report": str(config.report_path),
        },
        "summary": {
            "trace_count": len(records),
            "source_count": len(config.trace_paths),
            "trace_scan_workers": _effective_worker_count(
                config.trace_scan_workers,
                item_count=len(config.trace_paths),
            ),
        },
        "policy": {
            "signature": _policy_signature(policy),
            "payload": None if policy is None else policy.to_dict(),
        },
        "sources": [
            {
                "path": str(path),
                "fingerprint": fingerprint_path(path).to_dict(),
            }
            for path in config.trace_paths
        ],
        "records": [_trace_record_to_cache(record) for record in records],
    }


def _trace_record_to_cache(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(record.get("path")),
        "request_id": record.get("request_id"),
        "context": dict(_mapping(record.get("context"))),
        "metrics": dict(_mapping(record.get("metrics"))),
        "budget": None if record.get("budget") is None else dict(_mapping(record.get("budget"))),
    }


def _trace_record_from_cache(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("trace record cache entries must be objects.")
    if record.get("path") is None:
        raise ValueError("trace record cache entries require path.")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("trace record cache entries require metrics.")
    budget = record.get("budget")
    if budget is not None and not isinstance(budget, Mapping):
        raise ValueError("trace record cache budget must be an object or null.")
    return {
        "path": str(record.get("path")),
        "request_id": record.get("request_id"),
        "context": dict(_mapping(record.get("context"))),
        "metrics": dict(metrics),
        "budget": None if budget is None else dict(budget),
    }


def _policy_signature(policy: ProductRuntimeBudgetPolicy | None) -> str | None:
    if policy is None:
        return None
    return json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"))


def _fingerprint_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return all(
        expected.get(field_name) == actual.get(field_name)
        for field_name in ("exists", "kind", "sha256", "size_bytes", "file_count")
    )


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        "has_runtime_trace": bool(metrics.get("has_runtime_trace")),
        "total_seconds": metrics.get("total_seconds"),
        "accounted_seconds": metrics.get("accounted_seconds"),
        "unaccounted_seconds": metrics.get("unaccounted_seconds"),
        "measured_phases": metrics.get("measured_phases"),
        "phase_seconds": dict(_mapping(metrics.get("phase_seconds"))),
        "phase_counts": dict(_mapping(metrics.get("phase_counts"))),
        "phase_p95_seconds": dict(_mapping(metrics.get("phase_p95_seconds"))),
        "phase_p99_seconds": dict(_mapping(metrics.get("phase_p99_seconds"))),
        "slowest_phase": metrics.get("slowest_phase"),
        "cache_hit_rate": metrics.get("cache_hit_rate"),
        "named_cache_hit_rates": dict(_mapping(metrics.get("named_cache_hit_rates"))),
        "route_cost_summary": dict(_mapping(metrics.get("route_cost_summary"))),
        "mean_route_duration_seconds": metrics.get("mean_route_duration_seconds"),
        "p95_route_duration_seconds": metrics.get("p95_route_duration_seconds"),
        "p99_route_duration_seconds": metrics.get("p99_route_duration_seconds"),
        "max_route_duration_seconds": metrics.get("max_route_duration_seconds"),
        "mean_attempted_route_count": metrics.get("mean_attempted_route_count"),
        "route_budget_exhaustion_rate": metrics.get("route_budget_exhaustion_rate"),
        "route_budget_exhausted_count": metrics.get("route_budget_exhausted_count"),
        "unattempted_route_count": metrics.get("unattempted_route_count"),
        "retrieval_use_rate": metrics.get("retrieval_use_rate"),
        "retrieval_hit_count": metrics.get("retrieval_hit_count"),
        "mean_retrieval_hits": metrics.get("mean_retrieval_hits"),
        "verification_stage_summary": dict(_mapping(metrics.get("verification_stage_summary"))),
        "verification_stage_enabled": bool(metrics.get("verification_stage_enabled")),
        "verification_stage_skipped": bool(metrics.get("verification_stage_skipped")),
        "verification_skip_rate": metrics.get("verification_skip_rate"),
        "selective_claim_skip_rate": metrics.get("selective_claim_skip_rate"),
        "verified_claim_count": metrics.get("verified_claim_count"),
        "verifier_saved_claim_count": metrics.get("verifier_saved_claim_count"),
        "verification_plan_summary": dict(_mapping(metrics.get("verification_plan_summary"))),
        "verification_plan_available": bool(metrics.get("verification_plan_available")),
        "verification_plan_source": metrics.get("verification_plan_source"),
        "verification_plan_scope": metrics.get("verification_plan_scope"),
        "verification_plan_claim_count": metrics.get("verification_plan_claim_count"),
        "verification_plan_verify_claim_count": metrics.get("verification_plan_verify_claim_count"),
        "verification_plan_skipped_claim_count": metrics.get("verification_plan_skipped_claim_count"),
        "verification_plan_triggered_claim_count": metrics.get("verification_plan_triggered_claim_count"),
        "verification_plan_route_hint_count": metrics.get("verification_plan_route_hint_count"),
        "verification_plan_route_counts": dict(_mapping(metrics.get("verification_plan_route_counts"))),
        "verification_plan_retrieval_query_count": metrics.get("verification_plan_retrieval_query_count"),
        "verification_plan_calculation_check_count": metrics.get("verification_plan_calculation_check_count"),
        "verification_plan_state_check_count": metrics.get("verification_plan_state_check_count"),
        "verification_plan_world_model_check_count": metrics.get("verification_plan_world_model_check_count"),
        "verification_plan_dependency_count": metrics.get("verification_plan_dependency_count"),
        "action_audit_summary": dict(_mapping(metrics.get("action_audit_summary"))),
        "action_audit_available": bool(metrics.get("action_audit_available")),
        "action_audit_source": metrics.get("action_audit_source"),
        "action_audit_passed": metrics.get("action_audit_passed"),
        "action_audit_issue_count": metrics.get("action_audit_issue_count"),
        "action_audit_error_count": metrics.get("action_audit_error_count"),
        "action_audit_warning_count": metrics.get("action_audit_warning_count"),
        "action_audit_missing_decision_action_count": metrics.get(
            "action_audit_missing_decision_action_count"
        ),
        "action_audit_missing_retrieval_action_count": metrics.get(
            "action_audit_missing_retrieval_action_count"
        ),
        "action_audit_malformed_payload_count": metrics.get("action_audit_malformed_payload_count"),
        "action_audit_unexpected_action_count": metrics.get("action_audit_unexpected_action_count"),
        "action_audit_unknown_claim_id_count": metrics.get("action_audit_unknown_claim_id_count"),
        "triple_coverage_summary": dict(_mapping(metrics.get("triple_coverage_summary"))),
        "triple_coverage_source": metrics.get("triple_coverage_source"),
        "triple_claim_count": metrics.get("triple_claim_count"),
        "triple_claim_coverage_rate": metrics.get("triple_claim_coverage_rate"),
        "triple_audit_available": metrics.get("triple_audit_available"),
        "triple_audit_report_count": metrics.get("triple_audit_report_count"),
        "triple_audit_claim_covered_count": metrics.get("triple_audit_claim_covered_count"),
        "triple_audit_claim_coverage_rate": metrics.get("triple_audit_claim_coverage_rate"),
        "triple_audit_triple_count": metrics.get("triple_audit_triple_count"),
        "triple_audit_pass_rate": metrics.get("triple_audit_pass_rate"),
        "triple_slot_coverage_rate": metrics.get("triple_slot_coverage_rate"),
        "triple_structured_fact_result_count": metrics.get("triple_structured_fact_result_count"),
        "triple_claim_predicate_counts": dict(_mapping(metrics.get("triple_claim_predicate_counts"))),
        "triple_audit_predicate_counts": dict(_mapping(metrics.get("triple_audit_predicate_counts"))),
        "triple_missing_slot_counts": dict(_mapping(metrics.get("triple_missing_slot_counts"))),
        "triple_covered_slot_counts": dict(_mapping(metrics.get("triple_covered_slot_counts"))),
        "triple_structured_fact_status_counts": dict(
            _mapping(metrics.get("triple_structured_fact_status_counts"))
        ),
        "triple_structured_fact_predicate_counts": dict(
            _mapping(metrics.get("triple_structured_fact_predicate_counts"))
        ),
        "final_answer_summary": dict(_mapping(metrics.get("final_answer_summary"))),
        "final_answer_available": bool(metrics.get("final_answer_available")),
        "final_answer_source": metrics.get("final_answer_source"),
        "final_answer_status": metrics.get("final_answer_status"),
        "final_answer_action": metrics.get("final_answer_action"),
        "final_answer_risk_level": metrics.get("final_answer_risk_level"),
        "final_answer_answerable": metrics.get("final_answer_answerable"),
        "final_answer_confidence": metrics.get("final_answer_confidence"),
        "final_answer_evidence_count": metrics.get("final_answer_evidence_count"),
        "final_answer_total_claims": metrics.get("final_answer_total_claims"),
        "final_answer_blocked_claim_count": metrics.get("final_answer_blocked_claim_count"),
        "final_answer_requires_followup": metrics.get("final_answer_requires_followup"),
        "promotion_contract_summary": dict(_mapping(metrics.get("promotion_contract_summary"))),
        "promotion_contract_available": bool(metrics.get("promotion_contract_available")),
        "promotion_contract_source": metrics.get("promotion_contract_source"),
        "promotion_contract_source_status": metrics.get("promotion_contract_source_status"),
        "promotion_contract_budget_enabled": metrics.get("promotion_contract_budget_enabled"),
        "promotion_contract_recommended_route_covered_fact_property_count": metrics.get(
            "promotion_contract_recommended_route_covered_fact_property_count"
        ),
        "promotion_contract_recommended_route_covered_fact_properties": list(
            _sequence(metrics.get("promotion_contract_recommended_route_covered_fact_properties"))
        ),
        "promotion_contract_recommended_route_covered_fact_property_metrics": dict(
            _mapping(metrics.get("promotion_contract_recommended_route_covered_fact_property_metrics"))
        ),
        "promotion_contract_required_route_baseline_covered_fact_property_counts": dict(
            _mapping(metrics.get("promotion_contract_required_route_baseline_covered_fact_property_counts"))
        ),
        "promotion_contract_required_route_baseline_covered_fact_properties": {
            str(key): list(_sequence(value))
            for key, value in _mapping(
                metrics.get("promotion_contract_required_route_baseline_covered_fact_properties")
            ).items()
        },
        "promotion_contract_required_route_baseline_covered_fact_property_metrics": {
            str(key): dict(_mapping(value))
            for key, value in _mapping(
                metrics.get("promotion_contract_required_route_baseline_covered_fact_property_metrics")
            ).items()
        },
        "promotion_contract_structured_fact_robustness_property_counts": dict(
            _mapping(metrics.get("promotion_contract_structured_fact_robustness_property_counts"))
        ),
        "promotion_contract_structured_fact_robustness_properties": {
            str(key): list(_sequence(value))
            for key, value in _mapping(
                metrics.get("promotion_contract_structured_fact_robustness_properties")
            ).items()
        },
        "promotion_contract_structured_fact_robustness_property_metrics": {
            str(key): dict(_mapping(value))
            for key, value in _mapping(
                metrics.get("promotion_contract_structured_fact_robustness_property_metrics")
            ).items()
        },
        "promotion_contract_triple_extraction_fixture_matrix_available": bool(
            metrics.get("promotion_contract_triple_extraction_fixture_matrix_available")
        ),
        "promotion_contract_triple_extraction_fixture_matrix_source": metrics.get(
            "promotion_contract_triple_extraction_fixture_matrix_source"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_status": metrics.get(
            "promotion_contract_triple_extraction_fixture_matrix_status"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_manifest_verified": metrics.get(
            "promotion_contract_triple_extraction_fixture_matrix_manifest_verified"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_n_corpora": metrics.get(
            "promotion_contract_triple_extraction_fixture_matrix_n_corpora"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_promoted_corpora": metrics.get(
            "promotion_contract_triple_extraction_fixture_matrix_promoted_corpora"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_distinct_predicate_count": metrics.get(
            "promotion_contract_triple_extraction_fixture_matrix_distinct_predicate_count"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_mean_best_f1": metrics.get(
            "promotion_contract_triple_extraction_fixture_matrix_mean_best_f1"
        ),
        "promotion_contract_triple_extraction_fixture_matrix_mean_f1_lift": metrics.get(
            "promotion_contract_triple_extraction_fixture_matrix_mean_f1_lift"
        ),
    }
    for field_name in _PROMOTION_CONTRACT_PRODUCT_RUNTIME_DRIFT_FIELDS:
        compact[field_name] = metrics.get(field_name)
    for field_name in _PROMOTION_CONTRACT_COVERED_FACT_ROLLUP_FIELDS:
        compact[field_name] = metrics.get(field_name)
    for prefix in _PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            field_name = f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            compact[field_name] = metrics.get(field_name)
    for prefix in _PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            field_name = f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            compact[field_name] = metrics.get(field_name)
    for prefix in _PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_PREFIXES:
        for suffix in ("baseline", "current", "status"):
            field_name = f"promotion_contract_product_runtime_drift_{prefix}_{suffix}"
            compact[field_name] = metrics.get(field_name)
    return compact


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [_mapping(record.get("metrics")) for record in records]
    contexts = [_mapping(record.get("context")) for record in records]
    return {
        "n_traces": len(records),
        "runtime_trace_count": sum(1 for item in metrics if bool(item.get("has_runtime_trace"))),
        "total_seconds": _numeric_summary(item.get("total_seconds") for item in metrics),
        "accounted_seconds": _numeric_summary(item.get("accounted_seconds") for item in metrics),
        "measured_phases": _numeric_summary(item.get("measured_phases") for item in metrics),
        "mean_route_duration_seconds": _numeric_summary(
            item.get("mean_route_duration_seconds") for item in metrics
        ),
        "p95_route_duration_seconds": _numeric_summary(
            item.get("p95_route_duration_seconds") for item in metrics
        ),
        "p99_route_duration_seconds": _numeric_summary(
            item.get("p99_route_duration_seconds") for item in metrics
        ),
        "max_route_duration_seconds": _numeric_summary(
            item.get("max_route_duration_seconds") for item in metrics
        ),
        "mean_attempted_route_count": _numeric_summary(
            item.get("mean_attempted_route_count") for item in metrics
        ),
        "route_budget_exhaustion_rate": _numeric_summary(
            item.get("route_budget_exhaustion_rate") for item in metrics
        ),
        "route_budget_exhausted_count": _numeric_summary(
            item.get("route_budget_exhausted_count") for item in metrics
        ),
        "unattempted_route_count": _numeric_summary(
            item.get("unattempted_route_count") for item in metrics
        ),
        "retrieval_use_rate": _numeric_summary(item.get("retrieval_use_rate") for item in metrics),
        "retrieval_hit_count": _numeric_summary(item.get("retrieval_hit_count") for item in metrics),
        "cache_hit_rate": _numeric_summary(item.get("cache_hit_rate") for item in metrics),
        "verification_skip_rate": _numeric_summary(item.get("verification_skip_rate") for item in metrics),
        "selective_claim_skip_rate": _numeric_summary(
            item.get("selective_claim_skip_rate") for item in metrics
        ),
        "verified_claim_count": _numeric_summary(item.get("verified_claim_count") for item in metrics),
        "verifier_saved_claim_count": _numeric_summary(item.get("verifier_saved_claim_count") for item in metrics),
        "verification_stage": _aggregate_verification_stage(metrics),
        "verification_plan": _aggregate_verification_plan(metrics),
        "action_audit": _aggregate_action_audit(metrics),
        "triple_coverage": _aggregate_triple_coverage(metrics),
        "final_answer": _aggregate_final_answer(metrics),
        "promotion_contract": _aggregate_promotion_contract(metrics),
        "phases": _aggregate_phases(metrics),
        "routes": _aggregate_routes(metrics),
        "profiles": _aggregate_contexts(contexts),
    }


def _aggregate_contexts(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    staged_values = [context.get("staged_verification_enabled") for context in contexts]
    return {
        "source_trace_count": len(contexts),
        "runtime_profile_counts": _counts(context.get("runtime_profile") for context in contexts),
        "runtime_profile_source_counts": _counts(
            context.get("runtime_profile_source") for context in contexts
        ),
        "pre_generation_profile_requested_counts": _counts(
            context.get("pre_generation_profile_requested") for context in contexts
        ),
        "max_verifier_route_attempts": _numeric_summary(
            context.get("max_verifier_route_attempts") for context in contexts
        ),
        "risk_level_counts": _counts(context.get("risk_level") for context in contexts),
        "action_counts": _counts(context.get("action") for context in contexts),
        "staged_verification_enabled_count": sum(1 for value in staged_values if _truthy_flag(value)),
    }


def _optimization_report(
    summary: Mapping[str, Any],
    *,
    budget: Mapping[str, Any],
    trace_record_cache: Mapping[str, Any],
) -> dict[str, Any]:
    phase_hotspots = _phase_hotspots(summary)
    route_hotspots = _route_hotspots(summary)
    recommendations = _optimization_recommendations(
        summary,
        phase_hotspots=phase_hotspots,
        route_hotspots=route_hotspots,
        trace_record_cache=trace_record_cache,
    )
    return {
        "schema_version": 1,
        "status": _optimization_status(summary, recommendations),
        "summary": {
            "n_traces": summary.get("n_traces"),
            "runtime_trace_count": summary.get("runtime_trace_count"),
            "runtime_trace_coverage": _safe_div(
                _finite_float(summary.get("runtime_trace_count")),
                _finite_float(summary.get("n_traces")),
            ),
            "cache_hit_rate_mean": _nested(summary, "cache_hit_rate", "mean"),
            "retrieval_use_rate": _nested(summary, "routes", "overall", "retrieval_use_rate"),
            "route_budget_exhaustion_rate": _nested(
                summary,
                "routes",
                "overall",
                "route_budget_exhaustion_rate",
            ),
            "verification_claim_skip_rate": _nested(
                summary,
                "verification_stage",
                "claim_skip_rate",
            ),
            "verification_plan_coverage": _nested(
                summary,
                "verification_plan",
                "coverage_rate",
            ),
            "action_audit_error_rate": _nested(summary, "action_audit", "error_rate"),
            "action_audit_failed_trace_rate": _nested(summary, "action_audit", "failed_trace_rate"),
            "action_audit_missing_retrieval_action_rate": _nested(
                summary,
                "action_audit",
                "missing_retrieval_action_rate",
            ),
            "action_audit_malformed_payload_rate": _nested(
                summary,
                "action_audit",
                "malformed_payload_rate",
            ),
            "slowest_phase": None if not phase_hotspots else phase_hotspots[0]["phase"],
            "slowest_route": None if not route_hotspots else route_hotspots[0]["route"],
            "budget_enabled": budget.get("enabled"),
            "budget_passed": budget.get("passed"),
        },
        "hotspots": {
            "phases": phase_hotspots,
            "routes": route_hotspots,
        },
        "recommendations": recommendations,
        "policy_hints": _optimization_policy_hints(summary, phase_hotspots=phase_hotspots),
    }


def _optimization_status(
    summary: Mapping[str, Any],
    recommendations: Sequence[Mapping[str, Any]],
) -> str:
    if not _finite_float(summary.get("n_traces")):
        return "insufficient_data"
    if any(item.get("priority") == "high" for item in recommendations):
        return "needs_attention"
    if recommendations:
        return "has_recommendations"
    return "ok"


def _phase_hotspots(summary: Mapping[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, stats in _mapping(summary.get("phases")).items():
        phase_stats = _mapping(stats)
        seconds = _mapping(phase_stats.get("seconds"))
        count = _finite_float(seconds.get("count"))
        mean = _finite_float(seconds.get("mean"))
        total = None if count is None or mean is None else count * mean
        rows.append({
            "phase": str(phase),
            "observation_count": int(count or 0),
            "phase_count": phase_stats.get("phase_count"),
            "total_observed_seconds": total,
            "mean_seconds": mean,
            "p95_seconds": _finite_float(seconds.get("p95")),
            "p99_seconds": _finite_float(seconds.get("p99")),
            "max_seconds": _finite_float(seconds.get("max")),
        })
    rows.sort(
        key=lambda row: (
            _sort_value(row.get("total_observed_seconds")),
            _sort_value(row.get("mean_seconds")),
            row["phase"],
        ),
        reverse=True,
    )
    return rows[:limit]


def _route_hotspots(summary: Mapping[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route, stats in _mapping(_nested(summary, "routes", "by_route")).items():
        route_stats = _mapping(stats)
        rows.append({
            "route": str(route),
            "total": route_stats.get("total"),
            "routed_total": route_stats.get("routed_total"),
            "total_duration_seconds": _finite_float(route_stats.get("total_duration_seconds")),
            "mean_duration_seconds": _finite_float(route_stats.get("mean_duration_seconds")),
            "mean_selected_route_duration_seconds": _finite_float(
                route_stats.get("mean_selected_route_duration_seconds")
            ),
            "max_duration_seconds": _finite_float(route_stats.get("max_duration_seconds")),
            "mean_attempted_route_count": _finite_float(
                route_stats.get("mean_attempted_route_count")
            ),
            "retrieval_use_rate": _finite_float(route_stats.get("retrieval_use_rate")),
            "mean_retrieval_hits": _finite_float(route_stats.get("mean_retrieval_hits")),
        })
    rows.sort(
        key=lambda row: (
            _sort_value(row.get("total_duration_seconds")),
            _sort_value(row.get("mean_duration_seconds")),
            row["route"],
        ),
        reverse=True,
    )
    return rows[:limit]


def _optimization_recommendations(
    summary: Mapping[str, Any],
    *,
    phase_hotspots: Sequence[Mapping[str, Any]],
    route_hotspots: Sequence[Mapping[str, Any]],
    trace_record_cache: Mapping[str, Any],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    overall_routes = _mapping(_nested(summary, "routes", "overall"))
    mean_attempted = _finite_float(overall_routes.get("mean_attempted_route_count"))
    route_budget_exhaustion_rate = _finite_float(overall_routes.get("route_budget_exhaustion_rate"))
    retrieval_use_rate = _finite_float(overall_routes.get("retrieval_use_rate"))
    cache_hit_rate = _finite_float(_nested(summary, "cache_hit_rate", "mean"))
    verification_stage = _mapping(summary.get("verification_stage"))
    triple_coverage = _mapping(summary.get("triple_coverage"))
    claim_skip_rate = _finite_float(verification_stage.get("claim_skip_rate"))
    triple_claim_count = _finite_float(triple_coverage.get("claim_triple_count")) or 0.0
    triple_audit_claim_coverage_rate = _finite_float(
        triple_coverage.get("audit_claim_coverage_rate")
    )
    enabled_stage_count = _finite_float(verification_stage.get("enabled_trace_count")) or 0.0
    run_verifier_stage_count = _finite_float(verification_stage.get("run_verifier_trace_count")) or 0.0
    triggered_scope_count = _finite_float(verification_stage.get("triggered_scope_trace_count")) or 0.0
    partial_skip_count = _finite_float(verification_stage.get("partial_skip_trace_count")) or 0.0
    verified_claim_count_mean = _finite_float(_nested(summary, "verified_claim_count", "mean"))
    profiles = _mapping(summary.get("profiles"))
    profile_counts = _mapping(profiles.get("runtime_profile_counts"))
    recommended_route_attempts = _recommended_max_verifier_route_attempts(summary)
    audit_rate = _count_rate(profile_counts, "audit")
    n_traces = _finite_float(summary.get("n_traces")) or 0.0
    top_phase_name = None if not phase_hotspots else str(phase_hotspots[0].get("phase"))

    if phase_hotspots:
        top_phase = phase_hotspots[0]
        if _finite_float(top_phase.get("mean_seconds")):
            recommendations.append(_recommendation(
                "phase_hotspot_review",
                priority="medium",
                area="runtime_trace",
                title="Review the dominant runtime phase before widening model or verifier work.",
                reason=f"{top_phase['phase']} is the largest observed phase in this baseline.",
                evidence={
                    "phase": top_phase["phase"],
                    "mean_seconds": top_phase.get("mean_seconds"),
                    "total_observed_seconds": top_phase.get("total_observed_seconds"),
                },
                suggested_action=(
                    "Use this phase as the first optimization target; keep model-forward, "
                    "retrieval, and verifier changes gated by phase-level p95/p99 budgets."
                ),
            ))

    if cache_hit_rate is None:
        recommendations.append(_recommendation(
            "instrument_cache_hit_rates",
            priority="medium",
            area="cache",
            title="Record named cache hit rates in ProductTrace metadata.",
            reason="The baseline cannot evaluate cache effectiveness without cache hit/miss telemetry.",
            evidence={"cache_hit_rate_mean": None},
            suggested_action=(
                "Populate metadata.cache for verifier, retrieval, state, and score-dump caches "
                "before using this report as a release gate."
            ),
        ))
    elif cache_hit_rate < 0.60:
        recommendations.append(_recommendation(
            "improve_cache_keys",
            priority="high",
            area="cache",
            title="Normalize cache keys before adding more expensive verifier routes.",
            reason="The aggregate cache hit rate is below the default performance target.",
            evidence={"cache_hit_rate_mean": cache_hit_rate, "target": 0.60},
            suggested_action=(
                "Normalize claim text, retrieval queries, and state-source inputs; then rerun "
                "the trace baseline with trace-record caching enabled."
            ),
        ))

    if mean_attempted is not None and mean_attempted > 1.5:
        recommendations.append(_recommendation(
            "reduce_verifier_route_fanout",
            priority="high",
            area="verifier_routes",
            title="Reduce verifier route fanout for common requests.",
            reason="The mean attempted verifier route count is above the balanced-profile target.",
            evidence={"mean_attempted_route_count": mean_attempted, "target": 1.5},
            suggested_action=(
                "Prefer routed verifier adapters by claim metadata and stop after decisive support "
                "or refutation instead of trying broad fallback chains."
            ),
        ))

    if route_budget_exhaustion_rate is not None and route_budget_exhaustion_rate > 0.0:
        recommendations.append(_recommendation(
            "review_verifier_route_budget_exhaustion",
            priority="medium",
            area="verifier_routes",
            title="Review capped verifier routes before tightening latency budgets.",
            reason="Some routed verification decisions stopped because the route fanout cap was exhausted.",
            evidence={
                "route_budget_exhaustion_rate": route_budget_exhaustion_rate,
                "target": 0.0,
                "recommended_max_verifier_route_attempts": recommended_route_attempts,
            },
            suggested_action=(
                (
                    "Raise max_verifier_route_attempts to at least "
                    f"{recommended_route_attempts}, then inspect unattempted routes and move decisive "
                    "cheap routes earlier."
                )
                if recommended_route_attempts is not None
                else (
                    "Inspect unattempted routes and move decisive cheap routes earlier before reducing "
                    "max_verifier_route_attempts further."
                )
            ),
        ))

    if retrieval_use_rate is not None and retrieval_use_rate > 0.50:
        recommendations.append(_recommendation(
            "gate_retrieval_to_unsupported_claims",
            priority="high",
            area="retrieval",
            title="Gate retrieval to unsupported, high-risk, or time-sensitive claims.",
            reason="Retrieval is used on more than half of routed verification decisions.",
            evidence={"retrieval_use_rate": retrieval_use_rate, "target": 0.50},
            suggested_action=(
                "Run cheap structured/rule/self-consistency routes first, then retrieve only "
                "for unsupported or freshness-sensitive claims with bounded top_k."
            ),
        ))

    if triple_claim_count > 0.0 and (
        triple_audit_claim_coverage_rate is None or triple_audit_claim_coverage_rate < 1.0
    ):
        recommendations.append(_recommendation(
            "enable_strict_triple_evidence_audits",
            priority="medium",
            area="verifier_routes",
            title="Route structured factual triples through slot-level evidence audits.",
            reason=(
                "The trace corpus contains extracted claim triples, but not every triple-bearing "
                "claim has a recorded triple-evidence audit report."
            ),
            evidence={
                "claim_triple_count": triple_claim_count,
                "audit_claim_coverage_rate": triple_audit_claim_coverage_rate,
                "claim_predicate_counts": _mapping(triple_coverage.get("claim_predicate_counts")),
            },
            suggested_action=(
                "Enable the triple_evidence route for sensitive factual claims or attach a "
                "structured-fact verifier, then rerun the product runtime baseline and require "
                "slot_coverage_rate evidence before promoting covered-fact routes."
            ),
        ))

    if enabled_stage_count == 0.0:
        recommendations.append(_recommendation(
            "enable_staged_verification",
            priority="medium",
            area="runtime_profile",
            title="Enable staged verification for low-risk traces.",
            reason="No traces in this baseline recorded staged-verification decisions.",
            evidence={"enabled_trace_count": 0, "n_traces": n_traces},
            suggested_action=(
                "Use latency/balanced runtime profiles so low-risk, non-sensitive claims skip "
                "expensive verifier routes while high-risk claims still audit."
            ),
        ))
    elif claim_skip_rate is not None and claim_skip_rate < 0.25:
        recommendations.append(_recommendation(
            "tune_staged_verification_policy",
            priority="medium",
            area="runtime_profile",
            title="Tune staged verification to save more verifier work.",
            reason="Staged verification is enabled but saves fewer than 25% of claims.",
            evidence={"claim_skip_rate": claim_skip_rate, "target": 0.25},
            suggested_action=(
                "Replay runtime profiles over a trace corpus and adjust sensitive claim features "
                "or diagnostic-risk thresholds before changing verifier implementations."
            ),
        ))

    if (
        enabled_stage_count > 0
        and run_verifier_stage_count > 0
        and partial_skip_count == 0
        and triggered_scope_count == 0
        and (
            top_phase_name == "initial_verification"
            or (verified_claim_count_mean is not None and verified_claim_count_mean > 1.0)
        )
    ):
        recommendations.append(_recommendation(
            "enable_selective_staged_verification",
            priority="medium",
            area="runtime_profile",
            title="Enable triggered-claim-only staged verification before widening verifier coverage.",
            reason=(
                "Staged verification is enabled, but verifier-running traces did not record "
                "triggered-scope partial skips."
            ),
            evidence={
                "enabled_trace_count": enabled_stage_count,
                "run_verifier_trace_count": run_verifier_stage_count,
                "triggered_scope_trace_count": triggered_scope_count,
                "partial_skip_trace_count": partial_skip_count,
                "verified_claim_count_mean": verified_claim_count_mean,
                "slowest_phase": top_phase_name,
            },
            suggested_action=(
                "Set stage_verify_triggered_claims_only=true for latency or balanced profiles, "
                "then replay traces and confirm verification_stage.partial_skip_trace_count and "
                "verification_stage.selective_claim_skip_rate increase without changing risk decisions."
            ),
        ))

    if audit_rate is not None and audit_rate > 0.50 and n_traces >= 2:
        recommendations.append(_recommendation(
            "replay_runtime_profile_selector",
            priority="medium",
            area="runtime_profile",
            title="Replay selector policies before making audit the default path.",
            reason="More than half of observed traces ran with the audit profile.",
            evidence={"audit_profile_rate": audit_rate, "runtime_profile_counts": profile_counts},
            suggested_action=(
                "Run run_product_trace_replay_workflow.py with latency/balanced/audit candidates "
                "and gate observed selected runtime deltas before promoting a selector policy."
            ),
        ))

    if trace_record_cache.get("enabled") is not True and n_traces >= 10:
        recommendations.append(_recommendation(
            "enable_trace_record_cache",
            priority="low",
            area="benchmarking",
            title="Enable trace-record cache for repeated runtime baseline runs.",
            reason="Large trace baselines can avoid repeated ProductTrace JSON scans.",
            evidence={"n_traces": n_traces, "trace_record_cache_enabled": False},
            suggested_action="Pass --trace-records-cache-json on repeated baseline and replay workflows.",
        ))

    if route_hotspots:
        top_route = route_hotspots[0]
        if _finite_float(top_route.get("mean_duration_seconds")):
            recommendations.append(_recommendation(
                "route_hotspot_review",
                priority="medium",
                area="verifier_routes",
                title="Review the slowest verifier route before increasing verifier coverage.",
                reason=f"{top_route['route']} contributes the largest observed verifier duration.",
                evidence={
                    "route": top_route["route"],
                    "mean_duration_seconds": top_route.get("mean_duration_seconds"),
                    "retrieval_use_rate": top_route.get("retrieval_use_rate"),
                },
                suggested_action=(
                    "Compare this route against cheaper structured/state/self-consistency routes "
                    "and keep it behind explicit metadata or risk triggers."
                ),
            ))

    return recommendations


def _optimization_policy_hints(
    summary: Mapping[str, Any],
    *,
    phase_hotspots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total_seconds_p95 = _finite_float(_nested(summary, "total_seconds", "p95"))
    cache_hit_rate = _finite_float(_nested(summary, "cache_hit_rate", "mean"))
    mean_attempted = _finite_float(_nested(summary, "routes", "overall", "mean_attempted_route_count"))
    route_budget_exhaustion_rate = _finite_float(
        _nested(summary, "routes", "overall", "route_budget_exhaustion_rate")
    )
    retrieval_use_rate = _finite_float(_nested(summary, "routes", "overall", "retrieval_use_rate"))
    recommended_route_attempts = _recommended_max_verifier_route_attempts(summary)
    phase_p95_budget = {
        str(phase["phase"]): _with_headroom(phase.get("p95_seconds"))
        for phase in phase_hotspots[:3]
        if _with_headroom(phase.get("p95_seconds")) is not None
    }
    return {
        "source": "observed_baseline_with_25_percent_headroom",
        "candidate_control_defaults": {
            "max_verifier_route_attempts": recommended_route_attempts,
        },
        "candidate_runtime_budget_policy": {
            "max_total_seconds": _with_headroom(total_seconds_p95),
            "max_phase_p95_seconds": phase_p95_budget,
            "max_mean_attempted_route_count": _with_headroom(mean_attempted),
            "max_route_budget_exhaustion_rate": (
                None
                if route_budget_exhaustion_rate is None
                else min(1.0, _with_headroom(route_budget_exhaustion_rate) or 0.0)
            ),
            "max_retrieval_use_rate": (
                None
                if retrieval_use_rate is None
                else min(1.0, _with_headroom(retrieval_use_rate) or 1.0)
            ),
            "min_cache_hit_rate": None if cache_hit_rate is None else max(0.0, min(1.0, cache_hit_rate * 0.80)),
        },
        "next_workflows": (
            "run_product_trace_replay_workflow.py",
            "run_runtime_profile_selector_replay.py",
            "run_product_runtime_profile_sweep.py",
        ),
    }


def _recommended_max_verifier_route_attempts(summary: Mapping[str, Any]) -> int | None:
    observed = _finite_float(_nested(summary, "profiles", "max_verifier_route_attempts", "max"))
    if observed is None:
        return None
    route_budget_exhaustion_rate = _finite_float(
        _nested(summary, "routes", "overall", "route_budget_exhaustion_rate")
    )
    recommended = math.ceil(observed)
    if route_budget_exhaustion_rate is not None and route_budget_exhaustion_rate > 0.0:
        recommended += 1
    return max(1, recommended)


def _recommendation(
    recommendation_id: str,
    *,
    priority: str,
    area: str,
    title: str,
    reason: str,
    evidence: Mapping[str, Any],
    suggested_action: str,
) -> dict[str, Any]:
    return {
        "id": recommendation_id,
        "priority": priority,
        "area": area,
        "title": title,
        "reason": reason,
        "evidence": dict(evidence),
        "suggested_action": suggested_action,
    }


def _aggregate_phases(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    phase_names = sorted(
        {
            str(name)
            for item in metrics
            for name in _mapping(item.get("phase_seconds")).keys()
        }
    )
    phases = {}
    for phase in phase_names:
        values = [
            _mapping(item.get("phase_seconds")).get(phase)
            for item in metrics
            if phase in _mapping(item.get("phase_seconds"))
        ]
        counts = [
            _finite_float(_mapping(item.get("phase_counts")).get(phase))
            for item in metrics
            if phase in _mapping(item.get("phase_counts"))
        ]
        phases[phase] = {
            "trace_observations": len(values),
            "phase_count": int(sum(value for value in counts if value is not None)),
            "seconds": _numeric_summary(values),
            "p95_seconds": _numeric_summary(
                _mapping(item.get("phase_p95_seconds")).get(phase)
                for item in metrics
                if phase in _mapping(item.get("phase_p95_seconds"))
            ),
            "p99_seconds": _numeric_summary(
                _mapping(item.get("phase_p99_seconds")).get(phase)
                for item in metrics
                if phase in _mapping(item.get("phase_p99_seconds"))
            ),
        }
    return phases


def _aggregate_routes(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [_mapping(item.get("route_cost_summary")) for item in metrics]
    by_route_names = sorted(
        {
            str(route)
            for summary in summaries
            for route in _mapping(summary.get("by_route")).keys()
        }
    )
    return {
        "overall": _aggregate_route_summaries(summaries),
        "by_route": {
            route: _aggregate_route_summaries(
                _mapping(_mapping(summary.get("by_route")).get(route))
                for summary in summaries
                if route in _mapping(summary.get("by_route"))
            )
            for route in by_route_names
        },
    }


def _aggregate_verification_stage(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [_mapping(item.get("verification_stage_summary")) for item in metrics]
    reason_counts: dict[str, int] = {}
    triggered_feature_counts: dict[str, int] = {}
    triggered_metadata_counts: dict[str, int] = {}
    verification_scope_counts: dict[str, int] = {}
    selective_claim_count = 0.0
    selective_saved_claim_count = 0.0
    selective_verified_claim_count = 0.0
    selective_claim_observations = 0
    partial_skip_count = 0
    for summary in summaries:
        scope = _verification_scope(summary)
        verification_scope_counts[scope] = verification_scope_counts.get(scope, 0) + 1
        saved_claims = _finite_float(summary.get("saved_claim_count")) or 0.0
        if summary.get("run_verifier") is True and scope == "triggered" and saved_claims > 0:
            partial_skip_count += 1
        if scope == "triggered":
            claim_count = _finite_float(summary.get("claim_count"))
            verified_claim_count = _finite_float(summary.get("verified_claim_count"))
            if claim_count is not None:
                selective_claim_count += claim_count
                selective_saved_claim_count += saved_claims
                selective_claim_observations += 1
            if verified_claim_count is not None:
                selective_verified_claim_count += verified_claim_count
        reason = summary.get("reason")
        if reason is not None:
            reason_key = str(reason)
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
        _merge_counts(triggered_feature_counts, _mapping(summary.get("triggered_feature_counts")))
        _merge_counts(triggered_metadata_counts, _mapping(summary.get("triggered_metadata_counts")))
    enabled_count = sum(1 for summary in summaries if bool(summary.get("enabled")))
    skipped_count = sum(1 for summary in summaries if bool(summary.get("skipped")))
    saved_claim_count = _sum_float(summaries, "saved_claim_count")
    verified_claim_count = _sum_float(summaries, "verified_claim_count")
    claim_count = _sum_float(summaries, "claim_count")
    return {
        "source_trace_count": len(summaries),
        "enabled_trace_count": enabled_count,
        "skipped_trace_count": skipped_count,
        "run_verifier_trace_count": sum(1 for summary in summaries if summary.get("run_verifier") is True),
        "verification_scope_counts": verification_scope_counts,
        "none_scope_trace_count": verification_scope_counts.get("none", 0),
        "all_scope_trace_count": verification_scope_counts.get("all", 0),
        "triggered_scope_trace_count": verification_scope_counts.get("triggered", 0),
        "partial_skip_trace_count": partial_skip_count,
        "partial_skip_trace_rate": _safe_div(partial_skip_count, len(summaries)),
        "skip_decision_rate": _safe_div(skipped_count, len(summaries)),
        "claim_count": claim_count,
        "saved_claim_count": saved_claim_count,
        "verified_claim_count": verified_claim_count,
        "claim_skip_rate": _safe_div(saved_claim_count, claim_count),
        "selective_claim_count": selective_claim_count if selective_claim_observations else None,
        "selective_saved_claim_count": (
            selective_saved_claim_count if selective_claim_observations else None
        ),
        "selective_verified_claim_count": (
            selective_verified_claim_count if selective_claim_observations else None
        ),
        "selective_claim_skip_rate": _safe_div(
            selective_saved_claim_count if selective_claim_observations else None,
            selective_claim_count if selective_claim_observations else None,
        ),
        "per_trace_skip_rate": _numeric_summary(summary.get("skip_rate") for summary in summaries),
        "reason_counts": reason_counts,
        "triggered_feature_counts": triggered_feature_counts,
        "triggered_metadata_counts": triggered_metadata_counts,
    }


def _aggregate_verification_plan(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [_mapping(item.get("verification_plan_summary")) for item in metrics]
    available_count = sum(1 for item in metrics if bool(item.get("verification_plan_available")))
    route_counts: dict[str, int] = {}
    tool_payload_counts: dict[str, int] = {}
    for item in metrics:
        _merge_counts(route_counts, _mapping(item.get("verification_plan_route_counts")))
        summary = _mapping(item.get("verification_plan_summary"))
        _merge_counts(tool_payload_counts, _mapping(summary.get("tool_payload_counts")))
    return {
        "source_trace_count": len(metrics),
        "available_trace_count": available_count,
        "missing_trace_count": len(metrics) - available_count,
        "coverage_rate": _safe_div(available_count, len(metrics)),
        "source_counts": _counts(item.get("verification_plan_source") for item in metrics),
        "verification_scope_counts": _counts(item.get("verification_plan_scope") for item in metrics),
        "run_verifier_trace_count": sum(
            1 for item in metrics if item.get("verification_plan_run_verifier") is True
        ),
        "claim_count": _sum_float(metrics, "verification_plan_claim_count"),
        "verify_claim_count": _sum_float(metrics, "verification_plan_verify_claim_count"),
        "skipped_claim_count": _sum_float(metrics, "verification_plan_skipped_claim_count"),
        "triggered_claim_count": _sum_float(metrics, "verification_plan_triggered_claim_count"),
        "route_hint_count": _sum_float(metrics, "verification_plan_route_hint_count"),
        "dependency_count": _sum_float(metrics, "verification_plan_dependency_count"),
        "route_counts": route_counts,
        "tool_payload_counts": tool_payload_counts,
        "per_trace_claim_count": _numeric_summary(
            item.get("verification_plan_claim_count") for item in metrics
        ),
        "per_trace_verify_claim_count": _numeric_summary(
            item.get("verification_plan_verify_claim_count") for item in metrics
        ),
        "per_trace_route_hint_count": _numeric_summary(
            item.get("verification_plan_route_hint_count") for item in metrics
        ),
        "per_trace_dependency_count": _numeric_summary(
            item.get("verification_plan_dependency_count") for item in metrics
        ),
        "summary_observations": sum(1 for summary in summaries if summary),
    }


def _aggregate_action_audit(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [_mapping(item.get("action_audit_summary")) for item in metrics]
    n_traces = len(metrics)
    available_count = sum(1 for item in metrics if item.get("action_audit_available") is True)
    passed_count = sum(1 for item in metrics if item.get("action_audit_passed") is True)
    failed_count = sum(1 for item in metrics if item.get("action_audit_passed") is False)
    counts_by_code: dict[str, int] = {}
    counts_by_severity: dict[str, int] = {}
    for summary in summaries:
        _merge_counts(counts_by_code, _mapping(summary.get("counts_by_code")))
        _merge_counts(counts_by_severity, _mapping(summary.get("counts_by_severity")))
    issue_count = _sum_float(metrics, "action_audit_issue_count")
    error_count = _sum_float(metrics, "action_audit_error_count")
    warning_count = _sum_float(metrics, "action_audit_warning_count")
    missing_decision_action_count = _sum_float(
        metrics,
        "action_audit_missing_decision_action_count",
    ) or 0.0
    missing_retrieval_action_count = _sum_float(
        metrics,
        "action_audit_missing_retrieval_action_count",
    ) or 0.0
    malformed_payload_count = _sum_float(metrics, "action_audit_malformed_payload_count") or 0.0
    unexpected_action_count = _sum_float(metrics, "action_audit_unexpected_action_count") or 0.0
    unknown_claim_id_count = _sum_float(metrics, "action_audit_unknown_claim_id_count") or 0.0
    return {
        "source_trace_count": n_traces,
        "available_trace_count": available_count,
        "missing_trace_count": n_traces - available_count,
        "coverage_rate": _safe_div(available_count, n_traces),
        "passed_trace_count": passed_count,
        "failed_trace_count": failed_count,
        "passed_trace_rate": _safe_div(passed_count, available_count),
        "failed_trace_rate": _safe_div(failed_count, available_count),
        "source_counts": _counts(item.get("action_audit_source") for item in metrics),
        "issue_count": issue_count,
        "error_count": error_count,
        "warning_count": warning_count,
        "issue_rate": _safe_div(issue_count, n_traces),
        "error_rate": _safe_div(error_count, n_traces),
        "warning_rate": _safe_div(warning_count, n_traces),
        "missing_decision_action_count": missing_decision_action_count,
        "missing_decision_action_rate": _safe_div(missing_decision_action_count, n_traces),
        "missing_retrieval_action_count": missing_retrieval_action_count,
        "missing_retrieval_action_rate": _safe_div(missing_retrieval_action_count, n_traces),
        "malformed_payload_count": malformed_payload_count,
        "malformed_payload_rate": _safe_div(malformed_payload_count, n_traces),
        "unexpected_action_count": unexpected_action_count,
        "unexpected_action_rate": _safe_div(unexpected_action_count, n_traces),
        "unknown_claim_id_count": unknown_claim_id_count,
        "unknown_claim_id_rate": _safe_div(unknown_claim_id_count, n_traces),
        "counts_by_code": counts_by_code,
        "counts_by_severity": counts_by_severity,
        "per_trace_action_count": _numeric_summary(
            summary.get("action_count") for summary in summaries
        ),
        "per_trace_issue_count": _numeric_summary(
            item.get("action_audit_issue_count") for item in metrics
        ),
        "summary_observations": sum(1 for summary in summaries if summary),
    }


def _aggregate_triple_coverage(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [_mapping(item.get("triple_coverage_summary")) for item in metrics]
    claim_count = _sum_float(summaries, "claim_count")
    claims_with_triples = _sum_float(summaries, "claims_with_triples")
    audit_report_count = _sum_float(summaries, "audit_report_count")
    audit_claim_covered_count = _sum_float(summaries, "audit_claim_covered_count")
    if audit_claim_covered_count is None:
        audit_claim_covered_count = _fallback_audit_claim_covered_count(summaries)
    audit_triple_count = _sum_float(summaries, "audit_triple_count")
    audit_passed_count = _sum_float(summaries, "audit_passed_count")
    covered_slot_count = _sum_float(summaries, "covered_slot_count")
    missing_slot_count = _sum_float(summaries, "missing_slot_count")
    total_slot_count = None
    if covered_slot_count is not None or missing_slot_count is not None:
        total_slot_count = (covered_slot_count or 0.0) + (missing_slot_count or 0.0)
    claim_predicate_counts: dict[str, int] = {}
    audit_predicate_counts: dict[str, int] = {}
    missing_slot_counts: dict[str, int] = {}
    structured_fact_status_counts: dict[str, int] = {}
    structured_fact_predicate_counts: dict[str, int] = {}
    for item in metrics:
        _merge_counts(claim_predicate_counts, _mapping(item.get("triple_claim_predicate_counts")))
        _merge_counts(audit_predicate_counts, _mapping(item.get("triple_audit_predicate_counts")))
        _merge_counts(missing_slot_counts, _mapping(item.get("triple_missing_slot_counts")))
        _merge_counts(
            structured_fact_status_counts,
            _mapping(item.get("triple_structured_fact_status_counts")),
        )
        _merge_counts(
            structured_fact_predicate_counts,
            _mapping(item.get("triple_structured_fact_predicate_counts")),
        )
    return {
        "source_trace_count": len(metrics),
        "summary_observations": sum(1 for summary in summaries if summary),
        "source_counts": _counts(item.get("triple_coverage_source") for item in metrics),
        "claim_count": claim_count,
        "claims_with_triples": claims_with_triples,
        "claim_triple_count": _sum_float(summaries, "claim_triple_count"),
        "claim_triple_coverage_rate": _safe_div(claims_with_triples, claim_count),
        "claim_predicate_counts": claim_predicate_counts,
        "audit_available_trace_count": sum(1 for item in metrics if item.get("triple_audit_available") is True),
        "audit_report_count": audit_report_count,
        "audit_claim_covered_count": audit_claim_covered_count,
        "audit_claim_coverage_rate": _safe_div(audit_claim_covered_count, claims_with_triples),
        "audit_triple_count": audit_triple_count,
        "audit_passed_count": audit_passed_count,
        "audit_failed_count": _sum_float(summaries, "audit_failed_count"),
        "audit_pass_rate": _safe_div(audit_passed_count, audit_triple_count),
        "audit_predicate_counts": audit_predicate_counts,
        "covered_slot_count": covered_slot_count,
        "missing_slot_count": missing_slot_count,
        "slot_coverage_rate": _safe_div(covered_slot_count, total_slot_count),
        "missing_slot_counts": missing_slot_counts,
        "structured_fact_result_count": _sum_float(summaries, "structured_fact_result_count"),
        "structured_fact_status_counts": structured_fact_status_counts,
        "structured_fact_predicate_counts": structured_fact_predicate_counts,
        "per_trace_claim_triple_coverage_rate": _numeric_summary(
            item.get("triple_claim_coverage_rate") for item in metrics
        ),
        "per_trace_audit_pass_rate": _numeric_summary(
            item.get("triple_audit_pass_rate") for item in metrics
        ),
        "per_trace_slot_coverage_rate": _numeric_summary(
            item.get("triple_slot_coverage_rate") for item in metrics
        ),
    }


def _fallback_audit_claim_covered_count(summaries: Sequence[Mapping[str, Any]]) -> float | None:
    total = 0.0
    observed = False
    for summary in summaries:
        claims_with_triples = _finite_float(summary.get("claims_with_triples"))
        audit_report_count = _finite_float(summary.get("audit_report_count"))
        if claims_with_triples is None or audit_report_count is None:
            continue
        observed = True
        total += min(claims_with_triples, audit_report_count)
    return total if observed else None


def _aggregate_final_answer(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available_count = sum(1 for item in metrics if bool(item.get("final_answer_available")))
    answerable_count = sum(1 for item in metrics if item.get("final_answer_answerable") is True)
    followup_count = sum(1 for item in metrics if item.get("final_answer_requires_followup") is True)
    return {
        "source_trace_count": len(metrics),
        "available_trace_count": available_count,
        "missing_trace_count": len(metrics) - available_count,
        "coverage_rate": _safe_div(available_count, len(metrics)),
        "answerable_count": answerable_count,
        "answerable_rate": _safe_div(answerable_count, available_count),
        "followup_required_count": followup_count,
        "followup_required_rate": _safe_div(followup_count, available_count),
        "source_counts": _counts(item.get("final_answer_source") for item in metrics),
        "status_counts": _counts(item.get("final_answer_status") for item in metrics),
        "action_counts": _counts(item.get("final_answer_action") for item in metrics),
        "risk_level_counts": _counts(item.get("final_answer_risk_level") for item in metrics),
        "confidence": _numeric_summary(item.get("final_answer_confidence") for item in metrics),
        "evidence_count": _numeric_summary(item.get("final_answer_evidence_count") for item in metrics),
        "total_claims": _numeric_summary(item.get("final_answer_total_claims") for item in metrics),
        "blocked_claim_count": _numeric_summary(
            item.get("final_answer_blocked_claim_count") for item in metrics
        ),
        "summary_observations": sum(1 for item in metrics if _mapping(item.get("final_answer_summary"))),
    }


def _aggregate_promotion_contract(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available_count = sum(1 for item in metrics if bool(item.get("promotion_contract_available")))
    property_count_values = [
        item.get("promotion_contract_recommended_route_covered_fact_property_count")
        for item in metrics
    ]
    property_sets = [
        _string_sequence(item.get("promotion_contract_recommended_route_covered_fact_properties"))
        for item in metrics
    ]
    property_scope_observations = sum(
        (value is not None and _finite_float(value) is not None) or bool(properties)
        for value, properties in zip(property_count_values, property_sets, strict=True)
    )
    matrix_available_count = sum(
        1
        for item in metrics
        if bool(item.get("promotion_contract_triple_extraction_fixture_matrix_available"))
    )
    manifest_values = [
        item.get("promotion_contract_triple_extraction_fixture_matrix_manifest_verified")
        for item in metrics
    ]
    manifest_observations = sum(value is not None for value in manifest_values)
    product_runtime_drift = _aggregate_promotion_contract_product_runtime_drift(metrics)
    return {
        "source_trace_count": len(metrics),
        "available_trace_count": available_count,
        "missing_trace_count": len(metrics) - available_count,
        "coverage_rate": _safe_div(available_count, len(metrics)),
        "source_counts": _counts(item.get("promotion_contract_source") for item in metrics),
        "source_status_counts": _counts(
            item.get("promotion_contract_source_status") for item in metrics
        ),
        "budget_enabled_counts": _counts(
            item.get("promotion_contract_budget_enabled") for item in metrics
        ),
        "summary_observations": sum(
            1 for item in metrics if _mapping(item.get("promotion_contract_summary"))
        ),
        "product_runtime_drift": product_runtime_drift,
        "covered_fact_properties": {
            "recommended_route_observation_count": property_scope_observations,
            "recommended_route_coverage_rate": _safe_div(
                property_scope_observations,
                len(metrics),
            ),
            "recommended_route_count": _numeric_summary(property_count_values),
            "recommended_route_properties": _counts_from_sequence_items(property_sets),
            "required_route_baseline_records": _counts_from_mapping_keys(
                item.get("promotion_contract_required_route_baseline_covered_fact_property_counts")
                for item in metrics
            ),
            "recommended_route_property_metrics": _covered_fact_rollup_summary(
                metrics,
                prefix="promotion_contract_recommended_route_covered_fact",
            ),
            "required_route_baseline_property_metrics": _covered_fact_rollup_summary(
                metrics,
                prefix="promotion_contract_required_route_baseline_covered_fact",
            ),
            "structured_fact_robustness_records": _counts_from_mapping_keys(
                item.get("promotion_contract_structured_fact_robustness_property_counts")
                for item in metrics
            ),
            "structured_fact_robustness_property_metrics": _covered_fact_rollup_summary(
                metrics,
                prefix="promotion_contract_structured_fact_robustness",
            ),
        },
        "triple_extraction_fixture_matrix": {
            "available_trace_count": matrix_available_count,
            "missing_trace_count": len(metrics) - matrix_available_count,
            "coverage_rate": _safe_div(matrix_available_count, len(metrics)),
            "source_counts": _counts(
                item.get("promotion_contract_triple_extraction_fixture_matrix_source")
                for item in metrics
            ),
            "status_counts": _counts(
                item.get("promotion_contract_triple_extraction_fixture_matrix_status")
                for item in metrics
            ),
            "manifest_verification_observations": manifest_observations,
            "manifest_verified_count": sum(value is True for value in manifest_values),
            "manifest_failed_count": sum(value is False for value in manifest_values),
            "manifest_unknown_count": len(metrics) - manifest_observations,
            "n_corpora": _numeric_summary(
                item.get("promotion_contract_triple_extraction_fixture_matrix_n_corpora")
                for item in metrics
            ),
            "promoted_corpora": _numeric_summary(
                item.get("promotion_contract_triple_extraction_fixture_matrix_promoted_corpora")
                for item in metrics
            ),
            "distinct_predicate_count": _numeric_summary(
                item.get(
                    "promotion_contract_triple_extraction_fixture_matrix_distinct_predicate_count"
                )
                for item in metrics
            ),
            "mean_best_f1": _numeric_summary(
                item.get("promotion_contract_triple_extraction_fixture_matrix_mean_best_f1")
                for item in metrics
            ),
            "mean_f1_lift": _numeric_summary(
                item.get("promotion_contract_triple_extraction_fixture_matrix_mean_f1_lift")
                for item in metrics
            ),
        },
    }


def _covered_fact_rollup_summary(
    metrics: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any]:
    return {
        "property_metric_count": _numeric_summary(
            item.get(f"{prefix}_property_metric_count") for item in metrics
        ),
        "min_records": _numeric_summary(item.get(f"{prefix}_min_records") for item in metrics),
        "min_source_documents": _numeric_summary(
            item.get(f"{prefix}_min_source_documents") for item in metrics
        ),
        "min_decision_accuracy": _numeric_summary(
            item.get(f"{prefix}_min_decision_accuracy") for item in metrics
        ),
        "max_false_supported_rate": _numeric_summary(
            item.get(f"{prefix}_max_false_supported_rate") for item in metrics
        ),
        "min_false_refuted_rate": _numeric_summary(
            item.get(f"{prefix}_min_false_refuted_rate") for item in metrics
        ),
    }


def _aggregate_promotion_contract_product_runtime_drift(
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    available_count = sum(
        1
        for item in metrics
        if bool(item.get("promotion_contract_product_runtime_drift_available"))
    )
    return {
        "available_trace_count": available_count,
        "missing_trace_count": len(metrics) - available_count,
        "coverage_rate": _safe_div(available_count, len(metrics)),
        "status_counts": _counts(
            item.get("promotion_contract_product_runtime_drift_status") for item in metrics
        ),
        "gate_enabled_counts": _counts(
            item.get("promotion_contract_product_runtime_drift_gate_enabled")
            for item in metrics
        ),
        "promotion_evidence_required_counts": _counts(
            item.get("promotion_contract_product_runtime_drift_promotion_evidence_required")
            for item in metrics
        ),
        "triple_audit_evidence_required_counts": _counts(
            item.get("promotion_contract_product_runtime_drift_triple_audit_evidence_required")
            for item in metrics
        ),
        "covered_fact_property_evidence_required_counts": _counts(
            item.get(
                "promotion_contract_product_runtime_drift_covered_fact_property_evidence_required"
            )
            for item in metrics
        ),
        "compared_metric_count": _numeric_summary(
            item.get("promotion_contract_product_runtime_drift_compared_metric_count")
            for item in metrics
        ),
        "blocked_metric_count": _numeric_summary(
            item.get("promotion_contract_product_runtime_drift_blocked_metric_count")
            for item in metrics
        ),
        "promotion_evidence_metric_count": _numeric_summary(
            item.get("promotion_contract_product_runtime_drift_promotion_evidence_metric_count")
            for item in metrics
        ),
        "promotion_evidence_blocked_metric_count": _numeric_summary(
            item.get(
                "promotion_contract_product_runtime_drift_promotion_evidence_blocked_metric_count"
            )
            for item in metrics
        ),
        "triple_audit_evidence_metric_count": _numeric_summary(
            item.get("promotion_contract_product_runtime_drift_triple_audit_evidence_metric_count")
            for item in metrics
        ),
        "triple_audit_evidence_blocked_metric_count": _numeric_summary(
            item.get(
                "promotion_contract_product_runtime_drift_triple_audit_evidence_blocked_metric_count"
            )
            for item in metrics
        ),
        "covered_fact_property_evidence_metric_count": _numeric_summary(
            item.get(
                "promotion_contract_product_runtime_drift_covered_fact_property_evidence_metric_count"
            )
            for item in metrics
        ),
        "covered_fact_property_evidence_blocked_metric_count": _numeric_summary(
            item.get(
                "promotion_contract_product_runtime_drift_covered_fact_property_evidence_blocked_metric_count"
            )
            for item in metrics
        ),
        "promotion_evidence": _aggregate_product_runtime_drift_evidence(
            metrics,
            prefixes=_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES,
        ),
        "triple_audit_evidence": _aggregate_product_runtime_drift_evidence(
            metrics,
            prefixes=_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES,
        ),
        "covered_fact_property_evidence": _aggregate_product_runtime_drift_evidence(
            metrics,
            prefixes=_PRODUCT_RUNTIME_DRIFT_COVERED_FACT_PROPERTY_EVIDENCE_PREFIXES,
        ),
    }


def _aggregate_product_runtime_drift_evidence(
    metrics: Sequence[Mapping[str, Any]],
    *,
    prefixes: Sequence[str],
) -> dict[str, Any]:
    return {
        prefix: {
            "baseline": _numeric_summary(
                item.get(f"promotion_contract_product_runtime_drift_{prefix}_baseline")
                for item in metrics
            ),
            "current": _numeric_summary(
                item.get(f"promotion_contract_product_runtime_drift_{prefix}_current")
                for item in metrics
            ),
            "status_counts": _counts(
                item.get(f"promotion_contract_product_runtime_drift_{prefix}_status")
                for item in metrics
            ),
        }
        for prefix in prefixes
    }


def _verification_scope(summary: Mapping[str, Any]) -> str:
    scope = _optional_string(summary.get("verification_scope"))
    if scope is not None:
        return scope.strip().lower()
    if summary.get("run_verifier") is False:
        return "none"
    if summary.get("run_verifier") is True:
        return "all"
    return "unknown"


def _merge_counts(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        numeric = _finite_float(value)
        if numeric is None:
            continue
        target[str(key)] = target.get(str(key), 0) + int(numeric)


def _aggregate_route_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = tuple(_mapping(summary) for summary in summaries)
    total = _sum_int(summaries, "total")
    routed_total = _sum_int(summaries, "routed_total")
    duration_observations = _sum_int(summaries, "duration_observations")
    selected_duration_observations = _sum_int(summaries, "selected_route_duration_observations")
    attempted_observations = _sum_int(summaries, "attempted_route_count_observations")
    route_budget_observations = _sum_int(summaries, "route_budget_limit_observations")
    total_duration = _sum_float(summaries, "total_duration_seconds")
    total_selected_duration = _sum_float(summaries, "total_selected_route_duration_seconds")
    total_attempted = _sum_float(summaries, "total_attempted_route_count")
    used_retrieval_count = _sum_int(summaries, "used_retrieval_count")
    retrieval_hit_count = _sum_int(summaries, "retrieval_hit_count")
    route_budget_exhausted_count = _sum_int(summaries, "route_budget_exhausted_count")
    selected_fallthrough_budget_stop_count = _sum_int(
        summaries,
        "selected_fallthrough_budget_stop_count",
    )
    unattempted_route_count = _sum_int(summaries, "unattempted_route_count")
    return {
        "source_trace_count": len(summaries),
        "total": total,
        "routed_total": routed_total,
        "unrouted_total": None if total is None or routed_total is None else total - routed_total,
        "duration_observations": duration_observations,
        "total_duration_seconds": total_duration,
        "mean_duration_seconds": _safe_div(total_duration, duration_observations),
        "per_trace_mean_duration_seconds": _numeric_summary(
            summary.get("mean_duration_seconds") for summary in summaries
        ),
        "per_trace_p95_duration_seconds": _numeric_summary(
            summary.get("p95_duration_seconds") for summary in summaries
        ),
        "per_trace_p99_duration_seconds": _numeric_summary(
            summary.get("p99_duration_seconds") for summary in summaries
        ),
        "max_duration_seconds": _max_numeric(summary.get("max_duration_seconds") for summary in summaries),
        "selected_route_duration_observations": selected_duration_observations,
        "total_selected_route_duration_seconds": total_selected_duration,
        "mean_selected_route_duration_seconds": _safe_div(
            total_selected_duration,
            selected_duration_observations,
        ),
        "attempted_route_count_observations": attempted_observations,
        "total_attempted_route_count": total_attempted,
        "mean_attempted_route_count": _safe_div(total_attempted, attempted_observations),
        "route_budget_limit_observations": route_budget_observations,
        "route_budget_exhausted_count": route_budget_exhausted_count,
        "route_budget_exhaustion_rate": _safe_div(
            route_budget_exhausted_count,
            route_budget_observations,
        ),
        "selected_fallthrough_budget_stop_count": selected_fallthrough_budget_stop_count,
        "unattempted_route_count": unattempted_route_count,
        "mean_unattempted_route_count": _safe_div(unattempted_route_count, total),
        "used_retrieval_count": used_retrieval_count,
        "retrieval_use_rate": _safe_div(used_retrieval_count, total),
        "retrieval_hit_count": retrieval_hit_count,
        "mean_retrieval_hits": _safe_div(retrieval_hit_count, total),
    }


def _budget_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: ProductRuntimeBudgetPolicy | None,
) -> dict[str, Any]:
    if policy is None:
        return {
            "enabled": False,
            "passed": None,
            "policy": None,
            "passed_count": None,
            "failed_count": None,
            "failure_counts_by_metric": {},
        }
    budgets = [_mapping(record.get("budget")) for record in records]
    failed = [budget for budget in budgets if budget.get("passed") is not True]
    failure_counts: dict[str, int] = {}
    for budget in failed:
        for failure in _sequence(budget.get("failures")):
            if not isinstance(failure, Mapping):
                continue
            metric = str(failure.get("metric", "unknown"))
            failure_counts[metric] = failure_counts.get(metric, 0) + 1
    return {
        "enabled": policy.enabled(),
        "passed": not failed,
        "policy": policy.to_dict(),
        "passed_count": len(budgets) - len(failed),
        "failed_count": len(failed),
        "failure_counts_by_metric": failure_counts,
    }


def _status_from_budget(budget: Mapping[str, Any]) -> str:
    if not bool(budget.get("enabled")):
        return "observed"
    return "promote" if budget.get("passed") is True else "blocked"


def _blocking_reasons(budget: Mapping[str, Any]) -> tuple[str, ...]:
    if not bool(budget.get("enabled")):
        return ()
    if budget.get("passed") is True:
        return ()
    counts = _mapping(budget.get("failure_counts_by_metric"))
    if not counts:
        return ("one or more runtime budget checks failed",)
    return tuple(
        f"{metric}: failed {count} trace(s)"
        for metric, count in sorted(counts.items())
    )


def _write_recommended_policy(
    config: ProductRuntimeBaselineConfig,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    if config.recommended_policy_path is None:
        return None
    policy_payload = _recommended_policy_payload(config, report)
    _write_report(config.recommended_policy_path, policy_payload, compact=config.compact_json)
    policy_config = _mapping(_nested(report, "config", "recommended_policy"))
    policy_config.update({
        "enabled": True,
        "written": True,
        "path": str(config.recommended_policy_path),
        "policy_enabled": ProductRuntimeBudgetPolicy.from_mapping(policy_payload).enabled(),
        "threshold_count": _policy_threshold_count(policy_payload),
        "source": _nested(report, "optimization", "policy_hints", "source"),
    })
    report["config"]["recommended_policy"] = policy_config
    return policy_payload


def _recommended_policy_payload(
    config: ProductRuntimeBaselineConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _mapping(
        _nested(
            report,
            "optimization",
            "policy_hints",
            "candidate_runtime_budget_policy",
        )
    )
    policy_payload = ProductRuntimeBudgetPolicy.from_mapping(candidate).to_dict()
    policy_payload["metadata"] = {
        "source": "run_product_runtime_baseline.optimization.policy_hints",
        "optimization_source": _nested(report, "optimization", "policy_hints", "source"),
        "baseline_report": str(config.report_path),
        "baseline_status": report.get("status"),
        "optimization_status": _nested(report, "optimization", "status"),
        "trace_count": _nested(report, "summary", "n_traces"),
    }
    return policy_payload


def _policy_threshold_count(policy_payload: Mapping[str, Any]) -> int:
    threshold_fields = (
        "max_total_seconds",
        "max_phase_seconds",
        "max_phase_p95_seconds",
        "max_phase_p99_seconds",
        "max_mean_route_duration_seconds",
        "max_p95_route_duration_seconds",
        "max_p99_route_duration_seconds",
        "max_route_duration_seconds",
        "max_mean_attempted_route_count",
        "max_route_budget_exhaustion_rate",
        "max_retrieval_use_rate",
        "max_retrieval_hit_count",
        "min_cache_hit_rate",
        "min_named_cache_hit_rate",
        "min_verification_skip_rate",
        "min_selective_claim_skip_rate",
        "max_verified_claim_count",
    )
    count = 0
    for field_name in threshold_fields:
        value = policy_payload.get(field_name)
        if isinstance(value, Mapping):
            count += len(value)
        elif value is not None:
            count += 1
    return count


def _load_policy(config: ProductRuntimeBaselineConfig) -> tuple[ProductRuntimeBudgetPolicy | None, str | None]:
    if config.policy is not None:
        return (
            config.policy
            if isinstance(config.policy, ProductRuntimeBudgetPolicy)
            else ProductRuntimeBudgetPolicy.from_mapping(config.policy),
            "inline",
        )
    if config.policy_path is not None:
        payload = _load_json(config.policy_path)
        return ProductRuntimeBudgetPolicy.from_mapping(payload), str(config.policy_path)
    if config.promotion_contract_path is not None:
        contract = ProductPromotionContract.from_json(config.promotion_contract_path)
        return contract.runtime_budget_policy, str(config.promotion_contract_path)
    return None, None


def _write_artifact_manifest(
    config: ProductRuntimeBaselineConfig,
    report: Mapping[str, Any],
    *,
    artifacts: Mapping[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    promotion_contract_drift_metadata = _promotion_contract_runtime_drift_flat_metadata(report)
    manifest = build_artifact_manifest(
        _artifact_paths(config) if artifacts is None else artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_product_runtime_baseline",
            "status": report.get("status"),
            "trace_count": len(config.trace_paths),
            "trace_records_storage": _mapping(report.get("trace_records")).get("storage"),
            "trace_records_cache_path": _nested(report, "paths", "trace_records_cache"),
            "trace_records_cache_source": _nested(report, "config", "trace_record_cache", "source"),
            "trace_records_cache_hit": _nested(report, "config", "trace_record_cache", "cache_hit"),
            "trace_records_cache_written": _nested(report, "config", "trace_record_cache", "cache_written"),
            "trace_scan_workers": config.trace_scan_workers,
            "trace_scan_effective_workers": _nested(
                report,
                "config",
                "trace_record_cache",
                "trace_scan_workers",
            ),
            "recommended_policy_path": _nested(report, "paths", "recommended_policy"),
            "recommended_policy_written": _nested(report, "config", "recommended_policy", "written"),
            "recommended_policy_enabled": _nested(report, "config", "recommended_policy", "policy_enabled"),
            "recommended_policy_threshold_count": _nested(
                report,
                "config",
                "recommended_policy",
                "threshold_count",
            ),
            "budget_enabled": _mapping(report.get("budget")).get("enabled"),
            "budget_passed": _mapping(report.get("budget")).get("passed"),
            "compact_json": config.compact_json,
            **promotion_contract_drift_metadata,
            **dict(config.metadata),
        },
    )
    _write_report(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _artifact_paths(config: ProductRuntimeBaselineConfig) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "product_runtime_baseline_report": config.report_path,
        "product_runtime_trace_records": config.trace_records_path,
        "product_runtime_trace_record_cache": config.trace_records_cache_path,
        "recommended_runtime_budget_policy": config.recommended_policy_path,
        "policy": config.policy_path,
        "promotion_contract": config.promotion_contract_path,
    }
    for index, trace_path in enumerate(config.trace_paths):
        artifacts[f"trace_{index:04d}_{_safe_artifact_name(trace_path.stem)}"] = trace_path
    return artifacts


def _record_registry(config: ProductRuntimeBaselineConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
    promotion_contract_drift_metadata = _promotion_contract_runtime_drift_flat_metadata(report)
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_product_runtime_baseline(
        name=str(config.name),
        path=config.report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_product_runtime_baseline",
            "status": report.get("status"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "trace_count": len(config.trace_paths),
            "trace_records_storage": _mapping(report.get("trace_records")).get("storage"),
            "trace_records_path": _mapping(report.get("trace_records")).get("path"),
            "trace_records_cache_path": _nested(report, "paths", "trace_records_cache"),
            "trace_records_cache_source": _nested(report, "config", "trace_record_cache", "source"),
            "trace_records_cache_hit": _nested(report, "config", "trace_record_cache", "cache_hit"),
            "trace_records_cache_written": _nested(report, "config", "trace_record_cache", "cache_written"),
            "trace_scan_workers": config.trace_scan_workers,
            "trace_scan_effective_workers": _nested(
                report,
                "config",
                "trace_record_cache",
                "trace_scan_workers",
            ),
            "recommended_policy_path": _nested(report, "paths", "recommended_policy"),
            "recommended_policy_written": _nested(report, "config", "recommended_policy", "written"),
            "recommended_policy_enabled": _nested(report, "config", "recommended_policy", "policy_enabled"),
            "recommended_policy_threshold_count": _nested(
                report,
                "config",
                "recommended_policy",
                "threshold_count",
            ),
            "budget_enabled": _mapping(report.get("budget")).get("enabled"),
            "budget_passed": _mapping(report.get("budget")).get("passed"),
            "failed_count": _mapping(report.get("budget")).get("failed_count"),
            "compact_json": config.compact_json,
            **promotion_contract_drift_metadata,
            **dict(config.metadata),
        },
    )
    if config.recommended_policy_path is not None:
        registry.record_product_runtime_budget_policy(
            name=f"{config.name}-recommended-policy",
            path=config.recommended_policy_path,
            version=str(config.version),
            metadata={
                "workflow": "run_product_runtime_baseline",
                "source_baseline_record": f"product_runtime_baseline:{config.name}:{config.version}",
                "source_baseline_report": str(config.report_path),
                "artifact_manifest": str(config.resolved_artifact_manifest_path),
                "policy_enabled": _nested(report, "config", "recommended_policy", "policy_enabled"),
                "threshold_count": _nested(report, "config", "recommended_policy", "threshold_count"),
                "optimization_status": _nested(report, "optimization", "status"),
                "compact_json": config.compact_json,
                **dict(config.metadata),
            },
        )
    registry.save_json()


def _promotion_contract_runtime_drift_flat_metadata(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    drift = _mapping(
        _nested(
            report,
            "summary",
            "promotion_contract",
            "product_runtime_drift",
        )
    )
    if not drift:
        return {}
    metadata = {
        "promotion_contract_product_runtime_drift_available_trace_count": drift.get(
            "available_trace_count"
        ),
        "promotion_contract_product_runtime_drift_missing_trace_count": drift.get(
            "missing_trace_count"
        ),
        "promotion_contract_product_runtime_drift_coverage_rate": drift.get("coverage_rate"),
        "promotion_contract_product_runtime_drift_status_counts": dict(
            _mapping(drift.get("status_counts"))
        ),
        "promotion_contract_product_runtime_drift_gate_enabled_counts": dict(
            _mapping(drift.get("gate_enabled_counts"))
        ),
        "promotion_contract_product_runtime_drift_promotion_evidence_required_counts": dict(
            _mapping(drift.get("promotion_evidence_required_counts"))
        ),
        "promotion_contract_product_runtime_drift_triple_audit_evidence_required_counts": dict(
            _mapping(drift.get("triple_audit_evidence_required_counts"))
        ),
        "promotion_contract_product_runtime_drift_compared_metric_count_mean": _nested(
            drift,
            "compared_metric_count",
            "mean",
        ),
        "promotion_contract_product_runtime_drift_blocked_metric_count_mean": _nested(
            drift,
            "blocked_metric_count",
            "mean",
        ),
        "promotion_contract_product_runtime_drift_promotion_evidence_metric_count_mean": _nested(
            drift,
            "promotion_evidence_metric_count",
            "mean",
        ),
        "promotion_contract_product_runtime_drift_promotion_evidence_blocked_metric_count_mean": _nested(
            drift,
            "promotion_evidence_blocked_metric_count",
            "mean",
        ),
        "promotion_contract_product_runtime_drift_triple_audit_evidence_metric_count_mean": _nested(
            drift,
            "triple_audit_evidence_metric_count",
            "mean",
        ),
        "promotion_contract_product_runtime_drift_triple_audit_evidence_blocked_metric_count_mean": _nested(
            drift,
            "triple_audit_evidence_blocked_metric_count",
            "mean",
        ),
    }
    metadata.update(
        _product_runtime_drift_evidence_flat_metadata(
            _mapping(drift.get("promotion_evidence")),
            prefixes=_PRODUCT_RUNTIME_DRIFT_PROMOTION_EVIDENCE_PREFIXES,
        )
    )
    metadata.update(
        _product_runtime_drift_evidence_flat_metadata(
            _mapping(drift.get("triple_audit_evidence")),
            prefixes=_PRODUCT_RUNTIME_DRIFT_TRIPLE_AUDIT_EVIDENCE_PREFIXES,
        )
    )
    return metadata


def _product_runtime_drift_evidence_flat_metadata(
    evidence: Mapping[str, Any],
    *,
    prefixes: Sequence[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for prefix in prefixes:
        values = _mapping(evidence.get(prefix))
        metadata[f"promotion_contract_product_runtime_drift_{prefix}_baseline_mean"] = _nested(
            values,
            "baseline",
            "mean",
        )
        metadata[f"promotion_contract_product_runtime_drift_{prefix}_current_mean"] = _nested(
            values,
            "current",
            "mean",
        )
        metadata[f"promotion_contract_product_runtime_drift_{prefix}_status_counts"] = dict(
            _mapping(values.get("status_counts"))
        )
    return metadata


def _load_trace(path: str | Path) -> dict[str, Any]:
    payload = _load_json(path)
    reject_bounded_product_trace(payload, path=path)
    if "runtime_trace" not in payload and "verification_results" not in payload:
        raise ValueError(f"ProductTrace JSON is missing runtime/control fields: {path}")
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


def _write_report(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_text(payload, compact=compact), encoding="utf-8")


def _json_text(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _jsonl_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def _numeric_summary(values: Sequence[Any] | Any) -> dict[str, Any]:
    raw_values = tuple(values)
    finite_values = [
        numeric
        for value in raw_values
        if (numeric := _finite_float(value)) is not None
    ]
    if not finite_values:
        return {
            "count": 0,
            "missing_or_nonfinite": len(raw_values),
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    total = sum(finite_values)
    return {
        "count": len(finite_values),
        "missing_or_nonfinite": len(raw_values) - len(finite_values),
        "mean": total / len(finite_values),
        "min": min(finite_values),
        "p50": _percentile(finite_values, 50.0),
        "p95": _percentile(finite_values, 95.0),
        "p99": _percentile(finite_values, 99.0),
        "max": max(finite_values),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
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


def _sum_int(items: Sequence[Mapping[str, Any]], field_name: str) -> int | None:
    values = [_finite_float(item.get(field_name)) for item in items]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return int(sum(finite_values))


def _sum_float(items: Sequence[Mapping[str, Any]], field_name: str) -> float | None:
    values = [_finite_float(item.get(field_name)) for item in items]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return float(sum(finite_values))


def _max_numeric(values: Sequence[Any] | Any) -> float | None:
    finite_values = [
        numeric
        for value in values
        if (numeric := _finite_float(value)) is not None
    ]
    return None if not finite_values else max(finite_values)


def _safe_div(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _string_sequence(value: Any) -> tuple[str, ...]:
    values = []
    for item in _sequence(value):
        if isinstance(item, Mapping) and item.get("_truncated") is True:
            continue
        text = _optional_string(item)
        if text is not None:
            values.append(text)
    return tuple(values)


def _counts_from_sequence_items(values: Sequence[Sequence[Any]] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sequence in values:
        for item in set(_string_sequence(sequence)):
            counts[item] = counts.get(item, 0) + 1
    return counts


def _counts_from_mapping_keys(values: Sequence[Any] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        for key in _mapping(value):
            text = _optional_string(key)
            if text is None:
                continue
            counts[text] = counts.get(text, 0) + 1
    return counts


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned or "trace"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _counts(values: Sequence[Any] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = _optional_string(value)
        if text is None:
            continue
        counts[text] = counts.get(text, 0) + 1
    return counts


def _count_rate(counts: Mapping[str, Any], key: str) -> float | None:
    values = [_finite_float(value) for value in counts.values()]
    finite = [value for value in values if value is not None]
    total = sum(finite)
    if total <= 0:
        return None
    return (_finite_float(counts.get(key)) or 0.0) / total


def _sort_value(value: Any) -> float:
    numeric = _finite_float(value)
    return float("-inf") if numeric is None else numeric


def _with_headroom(value: Any, *, ratio: float = 1.25) -> float | None:
    numeric = _finite_float(value)
    if numeric is None:
        return None
    return round(numeric * ratio, 6)


def _parse_metadata(values: Sequence[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--metadata entries must be formatted as key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def _config_from_args(args: argparse.Namespace) -> ProductRuntimeBaselineConfig:
    return ProductRuntimeBaselineConfig(
        trace_paths=tuple(args.trace),
        report_path=Path(args.json),
        policy_path=Path(args.policy) if args.policy else None,
        promotion_contract_path=Path(args.promotion_contract) if args.promotion_contract else None,
        trace_records_path=Path(args.trace_records_jsonl) if args.trace_records_jsonl else None,
        trace_records_cache_path=(
            Path(args.trace_records_cache_json) if args.trace_records_cache_json else None
        ),
        refresh_trace_records_cache=bool(args.refresh_trace_records_cache),
        trace_scan_workers=args.trace_scan_workers,
        recommended_policy_path=Path(args.save_recommended_policy) if args.save_recommended_policy else None,
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = build_product_runtime_baseline(_config_from_args(args))
    print(_json_text(report, compact=bool(args.compact_json)), end="")
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a ProductTrace runtime baseline report")
    parser.add_argument("--trace", action="append", required=True, help="ProductTrace JSON path; repeatable")
    parser.add_argument("--json", required=True, help="output baseline report JSON path")
    parser.add_argument("--policy", default=None, help="ProductRuntimeBudgetPolicy JSON path")
    parser.add_argument("--promotion-contract", default=None, help="ProductPromotionContract/release report JSON path")
    parser.add_argument("--trace-records-jsonl", default=None,
                        help="write per-trace records to JSONL sidecar instead of embedding them in the report")
    parser.add_argument("--trace-records-cache-json", default=None,
                        help="optional cache path for compact per-trace runtime metric/budget records")
    parser.add_argument("--refresh-trace-records-cache", action="store_true",
                        help="rebuild --trace-records-cache-json even when a valid cache exists")
    parser.add_argument("--trace-scan-workers", type=int, default=1,
                        help="maximum worker threads for ProductTrace JSON scan and metric extraction")
    parser.add_argument("--save-recommended-policy", default=None,
                        help="write the optimization candidate ProductRuntimeBudgetPolicy JSON")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest output path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry product runtime baseline name")
    parser.add_argument("--version", default=None, help="registry product runtime baseline version")
    parser.add_argument("--metadata", action="append", default=[], help="metadata key=value; repeatable")
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified baseline report and manifest JSON")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
