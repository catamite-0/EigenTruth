"""Compare registered verifier-route promotion baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import artifact_json_cache as _artifact_json_cache  # noqa: E402
from benchmarks.runtime_budget_policy import (  # noqa: E402
    RuntimeBudgetPolicy,
    evaluate_runtime_budget,
    runtime_metrics_from_metadata,
)
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    ArtifactVerificationContext,
    RegistryRecord,
    verify_artifact_manifest,
)

_load_optional_json = _artifact_json_cache.load_optional_json

DEFAULT_MIN_STRESS_FALSE_SUPPORTED_RATE = 0.50
DEFAULT_MAX_STRESS_FALSE_REFUTED_RATE = 0.05
ANSWER_ECHO_CORPUS_TYPE = "retrieval_stress_answer_echo"


def compare_route_baselines(
    *,
    registry_path: str | Path,
    baseline_keys: Sequence[str] = (),
    recursive: bool = True,
    allow_unverified: bool = False,
    min_selected: int | None = None,
    min_decision_accuracy: float | None = None,
    max_false_supported_rate: float | None = None,
    min_false_refuted_rate: float | None = None,
    max_verified_false_alarm: float | None = None,
    min_verified_detection: float | None = None,
    max_mean_duration_seconds: float | None = None,
    max_p99_duration_seconds: float | None = None,
    max_max_duration_seconds: float | None = None,
    max_mean_attempted_route_count: float | None = None,
    max_retrieval_use_rate: float | None = None,
    max_runtime_total_seconds: float | None = None,
    max_retrieval_hit_count: float | None = None,
    min_claims_cache_hit_rate: float | None = None,
    min_verifier_trace_cache_hit_rate: float | None = None,
    min_covered_fact_properties: int | None = None,
    min_covered_fact_property_records: int | None = None,
    min_covered_fact_property_source_documents: int | None = None,
    min_covered_fact_property_decision_accuracy: float | None = None,
    max_covered_fact_property_false_supported_rate: float | None = None,
    min_covered_fact_property_false_refuted_rate: float | None = None,
    require_non_oracle_evidence: bool = False,
    require_retrieval_provenance_filter: bool = False,
    required_retrieval_source_prefixes: Sequence[str] = (),
    required_retrieval_metadata: Mapping[str, Any] | None = None,
    min_retrieval_filter_score: float | None = None,
    require_retrieval_stress_control: bool = False,
    retrieval_stress_manifest: str | Path | None = None,
    min_stress_false_supported_rate: float | None = None,
    max_stress_false_refuted_rate: float | None = None,
    notes: Sequence[str] = (),
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache: MutableMapping[str, dict[str, Any]] | None = None,
    json_cache_stats: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed comparison of registered route-promotion baselines."""
    verification_context = ArtifactVerificationContext(
        fingerprint_cache=fingerprint_cache,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
    )
    cache = verification_context.fingerprint_cache
    payload_cache = verification_context.json_cache
    payload_cache_stats = verification_context.json_cache_stats
    registry = ArtifactRegistry.load_json(registry_path)
    records = _select_records(registry, baseline_keys=baseline_keys)
    rows = [
        _route_baseline_row(
            record,
            recursive=recursive,
            allow_unverified=allow_unverified,
            min_selected=min_selected,
            min_decision_accuracy=min_decision_accuracy,
            max_false_supported_rate=max_false_supported_rate,
            min_false_refuted_rate=min_false_refuted_rate,
            max_verified_false_alarm=max_verified_false_alarm,
            min_verified_detection=min_verified_detection,
            max_mean_duration_seconds=max_mean_duration_seconds,
            max_p99_duration_seconds=max_p99_duration_seconds,
            max_max_duration_seconds=max_max_duration_seconds,
            max_mean_attempted_route_count=max_mean_attempted_route_count,
            max_retrieval_use_rate=max_retrieval_use_rate,
            max_runtime_total_seconds=max_runtime_total_seconds,
            max_retrieval_hit_count=max_retrieval_hit_count,
            min_claims_cache_hit_rate=min_claims_cache_hit_rate,
            min_verifier_trace_cache_hit_rate=min_verifier_trace_cache_hit_rate,
            min_covered_fact_properties=min_covered_fact_properties,
            min_covered_fact_property_records=min_covered_fact_property_records,
            min_covered_fact_property_source_documents=min_covered_fact_property_source_documents,
            min_covered_fact_property_decision_accuracy=min_covered_fact_property_decision_accuracy,
            max_covered_fact_property_false_supported_rate=max_covered_fact_property_false_supported_rate,
            min_covered_fact_property_false_refuted_rate=min_covered_fact_property_false_refuted_rate,
            require_non_oracle_evidence=require_non_oracle_evidence,
            require_retrieval_provenance_filter=require_retrieval_provenance_filter,
            required_retrieval_source_prefixes=required_retrieval_source_prefixes,
            required_retrieval_metadata=required_retrieval_metadata,
            min_retrieval_filter_score=min_retrieval_filter_score,
            require_retrieval_stress_control=require_retrieval_stress_control,
            retrieval_stress_manifest=retrieval_stress_manifest,
            min_stress_false_supported_rate=min_stress_false_supported_rate,
            max_stress_false_refuted_rate=max_stress_false_refuted_rate,
            fingerprint_cache=cache,
            json_cache=payload_cache,
            json_cache_stats=payload_cache_stats,
        )
        for record in records
    ]
    leaderboard = sorted(rows, key=_leaderboard_key)
    recommendation = next((row for row in leaderboard if row["gate"]["passed"]), None)
    decision = _decision(leaderboard, recommendation)
    return {
        "schema_version": 1,
        "workflow": "route_baseline_comparison",
        "registry": str(registry_path),
        "config": {
            "baseline_keys": list(baseline_keys),
            "recursive": recursive,
            "allow_unverified": allow_unverified,
            "min_selected": min_selected,
            "min_decision_accuracy": min_decision_accuracy,
            "max_false_supported_rate": max_false_supported_rate,
            "min_false_refuted_rate": min_false_refuted_rate,
            "max_verified_false_alarm": max_verified_false_alarm,
            "min_verified_detection": min_verified_detection,
            "max_mean_duration_seconds": max_mean_duration_seconds,
            "max_p99_duration_seconds": max_p99_duration_seconds,
            "max_max_duration_seconds": max_max_duration_seconds,
            "max_mean_attempted_route_count": max_mean_attempted_route_count,
            "max_retrieval_use_rate": max_retrieval_use_rate,
            "max_runtime_total_seconds": max_runtime_total_seconds,
            "max_retrieval_hit_count": max_retrieval_hit_count,
            "min_claims_cache_hit_rate": min_claims_cache_hit_rate,
            "min_verifier_trace_cache_hit_rate": min_verifier_trace_cache_hit_rate,
            "min_covered_fact_properties": min_covered_fact_properties,
            "min_covered_fact_property_records": min_covered_fact_property_records,
            "min_covered_fact_property_source_documents": min_covered_fact_property_source_documents,
            "min_covered_fact_property_decision_accuracy": min_covered_fact_property_decision_accuracy,
            "max_covered_fact_property_false_supported_rate": max_covered_fact_property_false_supported_rate,
            "min_covered_fact_property_false_refuted_rate": min_covered_fact_property_false_refuted_rate,
            "require_non_oracle_evidence": require_non_oracle_evidence,
            "require_retrieval_provenance_filter": require_retrieval_provenance_filter,
            "required_retrieval_source_prefixes": list(required_retrieval_source_prefixes),
            "required_retrieval_metadata": dict(required_retrieval_metadata or {}),
            "min_retrieval_filter_score": min_retrieval_filter_score,
            "require_retrieval_stress_control": require_retrieval_stress_control,
            "retrieval_stress_manifest": None if retrieval_stress_manifest is None else str(retrieval_stress_manifest),
            "min_stress_false_supported_rate": _resolved_min_stress_false_supported_rate(
                require_retrieval_stress_control=require_retrieval_stress_control,
                min_stress_false_supported_rate=min_stress_false_supported_rate,
            ),
            "max_stress_false_refuted_rate": _resolved_max_stress_false_refuted_rate(
                require_retrieval_stress_control=require_retrieval_stress_control,
                max_stress_false_refuted_rate=max_stress_false_refuted_rate,
            ),
        },
        "summary": {
            "record_count": len(rows),
            "passing_count": sum(1 for row in rows if row["gate"]["passed"]),
            "recommended_record": None if recommendation is None else recommendation["record_key"],
            "recommended_route": None if recommendation is None else recommendation.get("recommended_route"),
            "artifact_json_cache": verification_context.json_cache_summary(),
        },
        "decision": decision,
        "leaderboard": leaderboard,
        "notes": list(notes),
    }


def _select_records(
    registry: ArtifactRegistry,
    *,
    baseline_keys: Sequence[str],
) -> tuple[RegistryRecord, ...]:
    if baseline_keys:
        records = tuple(registry.get(key) for key in baseline_keys)
    else:
        records = tuple(
            record
            for record in registry.list_records(artifact_type="benchmark_manifest")
            if _is_route_record(record)
        )
    for record in records:
        if record.artifact_type != "benchmark_manifest":
            raise ValueError(f"registry record {record.key()!r} is not a benchmark_manifest.")
    return records


def _is_route_record(record: RegistryRecord) -> bool:
    metadata = dict(record.metadata)
    manifest_metadata = _mapping(metadata.get("manifest_metadata"))
    return (
        metadata.get("workflow") == "run_adapter_promotion_workflow"
        or metadata.get("workflow") == "adapter_promotion_workflow"
        or metadata.get("workflow") == "wikidata_structured_qa_route_workflow"
        or manifest_metadata.get("runner") == "run_adapter_promotion_workflow"
        or manifest_metadata.get("workflow") == "wikidata_structured_qa_route_workflow"
        or manifest_metadata.get("promotes_covered_facts_route") is not None
        or metadata.get("route_promotion_status") is not None
    )


def _route_baseline_row(
    record: RegistryRecord,
    *,
    recursive: bool,
    allow_unverified: bool,
    min_selected: int | None,
    min_decision_accuracy: float | None,
    max_false_supported_rate: float | None,
    min_false_refuted_rate: float | None,
    max_verified_false_alarm: float | None,
    min_verified_detection: float | None,
    max_mean_duration_seconds: float | None,
    max_p99_duration_seconds: float | None,
    max_max_duration_seconds: float | None,
    max_mean_attempted_route_count: float | None,
    max_retrieval_use_rate: float | None,
    max_runtime_total_seconds: float | None,
    max_retrieval_hit_count: float | None,
    min_claims_cache_hit_rate: float | None,
    min_verifier_trace_cache_hit_rate: float | None,
    min_covered_fact_properties: int | None,
    min_covered_fact_property_records: int | None,
    min_covered_fact_property_source_documents: int | None,
    min_covered_fact_property_decision_accuracy: float | None,
    max_covered_fact_property_false_supported_rate: float | None,
    min_covered_fact_property_false_refuted_rate: float | None,
    require_non_oracle_evidence: bool,
    require_retrieval_provenance_filter: bool,
    required_retrieval_source_prefixes: Sequence[str],
    required_retrieval_metadata: Mapping[str, Any] | None,
    min_retrieval_filter_score: float | None,
    require_retrieval_stress_control: bool,
    retrieval_stress_manifest: str | Path | None,
    min_stress_false_supported_rate: float | None,
    max_stress_false_refuted_rate: float | None,
    fingerprint_cache: MutableMapping[str, dict[str, Any]],
    json_cache: MutableMapping[str, dict[str, Any]],
    json_cache_stats: MutableMapping[str, int],
) -> dict[str, Any]:
    manifest_path = Path(record.path)
    manifest, manifest_error = _load_optional_json(
        manifest_path,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
    )
    verification = _verify_manifest(
        manifest_path,
        manifest=manifest,
        manifest_error=manifest_error,
        recursive=recursive,
        fingerprint_cache=fingerprint_cache,
    )
    manifest_metadata = _manifest_metadata(record, manifest)
    route_report_path = _resolve_artifact_path(
        manifest_path,
        manifest,
        artifact_name="route_comparison_report",
    )
    route_summary_path = _resolve_artifact_path(
        manifest_path,
        manifest,
        artifact_name="route_summary",
    )
    route_summary: dict[str, Any] = {}
    route_summary_error = None
    if route_summary_path is not None:
        route_summary, route_summary_error = _load_optional_json(
            route_summary_path,
            json_cache=json_cache,
            json_cache_stats=json_cache_stats,
        )
    if route_report_path is None:
        route_report_error = (
            "route_comparison_report and route_summary artifacts missing"
            if route_summary_path is None
            else route_summary_error
        )
        route_comparison = (
            {}
            if route_report_error is not None
            else _route_comparison_from_summary(route_summary, manifest_metadata)
        )
    else:
        route_comparison, route_report_error = _load_optional_json(
            route_report_path,
            json_cache=json_cache,
            json_cache_stats=json_cache_stats,
        )
    claims_path = _resolve_artifact_path(
        manifest_path,
        manifest,
        artifact_name="retrieval_claims",
    )
    claims_fixture: dict[str, Any] = {}
    claims_error = None
    if require_non_oracle_evidence or _retrieval_provenance_gate_enabled(
        require_retrieval_provenance_filter=require_retrieval_provenance_filter,
        required_retrieval_source_prefixes=required_retrieval_source_prefixes,
        required_retrieval_metadata=required_retrieval_metadata,
        min_retrieval_filter_score=min_retrieval_filter_score,
    ):
        claims_fixture, claims_error = (
            ({}, "retrieval_claims artifact missing")
            if claims_path is None
            else _load_optional_json(
                claims_path,
                json_cache=json_cache,
                json_cache_stats=json_cache_stats,
            )
        )
    route_decision = _mapping(route_comparison.get("promotion_decision"))
    recommended_route = (
        route_decision.get("recommended_route")
        or manifest_metadata.get("recommended_route")
    )
    metrics = _recommended_route_metrics(
        route_comparison,
        manifest_metadata,
        recommended_route=None if recommended_route is None else str(recommended_route),
    )
    covered_fact_property_metrics = _mapping(route_summary.get("property_metrics"))
    covered_fact_property_count = _int_or_none(route_summary.get("property_count"))
    if covered_fact_property_count is None and covered_fact_property_metrics:
        covered_fact_property_count = len(covered_fact_property_metrics)
    route_status = (
        route_decision.get("status")
        or manifest_metadata.get("route_promotion_status")
        or manifest_metadata.get("promotion_status")
    )
    runtime_budget = evaluate_runtime_budget(
        runtime_metrics_from_metadata(manifest_metadata),
        RuntimeBudgetPolicy(
            max_total_seconds=max_runtime_total_seconds,
            max_retrieval_hit_count=max_retrieval_hit_count,
            min_claims_cache_hit_rate=min_claims_cache_hit_rate,
            min_verifier_trace_cache_hit_rate=min_verifier_trace_cache_hit_rate,
        ),
    )
    retrieval_stress_audit = _retrieval_stress_audit(
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_metadata=manifest_metadata,
        recursive=recursive,
        require_retrieval_stress_control=require_retrieval_stress_control,
        retrieval_stress_manifest=retrieval_stress_manifest,
        min_stress_false_supported_rate=min_stress_false_supported_rate,
        max_stress_false_refuted_rate=max_stress_false_refuted_rate,
        fingerprint_cache=fingerprint_cache,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
    )
    gate = _gate(
        verification=verification,
        allow_unverified=allow_unverified,
        manifest_error=manifest_error,
        route_report_error=route_report_error,
        route_status=route_status,
        recommended_route=recommended_route,
        metrics=metrics,
        min_selected=min_selected,
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
        max_mean_duration_seconds=max_mean_duration_seconds,
        max_p99_duration_seconds=max_p99_duration_seconds,
        max_max_duration_seconds=max_max_duration_seconds,
        max_mean_attempted_route_count=max_mean_attempted_route_count,
        max_retrieval_use_rate=max_retrieval_use_rate,
        covered_fact_property_count=covered_fact_property_count,
        covered_fact_property_metrics=covered_fact_property_metrics,
        min_covered_fact_properties=min_covered_fact_properties,
        min_covered_fact_property_records=min_covered_fact_property_records,
        min_covered_fact_property_source_documents=min_covered_fact_property_source_documents,
        min_covered_fact_property_decision_accuracy=min_covered_fact_property_decision_accuracy,
        max_covered_fact_property_false_supported_rate=max_covered_fact_property_false_supported_rate,
        min_covered_fact_property_false_refuted_rate=min_covered_fact_property_false_refuted_rate,
        runtime_budget=runtime_budget,
        evidence_audit=_evidence_audit(
            manifest_metadata,
            claims_fixture,
            claims_error=claims_error,
            require_non_oracle_evidence=require_non_oracle_evidence,
        ),
        retrieval_provenance_audit=_retrieval_provenance_audit(
            manifest_metadata,
            claims_fixture,
            claims_error=claims_error,
            require_retrieval_provenance_filter=require_retrieval_provenance_filter,
            required_retrieval_source_prefixes=required_retrieval_source_prefixes,
            required_retrieval_metadata=required_retrieval_metadata,
            min_retrieval_filter_score=min_retrieval_filter_score,
        ),
        retrieval_stress_audit=retrieval_stress_audit,
    )
    runtime_metrics = dict(runtime_budget.get("metrics") or {})
    return {
        "record_key": record.key(),
        "name": record.name,
        "version": record.version,
        "manifest_path": str(manifest_path),
        "route_report_path": None if route_report_path is None else str(route_report_path),
        "route_summary_path": None if route_summary_path is None else str(route_summary_path),
        "claims_path": None if claims_path is None else str(claims_path),
        "verification": verification,
        "gate": gate,
        "route_promotion_status": route_status,
        "recommended_route": None if recommended_route is None else str(recommended_route),
        "selected": _int_or_none(metrics.get("selected")),
        "decision_accuracy": _float_or_none(metrics.get("decision_accuracy")),
        "false_supported_rate": _float_or_none(metrics.get("false_supported_rate")),
        "false_refuted_rate": _float_or_none(metrics.get("false_refuted_rate")),
        "verified_false_alarm": _float_or_none(metrics.get("verified_false_alarm")),
        "verified_detection": _float_or_none(metrics.get("verified_detection")),
        "mean_duration_seconds": _float_or_none(metrics.get("mean_duration_seconds")),
        "p95_duration_seconds": _float_or_none(metrics.get("p95_duration_seconds")),
        "p99_duration_seconds": _float_or_none(metrics.get("p99_duration_seconds")),
        "max_duration_seconds": _float_or_none(metrics.get("max_duration_seconds")),
        "mean_attempted_route_count": _float_or_none(metrics.get("mean_attempted_route_count")),
        "retrieval_use_rate": _float_or_none(metrics.get("retrieval_use_rate")),
        "invalid_metric_counts": _mapping(metrics.get("invalid_metric_counts")),
        "runtime_total_seconds": _float_or_none(runtime_metrics.get("total_seconds")),
        "runtime_retrieval_hit_count": _float_or_none(runtime_metrics.get("retrieval_hit_count")),
        "claims_cache_hit_rate": _float_or_none(runtime_metrics.get("claims_cache_hit_rate")),
        "verifier_trace_cache_hit_rate": _float_or_none(runtime_metrics.get("verifier_trace_cache_hit_rate")),
        "covered_fact_property_count": covered_fact_property_count,
        "covered_fact_property_metrics": dict(covered_fact_property_metrics),
        "runtime_budget": runtime_budget,
        "evidence_audit": gate["evidence_audit"],
        "retrieval_provenance_audit": gate["retrieval_provenance_audit"],
        "retrieval_stress_audit": gate["retrieval_stress_audit"],
    }


def _recommended_route_metrics(
    route_comparison: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    *,
    recommended_route: str | None,
) -> dict[str, Any]:
    by_route = _mapping(route_comparison.get("by_route"))
    if recommended_route is not None:
        metrics = _mapping(by_route.get(recommended_route))
        if metrics:
            return metrics
    return {
        "selected": manifest_metadata.get("recommended_selected"),
        "decision_accuracy": manifest_metadata.get("recommended_decision_accuracy"),
        "false_supported_rate": manifest_metadata.get("recommended_false_supported_rate"),
        "false_refuted_rate": manifest_metadata.get("recommended_false_refuted_rate"),
        "verified_false_alarm": manifest_metadata.get("recommended_verified_false_alarm"),
        "verified_detection": manifest_metadata.get("recommended_verified_detection"),
        "mean_duration_seconds": manifest_metadata.get("recommended_mean_duration_seconds"),
        "p95_duration_seconds": manifest_metadata.get("recommended_p95_duration_seconds"),
        "p99_duration_seconds": manifest_metadata.get("recommended_p99_duration_seconds"),
        "max_duration_seconds": manifest_metadata.get("recommended_max_duration_seconds"),
        "mean_attempted_route_count": manifest_metadata.get("recommended_mean_attempted_route_count"),
        "retrieval_use_rate": manifest_metadata.get("recommended_retrieval_use_rate"),
        "invalid_metric_counts": manifest_metadata.get("recommended_invalid_metric_counts"),
    }


def _route_comparison_from_summary(
    route_summary: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    route = (
        route_summary.get("route")
        or manifest_metadata.get("route")
        or manifest_metadata.get("recommended_route")
    )
    route_name = None if route is None else str(route)
    metrics = dict(_mapping(route_summary.get("route_metrics")))
    if not metrics and route_name:
        metrics = dict(_mapping(route_summary.get(f"{route_name}_metrics")))
    if route_name:
        selected_counts = _mapping(route_summary.get("selected_route_counts"))
        score_dump_summary = _mapping(route_summary.get("score_dump_summary"))
        metrics.setdefault("selected", selected_counts.get(route_name))
        if metrics.get("selected") is None:
            metrics["selected"] = score_dump_summary.get("n_records")
        metrics.setdefault("decision_accuracy", manifest_metadata.get(f"{route_name}_decision_accuracy"))
        metrics.setdefault("false_supported_rate", manifest_metadata.get(f"{route_name}_false_supported_rate"))
        metrics.setdefault("false_refuted_rate", manifest_metadata.get(f"{route_name}_false_refuted_rate"))
        metrics.setdefault("verified_false_alarm", metrics.get("false_supported_rate"))
        metrics.setdefault("verified_detection", metrics.get("false_refuted_rate"))
    status = route_summary.get("status") or manifest_metadata.get("status")
    return {
        "schema_version": 1,
        "workflow": "route_baseline_comparison_from_summary",
        "source_workflow": route_summary.get("workflow") or manifest_metadata.get("workflow"),
        "promotion_decision": {
            "status": "promote" if status == "promote" else status,
            "recommended_route": route_name,
        },
        "by_route": {} if route_name is None else {route_name: metrics},
    }


def _gate(
    *,
    verification: Mapping[str, Any],
    allow_unverified: bool,
    manifest_error: str | None,
    route_report_error: str | None,
    route_status: Any,
    recommended_route: Any,
    metrics: Mapping[str, Any],
    min_selected: int | None,
    min_decision_accuracy: float | None,
    max_false_supported_rate: float | None,
    min_false_refuted_rate: float | None,
    max_verified_false_alarm: float | None,
    min_verified_detection: float | None,
    max_mean_duration_seconds: float | None,
    max_p99_duration_seconds: float | None,
    max_max_duration_seconds: float | None,
    max_mean_attempted_route_count: float | None,
    max_retrieval_use_rate: float | None,
    covered_fact_property_count: int | None,
    covered_fact_property_metrics: Mapping[str, Any],
    min_covered_fact_properties: int | None,
    min_covered_fact_property_records: int | None,
    min_covered_fact_property_source_documents: int | None,
    min_covered_fact_property_decision_accuracy: float | None,
    max_covered_fact_property_false_supported_rate: float | None,
    min_covered_fact_property_false_refuted_rate: float | None,
    runtime_budget: Mapping[str, Any],
    evidence_audit: Mapping[str, Any],
    retrieval_provenance_audit: Mapping[str, Any],
    retrieval_stress_audit: Mapping[str, Any],
) -> dict[str, Any]:
    failures = []
    if manifest_error is not None:
        failures.append(f"manifest could not be loaded: {manifest_error}")
    if route_report_error is not None:
        failures.append(f"route comparison report could not be loaded: {route_report_error}")
    if not bool(verification.get("passed", False)) and not allow_unverified:
        failures.append("manifest verification failed")
    if route_status != "promote":
        failures.append(f"route_promotion_status is {route_status!r}, expected 'promote'")
    if recommended_route is None:
        failures.append("recommended route is missing")
    invalid_metric_counts = {
        key: int(value)
        for key, value in _mapping(metrics.get("invalid_metric_counts")).items()
        if _int_or_none(value) is not None and int(value) > 0
    }
    if invalid_metric_counts:
        failures.append(f"recommended route has invalid source metrics: {invalid_metric_counts}")
    _check_min(failures, "selected", _int_or_none(metrics.get("selected")), min_selected)
    _check_min(failures, "decision_accuracy", _float_or_none(metrics.get("decision_accuracy")), min_decision_accuracy)
    _check_max(
        failures,
        "false_supported_rate",
        _float_or_none(metrics.get("false_supported_rate")),
        max_false_supported_rate,
    )
    _check_min(
        failures,
        "false_refuted_rate",
        _float_or_none(metrics.get("false_refuted_rate")),
        min_false_refuted_rate,
    )
    _check_max(
        failures,
        "verified_false_alarm",
        _float_or_none(metrics.get("verified_false_alarm")),
        max_verified_false_alarm,
    )
    _check_min(
        failures,
        "verified_detection",
        _float_or_none(metrics.get("verified_detection")),
        min_verified_detection,
    )
    _check_max(
        failures,
        "mean_duration_seconds",
        _float_or_none(metrics.get("mean_duration_seconds")),
        max_mean_duration_seconds,
    )
    _check_max(
        failures,
        "p99_duration_seconds",
        _float_or_none(metrics.get("p99_duration_seconds")),
        max_p99_duration_seconds,
    )
    _check_max(
        failures,
        "max_duration_seconds",
        _float_or_none(metrics.get("max_duration_seconds")),
        max_max_duration_seconds,
    )
    _check_max(
        failures,
        "mean_attempted_route_count",
        _float_or_none(metrics.get("mean_attempted_route_count")),
        max_mean_attempted_route_count,
    )
    _check_max(
        failures,
        "retrieval_use_rate",
        _float_or_none(metrics.get("retrieval_use_rate")),
        max_retrieval_use_rate,
    )
    if _covered_fact_property_gate_enabled(
        min_covered_fact_properties=min_covered_fact_properties,
        min_covered_fact_property_records=min_covered_fact_property_records,
        min_covered_fact_property_source_documents=min_covered_fact_property_source_documents,
        min_covered_fact_property_decision_accuracy=min_covered_fact_property_decision_accuracy,
        max_covered_fact_property_false_supported_rate=max_covered_fact_property_false_supported_rate,
        min_covered_fact_property_false_refuted_rate=min_covered_fact_property_false_refuted_rate,
    ):
        if not covered_fact_property_metrics:
            failures.append("covered-facts property metrics are missing")
        _check_min(
            failures,
            "covered_fact_properties",
            covered_fact_property_count,
            min_covered_fact_properties,
        )
        for property_id, property_metrics in sorted(covered_fact_property_metrics.items()):
            metric_payload = _mapping(property_metrics)
            prefix = f"covered_fact_property[{property_id}]"
            _check_min(
                failures,
                f"{prefix}.records",
                _int_or_none(metric_payload.get("n_records")),
                min_covered_fact_property_records,
            )
            _check_min(
                failures,
                f"{prefix}.source_documents",
                _int_or_none(metric_payload.get("n_source_documents")),
                min_covered_fact_property_source_documents,
            )
            _check_min(
                failures,
                f"{prefix}.decision_accuracy",
                _float_or_none(metric_payload.get("decision_accuracy")),
                min_covered_fact_property_decision_accuracy,
            )
            _check_max(
                failures,
                f"{prefix}.false_supported_rate",
                _float_or_none(metric_payload.get("false_supported_rate")),
                max_covered_fact_property_false_supported_rate,
            )
            _check_min(
                failures,
                f"{prefix}.false_refuted_rate",
                _float_or_none(metric_payload.get("false_refuted_rate")),
                min_covered_fact_property_false_refuted_rate,
            )
    if runtime_budget.get("enabled") and not runtime_budget.get("passed"):
        failures.extend(_runtime_budget_reasons(runtime_budget))
    if evidence_audit.get("enabled") and not evidence_audit.get("passed"):
        failures.extend(f"evidence_audit: {reason}" for reason in evidence_audit.get("blocking_reasons", ()))
    if retrieval_provenance_audit.get("enabled") and not retrieval_provenance_audit.get("passed"):
        failures.extend(
            f"retrieval_provenance_audit: {reason}"
            for reason in retrieval_provenance_audit.get("blocking_reasons", ())
        )
    if retrieval_stress_audit.get("enabled") and not retrieval_stress_audit.get("passed"):
        failures.extend(
            f"retrieval_stress_audit: {reason}"
            for reason in retrieval_stress_audit.get("blocking_reasons", ())
        )
    return {
        "passed": not failures,
        "blocking_reasons": failures,
        "evidence_audit": dict(evidence_audit),
        "retrieval_provenance_audit": dict(retrieval_provenance_audit),
        "retrieval_stress_audit": dict(retrieval_stress_audit),
    }


def _covered_fact_property_gate_enabled(
    *,
    min_covered_fact_properties: int | None,
    min_covered_fact_property_records: int | None,
    min_covered_fact_property_source_documents: int | None,
    min_covered_fact_property_decision_accuracy: float | None,
    max_covered_fact_property_false_supported_rate: float | None,
    min_covered_fact_property_false_refuted_rate: float | None,
) -> bool:
    return any(
        value is not None
        for value in (
            min_covered_fact_properties,
            min_covered_fact_property_records,
            min_covered_fact_property_source_documents,
            min_covered_fact_property_decision_accuracy,
            max_covered_fact_property_false_supported_rate,
            min_covered_fact_property_false_refuted_rate,
        )
    )


def _evidence_audit(
    manifest_metadata: Mapping[str, Any],
    claims_fixture: Mapping[str, Any],
    *,
    claims_error: str | None,
    require_non_oracle_evidence: bool,
) -> dict[str, Any]:
    label_usage = _mapping(claims_fixture.get("label_usage"))
    provenance = _mapping(claims_fixture.get("input_provenance"))
    score_dump = _mapping(provenance.get("score_dump"))
    corpora = provenance.get("corpora")
    corpus_fingerprints = (
        tuple(item for item in corpora if isinstance(item, Mapping) and item.get("sha256"))
        if isinstance(corpora, Sequence) and not isinstance(corpora, (str, bytes, bytearray))
        else ()
    )
    labels_used = _first_present(
        label_usage.get("labels_used_for_retrieval"),
        manifest_metadata.get("labels_used_for_retrieval"),
    )
    labels_copied = _first_present(
        label_usage.get("labels_copied_to_record_metadata"),
        manifest_metadata.get("labels_copied_to_record_metadata"),
    )
    failures: list[str] = []
    if require_non_oracle_evidence:
        if claims_error is not None:
            failures.append(f"retrieval claims fixture could not be loaded: {claims_error}")
        if labels_used is not False:
            failures.append("labels_used_for_retrieval must be false")
        if labels_copied is not False:
            failures.append("labels_copied_to_record_metadata must be false")
        if not provenance:
            failures.append("input_provenance is missing")
        if not score_dump:
            failures.append("input_provenance.score_dump is missing")
        if not corpus_fingerprints:
            failures.append("input_provenance.corpora must include at least one corpus fingerprint")
    return {
        "enabled": bool(require_non_oracle_evidence),
        "passed": not failures,
        "blocking_reasons": failures,
        "claims_loaded": claims_error is None and bool(claims_fixture),
        "claims_error": claims_error,
        "labels_used_for_retrieval": labels_used,
        "labels_copied_to_record_metadata": labels_copied,
        "input_provenance_present": bool(provenance),
        "score_dump_provenance_present": bool(score_dump),
        "corpus_fingerprint_count": len(corpus_fingerprints),
    }


def _retrieval_provenance_gate_enabled(
    *,
    require_retrieval_provenance_filter: bool,
    required_retrieval_source_prefixes: Sequence[str],
    required_retrieval_metadata: Mapping[str, Any] | None,
    min_retrieval_filter_score: float | None,
) -> bool:
    return bool(
        require_retrieval_provenance_filter
        or tuple(required_retrieval_source_prefixes)
        or dict(required_retrieval_metadata or {})
        or min_retrieval_filter_score is not None
    )


def _retrieval_provenance_audit(
    manifest_metadata: Mapping[str, Any],
    claims_fixture: Mapping[str, Any],
    *,
    claims_error: str | None,
    require_retrieval_provenance_filter: bool,
    required_retrieval_source_prefixes: Sequence[str],
    required_retrieval_metadata: Mapping[str, Any] | None,
    min_retrieval_filter_score: float | None,
) -> dict[str, Any]:
    required_prefixes = _clean_string_tuple(required_retrieval_source_prefixes)
    required_metadata = {
        str(key): value
        for key, value in dict(required_retrieval_metadata or {}).items()
        if str(key)
    }
    enabled = _retrieval_provenance_gate_enabled(
        require_retrieval_provenance_filter=require_retrieval_provenance_filter,
        required_retrieval_source_prefixes=required_prefixes,
        required_retrieval_metadata=required_metadata,
        min_retrieval_filter_score=min_retrieval_filter_score,
    )
    config, source = _resolve_retrieval_provenance_filter(
        manifest_metadata,
        claims_fixture,
    )
    failures: list[str] = []
    normalized = _normalize_retrieval_provenance_filter(config)
    if enabled:
        if claims_error is not None and not normalized:
            failures.append(f"retrieval claims fixture could not be loaded: {claims_error}")
        if not normalized:
            failures.append("retrieval provenance filter config is missing")
        if require_retrieval_provenance_filter and normalized.get("require_source") is not True:
            failures.append("retrieval provenance filter require_source must be true")
        allowed_prefixes = tuple(normalized.get("allowed_source_prefixes") or ())
        for prefix in required_prefixes:
            if prefix not in allowed_prefixes:
                failures.append(
                    f"retrieval provenance filter allowed_source_prefixes missing required prefix {prefix!r}"
                )
        configured_metadata = _mapping(normalized.get("required_metadata"))
        for key, expected in required_metadata.items():
            actual = configured_metadata.get(key)
            if actual != expected:
                failures.append(
                    f"retrieval provenance filter required_metadata.{key} is {actual!r}, expected {expected!r}"
                )
        if min_retrieval_filter_score is not None:
            configured_score = _float_or_none(normalized.get("min_score"))
            if configured_score is None or configured_score < min_retrieval_filter_score:
                failures.append(
                    f"retrieval provenance filter min_score below {min_retrieval_filter_score}"
                )
    return {
        "enabled": enabled,
        "passed": not failures,
        "blocking_reasons": failures,
        "claims_loaded": claims_error is None and bool(claims_fixture),
        "claims_error": claims_error,
        "source": source,
        "filter": normalized,
        "require_retrieval_provenance_filter": require_retrieval_provenance_filter,
        "required_retrieval_source_prefixes": required_prefixes,
        "required_retrieval_metadata": required_metadata,
        "min_retrieval_filter_score": min_retrieval_filter_score,
    }


def _resolve_retrieval_provenance_filter(
    manifest_metadata: Mapping[str, Any],
    claims_fixture: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    manifest_filter = _mapping(manifest_metadata.get("retrieval_provenance_filter"))
    if manifest_filter:
        return manifest_filter, "manifest_metadata"
    provenance = _mapping(claims_fixture.get("input_provenance"))
    provenance_config = _mapping(provenance.get("config"))
    input_filter = _mapping(provenance_config.get("provenance_filter"))
    if input_filter:
        return input_filter, "claims_input_provenance"
    retriever = _mapping(claims_fixture.get("retriever"))
    retriever_filter = _mapping(retriever.get("provenance_filter"))
    if retriever_filter:
        return retriever_filter, "claims_retriever"
    return {}, None


def _normalize_retrieval_provenance_filter(config: Mapping[str, Any]) -> dict[str, Any]:
    if not config:
        return {}
    normalized = dict(config)
    normalized["require_source"] = bool(config.get("require_source"))
    normalized["allowed_source_prefixes"] = _clean_string_tuple(config.get("allowed_source_prefixes", ()))
    normalized["denied_source_prefixes"] = _clean_string_tuple(config.get("denied_source_prefixes", ()))
    normalized["required_metadata"] = _mapping(config.get("required_metadata"))
    normalized["min_score"] = _float_or_none(config.get("min_score"))
    normalized["max_hits_per_source"] = _int_or_none(config.get("max_hits_per_source"))
    return normalized


def _clean_string_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence):
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


def _retrieval_stress_audit(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    recursive: bool,
    require_retrieval_stress_control: bool,
    retrieval_stress_manifest: str | Path | None,
    min_stress_false_supported_rate: float | None,
    max_stress_false_refuted_rate: float | None,
    fingerprint_cache: MutableMapping[str, dict[str, Any]],
    json_cache: MutableMapping[str, dict[str, Any]],
    json_cache_stats: MutableMapping[str, int],
) -> dict[str, Any]:
    min_false_supported = _resolved_min_stress_false_supported_rate(
        require_retrieval_stress_control=require_retrieval_stress_control,
        min_stress_false_supported_rate=min_stress_false_supported_rate,
    )
    max_false_refuted = _resolved_max_stress_false_refuted_rate(
        require_retrieval_stress_control=require_retrieval_stress_control,
        max_stress_false_refuted_rate=max_stress_false_refuted_rate,
    )
    enabled = bool(
        require_retrieval_stress_control
        or retrieval_stress_manifest is not None
        or min_false_supported is not None
        or max_false_refuted is not None
    )
    stress_manifest_path = _resolve_retrieval_stress_manifest_path(
        manifest_path,
        manifest,
        manifest_metadata,
        override=retrieval_stress_manifest,
    )
    audit: dict[str, Any] = {
        "enabled": enabled,
        "passed": True,
        "blocking_reasons": [],
        "stress_manifest_path": None if stress_manifest_path is None else str(stress_manifest_path),
        "min_stress_false_supported_rate": min_false_supported,
        "max_stress_false_refuted_rate": max_false_refuted,
        "verification_passed": None,
        "corpus_path": None,
        "corpus_type": None,
        "verifier_report_path": None,
        "run_count": 0,
        "min_false_supported_rate": None,
        "max_false_refuted_rate": None,
        "runs": [],
    }
    failures: list[str] = []
    if not enabled:
        return audit
    if stress_manifest_path is None:
        failures.append("retrieval stress manifest is required")
        audit["passed"] = False
        audit["blocking_reasons"] = failures
        return audit
    stress_manifest, manifest_error = _load_optional_json(
        stress_manifest_path,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
    )
    verification = _verify_manifest(
        stress_manifest_path,
        manifest=stress_manifest,
        manifest_error=manifest_error,
        recursive=recursive,
        fingerprint_cache=fingerprint_cache,
    )
    audit["verification_passed"] = bool(verification.get("passed", False))
    audit["verification"] = verification
    if manifest_error is not None:
        failures.append(f"retrieval stress manifest could not be loaded: {manifest_error}")
    elif not bool(verification.get("passed", False)):
        failures.append("retrieval stress manifest verification failed")
    corpus_path = _resolve_prefixed_artifact_path(
        stress_manifest_path,
        stress_manifest,
        artifact_prefix="retrieval_corpora.",
    )
    verifier_report_path = _resolve_artifact_path(
        stress_manifest_path,
        stress_manifest,
        artifact_name="verifier_report",
    )
    audit["corpus_path"] = None if corpus_path is None else str(corpus_path)
    audit["verifier_report_path"] = None if verifier_report_path is None else str(verifier_report_path)
    corpus, corpus_error = (
        ({}, "retrieval stress corpus artifact missing")
        if corpus_path is None
        else _load_optional_json(corpus_path, json_cache=json_cache, json_cache_stats=json_cache_stats)
    )
    verifier_report, verifier_report_error = (
        ({}, "retrieval stress verifier_report artifact missing")
        if verifier_report_path is None
        else _load_optional_json(verifier_report_path, json_cache=json_cache, json_cache_stats=json_cache_stats)
    )
    if corpus_error is not None:
        failures.append(f"retrieval stress corpus could not be loaded: {corpus_error}")
    if verifier_report_error is not None:
        failures.append(f"retrieval stress verifier report could not be loaded: {verifier_report_error}")
    corpus_type = corpus.get("corpus_type")
    label_usage = _mapping(corpus.get("label_usage"))
    audit["corpus_type"] = corpus_type
    audit["label_usage"] = label_usage
    if corpus_type != ANSWER_ECHO_CORPUS_TYPE:
        failures.append(f"retrieval stress corpus_type is {corpus_type!r}, expected {ANSWER_ECHO_CORPUS_TYPE!r}")
    if label_usage.get("labels_used_for_documents") is not False:
        failures.append("retrieval stress labels_used_for_documents must be false")
    if label_usage.get("labels_copied_to_document_metadata") is not False:
        failures.append("retrieval stress labels_copied_to_document_metadata must be false")
    stress_runs = _stress_quality_runs(verifier_report)
    audit["runs"] = stress_runs
    audit["run_count"] = len(stress_runs)
    if not stress_runs:
        failures.append("retrieval stress verifier report has no finite false-supported/refuted metrics")
    else:
        observed_false_supported = tuple(run["false_supported_rate"] for run in stress_runs)
        observed_false_refuted = tuple(run["false_refuted_rate"] for run in stress_runs)
        audit["min_false_supported_rate"] = min(observed_false_supported)
        audit["max_false_supported_rate"] = max(observed_false_supported)
        audit["min_false_refuted_rate"] = min(observed_false_refuted)
        audit["max_false_refuted_rate"] = max(observed_false_refuted)
        if min_false_supported is not None and audit["min_false_supported_rate"] < min_false_supported:
            failures.append(f"stress false_supported_rate below {min_false_supported}")
        if max_false_refuted is not None and audit["max_false_refuted_rate"] > max_false_refuted:
            failures.append(f"stress false_refuted_rate above {max_false_refuted}")
    audit["passed"] = not failures
    audit["blocking_reasons"] = failures
    return audit


def _resolved_min_stress_false_supported_rate(
    *,
    require_retrieval_stress_control: bool,
    min_stress_false_supported_rate: float | None,
) -> float | None:
    if min_stress_false_supported_rate is not None:
        return min_stress_false_supported_rate
    if require_retrieval_stress_control:
        return DEFAULT_MIN_STRESS_FALSE_SUPPORTED_RATE
    return None


def _resolved_max_stress_false_refuted_rate(
    *,
    require_retrieval_stress_control: bool,
    max_stress_false_refuted_rate: float | None,
) -> float | None:
    if max_stress_false_refuted_rate is not None:
        return max_stress_false_refuted_rate
    if require_retrieval_stress_control:
        return DEFAULT_MAX_STRESS_FALSE_REFUTED_RATE
    return None


def _resolve_retrieval_stress_manifest_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    *,
    override: str | Path | None,
) -> Path | None:
    if override is not None:
        return Path(override)
    artifact_path = _resolve_artifact_path(
        manifest_path,
        manifest,
        artifact_name="retrieval_stress_manifest",
    )
    if artifact_path is not None:
        return artifact_path
    raw_path = manifest_metadata.get("retrieval_stress_manifest_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _resolve_prefixed_artifact_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    artifact_prefix: str,
) -> Path | None:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return None
    for name in sorted(str(key) for key in artifacts):
        if not name.startswith(artifact_prefix):
            continue
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            continue
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path)
        return path if path.is_absolute() else manifest_path.parent / path
    return None


def _stress_quality_runs(verifier_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, run in enumerate(verifier_report.get("runs", ())):
        if not isinstance(run, Mapping):
            continue
        quality = _mapping(run.get("verification_quality"))
        false_supported_rate = _float_or_none(quality.get("false_supported_rate"))
        false_refuted_rate = _float_or_none(quality.get("false_refuted_rate"))
        if false_supported_rate is None or false_refuted_rate is None:
            continue
        rows.append({
            "name": str(run.get("name", f"run_{idx}")),
            "false_supported_rate": false_supported_rate,
            "false_refuted_rate": false_refuted_rate,
            "decision_accuracy": _float_or_none(quality.get("decision_accuracy")),
            "true_supported_rate": _float_or_none(quality.get("true_supported_rate")),
        })
    return rows


def _runtime_budget_reasons(runtime_budget: Mapping[str, Any]) -> list[str]:
    reasons = []
    for failure in runtime_budget.get("failures", ()):
        if not isinstance(failure, Mapping):
            continue
        metric = failure.get("metric")
        reason = failure.get("reason") or "failed"
        reasons.append(f"runtime_budget: {metric} {reason}")
    return reasons


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _check_min(failures: list[str], metric: str, value: float | int | None, limit: float | int | None) -> None:
    if limit is None:
        return
    if value is None or value < limit:
        failures.append(f"{metric} below {limit}")


def _check_max(failures: list[str], metric: str, value: float | int | None, limit: float | int | None) -> None:
    if limit is None:
        return
    if value is None or value > limit:
        failures.append(f"{metric} above {limit}")


def _decision(
    leaderboard: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not leaderboard:
        return {
            "status": "no_candidate",
            "recommended_record": None,
            "blocking_reasons": ("no route benchmark_manifest records selected",),
        }
    if recommendation is not None:
        return {
            "status": "promote",
            "recommended_record": recommendation["record_key"],
            "recommended_route": recommendation.get("recommended_route"),
            "blocking_reasons": (),
        }
    return {
        "status": "blocked",
        "recommended_record": None,
        "recommended_route": None,
        "blocking_reasons": tuple(
            f"{row['record_key']}: {reason}"
            for row in leaderboard
            for reason in row["gate"]["blocking_reasons"]
        ),
    }


def _leaderboard_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        not _mapping(row.get("gate")).get("passed", False),
        -(_float_or_none(row.get("decision_accuracy")) or -math.inf),
        _float_or_none(row.get("false_supported_rate")) if row.get("false_supported_rate") is not None else math.inf,
        -(_float_or_none(row.get("false_refuted_rate")) or -math.inf),
        _float_or_none(row.get("p99_duration_seconds")) if row.get("p99_duration_seconds") is not None else math.inf,
        _float_or_none(row.get("mean_duration_seconds")) if row.get("mean_duration_seconds") is not None else math.inf,
        _float_or_none(row.get("retrieval_use_rate")) if row.get("retrieval_use_rate") is not None else math.inf,
        _float_or_none(row.get("runtime_total_seconds")) if row.get("runtime_total_seconds") is not None else math.inf,
        -(_int_or_none(row.get("selected")) or -1),
        str(row.get("record_key")),
    )


def _manifest_metadata(record: RegistryRecord, manifest: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(record.metadata)
    manifest_metadata = _mapping(manifest.get("metadata"))
    merged = dict(manifest_metadata)
    merged.update(metadata)
    nested = _mapping(metadata.get("manifest_metadata"))
    merged.update(nested)
    return merged


def _verify_manifest(
    manifest_path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_error: str | None,
    recursive: bool,
    fingerprint_cache: MutableMapping[str, dict[str, Any]],
) -> dict[str, Any]:
    if manifest_error is not None:
        return _manifest_load_failure(manifest_path, manifest_error)
    try:
        return verify_artifact_manifest(
            manifest,
            manifest_path=manifest_path,
            recursive=recursive,
            fingerprint_cache=fingerprint_cache,
        ).to_dict()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _manifest_load_failure(manifest_path, str(exc))


def _manifest_load_failure(manifest_path: Path, error: str) -> dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "passed": False,
        "checked": 0,
        "failures": [
            {
                "name": "manifest",
                "path": str(manifest_path),
                "field": "load",
                "expected": "readable artifact manifest",
                "actual": error,
            }
        ],
        "nested": [],
    }


def _resolve_artifact_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    artifact_name: str,
) -> Path | None:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return None
    artifact = artifacts.get(artifact_name)
    if not isinstance(artifact, Mapping):
        return None
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _parse_non_negative_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{flag} must be a non-negative finite number.")
    return numeric


def _parse_non_negative_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{flag} must be a non-negative integer.")
    return numeric


def _parse_csv(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tuple(parsed)


def _parse_key_values(values: Sequence[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not values:
        return parsed
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"metadata requirement {item!r} must use key=value format.")
            key, raw = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("metadata requirement key must be non-empty.")
            parsed[key] = raw.strip()
    return parsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = compare_route_baselines(
        registry_path=args.registry,
        baseline_keys=tuple(args.baseline_key or ()),
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        min_selected=args.min_selected,
        min_decision_accuracy=args.min_decision_accuracy,
        max_false_supported_rate=args.max_false_supported_rate,
        min_false_refuted_rate=args.min_false_refuted_rate,
        max_verified_false_alarm=args.max_verified_false_alarm,
        min_verified_detection=args.min_verified_detection,
        max_mean_duration_seconds=args.max_mean_duration_seconds,
        max_p99_duration_seconds=args.max_p99_duration_seconds,
        max_max_duration_seconds=args.max_max_duration_seconds,
        max_mean_attempted_route_count=args.max_mean_attempted_route_count,
        max_retrieval_use_rate=args.max_retrieval_use_rate,
        max_runtime_total_seconds=args.max_runtime_total_seconds,
        max_retrieval_hit_count=args.max_retrieval_hit_count,
        min_claims_cache_hit_rate=args.min_claims_cache_hit_rate,
        min_verifier_trace_cache_hit_rate=args.min_verifier_trace_cache_hit_rate,
        min_covered_fact_properties=args.min_covered_fact_properties,
        min_covered_fact_property_records=args.min_covered_fact_property_records,
        min_covered_fact_property_source_documents=args.min_covered_fact_property_source_documents,
        min_covered_fact_property_decision_accuracy=args.min_covered_fact_property_decision_accuracy,
        max_covered_fact_property_false_supported_rate=args.max_covered_fact_property_false_supported_rate,
        min_covered_fact_property_false_refuted_rate=args.min_covered_fact_property_false_refuted_rate,
        require_non_oracle_evidence=bool(args.require_non_oracle_evidence),
        require_retrieval_provenance_filter=bool(args.require_retrieval_provenance_filter),
        required_retrieval_source_prefixes=_parse_csv(args.required_retrieval_source_prefix),
        required_retrieval_metadata=_parse_key_values(args.required_retrieval_metadata),
        min_retrieval_filter_score=args.min_retrieval_filter_score,
        require_retrieval_stress_control=bool(args.require_retrieval_stress_control),
        retrieval_stress_manifest=args.retrieval_stress_manifest,
        min_stress_false_supported_rate=args.min_stress_false_supported_rate,
        max_stress_false_refuted_rate=args.max_stress_false_refuted_rate,
        notes=args.note,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote route baseline comparison to {output_path}")
    decision = payload["decision"]
    print(
        "route_baseline_comparison="
        f"{decision['status']} recommended={decision.get('recommended_record')} "
        f"route={decision.get('recommended_route')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare registered verifier-route promotion benchmark manifests"
    )
    parser.add_argument("--registry", required=True, help="local ArtifactRegistry JSON path")
    parser.add_argument("--baseline-key", action="append", default=[],
                        help="benchmark_manifest registry key to compare; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the comparison report; repeatable")
    parser.add_argument("--no-recursive", action="store_true",
                        help="only verify root manifests")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="allow unverified manifests to become candidates")
    parser.add_argument("--min-selected", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-selected",
    ), default=None)
    parser.add_argument("--min-decision-accuracy", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-decision-accuracy",
    ), default=None)
    parser.add_argument("--max-false-supported-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-false-supported-rate",
    ), default=None)
    parser.add_argument("--min-false-refuted-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-false-refuted-rate",
    ), default=None)
    parser.add_argument("--max-verified-false-alarm", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-verified-false-alarm",
    ), default=None)
    parser.add_argument("--min-verified-detection", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-verified-detection",
    ), default=None)
    parser.add_argument("--max-mean-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-mean-duration-seconds",
    ), default=None)
    parser.add_argument("--max-p99-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-p99-duration-seconds",
    ), default=None)
    parser.add_argument("--max-max-duration-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-max-duration-seconds",
    ), default=None)
    parser.add_argument("--max-mean-attempted-route-count", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-mean-attempted-route-count",
    ), default=None)
    parser.add_argument("--max-retrieval-use-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-retrieval-use-rate",
    ), default=None)
    parser.add_argument("--max-runtime-total-seconds", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-runtime-total-seconds",
    ), default=None)
    parser.add_argument("--max-retrieval-hit-count", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-retrieval-hit-count",
    ), default=None)
    parser.add_argument("--min-claims-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-claims-cache-hit-rate",
    ), default=None)
    parser.add_argument("--min-verifier-trace-cache-hit-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-verifier-trace-cache-hit-rate",
    ), default=None)
    parser.add_argument("--min-covered-fact-properties", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-covered-fact-properties",
    ), default=None)
    parser.add_argument("--min-covered-fact-property-records", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-covered-fact-property-records",
    ), default=None)
    parser.add_argument("--min-covered-fact-property-source-documents", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-covered-fact-property-source-documents",
    ), default=None)
    parser.add_argument("--min-covered-fact-property-decision-accuracy", type=lambda value: (
        _parse_non_negative_float(
            value,
            flag="--min-covered-fact-property-decision-accuracy",
        )
    ), default=None)
    parser.add_argument("--max-covered-fact-property-false-supported-rate", type=lambda value: (
        _parse_non_negative_float(
            value,
            flag="--max-covered-fact-property-false-supported-rate",
        )
    ), default=None)
    parser.add_argument("--min-covered-fact-property-false-refuted-rate", type=lambda value: (
        _parse_non_negative_float(
            value,
            flag="--min-covered-fact-property-false-refuted-rate",
        )
    ), default=None)
    parser.add_argument(
        "--require-non-oracle-evidence",
        action="store_true",
        help="require local retrieval claims to omit labels and include input provenance",
    )
    parser.add_argument(
        "--require-retrieval-provenance-filter",
        action="store_true",
        help="require route manifests or retrieval claims to record a source-requiring provenance filter",
    )
    parser.add_argument(
        "--required-retrieval-source-prefix",
        action="append",
        default=None,
        help="source prefix that must appear in the retrieval provenance filter allow-list; "
             "comma-separated or repeatable",
    )
    parser.add_argument(
        "--required-retrieval-metadata",
        action="append",
        default=None,
        help="required provenance-filter metadata key=value; comma-separated or repeatable",
    )
    parser.add_argument(
        "--min-retrieval-filter-score",
        type=lambda value: _parse_non_negative_float(
            value,
            flag="--min-retrieval-filter-score",
        ),
        default=None,
        help="minimum min_score required in the retrieval provenance filter",
    )
    parser.add_argument(
        "--require-retrieval-stress-control",
        action="store_true",
        help="require an answer-echo retrieval stress manifest proving self-support failure",
    )
    parser.add_argument(
        "--retrieval-stress-manifest",
        default=None,
        help="optional answer-echo retrieval stress artifact manifest; overrides per-record metadata",
    )
    parser.add_argument(
        "--min-stress-false-supported-rate",
        type=lambda value: _parse_non_negative_float(
            value,
            flag="--min-stress-false-supported-rate",
        ),
        default=None,
        help="minimum false-supported rate required from the answer-echo stress control",
    )
    parser.add_argument(
        "--max-stress-false-refuted-rate",
        type=lambda value: _parse_non_negative_float(
            value,
            flag="--max-stress-false-refuted-rate",
        ),
        default=None,
        help="maximum false-refuted rate allowed from the answer-echo stress control",
    )
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless a route baseline promotes")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
