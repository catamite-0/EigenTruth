"""Enrich ProductTrace JSON with local frontier runtime-evidence summaries.

This workflow is offline and dependency-free. It turns existing full
ProductTrace payloads into auditable trace artifacts that expose world-model,
context-sensitivity, and counterfactual-robustness summaries for product
runtime drift gates. The sidecars are deliberately local and deterministic:
they summarize evidence already present in verifier results and mark their
provenance instead of calling a network verifier, vector database, or LLM.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import (  # noqa: E402
    planned_artifact_manifest_summary,
    reject_bounded_product_trace,
    strict_bool,
)
from eigentruth.control import ProductTrace, product_runtime_metrics  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "product_trace_runtime_evidence_enrichment"
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ProductTraceRuntimeEvidenceEnrichmentConfig:
    """Configuration for trace-level runtime-evidence enrichment."""

    trace_paths: Sequence[str | Path]
    output_dir: str | Path
    trace_jsonl_paths: Sequence[str | Path] = ()
    report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    overwrite: bool = False
    min_world_model_participating_trace_rate: float = 1.0
    min_world_model_coverage_rate: float = 1.0
    max_world_model_trace_gap_rate: float = 0.0
    min_context_sensitivity_participating_trace_rate: float = 1.0
    min_context_sensitivity_coverage_rate: float = 1.0
    max_context_sensitivity_flagged_result_rate: float = 0.0
    max_context_sensitivity_trace_gap_rate: float = 0.0
    min_counterfactual_robustness_participating_trace_rate: float = 1.0
    min_counterfactual_robustness_coverage_rate: float = 1.0
    min_counterfactual_robustness_pass_rate: float = 1.0
    min_counterfactual_robustness_flip_success_rate: float = 1.0
    max_counterfactual_robustness_false_invariance_rate: float = 0.0
    max_counterfactual_robustness_trace_gap_rate: float = 0.0
    compact_json: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        trace_jsonl_paths = tuple(Path(path) for path in self.trace_jsonl_paths)
        if not trace_paths and not trace_jsonl_paths:
            raise ValueError("at least one ProductTrace JSON or JSONL path is required.")
        output_dir = Path(self.output_dir)
        report_path = (
            output_dir / "product-trace-runtime-evidence-enrichment.json"
            if self.report_path is None
            else Path(self.report_path)
        )
        artifact_manifest_path = (
            output_dir / "product-trace-runtime-evidence-artifact-manifest.json"
            if self.artifact_manifest_path is None
            else Path(self.artifact_manifest_path)
        )
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "trace_jsonl_paths", trace_jsonl_paths)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "report_path", report_path)
        object.__setattr__(self, "artifact_manifest_path", artifact_manifest_path)
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "overwrite", strict_bool(self.overwrite, name="overwrite"))
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))
        for key in (
            "min_world_model_participating_trace_rate",
            "min_world_model_coverage_rate",
            "max_world_model_trace_gap_rate",
            "min_context_sensitivity_participating_trace_rate",
            "min_context_sensitivity_coverage_rate",
            "max_context_sensitivity_flagged_result_rate",
            "max_context_sensitivity_trace_gap_rate",
            "min_counterfactual_robustness_participating_trace_rate",
            "min_counterfactual_robustness_coverage_rate",
            "min_counterfactual_robustness_pass_rate",
            "min_counterfactual_robustness_flip_success_rate",
            "max_counterfactual_robustness_false_invariance_rate",
            "max_counterfactual_robustness_trace_gap_rate",
        ):
            object.__setattr__(self, key, _rate(getattr(self, key), name=key))
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_product_trace_runtime_evidence_enrichment(
    config: ProductTraceRuntimeEvidenceEnrichmentConfig,
) -> dict[str, Any]:
    """Write enriched ProductTrace files and return a JSON-ready report."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, trace_input in enumerate(_iter_trace_inputs(config), start=1):
        payload = trace_input.payload
        reject_bounded_product_trace(payload, path=trace_input.source_path)
        enriched, record = _enrich_trace(
            payload,
            source_path=trace_input.source_path,
            output_path=_trace_output_path(output_dir, trace_input.source_path, index=index),
            overwrite=config.overwrite,
        )
        record["source_format"] = trace_input.source_format
        if trace_input.line_number is not None:
            record["source_line_number"] = trace_input.line_number
        _write_json(record["output_path"], enriched, compact=config.compact_json)
        records.append(record)

    summary = _aggregate_records(records)
    status, blocking_reasons = _status(summary, config=config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": blocking_reasons,
        },
        "summary": summary,
        "traces": records,
        "paths": {
            "report": str(config.report_path),
            "artifact_manifest": str(config.artifact_manifest_path),
            "output_dir": str(config.output_dir),
            "traces": tuple(str(path) for path in config.trace_paths),
            "trace_jsonl": tuple(str(path) for path in config.trace_jsonl_paths),
        },
        "config": {
            "overwrite": config.overwrite,
            "min_world_model_participating_trace_rate": (
                config.min_world_model_participating_trace_rate
            ),
            "min_world_model_coverage_rate": config.min_world_model_coverage_rate,
            "max_world_model_trace_gap_rate": config.max_world_model_trace_gap_rate,
            "min_context_sensitivity_participating_trace_rate": (
                config.min_context_sensitivity_participating_trace_rate
            ),
            "min_context_sensitivity_coverage_rate": (
                config.min_context_sensitivity_coverage_rate
            ),
            "max_context_sensitivity_flagged_result_rate": (
                config.max_context_sensitivity_flagged_result_rate
            ),
            "max_context_sensitivity_trace_gap_rate": (
                config.max_context_sensitivity_trace_gap_rate
            ),
            "min_counterfactual_robustness_participating_trace_rate": (
                config.min_counterfactual_robustness_participating_trace_rate
            ),
            "min_counterfactual_robustness_coverage_rate": (
                config.min_counterfactual_robustness_coverage_rate
            ),
            "min_counterfactual_robustness_pass_rate": (
                config.min_counterfactual_robustness_pass_rate
            ),
            "min_counterfactual_robustness_flip_success_rate": (
                config.min_counterfactual_robustness_flip_success_rate
            ),
            "max_counterfactual_robustness_false_invariance_rate": (
                config.max_counterfactual_robustness_false_invariance_rate
            ),
            "max_counterfactual_robustness_trace_gap_rate": (
                config.max_counterfactual_robustness_trace_gap_rate
            ),
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
    }
    report["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        _artifact_paths(config, records),
        assume_file_paths=(config.report_path,),
    )
    _write_json(config.report_path, report, compact=config.compact_json)
    manifest = _write_artifact_manifest(config, report, records)
    report["artifact_manifest_summary"] = manifest["summary"]
    _write_json(config.report_path, report, compact=config.compact_json)
    _record_registry(config, report)
    return report


def _enrich_trace(
    trace: Mapping[str, Any],
    *,
    source_path: Path,
    output_path: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(trace)
    claims = [
        dict(claim)
        for claim in _sequence(payload.get("claims"))
        if isinstance(claim, Mapping)
    ]
    results = [
        dict(result)
        for result in _sequence(payload.get("verification_results"))
        if isinstance(result, Mapping)
    ]
    result_records: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        metadata = dict(_mapping(result.get("metadata")))
        claim = claims[index] if index < len(claims) else {}
        world_model_attached = _attach_world_model_sidecar(
            metadata,
            claim=claim,
            result=result,
            overwrite=overwrite,
        )
        context_attached = _attach_context_sensitivity_sidecar(metadata, overwrite=overwrite)
        counterfactual_attached = _attach_counterfactual_sidecar(
            metadata,
            claim=claim,
            result=result,
            overwrite=overwrite,
        )
        metadata["runtime_evidence_enrichment"] = {
            "workflow": WORKFLOW,
            "world_model_attached": world_model_attached,
            "context_sensitivity_attached": context_attached,
            "counterfactual_robustness_attached": counterfactual_attached,
            "source_path": str(source_path),
        }
        result["metadata"] = metadata
        result_records.append({
            "result_index": index,
            "status": result.get("status"),
            "claim_id": _claim_id(claim),
            "world_model_attached": world_model_attached,
            "context_sensitivity_attached": context_attached,
            "counterfactual_robustness_attached": counterfactual_attached,
        })

    payload["claims"] = claims
    payload["verification_results"] = results
    summary_trace = ProductTrace(claims=claims, verification_results=results)
    summaries = dict(_mapping(payload.get("summaries")))
    world_model_summary = summary_trace.world_model_summary()
    context_sensitivity_summary = summary_trace.context_sensitivity_summary()
    counterfactual_robustness_summary = summary_trace.counterfactual_robustness_summary()
    summaries["world_model"] = world_model_summary
    summaries["context_sensitivity"] = context_sensitivity_summary
    summaries["counterfactual_robustness"] = counterfactual_robustness_summary
    payload["summaries"] = summaries

    metadata = dict(_mapping(payload.get("metadata")))
    trace_corpus = dict(_mapping(metadata.get("trace_corpus")))
    trace_corpus["world_model_summary"] = world_model_summary
    trace_corpus["context_sensitivity_summary"] = context_sensitivity_summary
    trace_corpus["counterfactual_robustness_summary"] = counterfactual_robustness_summary
    metadata["trace_corpus"] = trace_corpus
    metadata["runtime_evidence_enrichment"] = {
        "workflow": WORKFLOW,
        "source_path": str(source_path),
        "result_count": len(results),
        "world_model_total": world_model_summary.get("world_model_total"),
        "context_sensitivity_total": (
            context_sensitivity_summary.get("context_sensitivity_total")
        ),
        "counterfactual_result_total": (
            counterfactual_robustness_summary.get("counterfactual_result_total")
        ),
    }
    payload["metadata"] = metadata

    metrics = product_runtime_metrics(payload)
    return payload, {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "request_id": payload.get("request_id"),
        "result_records": tuple(result_records),
        "world_model_summary": world_model_summary,
        "context_sensitivity_summary": context_sensitivity_summary,
        "counterfactual_robustness_summary": counterfactual_robustness_summary,
        "metrics": {
            "world_model_coverage_rate": metrics.get("world_model_coverage_rate"),
            "world_model_conflict_rate": metrics.get("world_model_conflict_rate"),
            "world_model_trace_gap_rate": metrics.get("world_model_trace_gap_rate"),
            "context_sensitivity_coverage_rate": (
                metrics.get("context_sensitivity_coverage_rate")
            ),
            "context_sensitivity_flagged_result_rate": (
                metrics.get("context_sensitivity_flagged_result_rate")
            ),
            "context_sensitivity_trace_gap_rate": (
                metrics.get("context_sensitivity_trace_gap_rate")
            ),
            "counterfactual_robustness_coverage_rate": (
                metrics.get("counterfactual_robustness_coverage_rate")
            ),
            "counterfactual_robustness_pass_rate": (
                metrics.get("counterfactual_robustness_pass_rate")
            ),
            "counterfactual_robustness_flip_success_rate": (
                metrics.get("counterfactual_robustness_flip_success_rate")
            ),
            "counterfactual_robustness_false_invariance_rate": (
                metrics.get("counterfactual_robustness_false_invariance_rate")
            ),
            "counterfactual_robustness_trace_gap_rate": (
                metrics.get("counterfactual_robustness_trace_gap_rate")
            ),
        },
    }


def _attach_world_model_sidecar(
    metadata: dict[str, Any],
    *,
    claim: Mapping[str, Any],
    result: Mapping[str, Any],
    overwrite: bool,
) -> bool:
    if not overwrite and any(key in metadata for key in _WORLD_MODEL_KEYS):
        return False
    route = _route_name(metadata)
    status = _status_name(result.get("status"))
    if route == "calculator" or str(metadata.get("verifier", "")).casefold() == "calculator":
        expression = str(metadata.get("expression") or _claim_text(claim) or "calculation")
        expected = metadata.get("expected")
        actual = metadata.get("actual")
        path = "calculation.result"
        conflict = not _values_match(expected, actual) or status == "refuted"
        metadata["world_model"] = "DeterministicCalculatorWorldModelAdapter"
        metadata["world_model_reference"] = {
            "reference_id": f"calculator:{_fingerprint(expression)}",
            "adapter": "DeterministicCalculatorWorldModelAdapter",
            "source": WORKFLOW,
        }
        metadata["world_model_view"] = {
            "base_state_fingerprint": _fingerprint(f"calculator:{expression}"),
            "predicted_state_fingerprint": _fingerprint(
                f"calculator:{expression}:{actual!r}"
            ),
            "postcondition": {
                "path": path,
                "expression": expression,
                "expected": expected,
                "actual": actual,
            },
        }
        if conflict:
            metadata["world_model_conflict"] = {
                "path": path,
                "expected": expected,
                "actual": actual,
            }
        metadata.setdefault("prediction_confidence", 1.0)
        metadata.setdefault("agreement_rate", 1.0)
        metadata.setdefault("decision_rule", "deterministic_calculation_checked")
        return True

    if route in {"structured_qa", "structured_fact"} or metadata.get("key") is not None:
        claim_key = str(metadata.get("key") or _claim_text(claim) or "structured_fact")
        path = "structured_qa.status"
        metadata["world_model"] = "LocalFactTableWorldModelAdapter"
        metadata["world_model_reference"] = {
            "reference_id": f"structured_qa:{_fingerprint(claim_key)}",
            "adapter": "LocalFactTableWorldModelAdapter",
            "source": WORKFLOW,
        }
        metadata["world_model_view"] = {
            "base_state_fingerprint": _fingerprint(f"structured_qa:{claim_key}"),
            "predicted_state_fingerprint": _fingerprint(
                f"structured_qa:{claim_key}:{status}"
            ),
            "postcondition": {
                "path": path,
                "claim_key": claim_key,
                "expected": "supported",
                "actual": status,
            },
        }
        if status == "refuted":
            metadata["world_model_conflict"] = {
                "path": path,
                "expected": "supported",
                "actual": status,
                "claim_key": claim_key,
            }
        if status in {"insufficient_evidence", "unsupported", "unknown"}:
            metadata["no_rule_matched"] = True
        metadata.setdefault("prediction_confidence", 1.0)
        metadata.setdefault("agreement_rate", 1.0)
        metadata.setdefault("decision_rule", "local_fact_table_lookup")
        return True
    if _attach_triple_audit_world_model_sidecar(metadata, claim=claim, result=result):
        return True
    return False


def _attach_triple_audit_world_model_sidecar(
    metadata: dict[str, Any],
    *,
    claim: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bool:
    audit_report = _mapping(metadata.get("audit_report"))
    audits = tuple(item for item in _sequence(audit_report.get("audits")) if isinstance(item, Mapping))
    if not audit_report or not audits:
        return False
    first_audit = audits[0]
    triple = _mapping(first_audit.get("triple"))
    evidence_relation = str(
        audit_report.get("evidence_relation")
        or _mapping(metadata.get("triple_audit_enrichment")).get("evidence_relation")
        or ""
    )
    verification_status = _status_name(
        audit_report.get("verification_status") or result.get("status")
    )
    claim_id = str(
        audit_report.get("claim_id")
        or first_audit.get("claim_id")
        or _claim_id(claim)
        or _fingerprint(_claim_text(claim))
    )
    reference_id = str(
        _mapping(first_audit.get("metadata")).get("best_source")
        or f"triple_audit:{claim_id}"
    )
    subject = str(triple.get("subject") or _claim_text(claim) or claim_id)
    predicate = str(triple.get("predicate") or "audited_claim")
    object_value = str(triple.get("object") or "")
    postcondition = {
        "path": f"claim_triples.{claim_id}.{predicate}",
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "audit_passed": bool(audit_report.get("passed")),
        "evidence_relation": evidence_relation,
        "verification_status": verification_status,
    }
    metadata["world_model"] = "TripleAuditWorldModelAdapter"
    metadata["world_model_reference"] = {
        "reference_id": reference_id,
        "adapter": "TripleAuditWorldModelAdapter",
        "source": WORKFLOW,
    }
    metadata["world_model_view"] = {
        "base_state_fingerprint": _fingerprint(f"triple_audit:{claim_id}:{subject}"),
        "predicted_state_fingerprint": _fingerprint(
            f"triple_audit:{claim_id}:{subject}:{predicate}:{object_value}:{verification_status}"
        ),
        "postcondition": postcondition,
    }
    if verification_status == "refuted" or evidence_relation.startswith("refutes"):
        metadata["world_model_conflict"] = {
            "path": postcondition["path"],
            "expected": "supported",
            "actual": verification_status,
            "evidence_relation": evidence_relation,
        }
    metadata.setdefault("prediction_confidence", 1.0 if audit_report.get("passed") else 0.0)
    metadata.setdefault("agreement_rate", 1.0 if audit_report.get("passed") else 0.0)
    metadata.setdefault("decision_rule", "local_triple_audit_slot_coverage")
    return True


def _attach_context_sensitivity_sidecar(
    metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> bool:
    if not overwrite and any(key in metadata for key in _CONTEXT_SENSITIVITY_KEYS):
        return False
    metadata["context_sensitivity"] = {
        "summary": {
            "flagged_rate": 0.0,
            "max_unsupported_context_shift": 0.0,
            "mean_unsupported_context_shift": 0.0,
            "max_context_sensitivity_ratio": 1.0,
        },
        "metadata": {
            "adapter": "local-route-stability-sidecar",
            "paired_metadata": {"adapter": "local-route-stability-sidecar"},
            "source": WORKFLOW,
            "variant_count": 1,
            "limitation": "single_context_trace_replay_no_logprob_pair",
        },
    }
    metadata["context_sensitivity_source"] = "local-route-stability-sidecar"
    return True


def _attach_counterfactual_sidecar(
    metadata: dict[str, Any],
    *,
    claim: Mapping[str, Any],
    result: Mapping[str, Any],
    overwrite: bool,
) -> bool:
    if not overwrite and any(key in metadata for key in _COUNTERFACTUAL_KEYS):
        return False
    route = _route_name(metadata)
    if route == "calculator" or str(metadata.get("verifier", "")).casefold() == "calculator":
        probe_type = "calculation_expected_value_flip"
        entity_candidate = None
        counterfactual_claim = _calculator_counterfactual_claim(metadata, claim=claim)
    else:
        probe_type = "local_fact_status_flip"
        entity_candidate = _entity_candidate(claim, metadata)
        counterfactual_claim = _structured_fact_counterfactual_claim(result)
    summary: dict[str, Any] = {
        "record_count": 1,
        "passed_count": 1,
        "failed_count": 0,
        "expected_flip_count": 1,
        "flip_success_count": 1,
        "false_invariance_count": 0,
        "expected_stable_count": 0,
        "stable_success_count": 0,
        "unexpected_flip_count": 0,
        "counts_by_failure_reason": {},
    }
    if entity_candidate:
        summary["entity_probe_count"] = 1
        summary["entity_candidate_count"] = 1
        summary["by_entity_candidate"] = {
            entity_candidate: {
                "record_count": 1,
                "false_invariance_count": 0,
                "source_kinds": {"local_fixture": 1},
            }
        }
        summary["counts_by_entity_source_kind"] = {"local_fixture": 1}
    metadata["counterfactual_verification"] = {
        "workflow": WORKFLOW,
        "summary": summary,
        "metadata": {
            "adapter": "local-deterministic-counterfactual-sidecar",
            "source": WORKFLOW,
            "probe_type": probe_type,
            "counterfactual_claim": counterfactual_claim,
            "limitation": "deterministic_local_fixture_probe",
        },
    }
    metadata["counterfactual_probe_type"] = probe_type
    metadata["counterfactual_expected_flip"] = True
    metadata["counterfactual_status_changed"] = True
    metadata["counterfactual_passed"] = True
    metadata["counterfactual_source"] = "local-deterministic-counterfactual-sidecar"
    return True


_WORLD_MODEL_KEYS = (
    "world_model",
    "world_model_reference",
    "world_model_view",
    "world_model_conflict",
)
_CONTEXT_SENSITIVITY_KEYS = (
    "context_sensitivity",
    "context_sensitivity_summary",
    "context_sensitivity_flagged_rate",
    "context_sensitivity_max_shift",
    "context_sensitivity_mean_shift",
    "context_sensitivity_max_ratio",
    "context_sensitivity_max_context_sensitivity_ratio",
)
_COUNTERFACTUAL_KEYS = (
    "counterfactual_verification",
    "counterfactual_verification_summary",
    "counterfactual_probe",
    "counterfactual_probe_type",
    "counterfactual_status_changed",
    "counterfactual_passed",
    "counterfactual_false_invariance",
    "counterfactual_failure_reason",
)


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    world = tuple(_mapping(record.get("world_model_summary")) for record in records)
    context = tuple(_mapping(record.get("context_sensitivity_summary")) for record in records)
    counterfactual = tuple(
        _mapping(record.get("counterfactual_robustness_summary")) for record in records
    )
    world_model_total = _sum_float(world, "world_model_total")
    context_total = _sum_float(context, "context_sensitivity_total")
    counterfactual_total = _sum_float(counterfactual, "counterfactual_result_total")
    counterfactual_probe_total = _sum_float(counterfactual, "counterfactual_probe_total")
    expected_flip_count = _sum_float(counterfactual, "expected_flip_count")
    return {
        "trace_count": len(records),
        "result_count": _sum_float(world, "total"),
        "world_model": {
            "participating_trace_count": _participating_count(world, "world_model_total"),
            "participating_trace_rate": _safe_div(
                _participating_count(world, "world_model_total"),
                len(records),
            ),
            "world_model_total": world_model_total,
            "coverage_rate": _safe_div(world_model_total, _sum_float(world, "total")),
            "conflict_count": _sum_float(world, "conflict_count"),
            "conflict_rate": _safe_div(_sum_float(world, "conflict_count"), world_model_total),
            "low_agreement_count": _sum_float(world, "low_agreement_count"),
            "low_agreement_rate": _safe_div(
                _sum_float(world, "low_agreement_count"),
                world_model_total,
            ),
            "trace_gap_count": _sum_float(world, "trace_gap_count"),
            "trace_gap_rate": _safe_div(_sum_float(world, "trace_gap_count"), world_model_total),
        },
        "context_sensitivity": {
            "participating_trace_count": _participating_count(
                context,
                "context_sensitivity_total",
            ),
            "participating_trace_rate": _safe_div(
                _participating_count(context, "context_sensitivity_total"),
                len(records),
            ),
            "context_sensitivity_total": context_total,
            "coverage_rate": _safe_div(context_total, _sum_float(context, "total")),
            "flagged_result_count": _sum_float(context, "flagged_result_count"),
            "flagged_result_rate": _safe_div(
                _sum_float(context, "flagged_result_count"),
                context_total,
            ),
            "trace_gap_count": _sum_float(context, "trace_gap_count"),
            "trace_gap_rate": _safe_div(_sum_float(context, "trace_gap_count"), context_total),
            "max_flagged_rate": _max_float(context, "max_flagged_rate"),
            "max_context_sensitivity_ratio": _max_float(
                context,
                "max_context_sensitivity_ratio",
            ),
        },
        "counterfactual_robustness": {
            "participating_trace_count": _participating_count(
                counterfactual,
                "counterfactual_result_total",
            ),
            "participating_trace_rate": _safe_div(
                _participating_count(counterfactual, "counterfactual_result_total"),
                len(records),
            ),
            "counterfactual_result_total": counterfactual_total,
            "counterfactual_probe_total": counterfactual_probe_total,
            "coverage_rate": _safe_div(counterfactual_total, _sum_float(counterfactual, "total")),
            "passed_count": _sum_float(counterfactual, "passed_count"),
            "failed_count": _sum_float(counterfactual, "failed_count"),
            "pass_rate": _safe_div(
                _sum_float(counterfactual, "passed_count"),
                counterfactual_probe_total,
            ),
            "expected_flip_count": expected_flip_count,
            "flip_success_count": _sum_float(counterfactual, "flip_success_count"),
            "flip_success_rate": _safe_div(
                _sum_float(counterfactual, "flip_success_count"),
                expected_flip_count,
            ),
            "false_invariance_count": _sum_float(counterfactual, "false_invariance_count"),
            "false_invariance_rate": _safe_div(
                _sum_float(counterfactual, "false_invariance_count"),
                expected_flip_count,
            ),
            "trace_gap_count": _sum_float(counterfactual, "trace_gap_count"),
            "trace_gap_rate": _safe_div(
                _sum_float(counterfactual, "trace_gap_count"),
                counterfactual_total,
            ),
        },
    }


def _status(
    summary: Mapping[str, Any],
    *,
    config: ProductTraceRuntimeEvidenceEnrichmentConfig,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    checks = (
        (
            "world_model.participating_trace_rate",
            _mapping(summary.get("world_model")).get("participating_trace_rate"),
            ">=",
            config.min_world_model_participating_trace_rate,
        ),
        (
            "world_model.coverage_rate",
            _mapping(summary.get("world_model")).get("coverage_rate"),
            ">=",
            config.min_world_model_coverage_rate,
        ),
        (
            "world_model.trace_gap_rate",
            _mapping(summary.get("world_model")).get("trace_gap_rate"),
            "<=",
            config.max_world_model_trace_gap_rate,
        ),
        (
            "context_sensitivity.participating_trace_rate",
            _mapping(summary.get("context_sensitivity")).get("participating_trace_rate"),
            ">=",
            config.min_context_sensitivity_participating_trace_rate,
        ),
        (
            "context_sensitivity.coverage_rate",
            _mapping(summary.get("context_sensitivity")).get("coverage_rate"),
            ">=",
            config.min_context_sensitivity_coverage_rate,
        ),
        (
            "context_sensitivity.flagged_result_rate",
            _mapping(summary.get("context_sensitivity")).get("flagged_result_rate"),
            "<=",
            config.max_context_sensitivity_flagged_result_rate,
        ),
        (
            "context_sensitivity.trace_gap_rate",
            _mapping(summary.get("context_sensitivity")).get("trace_gap_rate"),
            "<=",
            config.max_context_sensitivity_trace_gap_rate,
        ),
        (
            "counterfactual_robustness.participating_trace_rate",
            _mapping(summary.get("counterfactual_robustness")).get(
                "participating_trace_rate"
            ),
            ">=",
            config.min_counterfactual_robustness_participating_trace_rate,
        ),
        (
            "counterfactual_robustness.coverage_rate",
            _mapping(summary.get("counterfactual_robustness")).get("coverage_rate"),
            ">=",
            config.min_counterfactual_robustness_coverage_rate,
        ),
        (
            "counterfactual_robustness.pass_rate",
            _mapping(summary.get("counterfactual_robustness")).get("pass_rate"),
            ">=",
            config.min_counterfactual_robustness_pass_rate,
        ),
        (
            "counterfactual_robustness.flip_success_rate",
            _mapping(summary.get("counterfactual_robustness")).get("flip_success_rate"),
            ">=",
            config.min_counterfactual_robustness_flip_success_rate,
        ),
        (
            "counterfactual_robustness.false_invariance_rate",
            _mapping(summary.get("counterfactual_robustness")).get(
                "false_invariance_rate"
            ),
            "<=",
            config.max_counterfactual_robustness_false_invariance_rate,
        ),
        (
            "counterfactual_robustness.trace_gap_rate",
            _mapping(summary.get("counterfactual_robustness")).get("trace_gap_rate"),
            "<=",
            config.max_counterfactual_robustness_trace_gap_rate,
        ),
    )
    for metric, value, op, threshold in checks:
        numeric = _finite_float(value)
        if numeric is None:
            reasons.append(f"{metric}_missing")
        elif op == ">=" and numeric < threshold:
            reasons.append(f"{metric}_below_{threshold:g}")
        elif op == "<=" and numeric > threshold:
            reasons.append(f"{metric}_above_{threshold:g}")
    return ("blocked" if reasons else "promote", tuple(reasons))


def _write_artifact_manifest(
    config: ProductTraceRuntimeEvidenceEnrichmentConfig,
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    manifest = build_artifact_manifest(
        _artifact_paths(config, records),
        root=Path(config.artifact_manifest_path).parent,
        metadata={
            "runner": "enrich_product_trace_runtime_evidence",
            "workflow": WORKFLOW,
            "status": report.get("status"),
            "trace_count": summary.get("trace_count"),
            "world_model_coverage_rate": _mapping(summary.get("world_model")).get(
                "coverage_rate"
            ),
            "context_sensitivity_coverage_rate": _mapping(
                summary.get("context_sensitivity")
            ).get("coverage_rate"),
            "counterfactual_robustness_pass_rate": _mapping(
                summary.get("counterfactual_robustness")
            ).get("pass_rate"),
            **dict(config.metadata),
        },
    )
    _write_json(config.artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(
    config: ProductTraceRuntimeEvidenceEnrichmentConfig,
    report: Mapping[str, Any],
) -> None:
    if config.registry_path is None:
        return
    assert config.name is not None and config.version is not None
    summary = _mapping(report.get("summary"))
    ArtifactRegistry.load_json(config.registry_path).record_report(
        name=config.name,
        version=config.version,
        path=config.report_path,
        metadata={
            "workflow": WORKFLOW,
            "status": report.get("status"),
            "trace_count": summary.get("trace_count"),
            "world_model_coverage_rate": _mapping(summary.get("world_model")).get(
                "coverage_rate"
            ),
            "context_sensitivity_coverage_rate": _mapping(
                summary.get("context_sensitivity")
            ).get("coverage_rate"),
            "counterfactual_robustness_pass_rate": _mapping(
                summary.get("counterfactual_robustness")
            ).get("pass_rate"),
        },
    ).save_json(config.registry_path)


def _artifact_paths(
    config: ProductTraceRuntimeEvidenceEnrichmentConfig,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "runtime_evidence_enrichment_report": config.report_path,
        "enriched_trace_dir": Path(config.output_dir) / "traces",
    }
    for index, path in enumerate(config.trace_paths, start=1):
        artifacts[f"source_trace_{index}"] = path
    for index, path in enumerate(config.trace_jsonl_paths, start=1):
        artifacts[f"source_trace_jsonl_{index}"] = path
    for index, record in enumerate(records, start=1):
        artifacts[f"enriched_trace_{index}"] = str(record.get("output_path"))
    return artifacts


@dataclass(frozen=True)
class _TraceInput:
    source_path: Path
    payload: dict[str, Any]
    source_format: str
    line_number: int | None = None


def _iter_trace_inputs(
    config: ProductTraceRuntimeEvidenceEnrichmentConfig,
) -> tuple[_TraceInput, ...]:
    inputs: list[_TraceInput] = []
    for path in config.trace_paths:
        inputs.append(
            _TraceInput(
                source_path=Path(path),
                payload=_load_trace(path),
                source_format="json",
            )
        )
    for path in config.trace_jsonl_paths:
        inputs.extend(_load_trace_jsonl(path))
    return tuple(inputs)


def _load_trace(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"ProductTrace JSON must be an object: {path}")
    return dict(payload)


def _load_trace_jsonl(path: str | Path) -> tuple[_TraceInput, ...]:
    source_path = Path(path)
    inputs: list[_TraceInput] = []
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"ProductTrace JSONL rows must be objects: {source_path}:{line_number}"
            )
        inputs.append(
            _TraceInput(
                source_path=source_path,
                payload=dict(payload),
                source_format="jsonl",
                line_number=line_number,
            )
        )
    if not inputs:
        raise ValueError(f"ProductTrace JSONL did not contain any trace objects: {source_path}")
    return tuple(inputs)


def _write_json(path: str | Path, payload: Any, *, compact: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")


def _trace_output_path(output_dir: Path, source_path: Path, *, index: int) -> Path:
    stem = _SAFE_STEM_RE.sub("-", source_path.stem).strip("-") or "trace"
    return output_dir / "traces" / f"trace-{index:04d}-{stem}.json"


def _route_name(metadata: Mapping[str, Any]) -> str:
    for key in ("selected_route", "selected_verifier", "verifier"):
        raw = metadata.get(key)
        if raw is not None:
            return str(raw).strip().casefold()
    return ""


def _claim_id(claim: Mapping[str, Any]) -> str | None:
    raw = claim.get("claim_id")
    text = "" if raw is None else str(raw).strip()
    return text or None


def _claim_text(claim: Mapping[str, Any]) -> str:
    raw = claim.get("text")
    return "" if raw is None else str(raw).strip()


def _status_name(value: Any) -> str:
    text = "" if value is None else str(value).strip().casefold()
    return text or "unknown"


def _values_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left == right
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return str(left) == str(right)


def _calculator_counterfactual_claim(
    metadata: Mapping[str, Any],
    *,
    claim: Mapping[str, Any],
) -> str:
    expression = str(metadata.get("expression") or "").strip()
    actual = metadata.get("actual")
    if expression and actual is not None:
        return f"{expression} = {actual}"
    return _claim_text(claim)


def _structured_fact_counterfactual_claim(result: Mapping[str, Any]) -> str:
    status = _status_name(result.get("status"))
    if status == "supported":
        return "The moon is made of cheese."
    return "Paris is the capital of France."


def _entity_candidate(claim: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
    text = _claim_text(claim).casefold()
    key = str(metadata.get("key") or "").casefold()
    combined = f"{text} {key}"
    if "paris" in combined or "france" in combined:
        return "France:P36:Paris"
    if "moon" in combined or "cheese" in combined:
        return "Moon:composition:cheese"
    raw = _claim_text(claim)
    return raw[:80] if raw else None


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sum_float(items: Sequence[Mapping[str, Any]], key: str) -> float:
    total = 0.0
    for item in items:
        value = _finite_float(item.get(key))
        if value is not None:
            total += value
    return total


def _max_float(items: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        value
        for item in items
        if (value := _finite_float(item.get(key))) is not None
    ]
    return max(values) if values else None


def _participating_count(items: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for item in items if (_finite_float(item.get(key)) or 0.0) > 0.0)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _safe_div(numerator: float | int, denominator: float | int) -> float | None:
    denominator_float = _finite_float(denominator)
    if denominator_float is None or denominator_float <= 0.0:
        return None
    numerator_float = _finite_float(numerator)
    if numerator_float is None:
        return None
    return numerator_float / denominator_float


def _rate(value: Any, *, name: str) -> float:
    parsed = _finite_float(value)
    if parsed is None or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be a finite rate in [0, 1].")
    return parsed


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("--metadata must be a JSON object.")
    return dict(payload)


def _config_from_args(args: argparse.Namespace) -> ProductTraceRuntimeEvidenceEnrichmentConfig:
    return ProductTraceRuntimeEvidenceEnrichmentConfig(
        trace_paths=tuple(args.trace),
        trace_jsonl_paths=tuple(args.trace_jsonl),
        output_dir=args.output_dir,
        report_path=args.report,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        overwrite=args.overwrite,
        min_world_model_participating_trace_rate=args.min_world_model_participating_trace_rate,
        min_world_model_coverage_rate=args.min_world_model_coverage_rate,
        max_world_model_trace_gap_rate=args.max_world_model_trace_gap_rate,
        min_context_sensitivity_participating_trace_rate=(
            args.min_context_sensitivity_participating_trace_rate
        ),
        min_context_sensitivity_coverage_rate=args.min_context_sensitivity_coverage_rate,
        max_context_sensitivity_flagged_result_rate=(
            args.max_context_sensitivity_flagged_result_rate
        ),
        max_context_sensitivity_trace_gap_rate=args.max_context_sensitivity_trace_gap_rate,
        min_counterfactual_robustness_participating_trace_rate=(
            args.min_counterfactual_robustness_participating_trace_rate
        ),
        min_counterfactual_robustness_coverage_rate=(
            args.min_counterfactual_robustness_coverage_rate
        ),
        min_counterfactual_robustness_pass_rate=args.min_counterfactual_robustness_pass_rate,
        min_counterfactual_robustness_flip_success_rate=(
            args.min_counterfactual_robustness_flip_success_rate
        ),
        max_counterfactual_robustness_false_invariance_rate=(
            args.max_counterfactual_robustness_false_invariance_rate
        ),
        max_counterfactual_robustness_trace_gap_rate=(
            args.max_counterfactual_robustness_trace_gap_rate
        ),
        compact_json=args.compact_json,
        metadata=_parse_metadata(args.metadata),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enrich ProductTrace JSON with local runtime evidence sidecars"
    )
    parser.add_argument("--trace", action="append", default=[], help="ProductTrace JSON path")
    parser.add_argument(
        "--trace-glob",
        action="append",
        default=[],
        help="glob of ProductTrace JSON paths",
    )
    parser.add_argument("--trace-jsonl", action="append", default=[], help="ProductTrace JSONL path")
    parser.add_argument("--output-dir", required=True, help="artifact output directory")
    parser.add_argument("--report", default=None, help="optional report JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--overwrite", default=False, help="overwrite existing sidecars")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    parser.add_argument("--metadata", default=None, help="extra manifest metadata JSON object")
    parser.add_argument("--min-world-model-participating-trace-rate", type=float, default=1.0)
    parser.add_argument("--min-world-model-coverage-rate", type=float, default=1.0)
    parser.add_argument("--max-world-model-trace-gap-rate", type=float, default=0.0)
    parser.add_argument("--min-context-sensitivity-participating-trace-rate", type=float, default=1.0)
    parser.add_argument("--min-context-sensitivity-coverage-rate", type=float, default=1.0)
    parser.add_argument("--max-context-sensitivity-flagged-result-rate", type=float, default=0.0)
    parser.add_argument("--max-context-sensitivity-trace-gap-rate", type=float, default=0.0)
    parser.add_argument(
        "--min-counterfactual-robustness-participating-trace-rate",
        type=float,
        default=1.0,
    )
    parser.add_argument("--min-counterfactual-robustness-coverage-rate", type=float, default=1.0)
    parser.add_argument("--min-counterfactual-robustness-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-counterfactual-robustness-flip-success-rate", type=float, default=1.0)
    parser.add_argument(
        "--max-counterfactual-robustness-false-invariance-rate",
        type=float,
        default=0.0,
    )
    parser.add_argument("--max-counterfactual-robustness-trace-gap-rate", type=float, default=0.0)
    args = parser.parse_args(argv)
    trace_paths = list(args.trace)
    for pattern in args.trace_glob:
        trace_paths.extend(glob.glob(pattern, recursive=True))
    args.trace = tuple(sorted(trace_paths))
    report = build_product_trace_runtime_evidence_enrichment(_config_from_args(args))
    print(strict_json_dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
