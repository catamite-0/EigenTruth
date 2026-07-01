"""Enrich ProductTrace JSON with signed action receipts and receipt references.

This workflow is intentionally offline and dependency-free. It takes existing
full ProductTrace payloads, assigns stable request ids when replay traces did not
have them, signs each action result with a local HMAC receipt, and adds explicit
claim/final-answer references that the receipt claim-support audit can verify.
"""

from __future__ import annotations

import argparse
import copy
import glob
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
from eigentruth.control import (  # noqa: E402
    ActionExecutionStatus,
    ActionReceiptSigner,
    ActionRequest,
    ActionResult,
    ReceiptClaimSupportPolicy,
    action_receipt_summary_from_results,
    attach_action_receipt,
    audit_receipt_claim_support,
    json_fingerprint,
    product_runtime_metrics,
)
from eigentruth.json_utils import strict_json_dumps, to_jsonable  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "product_trace_action_receipt_enrichment"
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_ACCEPTED_STATUSES = (
    ActionExecutionStatus.SUCCEEDED,
    ActionExecutionStatus.DRY_RUN,
)


@dataclass(frozen=True)
class ProductTraceActionReceiptEnrichmentConfig:
    """Configuration for trace-level action receipt enrichment."""

    trace_paths: Sequence[str | Path]
    output_dir: str | Path
    trace_jsonl_paths: Sequence[str | Path] = ()
    report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    secret: str = "eigentruth-local-action-receipts"
    key_id: str = "local-action-receipts"
    issuer: str = "eigentruth"
    created_at: str | None = None
    min_receipt_coverage: float = 1.0
    min_reference_support_rate: float = 1.0
    require_claim_reference: bool = True
    compact_json: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        trace_jsonl_paths = tuple(Path(path) for path in self.trace_jsonl_paths)
        if not trace_paths and not trace_jsonl_paths:
            raise ValueError("at least one ProductTrace JSON or JSONL path is required.")
        output_dir = Path(self.output_dir)
        report_path = (
            output_dir / "product-trace-action-receipt-enrichment.json"
            if self.report_path is None
            else Path(self.report_path)
        )
        artifact_manifest_path = (
            output_dir / "product-trace-action-receipt-artifact-manifest.json"
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
        object.__setattr__(self, "secret", _non_empty_text(self.secret, name="secret"))
        object.__setattr__(self, "key_id", _non_empty_text(self.key_id, name="key_id"))
        object.__setattr__(self, "issuer", _non_empty_text(self.issuer, name="issuer"))
        object.__setattr__(
            self,
            "min_receipt_coverage",
            _rate(self.min_receipt_coverage, name="min_receipt_coverage"),
        )
        object.__setattr__(
            self,
            "min_reference_support_rate",
            _rate(self.min_reference_support_rate, name="min_reference_support_rate"),
        )
        object.__setattr__(
            self,
            "require_claim_reference",
            strict_bool(self.require_claim_reference, name="require_claim_reference"),
        )
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_product_trace_action_receipt_enrichment(
    config: ProductTraceActionReceiptEnrichmentConfig,
) -> dict[str, Any]:
    """Write receipt-aware ProductTrace files and return a JSON-ready report."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signer = ActionReceiptSigner(config.secret, key_id=config.key_id, issuer=config.issuer)
    policy = ReceiptClaimSupportPolicy(
        accepted_statuses=_DEFAULT_ACCEPTED_STATUSES,
        require_signed_receipt=True,
        warn_on_unsigned_receipt=False,
    )

    records: list[dict[str, Any]] = []
    for index, trace_input in enumerate(_iter_trace_inputs(config), start=1):
        trace = trace_input.payload
        reject_bounded_product_trace(trace, path=trace_input.source_path)
        enriched, record = _enrich_trace(
            trace,
            source_path=trace_input.source_path,
            output_path=_trace_output_path(output_dir, trace_input.source_path, index=index),
            signer=signer,
            policy=policy,
            created_at=config.created_at,
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
            "key_id": config.key_id,
            "issuer": config.issuer,
            "created_at": config.created_at,
            "accepted_statuses": tuple(status.value for status in _DEFAULT_ACCEPTED_STATUSES),
            "require_signed_receipt": True,
            "min_receipt_coverage": config.min_receipt_coverage,
            "min_reference_support_rate": config.min_reference_support_rate,
            "require_claim_reference": config.require_claim_reference,
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
    signer: ActionReceiptSigner,
    policy: ReceiptClaimSupportPolicy,
    created_at: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = copy.deepcopy(dict(trace))
    actions = [dict(action) for action in _sequence(payload.get("actions")) if isinstance(action, Mapping)]
    results = [
        dict(result)
        for result in _sequence(payload.get("action_results"))
        if isinstance(result, Mapping)
    ]
    request_ids = _assign_stable_request_ids(payload, actions, results)
    receipt_verifications = []
    for index, result in enumerate(results):
        request = actions[index] if index < len(actions) else None
        result_obj = ActionResult.from_dict(result)
        request_obj = ActionRequest.from_dict(request) if request is not None else None
        receipt = signer.issue(
            result_obj,
            request=request_obj,
            created_at=created_at,
            metadata={
                "workflow": WORKFLOW,
                "source_path": str(source_path),
                "trace_request_id": payload.get("request_id"),
                "result_index": index,
            },
        )
        receipted = attach_action_receipt(result_obj, receipt).to_dict()
        results[index] = receipted
        receipt_verifications.append(
            signer.verify(receipt, result=receipted, request=request_obj).to_dict()
        )

    claims = _enrich_claim_references(payload.get("claims"), results)
    payload["actions"] = actions
    payload["action_results"] = results
    payload["claims"] = claims
    payload["final_answer"] = _enrich_final_answer_references(payload.get("final_answer"), results)

    action_receipts = action_receipt_summary_from_results(results)
    receipt_claim_support_report = audit_receipt_claim_support(payload, policy=policy).to_dict()
    receipt_claim_support = dict(receipt_claim_support_report["summary"])
    summaries = dict(_mapping(payload.get("summaries")))
    summaries["action_receipts"] = action_receipts
    summaries["receipt_claim_support"] = receipt_claim_support
    payload["summaries"] = summaries

    metadata = dict(_mapping(payload.get("metadata")))
    trace_corpus = dict(_mapping(metadata.get("trace_corpus")))
    trace_corpus["action_receipts_summary"] = action_receipts
    trace_corpus["receipt_claim_support_summary"] = receipt_claim_support
    metadata["trace_corpus"] = trace_corpus
    metadata["action_receipt_enrichment"] = {
        "workflow": WORKFLOW,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "request_ids_assigned": request_ids,
        "accepted_statuses": tuple(status.value for status in _DEFAULT_ACCEPTED_STATUSES),
        "require_signed_receipt": policy.require_signed_receipt,
        "action_receipts": action_receipts,
        "receipt_claim_support": receipt_claim_support,
    }
    payload["metadata"] = metadata
    metrics = product_runtime_metrics(payload)
    return payload, {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "request_id": payload.get("request_id"),
        "request_ids_assigned": request_ids,
        "action_receipts": action_receipts,
        "receipt_claim_support": receipt_claim_support,
        "receipt_claim_support_report": receipt_claim_support_report,
        "receipt_verifications": tuple(receipt_verifications),
        "metrics": {
            "action_receipts_coverage": metrics.get("action_receipts_coverage"),
            "action_receipts_missing_receipt_count": metrics.get("action_receipts_missing_receipt_count"),
            "action_receipts_invalid_receipt_count": metrics.get("action_receipts_invalid_receipt_count"),
            "action_receipts_fingerprint_mismatch_count": (
                metrics.get("action_receipts_fingerprint_mismatch_count")
            ),
            "receipt_claim_support_reference_count": metrics.get("receipt_claim_support_reference_count"),
            "receipt_claim_support_unsupported_reference_count": (
                metrics.get("receipt_claim_support_unsupported_reference_count")
            ),
            "receipt_claim_support_failed_result_reference_count": (
                metrics.get("receipt_claim_support_failed_result_reference_count")
            ),
            "receipt_claim_support_fingerprint_mismatch_reference_count": (
                metrics.get("receipt_claim_support_fingerprint_mismatch_reference_count")
            ),
        },
    }


def _assign_stable_request_ids(
    trace: Mapping[str, Any],
    actions: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
) -> tuple[str, ...]:
    assigned: list[str] = []
    seen: set[str] = set()
    trace_id = _safe_token(str(trace.get("request_id") or "trace"))
    for index in range(max(len(actions), len(results))):
        action = actions[index] if index < len(actions) else None
        result = results[index] if index < len(results) else None
        existing = _first_text(
            None if action is None else action.get("request_id"),
            None if result is None else result.get("request_id"),
        )
        action_name = _first_text(
            None if action is None else action.get("action"),
            None if result is None else result.get("action"),
            "action",
        )
        request_id = existing or f"{trace_id}-{_safe_token(action_name)}-{index + 1}"
        request_id = _dedupe_request_id(_safe_token(request_id), seen)
        seen.add(request_id)
        if action is not None:
            action["request_id"] = request_id
        if result is not None:
            result["request_id"] = request_id
        assigned.append(request_id)
    return tuple(assigned)


def _enrich_claim_references(
    claims_payload: Any,
    results: Sequence[Mapping[str, Any]],
) -> list[Any]:
    claims = []
    for claim in _sequence(claims_payload):
        if not isinstance(claim, Mapping):
            claims.append(claim)
            continue
        claim_payload = copy.deepcopy(dict(claim))
        metadata = dict(_mapping(claim_payload.get("metadata")))
        indexes = _matched_result_indexes_for_claim(claim_payload, results)
        request_ids = tuple(
            str(results[index].get("request_id"))
            for index in indexes
            if results[index].get("request_id") is not None
        )
        output_fingerprints = tuple(
            json_fingerprint(results[index].get("output", {}))
            for index in indexes
        )
        if request_ids:
            metadata["action_request_ids"] = _merged_unique_strings(
                metadata.get("action_request_ids"),
                request_ids,
            )
        if output_fingerprints:
            metadata["receipt_output_fingerprints"] = _merged_unique_strings(
                metadata.get("receipt_output_fingerprints"),
                output_fingerprints,
            )
        if request_ids or output_fingerprints:
            metadata["receipt_reference_source"] = WORKFLOW
        claim_payload["metadata"] = metadata
        claims.append(claim_payload)
    return claims


def _enrich_final_answer_references(
    final_answer_payload: Any,
    results: Sequence[Mapping[str, Any]],
) -> Any:
    if not isinstance(final_answer_payload, Mapping):
        return final_answer_payload
    final_answer = copy.deepcopy(dict(final_answer_payload))
    if not results:
        return final_answer
    request_id = results[0].get("request_id")
    output_fingerprint = json_fingerprint(results[0].get("output", {}))
    metadata = dict(_mapping(final_answer.get("metadata")))
    if request_id is not None:
        metadata["action_request_ids"] = _merged_unique_strings(
            metadata.get("action_request_ids"),
            (str(request_id),),
        )
    metadata["receipt_output_fingerprints"] = _merged_unique_strings(
        metadata.get("receipt_output_fingerprints"),
        (output_fingerprint,),
    )
    metadata["receipt_reference_source"] = WORKFLOW
    final_answer["metadata"] = metadata
    evidence_items = []
    for evidence in _sequence(final_answer.get("evidence")):
        if not isinstance(evidence, Mapping):
            evidence_items.append(evidence)
            continue
        evidence_payload = copy.deepcopy(dict(evidence))
        if request_id is not None:
            evidence_payload.setdefault("request_id", str(request_id))
        evidence_metadata = dict(_mapping(evidence_payload.get("metadata")))
        evidence_metadata["receipt_output_fingerprints"] = _merged_unique_strings(
            evidence_metadata.get("receipt_output_fingerprints"),
            (output_fingerprint,),
        )
        evidence_payload["metadata"] = evidence_metadata
        evidence_items.append(evidence_payload)
    if evidence_items:
        final_answer["evidence"] = evidence_items
    return final_answer


def _matched_result_indexes_for_claim(
    claim: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    claim_id = _optional_text(claim.get("claim_id"))
    if claim_id is not None:
        matches = [
            index
            for index, result in enumerate(results)
            if claim_id in _claim_ids_from_result(result)
        ]
        if matches:
            return tuple(matches)
    if len(results) == 1:
        return (0,)
    return ()


def _claim_ids_from_result(result: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    payloads = [result, _mapping(result.get("output")), _mapping(result.get("metadata"))]
    output = _mapping(result.get("output"))
    for key in ("blocked_claims", "claims", "retrieval_targets", "targets"):
        payloads.extend(item for item in _sequence(output.get(key)) if isinstance(item, Mapping))
    for payload in payloads:
        for key in ("claim_id", "selected_claim_id", "source_claim_id"):
            text = _optional_text(payload.get(key))
            if text is not None:
                ids.add(text)
    return ids


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    action_summaries = tuple(_mapping(record.get("action_receipts")) for record in records)
    support_summaries = tuple(_mapping(record.get("receipt_claim_support")) for record in records)
    result_count = _sum_float(action_summaries, "result_count")
    receipt_count = _sum_float(action_summaries, "receipt_count")
    missing_receipt_count = _sum_float(action_summaries, "missing_receipt_count")
    invalid_receipt_count = _sum_float(action_summaries, "invalid_receipt_count")
    unsigned_receipt_count = _sum_float(action_summaries, "unsigned_receipt_count")
    fingerprint_mismatch_count = _sum_float(action_summaries, "fingerprint_mismatch_count")
    reference_count = _sum_float(support_summaries, "reference_count")
    unsupported_reference_count = _sum_float(support_summaries, "unsupported_reference_count")
    missing_reference_count = _sum_float(support_summaries, "missing_reference_count")
    unreceipted_reference_count = _sum_float(support_summaries, "unreceipted_reference_count")
    failed_result_reference_count = _sum_float(support_summaries, "failed_result_reference_count")
    fingerprint_mismatch_reference_count = _sum_float(
        support_summaries,
        "fingerprint_mismatch_reference_count",
    )
    unsigned_reference_count = _sum_float(support_summaries, "unsigned_reference_count")
    supported_reference_count = max(reference_count - unsupported_reference_count, 0.0)
    signature_valid_count = sum(
        1
        for record in records
        for verification in _sequence(record.get("receipt_verifications"))
        if _mapping(verification).get("signature_valid") is True
    )
    return {
        "trace_count": len(records),
        "action_receipts": {
            "result_count": result_count,
            "receipt_count": receipt_count,
            "missing_receipt_count": missing_receipt_count,
            "invalid_receipt_count": invalid_receipt_count,
            "unsigned_receipt_count": unsigned_receipt_count,
            "fingerprint_mismatch_count": fingerprint_mismatch_count,
            "signature_valid_count": signature_valid_count,
            "coverage_rate": _safe_div(receipt_count, result_count),
            "missing_receipt_rate": _safe_div(missing_receipt_count, result_count),
            "invalid_receipt_rate": _safe_div(invalid_receipt_count, receipt_count),
            "unsigned_receipt_rate": _safe_div(unsigned_receipt_count, receipt_count),
            "fingerprint_mismatch_rate": _safe_div(fingerprint_mismatch_count, receipt_count),
            "signature_valid_rate": _safe_div(signature_valid_count, receipt_count),
        },
        "receipt_claim_support": {
            "reference_count": reference_count,
            "supported_reference_count": supported_reference_count,
            "unsupported_reference_count": unsupported_reference_count,
            "missing_reference_count": missing_reference_count,
            "unreceipted_reference_count": unreceipted_reference_count,
            "failed_result_reference_count": failed_result_reference_count,
            "fingerprint_mismatch_reference_count": fingerprint_mismatch_reference_count,
            "unsigned_reference_count": unsigned_reference_count,
            "reference_support_rate": _safe_div(supported_reference_count, reference_count),
            "unsupported_reference_rate": _safe_div(unsupported_reference_count, reference_count),
            "missing_reference_rate": _safe_div(missing_reference_count, reference_count),
            "unreceipted_reference_rate": _safe_div(unreceipted_reference_count, reference_count),
            "failed_result_reference_rate": _safe_div(failed_result_reference_count, reference_count),
            "fingerprint_mismatch_reference_rate": _safe_div(
                fingerprint_mismatch_reference_count,
                reference_count,
            ),
            "unsigned_reference_rate": _safe_div(unsigned_reference_count, reference_count),
        },
        "per_trace": {
            "action_receipt_coverage": tuple(
                _mapping(record.get("action_receipts")).get("coverage") for record in records
            ),
            "receipt_claim_support_reference_count": tuple(
                _mapping(record.get("receipt_claim_support")).get("reference_count") for record in records
            ),
            "receipt_claim_support_unsupported_reference_count": tuple(
                _mapping(record.get("receipt_claim_support")).get("unsupported_reference_count")
                for record in records
            ),
        },
    }


def _status(
    summary: Mapping[str, Any],
    *,
    config: ProductTraceActionReceiptEnrichmentConfig,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    action_receipts = _mapping(summary.get("action_receipts"))
    support = _mapping(summary.get("receipt_claim_support"))
    coverage = _optional_float(action_receipts.get("coverage_rate"))
    if coverage is None or coverage < config.min_receipt_coverage:
        reasons.append(f"action_receipts_coverage_below_{config.min_receipt_coverage:g}")
    for key in (
        "missing_receipt_count",
        "invalid_receipt_count",
        "fingerprint_mismatch_count",
        "unsigned_receipt_count",
    ):
        if (_optional_float(action_receipts.get(key)) or 0.0) > 0.0:
            reasons.append(f"action_receipts_{key}_nonzero")
    reference_count = _optional_float(support.get("reference_count")) or 0.0
    support_rate = _optional_float(support.get("reference_support_rate"))
    if config.require_claim_reference and reference_count <= 0.0:
        reasons.append("receipt_claim_support_reference_count_zero")
    if support_rate is None or support_rate < config.min_reference_support_rate:
        reasons.append(f"receipt_claim_support_rate_below_{config.min_reference_support_rate:g}")
    for key in (
        "unsupported_reference_count",
        "missing_reference_count",
        "unreceipted_reference_count",
        "failed_result_reference_count",
        "fingerprint_mismatch_reference_count",
        "unsigned_reference_count",
    ):
        if (_optional_float(support.get(key)) or 0.0) > 0.0:
            reasons.append(f"receipt_claim_support_{key}_nonzero")
    return ("blocked" if reasons else "promote", tuple(reasons))


def _write_artifact_manifest(
    config: ProductTraceActionReceiptEnrichmentConfig,
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    action_receipts = _mapping(summary.get("action_receipts"))
    support = _mapping(summary.get("receipt_claim_support"))
    manifest = build_artifact_manifest(
        _artifact_paths(config, records),
        root=Path(config.artifact_manifest_path).parent,
        metadata={
            "runner": "enrich_product_trace_action_receipts",
            "workflow": WORKFLOW,
            "status": report.get("status"),
            "trace_count": summary.get("trace_count"),
            "action_receipts_coverage_rate": action_receipts.get("coverage_rate"),
            "receipt_claim_support_reference_support_rate": support.get("reference_support_rate"),
            **dict(config.metadata),
        },
    )
    _write_json(config.artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(
    config: ProductTraceActionReceiptEnrichmentConfig,
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
            "action_receipts": dict(_mapping(summary.get("action_receipts"))),
            "receipt_claim_support": dict(_mapping(summary.get("receipt_claim_support"))),
        },
    ).save_json(config.registry_path)


def _artifact_paths(
    config: ProductTraceActionReceiptEnrichmentConfig,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "action_receipt_enrichment_report": config.report_path,
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


def _iter_trace_inputs(config: ProductTraceActionReceiptEnrichmentConfig) -> tuple[_TraceInput, ...]:
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
        text = strict_json_dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")


def _trace_output_path(output_dir: Path, source_path: Path, *, index: int) -> Path:
    stem = _SAFE_STEM_RE.sub("-", source_path.stem).strip("-") or "trace"
    return output_dir / "traces" / f"trace-{index:04d}-{stem}.json"


def _paths_from_args(values: Sequence[str], globs: Sequence[str]) -> tuple[Path, ...]:
    paths = [Path(value) for value in values]
    for pattern in globs:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern, recursive=True)))
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sum_float(items: Sequence[Mapping[str, Any]], key: str) -> float:
    total = 0.0
    for item in items:
        value = _optional_float(item.get(key))
        if value is not None:
            total += value
    return total


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number in [0, 1], not bool.")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _non_empty_text(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _optional_text(value)
        if text is not None:
            return text
    return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_token(value: Any) -> str:
    token = _SAFE_STEM_RE.sub("-", str(value).strip()).strip("-")
    return token or "item"


def _dedupe_request_id(request_id: str, seen: set[str]) -> str:
    if request_id not in seen:
        return request_id
    suffix = 2
    while f"{request_id}-{suffix}" in seen:
        suffix += 1
    return f"{request_id}-{suffix}"


def _merged_unique_strings(existing: Any, additions: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for item in (*_sequence(existing), *additions):
        text = _optional_text(item)
        if text is None or text in seen:
            continue
        values.append(text)
        seen.add(text)
    return tuple(values)


def _config_from_args(args: argparse.Namespace) -> ProductTraceActionReceiptEnrichmentConfig:
    trace_paths = _paths_from_args(args.trace or (), args.trace_glob or ())
    trace_jsonl_paths = _paths_from_args(args.trace_jsonl or (), args.trace_jsonl_glob or ())
    if not trace_paths and not trace_jsonl_paths:
        raise ValueError("at least one --trace, --trace-glob, --trace-jsonl, or --trace-jsonl-glob match is required.")
    return ProductTraceActionReceiptEnrichmentConfig(
        trace_paths=trace_paths,
        output_dir=args.output_dir,
        trace_jsonl_paths=trace_jsonl_paths,
        report_path=args.report,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        secret=args.secret,
        key_id=args.key_id,
        issuer=args.issuer,
        created_at=args.created_at,
        min_receipt_coverage=args.min_receipt_coverage,
        min_reference_support_rate=args.min_reference_support_rate,
        require_claim_reference=not bool(args.allow_no_claim_reference),
        compact_json=bool(args.compact_json),
        metadata={"cli": True},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrich ProductTrace JSON with signed action receipts")
    parser.add_argument("--trace", action="append", default=[], help="full ProductTrace JSON path; repeatable")
    parser.add_argument("--trace-glob", action="append", default=[], help="glob for full ProductTrace JSON files")
    parser.add_argument("--trace-jsonl", action="append", default=[], help="ProductTrace JSONL path; repeatable")
    parser.add_argument(
        "--trace-jsonl-glob",
        action="append",
        default=[],
        help="glob for ProductTrace JSONL files",
    )
    parser.add_argument("--output-dir", required=True, help="directory for enriched traces and default report paths")
    parser.add_argument("--report", default=None, help="report JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--secret", default="eigentruth-local-action-receipts")
    parser.add_argument("--key-id", default="local-action-receipts")
    parser.add_argument("--issuer", default="eigentruth")
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--min-receipt-coverage", type=float, default=1.0)
    parser.add_argument("--min-reference-support-rate", type=float, default=1.0)
    parser.add_argument("--allow-no-claim-reference", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    report = build_product_trace_action_receipt_enrichment(_config_from_args(args))
    print(strict_json_dumps(to_jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
