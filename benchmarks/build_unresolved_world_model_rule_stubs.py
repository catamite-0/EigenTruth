"""Bridge unresolved blind-spot rule requests into rule-authoring stubs.

The unresolved evidence queue mixes external citation/search requests with
world-model or calculator rule-authoring requests. This workflow filters the
rule branch into the JSONL contract consumed by
``run_world_model_rule_authoring_adapter.py``. The emitted stubs are still
non-evidence work items: they contain no labels or model answers, and they
require explicit deterministic inputs plus a later promotion gate before any
product handoff can consume a candidate result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "unresolved_world_model_rule_stubs"
SOURCE_WORKFLOW = "unresolved_blind_spot_evidence_queue"
RULE_REQUEST_TYPE = "world_model_or_calculator_rule"
RULE_ADAPTER_FAMILY = "world_model_rule_authoring"
RESERVED_STUB_FIELDS = {
    "answer",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "record_index",
    "score_label",
    "target_rank",
}
RULE_FAMILY_ALIASES = {
    "temporal_freshness": "temporal_consistency",
}
REQUIRED_INPUTS_BY_FAMILY = {
    "quantity_or_arithmetic": ("numeric_value", "unit", "reference_time"),
    "entity_disambiguation": ("subject_entity", "answer_entity", "requested_role"),
    "causal_or_procedural": ("mechanism", "precondition", "source_citation"),
    "temporal_consistency": ("claim_time", "source_time", "retrieved_at", "source_citation"),
}


def build_unresolved_world_model_rule_stubs(
    *,
    queue_report: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready report and rule-stub rows for unresolved rules."""
    _validate_queue_report(queue_report)
    source_requests = _mapping_sequence(queue_report.get("adapter_requests", ()))
    rule_requests = tuple(
        request for request in source_requests if str(request.get("request_type") or "") == RULE_REQUEST_TYPE
    )
    stubs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for ordinal, request in enumerate(rule_requests, start=1):
        failures = _request_failures(request)
        if failures:
            skipped.append(_skip(request, ordinal=ordinal, reason="invalid_rule_request", failures=failures))
            continue
        stubs.append(_rule_stub(request, ordinal=ordinal))

    summary = _summary(
        source_requests=source_requests,
        rule_requests=rule_requests,
        stubs=stubs,
        skipped=skipped,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_rule_authoring" if stubs else "empty",
        "scope": (
            "Sanitized rule-authoring stubs for unresolved blind spots. These "
            "rows are deterministic-rule work items, not verifier evidence, "
            "and they intentionally omit labels, model answers, record indices, "
            "and target ranks from the adapter boundary."
        ),
        "source": {
            "queue_workflow": queue_report.get("workflow"),
            "queue_status": queue_report.get("status"),
            "queue_summary": queue_report.get("summary"),
        },
        "label_usage": {
            "labels_used_for_stub_selection": False,
            "labels_copied_to_stubs": False,
            "model_answers_copied_to_stubs": False,
            "record_indices_copied_to_stubs": False,
            "stubs_are_verifier_evidence": False,
            "candidate_results_require_promotion_gate": True,
        },
        "config": {
            "request_type": RULE_REQUEST_TYPE,
            "adapter_family": RULE_ADAPTER_FAMILY,
            "rule_family_aliases": dict(RULE_FAMILY_ALIASES),
        },
        "summary": summary,
        "rule_stubs": tuple(stubs),
        "skipped_rule_requests": tuple(skipped),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    queue_report_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    rule_stubs_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register unresolved rule stubs."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "unresolved-world-model-rule-stubs.json")
    stubs_path = Path(rule_stubs_path or output / "world-model-rule-stubs.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    queue_report = _load_json_object(queue_report_path)
    payload = build_unresolved_world_model_rule_stubs(
        queue_report=queue_report,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "queue_report": str(queue_report_path),
        "report": str(report_path),
        "rule_stubs": str(stubs_path),
        "artifact_manifest": str(manifest_path),
    }

    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(stubs_path, payload["rule_stubs"], compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "unresolved_world_model_rule_stubs": report_path,
            "world_model_rule_stubs": stubs_path,
            "unresolved_blind_spot_evidence_queue": Path(queue_report_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "source_rule_request_count": payload["summary"]["source_rule_request_count"],
            "rule_stub_count": payload["summary"]["rule_stub_count"],
            "skipped_rule_request_count": payload["summary"]["skipped_rule_request_count"],
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)

    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "source_rule_request_count": payload["summary"]["source_rule_request_count"],
                "rule_stub_count": payload["summary"]["rule_stub_count"],
                "skipped_rule_request_count": payload["summary"]["skipped_rule_request_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _rule_stub(request: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    source_family = str(request.get("rule_family") or "world_model_consistency")
    family = _normalize_rule_family(source_family)
    request_id = _request_id(request, ordinal=ordinal)
    metadata = _stub_metadata(request, source_family=source_family, normalized_family=family)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": request_id,
        "target_id": str(request.get("target_id") or ""),
        "request_type": RULE_REQUEST_TYPE,
        "rule_family": family,
        "rule_seed": _rule_seed(request, family=family),
        "rule_reason": _rule_reason(request),
        "required_inputs": _required_inputs(family),
        "question": str(request.get("question") or ""),
        "question_type": str(request.get("question_type") or ""),
        "gap_type": str(request.get("evidence_status") or request.get("mapping_decision") or ""),
        "priority": str(request.get("priority") or ""),
        "not_verifier_evidence": True,
        "metadata": metadata,
    }


def _stub_metadata(
    request: Mapping[str, Any],
    *,
    source_family: str,
    normalized_family: str,
) -> dict[str, Any]:
    source_request_id = str(
        request.get("source_request_id")
        or _mapping(request.get("metadata")).get("request_id")
        or ""
    )
    metadata = {
        "source_workflow": SOURCE_WORKFLOW,
        "source_queue_id": str(request.get("queue_id") or ""),
        "source_request_id": source_request_id,
        "source_rule_family": source_family,
        "normalized_rule_family": normalized_family,
        "adapter_family": str(request.get("adapter_family") or ""),
        "evidence_status": str(request.get("evidence_status") or ""),
        "mapping_decision": str(request.get("mapping_decision") or ""),
        "usage": str(request.get("usage") or ""),
        "priority_score": _optional_float(request.get("priority_score")),
        "requires_timestamp": bool(request.get("requires_timestamp")),
        "not_verifier_evidence": True,
        "candidate_results_require_promotion_gate": True,
    }
    return {key: value for key, value in metadata.items() if value not in ("", None)}


def _rule_seed(request: Mapping[str, Any], *, family: str) -> str:
    question = str(request.get("question") or "").strip()
    if family == "quantity_or_arithmetic":
        prefix = "Author a deterministic numeric or arithmetic check"
    elif family == "entity_disambiguation":
        prefix = "Author a deterministic entity-role disambiguation check"
    elif family == "temporal_consistency":
        prefix = "Author a timestamped temporal consistency check"
    elif family == "causal_or_procedural":
        prefix = "Author a deterministic causal or procedural consistency check"
    else:
        prefix = "Author a deterministic world-model consistency check"
    if question:
        return f"{prefix} for: {question}"
    return prefix


def _rule_reason(request: Mapping[str, Any]) -> str:
    evidence_status = str(request.get("evidence_status") or "unknown")
    mapping_decision = str(request.get("mapping_decision") or "unknown")
    return (
        "Unresolved blind-spot queue selected this target for deterministic "
        f"rule authoring: evidence_status={evidence_status}; "
        f"mapping_decision={mapping_decision}."
    )


def _summary(
    *,
    source_requests: Sequence[Mapping[str, Any]],
    rule_requests: Sequence[Mapping[str, Any]],
    stubs: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    family_counts = Counter(str(row.get("rule_family") or "") for row in stubs)
    source_family_counts = Counter(str(row.get("rule_family") or "") for row in rule_requests)
    priority_counts = Counter(str(row.get("priority") or "") for row in stubs)
    evidence_counts = Counter(str(row.get("gap_type") or "") for row in stubs)
    reserved_counts: Counter[str] = Counter()
    for request in rule_requests:
        for field in RESERVED_STUB_FIELDS:
            if field in request:
                reserved_counts[field] += 1
    return {
        "source_adapter_request_count": len(source_requests),
        "source_rule_request_count": len(rule_requests),
        "source_non_rule_request_count": len(source_requests) - len(rule_requests),
        "rule_stub_count": len(stubs),
        "skipped_rule_request_count": len(skipped),
        "target_count": len({str(row.get("target_id")) for row in stubs if str(row.get("target_id"))}),
        "rule_family_counts": _sorted_counter(family_counts),
        "source_rule_family_counts": _sorted_counter(source_family_counts),
        "priority_counts": _sorted_counter(priority_counts),
        "evidence_status_counts": _sorted_counter(evidence_counts),
        "reserved_source_field_counts": _sorted_counter(reserved_counts),
        "top_stub": None
        if not stubs
        else {
            "request_id": stubs[0]["request_id"],
            "target_id": stubs[0]["target_id"],
            "rule_family": stubs[0]["rule_family"],
            "priority": stubs[0]["priority"],
        },
    }


def _request_failures(request: Mapping[str, Any]) -> tuple[str, ...]:
    failures = []
    if request.get("not_verifier_evidence") is not True:
        failures.append("source_request_not_marked_non_evidence")
    if str(request.get("request_type") or "") != RULE_REQUEST_TYPE:
        failures.append("unsupported_request_type")
    if not str(request.get("question") or "").strip():
        failures.append("missing_question")
    if not _request_id(request, ordinal=0):
        failures.append("missing_request_id")
    return tuple(failures)


def _skip(
    request: Mapping[str, Any],
    *,
    ordinal: int,
    reason: str,
    failures: Sequence[str],
) -> dict[str, Any]:
    return {
        "source_request_id": _request_id(request, ordinal=ordinal),
        "target_id": str(request.get("target_id") or ""),
        "rule_family": _normalize_rule_family(str(request.get("rule_family") or "")),
        "reason": reason,
        "failures": tuple(str(item) for item in failures),
        "not_verifier_evidence": True,
    }


def _validate_queue_report(queue_report: Mapping[str, Any]) -> None:
    if queue_report.get("workflow") != SOURCE_WORKFLOW:
        raise ValueError(f"queue_report must be a {SOURCE_WORKFLOW} report.")
    if "adapter_requests" not in queue_report:
        raise ValueError("queue_report must include adapter_requests.")


def _request_id(request: Mapping[str, Any], *, ordinal: int) -> str:
    metadata = _mapping(request.get("metadata"))
    return str(
        metadata.get("request_id")
        or request.get("source_request_id")
        or request.get("queue_id")
        or (f"rule:unresolved:{ordinal}" if ordinal > 0 else "")
    )


def _normalize_rule_family(value: str) -> str:
    family = value.strip() or "world_model_consistency"
    return RULE_FAMILY_ALIASES.get(family, family)


def _required_inputs(family: str) -> tuple[str, ...]:
    return REQUIRED_INPUTS_BY_FAMILY.get(family, ("state", "action", "postcondition"))


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--rule-stubs-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        queue_report_path=args.queue_report,
        output_dir=args.output_dir,
        report_json_path=args.json,
        rule_stubs_path=args.rule_stubs_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "unresolved_world_model_rule_stubs_ok "
        f"status={payload['status']} "
        f"source_rules={summary['source_rule_request_count']} "
        f"stubs={summary['rule_stub_count']} "
        f"skipped={summary['skipped_rule_request_count']}"
    )


if __name__ == "__main__":
    main()
