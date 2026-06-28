"""Requeue audited world-model rule stubs into corrected rule families.

The rule-input audit emits non-evidence requeue suggestions when a typed input
task appears to use the wrong deterministic rule family. This workflow applies
those suggestions to the original sanitized rule stubs and materializes a new
stub file for the existing rule-authoring adapter. It still does not execute
rules, collect values, copy labels, or promote verifier evidence.
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

WORKFLOW = "world_model_rule_stub_requeue"
SOURCE_AUDIT_WORKFLOW = "world_model_rule_input_plan_audit"
SOURCE_STUB_WORKFLOW = "unresolved_world_model_rule_stubs"
RULE_REQUEST_TYPE = "world_model_or_calculator_rule"
RESERVED_STUB_FIELDS = {
    "answer",
    "answers",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "record_index",
    "score_label",
    "target_rank",
}
REQUIRED_INPUTS_BY_FAMILY = {
    "quantity_or_arithmetic": ("numeric_value", "unit", "reference_time"),
    "entity_disambiguation": ("subject_entity", "answer_entity", "requested_role"),
    "causal_or_procedural": ("mechanism", "precondition", "source_citation"),
    "temporal_consistency": ("claim_time", "source_time", "retrieved_at", "source_citation"),
}
RULE_SEED_PREFIX_BY_FAMILY = {
    "quantity_or_arithmetic": "Author a deterministic numeric or arithmetic check",
    "entity_disambiguation": "Author a deterministic entity-role disambiguation check",
    "causal_or_procedural": "Author a deterministic causal or procedural consistency check",
    "temporal_consistency": "Author a timestamped temporal consistency check",
}


def requeue_world_model_rule_stubs(
    *,
    rule_stubs: Sequence[Mapping[str, Any]],
    requeue_suggestions: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready report with requeued rule stubs."""
    stubs_by_id = {str(stub.get("request_id") or ""): _sanitize_stub(stub) for stub in rule_stubs}
    requeued: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()

    for ordinal, suggestion in enumerate(requeue_suggestions, start=1):
        sanitized = _sanitize_suggestion(suggestion)
        source_request_id = str(sanitized.get("source_request_id") or "")
        failures = _suggestion_failures(sanitized)
        source_stub = stubs_by_id.get(source_request_id)
        if source_stub is None:
            failures = (*failures, "source_stub_not_found")
        elif source_stub.get("not_verifier_evidence") is not True:
            failures = (*failures, "source_stub_not_marked_non_evidence")
        if source_request_id in seen_request_ids:
            failures = (*failures, "duplicate_requeue_suggestion")
        if failures:
            skipped.append(_skip(sanitized, ordinal=ordinal, reason="invalid_requeue_suggestion", failures=failures))
            continue
        assert source_stub is not None
        seen_request_ids.add(source_request_id)
        requeued.append(_requeued_stub(source_stub, sanitized))

    summary = _summary(
        source_stub_count=len(rule_stubs),
        suggestion_count=len(requeue_suggestions),
        requeued=requeued,
        skipped=skipped,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_rule_authoring" if requeued else "empty",
        "scope": (
            "Applies non-evidence audit requeue suggestions to sanitized rule stubs. "
            "The emitted stubs are replacement work items for the rule-authoring adapter; "
            "they are not verifier evidence and do not execute candidate checks."
        ),
        "source": {
            "rule_stub_workflow": SOURCE_STUB_WORKFLOW,
            "audit_workflow": SOURCE_AUDIT_WORKFLOW,
            "source_stub_count": len(rule_stubs),
            "requeue_suggestion_count": len(requeue_suggestions),
        },
        "label_usage": {
            "labels_used_for_requeue": False,
            "labels_copied_to_requeued_stubs": False,
            "model_answers_used_for_requeue": False,
            "model_answers_copied_to_requeued_stubs": False,
            "requeued_stubs_are_verifier_evidence": False,
            "candidate_results_require_promotion_gate": True,
        },
        "summary": summary,
        "requeued_rule_stubs": tuple(requeued),
        "skipped_requeue_suggestions": tuple(skipped),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    rule_stubs_path: str | Path,
    requeue_suggestions_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    requeued_stubs_path: str | Path | None = None,
    skipped_suggestions_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Requeue stubs, write artifacts, and optionally register the report."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "world-model-rule-stub-requeue.json")
    requeued_path = Path(requeued_stubs_path or output / "requeued-world-model-rule-stubs.jsonl")
    skipped_path = Path(skipped_suggestions_path or output / "skipped-requeue-suggestions.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    rule_stubs = _load_jsonl_mappings(rule_stubs_path, sanitizer=_sanitize_stub)
    requeue_suggestions = _load_jsonl_mappings(requeue_suggestions_path, sanitizer=_sanitize_suggestion)
    payload = requeue_world_model_rule_stubs(
        rule_stubs=rule_stubs,
        requeue_suggestions=requeue_suggestions,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "rule_stubs": str(rule_stubs_path),
        "requeue_suggestions": str(requeue_suggestions_path),
        "report": str(report_path),
        "requeued_rule_stubs": str(requeued_path),
        "skipped_requeue_suggestions": str(skipped_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(requeued_path, payload["requeued_rule_stubs"], compact=compact_json)
    _write_jsonl(skipped_path, payload["skipped_requeue_suggestions"], compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "world_model_rule_stub_requeue": report_path,
            "requeued_world_model_rule_stubs": requeued_path,
            "skipped_requeue_suggestions": skipped_path,
            "source_world_model_rule_stubs": Path(rule_stubs_path),
            "rule_input_requeue_suggestions": Path(requeue_suggestions_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "requeued_stub_count": payload["summary"]["requeued_stub_count"],
            "skipped_suggestion_count": payload["summary"]["skipped_suggestion_count"],
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
                "requeued_stub_count": payload["summary"]["requeued_stub_count"],
                "skipped_suggestion_count": payload["summary"]["skipped_suggestion_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _requeued_stub(source_stub: Mapping[str, Any], suggestion: Mapping[str, Any]) -> dict[str, Any]:
    family = str(suggestion.get("recommended_rule_family") or "")
    original_family = str(source_stub.get("rule_family") or "")
    question = str(suggestion.get("question") or source_stub.get("question") or "")
    request_id = str(source_stub.get("request_id") or suggestion.get("source_request_id") or "")
    metadata = dict(_mapping(source_stub.get("metadata")))
    metadata.update({
        "source_workflow": SOURCE_STUB_WORKFLOW,
        "requeue_workflow": WORKFLOW,
        "source_audit_workflow": SOURCE_AUDIT_WORKFLOW,
        "audit_task_id": str(suggestion.get("task_id") or ""),
        "original_rule_family": original_family,
        "current_rule_family": str(suggestion.get("current_rule_family") or original_family),
        "recommended_rule_family": family,
        "requeue_reason_codes": tuple(str(item) for item in _sequence(suggestion.get("reason_codes"))),
        "requeued_from_audit": True,
        "not_verifier_evidence": True,
        "candidate_results_require_promotion_gate": True,
    })
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": request_id,
        "target_id": str(source_stub.get("target_id") or suggestion.get("target_id") or ""),
        "request_type": str(source_stub.get("request_type") or RULE_REQUEST_TYPE),
        "rule_family": family,
        "rule_seed": _rule_seed(question=question, family=family),
        "rule_reason": _rule_reason(source_stub, suggestion),
        "required_inputs": _required_inputs(family),
        "question": question,
        "question_type": str(suggestion.get("question_type") or source_stub.get("question_type") or ""),
        "gap_type": str(source_stub.get("gap_type") or ""),
        "priority": str(source_stub.get("priority") or ""),
        "not_verifier_evidence": True,
        "metadata": {key: value for key, value in metadata.items() if value not in ("", None)},
    }


def _rule_seed(*, question: str, family: str) -> str:
    prefix = RULE_SEED_PREFIX_BY_FAMILY.get(family, "Author a deterministic world-model consistency check")
    return f"{prefix} for: {question}" if question else prefix


def _rule_reason(source_stub: Mapping[str, Any], suggestion: Mapping[str, Any]) -> str:
    original_reason = str(source_stub.get("rule_reason") or "").strip()
    reason_codes = ", ".join(str(item) for item in _sequence(suggestion.get("reason_codes")) if str(item))
    suffix = (
        "Rule-input audit recommended requeue before execution: "
        f"current_family={suggestion.get('current_rule_family')}; "
        f"recommended_family={suggestion.get('recommended_rule_family')}; "
        f"reason_codes={reason_codes or 'unspecified'}."
    )
    return f"{original_reason} {suffix}".strip() if original_reason else suffix


def _suggestion_failures(suggestion: Mapping[str, Any]) -> tuple[str, ...]:
    failures = []
    if suggestion.get("not_verifier_evidence") is not True:
        failures.append("suggestion_not_marked_non_evidence")
    if str(suggestion.get("workflow") or "") != SOURCE_AUDIT_WORKFLOW:
        failures.append("unsupported_suggestion_workflow")
    if str(suggestion.get("recommended_action") or "") != "requeue_rule_input_task":
        failures.append("unsupported_requeue_action")
    if not str(suggestion.get("source_request_id") or ""):
        failures.append("missing_source_request_id")
    recommended_family = str(suggestion.get("recommended_rule_family") or "")
    if recommended_family not in REQUIRED_INPUTS_BY_FAMILY:
        failures.append("unsupported_recommended_rule_family")
    if recommended_family == str(suggestion.get("current_rule_family") or ""):
        failures.append("recommended_family_matches_current_family")
    return tuple(failures)


def _skip(
    suggestion: Mapping[str, Any],
    *,
    ordinal: int,
    reason: str,
    failures: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "source_request_id": str(suggestion.get("source_request_id") or f"requeue-suggestion-{ordinal:04d}"),
        "target_id": str(suggestion.get("target_id") or ""),
        "current_rule_family": str(suggestion.get("current_rule_family") or ""),
        "recommended_rule_family": str(suggestion.get("recommended_rule_family") or ""),
        "reason": reason,
        "failures": tuple(str(item) for item in failures),
        "not_verifier_evidence": True,
    }


def _summary(
    *,
    source_stub_count: int,
    suggestion_count: int,
    requeued: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    family_counts = Counter(str(row.get("rule_family") or "") for row in requeued)
    original_family_counts = Counter(
        str(_mapping(row.get("metadata")).get("original_rule_family") or "") for row in requeued
    )
    skipped_counts = Counter(str(row.get("reason") or "") for row in skipped)
    failure_counts: Counter[str] = Counter()
    for row in skipped:
        for failure in _sequence(row.get("failures")):
            failure_counts[str(failure)] += 1
    return {
        "source_stub_count": int(source_stub_count),
        "requeue_suggestion_count": int(suggestion_count),
        "requeued_stub_count": len(requeued),
        "skipped_suggestion_count": len(skipped),
        "target_count": len({str(row.get("target_id")) for row in requeued if str(row.get("target_id"))}),
        "rule_family_counts": _sorted_counter(family_counts),
        "original_rule_family_counts": _sorted_counter(original_family_counts),
        "skipped_suggestion_counts": _sorted_counter(skipped_counts),
        "failure_counts": _sorted_counter(failure_counts),
        "top_requeued_stub": None
        if not requeued
        else {
            "request_id": requeued[0]["request_id"],
            "target_id": requeued[0]["target_id"],
            "rule_family": requeued[0]["rule_family"],
        },
    }


def _required_inputs(family: str) -> tuple[str, ...]:
    return REQUIRED_INPUTS_BY_FAMILY.get(family, ("state", "action", "postcondition"))


def _sanitize_stub(stub: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in stub.items() if str(key) not in RESERVED_STUB_FIELDS}


def _sanitize_suggestion(suggestion: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in suggestion.items() if str(key) not in RESERVED_STUB_FIELDS}


def _load_jsonl_mappings(
    path: str | Path,
    *,
    sanitizer: Any,
) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(sanitizer(dict(row)))
    return tuple(rows)


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


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return ()


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
    parser.add_argument("--rule-stubs", required=True)
    parser.add_argument("--requeue-suggestions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--requeued-stubs-jsonl", default=None)
    parser.add_argument("--skipped-suggestions-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        rule_stubs_path=args.rule_stubs,
        requeue_suggestions_path=args.requeue_suggestions,
        output_dir=args.output_dir,
        report_json_path=args.json,
        requeued_stubs_path=args.requeued_stubs_jsonl,
        skipped_suggestions_path=args.skipped_suggestions_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "world_model_rule_stub_requeue_ok "
        f"status={payload['status']} "
        f"requeued={summary['requeued_stub_count']} "
        f"skipped={summary['skipped_suggestion_count']}"
    )


if __name__ == "__main__":
    main()
