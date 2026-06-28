"""Promotion gate for deterministic world-model rule candidate results.

Rule adapter outputs are candidate diagnostics until this gate validates that an
executed result is backed by explicit rule inputs and provenance. Pending
``needs_inputs`` rows are not failures; they remain queued for later fills.
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

WORKFLOW = "world_model_rule_candidate_promotion_gate"
SOURCE_WORKFLOW = "world_model_rule_authoring_adapter"
EXECUTED_STATUSES = {"supported", "refuted", "insufficient_evidence", "error"}
PROMOTABLE_STATUSES = {"supported", "refuted"}


def promote_world_model_rule_candidates(
    *,
    rule_results: Sequence[Mapping[str, Any]],
    rule_inputs: Sequence[Mapping[str, Any]],
    adapter_report: Mapping[str, Any] | None = None,
    min_confidence: float = 0.90,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return promoted candidates and fail-closed gate diagnostics."""
    if not (0.0 <= float(min_confidence) <= 1.0):
        raise ValueError("min_confidence must be between 0 and 1.")
    inputs_by_request = {
        str(row.get("request_id") or ""): row
        for row in rule_inputs
        if str(row.get("request_id") or "")
    }
    promoted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for result in rule_results:
        status = str(result.get("status") or "")
        request_id = str(result.get("request_id") or "")
        if status == "needs_inputs":
            pending.append(_pending(result))
            continue
        if status not in EXECUTED_STATUSES:
            blocked.append(_blocked(result, reason="unknown_status"))
            continue
        failures = _candidate_failures(
            result,
            rule_input=inputs_by_request.get(request_id),
            min_confidence=float(min_confidence),
        )
        if failures:
            blocked.append(_blocked(result, reason="gate_failed", failures=failures))
            continue
        promoted.append(_promoted_candidate(result, rule_input=inputs_by_request[request_id]))

    summary = _summary(
        rule_results=rule_results,
        promoted=promoted,
        blocked=blocked,
        pending=pending,
        adapter_report=adapter_report,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Promotion gate for deterministic rule candidate results. Promoted "
            "rows are provenance-backed rule candidates for downstream product "
            "handoff; pending input rows remain non-evidence work items."
        ),
        "source": {
            "adapter_workflow": None if adapter_report is None else adapter_report.get("workflow"),
            "adapter_status": None if adapter_report is None else adapter_report.get("status"),
            "adapter_summary": None if adapter_report is None else adapter_report.get("summary"),
            "rule_result_count": len(rule_results),
            "rule_input_count": len(rule_inputs),
        },
        "label_usage": {
            "labels_used_for_promotion": False,
            "labels_copied_to_promoted_candidates": False,
            "requires_explicit_rule_inputs": True,
            "requires_source_citation": True,
            "pending_rows_are_verifier_evidence": False,
        },
        "config": {
            "min_confidence": float(min_confidence),
            "promotable_statuses": tuple(sorted(PROMOTABLE_STATUSES)),
        },
        "summary": summary,
        "promoted_candidates": tuple(promoted),
        "blocked_candidates": tuple(blocked),
        "pending_inputs": tuple(pending),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    rule_results_path: str | Path,
    rule_inputs_path: str | Path,
    output_dir: str | Path,
    adapter_report_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
    promoted_jsonl_path: str | Path | None = None,
    blocked_jsonl_path: str | Path | None = None,
    pending_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    min_confidence: float = 0.90,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Promote, write, manifest, and optionally register rule candidates."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "world-model-rule-candidate-promotion-gate.json")
    promoted_path = Path(promoted_jsonl_path or output / "promoted-rule-candidates.jsonl")
    blocked_path = Path(blocked_jsonl_path or output / "blocked-rule-candidates.jsonl")
    pending_path = Path(pending_jsonl_path or output / "pending-rule-inputs.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    rule_results = _load_jsonl_mappings(rule_results_path)
    rule_inputs = _load_jsonl_mappings(rule_inputs_path)
    adapter_report = _load_json_object(adapter_report_path) if adapter_report_path is not None else None
    payload = promote_world_model_rule_candidates(
        rule_results=rule_results,
        rule_inputs=rule_inputs,
        adapter_report=adapter_report,
        min_confidence=min_confidence,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "rule_results": str(rule_results_path),
        "rule_inputs": str(rule_inputs_path),
        "adapter_report": None if adapter_report_path is None else str(adapter_report_path),
        "report": str(report_path),
        "promoted_candidates": str(promoted_path),
        "blocked_candidates": str(blocked_path),
        "pending_inputs": str(pending_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(promoted_path, payload["promoted_candidates"], compact=compact_json)
    _write_jsonl(blocked_path, payload["blocked_candidates"], compact=compact_json)
    _write_jsonl(pending_path, payload["pending_inputs"], compact=compact_json)

    artifacts: dict[str, str | Path | None] = {
        "rule_candidate_promotion_gate": report_path,
        "promoted_rule_candidates": promoted_path,
        "blocked_rule_candidates": blocked_path,
        "pending_rule_inputs": pending_path,
        "rule_results": Path(rule_results_path),
        "rule_inputs": Path(rule_inputs_path),
    }
    if adapter_report_path is not None:
        artifacts["adapter_report"] = Path(adapter_report_path)
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "promoted_count": payload["summary"]["promoted_count"],
            "blocked_count": payload["summary"]["blocked_count"],
            "pending_count": payload["summary"]["pending_count"],
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
                "promoted_count": payload["summary"]["promoted_count"],
                "blocked_count": payload["summary"]["blocked_count"],
                "pending_count": payload["summary"]["pending_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _candidate_failures(
    result: Mapping[str, Any],
    *,
    rule_input: Mapping[str, Any] | None,
    min_confidence: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    status = str(result.get("status") or "")
    if status not in PROMOTABLE_STATUSES:
        failures.append("status_not_promotable")
    if rule_input is None:
        failures.append("missing_rule_input")
    if result.get("not_verifier_evidence") is not True:
        failures.append("candidate_missing_non_evidence_marker")
    if _mapping(result.get("metadata")).get("candidate_results_require_promotion_gate") is not True:
        failures.append("candidate_missing_promotion_gate_marker")
    if _float(result.get("confidence")) is None or (_float(result.get("confidence")) or 0.0) < min_confidence:
        failures.append("confidence_below_minimum")
    if _sequence(result.get("missing_inputs")):
        failures.append("candidate_has_missing_inputs")
    if not _sequence(result.get("evidence")):
        failures.append("missing_candidate_evidence")
    if rule_input is not None:
        if rule_input.get("not_verifier_evidence") is not True:
            failures.append("rule_input_missing_non_evidence_marker")
        if rule_input.get("candidate_results_require_promotion_gate") is not True:
            failures.append("rule_input_missing_promotion_gate_marker")
        source_citation = str(rule_input.get("source_citation") or "")
        if not source_citation:
            failures.append("missing_source_citation")
        elif source_citation not in " ".join(str(item) for item in _sequence(result.get("evidence"))):
            failures.append("source_citation_not_in_candidate_evidence")
        if str(rule_input.get("request_id") or "") != str(result.get("request_id") or ""):
            failures.append("request_id_mismatch")
        if str(rule_input.get("rule_family") or "") != str(result.get("rule_family") or ""):
            failures.append("rule_family_mismatch")
        if str(result.get("rule_family") or "") == "entity_disambiguation":
            for key in ("subject_entity", "answer_entity", "expected_entity", "requested_role"):
                if not str(rule_input.get(key) or "").strip():
                    failures.append(f"missing_{key}")
        if str(result.get("rule_family") or "") == "temporal_consistency":
            for key in ("claim_time", "source_time", "retrieved_at"):
                if not str(rule_input.get(key) or "").strip():
                    failures.append(f"missing_{key}")
    return tuple(failures)


def _promoted_candidate(result: Mapping[str, Any], *, rule_input: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": str(result.get("request_id") or ""),
        "target_id": str(result.get("target_id") or ""),
        "rule_family": str(result.get("rule_family") or ""),
        "status": str(result.get("status") or ""),
        "confidence": _float(result.get("confidence")),
        "adapter": str(
            _mapping(result.get("authored_rule")).get("adapter")
            or _mapping(result.get("metadata")).get("adapter")
            or ""
        ),
        "question": str(result.get("question") or ""),
        "source_citation": str(rule_input.get("source_citation") or ""),
        "source_url": str(rule_input.get("source_url") or ""),
        "evidence": tuple(str(item) for item in _sequence(result.get("evidence"))),
        "rule_input": {
            key: rule_input.get(key)
            for key in (
                "subject_entity",
                "answer_entity",
                "expected_entity",
                "requested_role",
                "numeric_value",
                "candidate_numeric_value",
                "unit",
                "reference_time",
                "calculation",
                "claim_time",
                "source_time",
                "retrieved_at",
                "temporal_relation",
                "source_fact_type",
                "source_family",
                "provider",
            )
            if rule_input.get(key) not in (None, "")
        },
        "promotion": {
            "status": "promote",
            "gate": WORKFLOW,
            "candidate_only_requires_downstream_handoff": True,
        },
    }


def _blocked(result: Mapping[str, Any], *, reason: str, failures: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "request_id": str(result.get("request_id") or ""),
        "target_id": str(result.get("target_id") or ""),
        "rule_family": str(result.get("rule_family") or ""),
        "status": str(result.get("status") or ""),
        "reason": reason,
        "failures": tuple(str(item) for item in failures),
    }


def _pending(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(result.get("request_id") or ""),
        "target_id": str(result.get("target_id") or ""),
        "rule_family": str(result.get("rule_family") or ""),
        "missing_inputs": tuple(str(item) for item in _sequence(result.get("missing_inputs"))),
        "not_verifier_evidence": True,
    }


def _summary(
    *,
    rule_results: Sequence[Mapping[str, Any]],
    promoted: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    pending: Sequence[Mapping[str, Any]],
    adapter_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status") or "") for row in rule_results)
    family_counts = Counter(str(row.get("rule_family") or "") for row in promoted)
    blocked_reasons = Counter(str(row.get("reason") or "") for row in blocked)
    return {
        "rule_result_count": len(rule_results),
        "executed_count": sum(count for status, count in status_counts.items() if status in EXECUTED_STATUSES),
        "promoted_count": len(promoted),
        "blocked_count": len(blocked),
        "pending_count": len(pending),
        "status_counts": _sorted_counter(status_counts),
        "promoted_rule_family_counts": _sorted_counter(family_counts),
        "blocked_reason_counts": _sorted_counter(blocked_reasons),
        "promoted_request_ids": tuple(str(row.get("request_id") or "") for row in promoted),
        "adapter_report_status": None if adapter_report is None else adapter_report.get("status"),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("promoted_count", 0)) == 0 and int(summary.get("blocked_count", 0)) == 0:
        return "empty" if int(summary.get("rule_result_count", 0)) == 0 else "blocked"
    if int(summary.get("blocked_count", 0)) > 0:
        return "blocked"
    return "promote" if int(summary.get("promoted_count", 0)) > 0 else "blocked"


def _load_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(dict(row))
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


def _float(value: Any) -> float | None:
    try:
        return float(value)
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
    parser.add_argument("--rule-results", required=True)
    parser.add_argument("--rule-inputs", required=True)
    parser.add_argument("--adapter-report", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--promoted-jsonl", default=None)
    parser.add_argument("--blocked-jsonl", default=None)
    parser.add_argument("--pending-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        rule_results_path=args.rule_results,
        rule_inputs_path=args.rule_inputs,
        adapter_report_path=args.adapter_report,
        output_dir=args.output_dir,
        report_json_path=args.json,
        promoted_jsonl_path=args.promoted_jsonl,
        blocked_jsonl_path=args.blocked_jsonl,
        pending_jsonl_path=args.pending_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        min_confidence=args.min_confidence,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "world_model_rule_candidate_promotion_gate_ok "
        f"status={payload['status']} "
        f"promoted={summary['promoted_count']} "
        f"blocked={summary['blocked_count']} "
        f"pending={summary['pending_count']}"
    )


if __name__ == "__main__":
    main()
