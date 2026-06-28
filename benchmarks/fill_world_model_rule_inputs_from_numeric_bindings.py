"""Fill numeric rule inputs from explicit source-backed bindings.

This workflow is a narrow bridge for calculator rule tasks. It does not infer
which entity a numeric claim is about and does not recover model answers from
upstream queue rows. Callers must provide JSONL binding rows that explicitly
name the subject, candidate numeric value, source-backed numeric value, unit,
reference time, and source citation. Bindings that require review remain
unfilled so ambiguous numeric claims do not become calculator evidence.
"""

from __future__ import annotations

import argparse
import json
import math
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

WORKFLOW = "world_model_rule_input_numeric_binding_fill"
SOURCE_WORKFLOW = "world_model_rule_input_collection_plan"
NUMERIC_COLLECTION_FAMILY = "numeric_rule_input_collection"
RESERVED_FIELDS = {"answer", "answers", "is_false", "label", "labels", "model_answer", "score_label"}
READY_REVIEW_STATUSES = {"", "ready", "approved"}


def fill_world_model_rule_inputs_from_numeric_bindings(
    *,
    input_tasks: Sequence[Mapping[str, Any]],
    numeric_bindings: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return filled calculator rule-input rows plus fail-closed diagnostics."""
    bindings_by_request: dict[str, Mapping[str, Any]] = {}
    duplicate_binding_ids: set[str] = set()
    for binding in numeric_bindings:
        request_id = str(binding.get("request_id") or binding.get("source_request_id") or "")
        if not request_id:
            continue
        if request_id in bindings_by_request:
            duplicate_binding_ids.add(request_id)
        else:
            bindings_by_request[request_id] = _sanitize(binding)

    filled: list[dict[str, Any]] = []
    unfilled: list[dict[str, Any]] = []
    unused_bindings = set(bindings_by_request)
    for task in input_tasks:
        request_id = str(task.get("source_request_id") or "")
        if str(task.get("collection_family") or "") != NUMERIC_COLLECTION_FAMILY:
            unfilled.append(_unfilled(task, reason="unsupported_collection_family"))
            continue
        binding = bindings_by_request.get(request_id)
        if binding is None:
            unfilled.append(_unfilled(task, reason="missing_numeric_binding"))
            continue
        unused_bindings.discard(request_id)
        failures = _binding_failures(binding, duplicate=request_id in duplicate_binding_ids)
        if failures:
            unfilled.append(_unfilled(task, reason="invalid_numeric_binding", failures=failures))
            continue
        filled.append(_filled_numeric_input(task, binding=binding))

    skipped_bindings = tuple(
        _skipped_binding(bindings_by_request[request_id], reason="no_matching_input_task")
        for request_id in sorted(unused_bindings)
    )
    summary = _summary(
        input_tasks=input_tasks,
        numeric_bindings=numeric_bindings,
        filled=filled,
        unfilled=unfilled,
        skipped_bindings=skipped_bindings,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Fills explicit numeric calculator rule inputs from caller-supplied "
            "source-backed bindings. Filled rows are adapter inputs, not verifier "
            "evidence, and still require adapter execution plus promotion before "
            "product handoff."
        ),
        "source": {
            "input_task_workflow": SOURCE_WORKFLOW,
            "input_task_count": len(input_tasks),
            "numeric_binding_count": len(numeric_bindings),
        },
        "label_usage": {
            "labels_used_for_input_fill": False,
            "labels_copied_to_rule_inputs": False,
            "candidate_numeric_values_bound_to_rule_inputs": True,
            "source_backed_numeric_values_required": True,
            "filled_inputs_are_verifier_evidence": False,
            "requires_adapter_execution_and_promotion_gate": True,
        },
        "summary": summary,
        "rule_inputs": tuple(filled),
        "unfilled_tasks": tuple(unfilled),
        "skipped_numeric_bindings": skipped_bindings,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    input_tasks_path: str | Path,
    numeric_bindings_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    rule_inputs_path: str | Path | None = None,
    unfilled_tasks_path: str | Path | None = None,
    skipped_bindings_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Fill, write, manifest, and optionally register numeric rule-input rows."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "rule-input-numeric-binding-fill.json")
    inputs_path = Path(rule_inputs_path or output / "rule-inputs.jsonl")
    unfilled_path = Path(unfilled_tasks_path or output / "unfilled-rule-input-tasks.jsonl")
    skipped_path = Path(skipped_bindings_path or output / "skipped-numeric-bindings.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    input_tasks = _load_jsonl_mappings(input_tasks_path)
    numeric_bindings = _load_jsonl_mappings(numeric_bindings_path)
    payload = fill_world_model_rule_inputs_from_numeric_bindings(
        input_tasks=input_tasks,
        numeric_bindings=numeric_bindings,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "input_tasks": str(input_tasks_path),
        "numeric_bindings": str(numeric_bindings_path),
        "report": str(report_path),
        "rule_inputs": str(inputs_path),
        "unfilled_tasks": str(unfilled_path),
        "skipped_numeric_bindings": str(skipped_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(inputs_path, payload["rule_inputs"], compact=compact_json)
    _write_jsonl(unfilled_path, payload["unfilled_tasks"], compact=compact_json)
    _write_jsonl(skipped_path, payload["skipped_numeric_bindings"], compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "rule_input_numeric_binding_fill": report_path,
            "rule_inputs": inputs_path,
            "unfilled_rule_input_tasks": unfilled_path,
            "skipped_numeric_bindings": skipped_path,
            "rule_input_tasks": Path(input_tasks_path),
            "numeric_bindings": Path(numeric_bindings_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "filled_input_count": payload["summary"]["filled_input_count"],
            "unfilled_task_count": payload["summary"]["unfilled_task_count"],
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
                "filled_input_count": payload["summary"]["filled_input_count"],
                "unfilled_task_count": payload["summary"]["unfilled_task_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _filled_numeric_input(task: Mapping[str, Any], *, binding: Mapping[str, Any]) -> dict[str, Any]:
    source_value = _required_float(binding.get("source_numeric_value", binding.get("numeric_value")))
    candidate_value = _required_float(binding.get("candidate_numeric_value"))
    calculation = _calculation_payload(binding, source_value=source_value, candidate_value=candidate_value)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": str(task.get("source_request_id") or binding.get("request_id") or ""),
        "target_id": str(task.get("target_id") or binding.get("target_id") or ""),
        "rule_family": str(task.get("rule_family") or "quantity_or_arithmetic"),
        "subject_entity": _clean(binding.get("subject_entity")),
        "numeric_value": source_value,
        "candidate_numeric_value": candidate_value,
        "unit": _clean(binding.get("unit")),
        "reference_time": _clean(binding.get("reference_time")),
        "source_citation": _clean(binding.get("source_citation")),
        "source_url": _clean(binding.get("source_url")),
        "source_title": _clean(binding.get("source_title")),
        "source_family": _clean(binding.get("source_family")),
        "provider": _clean(binding.get("provider")),
        "calculation": calculation,
        "not_verifier_evidence": True,
        "candidate_results_require_promotion_gate": True,
        "provenance": {
            "fill_source": WORKFLOW,
            "question": str(task.get("question") or ""),
            "binding_id": str(binding.get("binding_id") or ""),
            "candidate_value_source": str(binding.get("candidate_value_source") or ""),
            "source_value_source": str(binding.get("source_value_source") or ""),
            "review_status": str(binding.get("review_status") or ""),
            "source_note": str(binding.get("source_note") or ""),
        },
    }


def _calculation_payload(
    binding: Mapping[str, Any],
    *,
    source_value: float,
    candidate_value: float,
) -> dict[str, Any]:
    raw = binding.get("calculation")
    if isinstance(raw, Mapping):
        expression = _clean(raw.get("expression"))
        expected = _required_float(raw.get("expected", raw.get("result", raw.get("answer"))))
        default_tolerance = _optional_float(binding.get("tolerance"), default=0.0)
        tolerance = _optional_float(raw.get("tolerance"), default=default_tolerance)
        return {"expression": expression, "expected": expected, "tolerance": tolerance}
    tolerance = _optional_float(binding.get("tolerance"), default=0.0)
    return {
        "expression": f"({_format_number(source_value)}) - ({_format_number(candidate_value)})",
        "expected": 0.0,
        "tolerance": tolerance,
    }


def _binding_failures(binding: Mapping[str, Any], *, duplicate: bool) -> tuple[str, ...]:
    failures = []
    if duplicate:
        failures.append("duplicate_numeric_binding")
    if binding.get("not_verifier_evidence") is not True:
        failures.append("binding_not_marked_non_evidence")
    review_status = _clean(binding.get("review_status")).lower()
    if review_status not in READY_REVIEW_STATUSES:
        failures.append("binding_requires_review")
    for key in ("subject_entity", "unit", "reference_time", "source_citation"):
        if not _clean(binding.get(key)):
            failures.append(f"missing_{key}")
    for key in ("candidate_numeric_value",):
        if _float_or_none(binding.get(key)) is None:
            failures.append(f"missing_or_invalid_{key}")
    if _float_or_none(binding.get("source_numeric_value", binding.get("numeric_value"))) is None:
        failures.append("missing_or_invalid_source_numeric_value")
    raw = binding.get("calculation")
    if isinstance(raw, Mapping):
        if not _clean(raw.get("expression")):
            failures.append("missing_calculation_expression")
        if _float_or_none(raw.get("expected", raw.get("result", raw.get("answer")))) is None:
            failures.append("missing_or_invalid_calculation_expected")
        if "tolerance" in raw and _float_or_none(raw.get("tolerance")) is None:
            failures.append("invalid_calculation_tolerance")
    if "tolerance" in binding and _float_or_none(binding.get("tolerance")) is None:
        failures.append("invalid_tolerance")
    return tuple(failures)


def _unfilled(
    task: Mapping[str, Any],
    *,
    reason: str,
    failures: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id") or ""),
        "source_request_id": str(task.get("source_request_id") or ""),
        "target_id": str(task.get("target_id") or ""),
        "rule_family": str(task.get("rule_family") or ""),
        "collection_family": str(task.get("collection_family") or ""),
        "question": str(task.get("question") or ""),
        "reason": reason,
        "failures": tuple(str(item) for item in failures),
        "not_verifier_evidence": True,
    }


def _skipped_binding(binding: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "binding_id": str(binding.get("binding_id") or ""),
        "request_id": str(binding.get("request_id") or binding.get("source_request_id") or ""),
        "target_id": str(binding.get("target_id") or ""),
        "reason": reason,
        "not_verifier_evidence": True,
    }


def _summary(
    *,
    input_tasks: Sequence[Mapping[str, Any]],
    numeric_bindings: Sequence[Mapping[str, Any]],
    filled: Sequence[Mapping[str, Any]],
    unfilled: Sequence[Mapping[str, Any]],
    skipped_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    filled_family = Counter(str(item.get("rule_family") or "") for item in filled)
    unfilled_reason = Counter(str(item.get("reason") or "") for item in unfilled)
    failure_counts: Counter[str] = Counter()
    for item in unfilled:
        for failure in _sequence(item.get("failures")):
            failure_counts[str(failure)] += 1
    collection_family = Counter(str(item.get("collection_family") or "") for item in input_tasks)
    provider_counts = Counter(str(item.get("provider") or "") for item in filled)
    review_status_counts = Counter(str(item.get("review_status") or "") for item in numeric_bindings)
    return {
        "input_task_count": len(input_tasks),
        "numeric_binding_count": len(numeric_bindings),
        "filled_input_count": len(filled),
        "unfilled_task_count": len(unfilled),
        "skipped_binding_count": len(skipped_bindings),
        "filled_rule_family_counts": _sorted_counter(filled_family),
        "input_collection_family_counts": _sorted_counter(collection_family),
        "unfilled_reason_counts": _sorted_counter(unfilled_reason),
        "invalid_binding_failure_counts": _sorted_counter(failure_counts),
        "provider_counts": _sorted_counter(provider_counts),
        "review_status_counts": _sorted_counter(review_status_counts),
        "filled_request_ids": tuple(str(item.get("request_id") or "") for item in filled),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("input_task_count", 0)) == 0:
        return "empty"
    if int(summary.get("filled_input_count", 0)) == 0:
        return "blocked"
    if int(summary.get("unfilled_task_count", 0)) > 0:
        return "partial"
    return "filled"


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(_sanitize(dict(row)))
    return tuple(rows)


def _sanitize(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if str(key) not in RESERVED_FIELDS}


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


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return ()


def _required_float(value: Any) -> float:
    parsed = _float_or_none(value)
    if parsed is None:
        raise ValueError("expected a finite numeric value.")
    return parsed


def _optional_float(value: Any, *, default: float) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _format_number(value: float) -> str:
    return format(value, ".17g")


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
    parser.add_argument("--input-tasks", required=True)
    parser.add_argument("--numeric-bindings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--rule-inputs-jsonl", default=None)
    parser.add_argument("--unfilled-tasks-jsonl", default=None)
    parser.add_argument("--skipped-bindings-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        input_tasks_path=args.input_tasks,
        numeric_bindings_path=args.numeric_bindings,
        output_dir=args.output_dir,
        report_json_path=args.json,
        rule_inputs_path=args.rule_inputs_jsonl,
        unfilled_tasks_path=args.unfilled_tasks_jsonl,
        skipped_bindings_path=args.skipped_bindings_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "world_model_rule_input_numeric_binding_fill_ok "
        f"status={payload['status']} "
        f"filled={summary['filled_input_count']} "
        f"unfilled={summary['unfilled_task_count']}"
    )


if __name__ == "__main__":
    main()
