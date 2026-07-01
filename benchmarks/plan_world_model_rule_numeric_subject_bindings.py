"""Plan source-backed subject bindings for blocked numeric rule inputs.

This planner starts from a numeric binding fill report that failed closed on a
missing or ambiguous subject. It does not infer the subject entity and does not
turn numeric bindings into verifier evidence. Instead, it emits a small
adapter-ready JSONL worklist for collecting reviewed subject-binding sidecars
that can later be passed to
``fill_world_model_rule_inputs_from_numeric_bindings.py --subject-bindings``.
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

WORKFLOW = "world_model_rule_numeric_subject_binding_plan"
SOURCE_WORKFLOW = "world_model_rule_input_numeric_binding_fill"
COLLECTION_FAMILY = "numeric_subject_binding_collection"
READY_REVIEW_STATUSES = {"", "ready", "approved"}
SUBJECT_RESOLVABLE_REVIEW_STATUSES = {"ambiguous_subject", "subject_ambiguous"}
RESERVED_FIELDS = {"answer", "answers", "is_false", "label", "labels", "model_answer", "score_label"}


def build_world_model_rule_numeric_subject_binding_plan(
    *,
    fill_report: Mapping[str, Any],
    numeric_bindings: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready subject-binding collection plan."""
    bindings_by_request: dict[str, Mapping[str, Any]] = {}
    duplicate_request_ids: set[str] = set()
    for binding in numeric_bindings:
        request_id = str(binding.get("request_id") or binding.get("source_request_id") or "")
        if not request_id:
            continue
        if request_id in bindings_by_request:
            duplicate_request_ids.add(request_id)
        else:
            bindings_by_request[request_id] = _sanitize(binding)

    requests: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for task in _sequence(fill_report.get("unfilled_tasks")):
        if not isinstance(task, Mapping):
            continue
        failures = tuple(str(item) for item in _sequence(task.get("failures")) if str(item))
        request_id = str(task.get("source_request_id") or "")
        if "missing_subject_entity" not in failures:
            skipped.append(_skipped_task(task, reason="not_a_subject_binding_gap", failures=failures))
            continue
        binding = bindings_by_request.get(request_id)
        if binding is None:
            skipped.append(_skipped_task(task, reason="missing_numeric_binding", failures=failures))
            continue
        if request_id in duplicate_request_ids:
            skipped.append(_skipped_task(task, reason="duplicate_numeric_binding", failures=failures))
            continue
        review_status = _clean(binding.get("review_status")).lower()
        if review_status not in READY_REVIEW_STATUSES and review_status not in SUBJECT_RESOLVABLE_REVIEW_STATUSES:
            skipped.append(
                _skipped_task(
                    task,
                    reason="numeric_review_status_not_subject_resolvable",
                    failures=failures,
                    numeric_binding=binding,
                )
            )
            continue
        requests.append(_subject_binding_request(task, binding=binding, failures=failures))

    summary = _summary(
        fill_report=fill_report,
        numeric_bindings=numeric_bindings,
        requests=requests,
        skipped=skipped,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_collection" if requests else "empty",
        "scope": (
            "Plans reviewed subject-binding sidecars for numeric rule inputs "
            "that failed closed on missing subject entities. Requests are "
            "collection tasks only, not verifier evidence."
        ),
        "source": {
            "fill_report_workflow": fill_report.get("workflow"),
            "fill_report_status": fill_report.get("status"),
            "expected_fill_report_workflow": SOURCE_WORKFLOW,
            "numeric_binding_count": len(numeric_bindings),
        },
        "label_usage": {
            "labels_used_for_collection_planning": False,
            "labels_copied_to_subject_binding_requests": False,
            "requests_are_verifier_evidence": False,
            "subject_binding_may_unblock_numeric_fill": True,
        },
        "summary": summary,
        "subject_binding_requests": tuple(requests),
        "skipped_tasks": tuple(skipped),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    fill_report_path: str | Path,
    numeric_bindings_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    requests_path: str | Path | None = None,
    skipped_tasks_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, optionally manifest, and optionally register a plan."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "numeric-subject-binding-plan.json")
    request_rows_path = Path(requests_path or output / "subject-binding-requests.jsonl")
    skipped_path = Path(skipped_tasks_path or output / "skipped-subject-binding-tasks.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")
    fill_report = _load_json_object(fill_report_path)
    numeric_bindings = _load_jsonl_mappings(numeric_bindings_path)
    payload = build_world_model_rule_numeric_subject_binding_plan(
        fill_report=fill_report,
        numeric_bindings=numeric_bindings,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "fill_report": str(fill_report_path),
        "numeric_bindings": str(numeric_bindings_path),
        "report": str(report_path),
        "subject_binding_requests": str(request_rows_path),
        "skipped_tasks": str(skipped_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(request_rows_path, payload["subject_binding_requests"], compact=compact_json)
    _write_jsonl(skipped_path, payload["skipped_tasks"], compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "numeric_subject_binding_plan": report_path,
            "subject_binding_requests": request_rows_path,
            "skipped_subject_binding_tasks": skipped_path,
            "numeric_binding_fill_report": Path(fill_report_path),
            "numeric_bindings": Path(numeric_bindings_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "request_count": payload["summary"]["request_count"],
            "skipped_task_count": payload["summary"]["skipped_task_count"],
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
                "request_count": payload["summary"]["request_count"],
                "skipped_task_count": payload["summary"]["skipped_task_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _subject_binding_request(
    task: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    failures: Sequence[str],
) -> dict[str, Any]:
    request_id = str(task.get("source_request_id") or binding.get("request_id") or "")
    review_status = _clean(binding.get("review_status")).lower()
    additional_failures = tuple(
        failure
        for failure in failures
        if failure not in {"missing_subject_entity", "binding_requires_review"}
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": f"subject-binding:{request_id}",
        "source_request_id": request_id,
        "target_id": str(task.get("target_id") or binding.get("target_id") or ""),
        "task_id": str(task.get("task_id") or ""),
        "collection_family": COLLECTION_FAMILY,
        "question": str(task.get("question") or ""),
        "numeric_binding_id": str(binding.get("binding_id") or ""),
        "numeric_binding_review_status": review_status,
        "numeric_binding_source": {
            "source_citation": _clean(binding.get("source_citation")),
            "source_url": _clean(binding.get("source_url")),
            "source_title": _clean(binding.get("source_title")),
            "source_family": _clean(binding.get("source_family")),
            "provider": _clean(binding.get("provider")),
            "unit": _clean(binding.get("unit")),
            "reference_time": _clean(binding.get("reference_time")),
        },
        "numeric_context": {
            "source_numeric_value": binding.get("source_numeric_value", binding.get("numeric_value")),
            "candidate_numeric_value": binding.get("candidate_numeric_value"),
            "calculation_expression": _calculation_expression(binding),
        },
        "required_subject_binding_fields": (
            "request_id",
            "target_id",
            "subject_entity",
            "source_citation",
            "review_status",
            "not_verifier_evidence",
        ),
        "recommended_subject_binding_skeleton": {
            "request_id": request_id,
            "target_id": str(task.get("target_id") or binding.get("target_id") or ""),
            "subject_entity": "",
            "source_citation": "",
            "review_status": "approved",
            "not_verifier_evidence": True,
        },
        "blocking_failures": tuple(failures),
        "additional_numeric_failures": additional_failures,
        "subject_binding_unblocks_numeric_fill": not additional_failures,
        "not_verifier_evidence": True,
    }


def _skipped_task(
    task: Mapping[str, Any],
    *,
    reason: str,
    failures: Sequence[str],
    numeric_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = numeric_binding or {}
    return {
        "source_request_id": str(task.get("source_request_id") or ""),
        "target_id": str(task.get("target_id") or binding.get("target_id") or ""),
        "task_id": str(task.get("task_id") or ""),
        "reason": reason,
        "failures": tuple(str(item) for item in failures),
        "numeric_binding_id": str(binding.get("binding_id") or ""),
        "numeric_binding_review_status": _clean(binding.get("review_status")).lower(),
        "not_verifier_evidence": True,
    }


def _summary(
    *,
    fill_report: Mapping[str, Any],
    numeric_bindings: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    skipped_reason_counts = Counter(str(item.get("reason") or "") for item in skipped)
    review_status_counts = Counter(str(item.get("numeric_binding_review_status") or "") for item in requests)
    source_family_counts = Counter(
        str(_mapping(item.get("numeric_binding_source")).get("source_family") or "") for item in requests
    )
    additional_blocker_counts: Counter[str] = Counter()
    for item in requests:
        for failure in _sequence(item.get("additional_numeric_failures")):
            additional_blocker_counts[str(failure)] += 1
    request_count = len(requests)
    return {
        "fill_report_status": str(fill_report.get("status") or ""),
        "numeric_binding_count": len(numeric_bindings),
        "unfilled_task_count": len(_sequence(fill_report.get("unfilled_tasks"))),
        "request_count": request_count,
        "skipped_task_count": len(skipped),
        "fully_unblocking_request_count": sum(
            1 for item in requests if bool(item.get("subject_binding_unblocks_numeric_fill"))
        ),
        "partial_request_count": sum(
            1 for item in requests if not bool(item.get("subject_binding_unblocks_numeric_fill"))
        ),
        "skipped_reason_counts": _sorted_counter(skipped_reason_counts),
        "numeric_review_status_counts": _sorted_counter(review_status_counts),
        "source_family_counts": _sorted_counter(source_family_counts),
        "additional_numeric_failure_counts": _sorted_counter(additional_blocker_counts),
    }


def _calculation_expression(binding: Mapping[str, Any]) -> str:
    raw = binding.get("calculation")
    if isinstance(raw, Mapping):
        return _clean(raw.get("expression"))
    return ""


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(_sanitize(row))
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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


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
    parser.add_argument("--fill-report", required=True)
    parser.add_argument("--numeric-bindings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--requests-jsonl", default=None)
    parser.add_argument("--skipped-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        fill_report_path=args.fill_report,
        numeric_bindings_path=args.numeric_bindings,
        output_dir=args.output_dir,
        report_json_path=args.json,
        requests_path=args.requests_jsonl,
        skipped_tasks_path=args.skipped_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "world_model_rule_numeric_subject_binding_plan_ok "
        f"status={payload['status']} "
        f"requests={summary['request_count']} "
        f"skipped={summary['skipped_task_count']}"
    )


if __name__ == "__main__":
    main()
