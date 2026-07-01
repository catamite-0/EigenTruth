"""Plan source-backed temporal bindings for blocked temporal rule inputs.

This planner starts from a temporal binding fill report that failed closed on
missing, incomplete, invalid, or unreviewed timestamp inputs. It does not infer
claim dates, source dates, retrieval dates, or citations. Instead, it emits an
adapter-ready JSONL worklist for collecting reviewed temporal-binding sidecars
that can later be passed to
``fill_world_model_rule_inputs_from_temporal_bindings.py --temporal-bindings``.
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

WORKFLOW = "world_model_rule_temporal_binding_plan"
SOURCE_WORKFLOW = "world_model_rule_input_temporal_binding_fill"
COLLECTION_FAMILY = "temporal_binding_collection"
RESERVED_FIELDS = {"answer", "answers", "is_false", "label", "labels", "model_answer", "score_label"}
TEMPORAL_BINDING_FAILURES = {
    "missing_temporal_binding",
    "invalid_temporal_binding",
    "missing_claim_time",
    "missing_source_time",
    "missing_retrieved_at",
    "missing_source_citation",
    "invalid_claim_time",
    "invalid_source_time",
    "invalid_retrieved_at",
    "binding_requires_review",
    "binding_not_marked_non_evidence",
}


def build_world_model_rule_temporal_binding_plan(
    *,
    fill_report: Mapping[str, Any],
    temporal_bindings: Sequence[Mapping[str, Any]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready temporal-binding collection plan."""
    bindings_by_request: dict[str, Mapping[str, Any]] = {}
    duplicate_request_ids: set[str] = set()
    for binding in temporal_bindings:
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
        reason = str(task.get("reason") or "")
        if not _is_temporal_binding_gap(reason=reason, failures=failures):
            skipped.append(_skipped_task(task, reason="not_a_temporal_binding_gap", failures=failures))
            continue
        request_id = str(task.get("source_request_id") or "")
        binding = bindings_by_request.get(request_id)
        if request_id in duplicate_request_ids:
            skipped.append(
                _skipped_task(
                    task,
                    reason="duplicate_temporal_binding",
                    failures=failures,
                    temporal_binding=binding,
                )
            )
            continue
        requests.append(
            _temporal_binding_request(
                task,
                reason=reason,
                failures=failures,
                binding=binding,
            )
        )

    summary = _summary(
        fill_report=fill_report,
        temporal_bindings=temporal_bindings,
        requests=requests,
        skipped=skipped,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready_for_collection" if requests else "empty",
        "scope": (
            "Plans reviewed temporal-binding sidecars for temporal consistency "
            "rule inputs that failed closed on missing, invalid, incomplete, or "
            "unreviewed timestamp/citation metadata. Requests are collection "
            "tasks only, not verifier evidence."
        ),
        "source": {
            "fill_report_workflow": fill_report.get("workflow"),
            "fill_report_status": fill_report.get("status"),
            "expected_fill_report_workflow": SOURCE_WORKFLOW,
            "temporal_binding_count": len(temporal_bindings),
        },
        "label_usage": {
            "labels_used_for_collection_planning": False,
            "labels_copied_to_temporal_binding_requests": False,
            "requests_are_verifier_evidence": False,
            "temporal_binding_may_unblock_temporal_fill": True,
        },
        "summary": summary,
        "temporal_binding_requests": tuple(requests),
        "skipped_tasks": tuple(skipped),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    fill_report_path: str | Path,
    output_dir: str | Path,
    temporal_bindings_path: str | Path | None = None,
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
    """Build, write, manifest, and optionally register a temporal binding plan."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "temporal-binding-plan.json")
    request_rows_path = Path(requests_path or output / "temporal-binding-requests.jsonl")
    skipped_path = Path(skipped_tasks_path or output / "skipped-temporal-binding-tasks.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")
    fill_report = _load_json_object(fill_report_path)
    temporal_bindings = () if temporal_bindings_path is None else _load_jsonl_mappings(temporal_bindings_path)
    payload = build_world_model_rule_temporal_binding_plan(
        fill_report=fill_report,
        temporal_bindings=temporal_bindings,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "fill_report": str(fill_report_path),
        "temporal_bindings": None if temporal_bindings_path is None else str(temporal_bindings_path),
        "report": str(report_path),
        "temporal_binding_requests": str(request_rows_path),
        "skipped_tasks": str(skipped_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(request_rows_path, payload["temporal_binding_requests"], compact=compact_json)
    _write_jsonl(skipped_path, payload["skipped_tasks"], compact=compact_json)
    manifest_artifacts: dict[str, str | Path | None] = {
        "temporal_binding_plan": report_path,
        "temporal_binding_requests": request_rows_path,
        "skipped_temporal_binding_tasks": skipped_path,
        "temporal_binding_fill_report": Path(fill_report_path),
    }
    if temporal_bindings_path is not None:
        manifest_artifacts["temporal_bindings"] = Path(temporal_bindings_path)
    manifest = build_artifact_manifest(
        manifest_artifacts,
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


def _temporal_binding_request(
    task: Mapping[str, Any],
    *,
    reason: str,
    failures: Sequence[str],
    binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = binding or {}
    request_id = str(task.get("source_request_id") or source.get("request_id") or "")
    target_id = str(task.get("target_id") or source.get("target_id") or "")
    existing_values = _temporal_source(source)
    missing_fields = tuple(
        field
        for field in ("claim_time", "source_time", "retrieved_at", "source_citation")
        if not existing_values[field]
    )
    invalid_fields = tuple(
        failure.removeprefix("invalid_")
        for failure in failures
        if failure in {"invalid_claim_time", "invalid_source_time", "invalid_retrieved_at"}
    )
    review_failures = tuple(
        failure
        for failure in failures
        if failure in {"binding_requires_review", "binding_not_marked_non_evidence"}
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "request_id": f"temporal-binding:{request_id}",
        "source_request_id": request_id,
        "target_id": target_id,
        "task_id": str(task.get("task_id") or ""),
        "collection_family": COLLECTION_FAMILY,
        "question": str(task.get("question") or ""),
        "temporal_binding_id": str(source.get("binding_id") or ""),
        "temporal_binding_review_status": _clean(source.get("review_status")).lower(),
        "existing_temporal_binding_source": existing_values,
        "required_temporal_binding_fields": (
            "request_id",
            "target_id",
            "claim_time",
            "source_time",
            "retrieved_at",
            "source_citation",
            "review_status",
            "not_verifier_evidence",
        ),
        "recommended_temporal_binding_skeleton": {
            "request_id": request_id,
            "target_id": target_id,
            "claim_time": existing_values["claim_time"],
            "source_time": existing_values["source_time"],
            "retrieved_at": existing_values["retrieved_at"],
            "source_citation": existing_values["source_citation"],
            "source_url": existing_values["source_url"],
            "source_title": existing_values["source_title"],
            "source_family": existing_values["source_family"],
            "provider": existing_values["provider"],
            "review_status": "approved",
            "not_verifier_evidence": True,
        },
        "source_task_reason": reason,
        "blocking_failures": tuple(failures),
        "missing_temporal_fields": missing_fields,
        "invalid_temporal_fields": invalid_fields,
        "review_failures": review_failures,
        "temporal_binding_unblocks_fill": not missing_fields and not invalid_fields and not review_failures,
        "not_verifier_evidence": True,
    }


def _temporal_source(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "claim_time": _clean(binding.get("claim_time")),
        "source_time": _clean(binding.get("source_time")),
        "retrieved_at": _clean(binding.get("retrieved_at")),
        "source_citation": _clean(binding.get("source_citation")),
        "source_url": _clean(binding.get("source_url")),
        "source_title": _clean(binding.get("source_title")),
        "source_family": _clean(binding.get("source_family")),
        "provider": _clean(binding.get("provider")),
        "temporal_relation": _clean(binding.get("temporal_relation")),
        "source_fact_type": _clean(binding.get("source_fact_type")),
    }


def _skipped_task(
    task: Mapping[str, Any],
    *,
    reason: str,
    failures: Sequence[str],
    temporal_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = temporal_binding or {}
    return {
        "source_request_id": str(task.get("source_request_id") or ""),
        "target_id": str(task.get("target_id") or binding.get("target_id") or ""),
        "task_id": str(task.get("task_id") or ""),
        "reason": reason,
        "failures": tuple(str(item) for item in failures),
        "temporal_binding_id": str(binding.get("binding_id") or ""),
        "temporal_binding_review_status": _clean(binding.get("review_status")).lower(),
        "not_verifier_evidence": True,
    }


def _summary(
    *,
    fill_report: Mapping[str, Any],
    temporal_bindings: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    skipped_reason_counts = Counter(str(item.get("reason") or "") for item in skipped)
    review_status_counts = Counter(str(item.get("temporal_binding_review_status") or "") for item in requests)
    source_family_counts = Counter(
        str(_mapping(item.get("existing_temporal_binding_source")).get("source_family") or "") for item in requests
    )
    missing_field_counts: Counter[str] = Counter()
    invalid_field_counts: Counter[str] = Counter()
    review_failure_counts: Counter[str] = Counter()
    for item in requests:
        for field in _sequence(item.get("missing_temporal_fields")):
            missing_field_counts[str(field)] += 1
        for field in _sequence(item.get("invalid_temporal_fields")):
            invalid_field_counts[str(field)] += 1
        for failure in _sequence(item.get("review_failures")):
            review_failure_counts[str(failure)] += 1
    request_count = len(requests)
    return {
        "fill_report_status": str(fill_report.get("status") or ""),
        "temporal_binding_count": len(temporal_bindings),
        "unfilled_task_count": len(_sequence(fill_report.get("unfilled_tasks"))),
        "request_count": request_count,
        "skipped_task_count": len(skipped),
        "fully_unblocking_request_count": sum(
            1 for item in requests if bool(item.get("temporal_binding_unblocks_fill"))
        ),
        "partial_request_count": sum(
            1 for item in requests if not bool(item.get("temporal_binding_unblocks_fill"))
        ),
        "skipped_reason_counts": _sorted_counter(skipped_reason_counts),
        "temporal_review_status_counts": _sorted_counter(review_status_counts),
        "source_family_counts": _sorted_counter(source_family_counts),
        "missing_temporal_field_counts": _sorted_counter(missing_field_counts),
        "invalid_temporal_field_counts": _sorted_counter(invalid_field_counts),
        "review_failure_counts": _sorted_counter(review_failure_counts),
    }


def _is_temporal_binding_gap(*, reason: str, failures: Sequence[str]) -> bool:
    if reason in {"missing_temporal_binding", "invalid_temporal_binding"}:
        return True
    return bool(TEMPORAL_BINDING_FAILURES.intersection(failures))


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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temporal-bindings", default=None)
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
        output_dir=args.output_dir,
        temporal_bindings_path=args.temporal_bindings,
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
        "world_model_rule_temporal_binding_plan_ok "
        f"status={payload['status']} "
        f"requests={summary['request_count']} "
        f"skipped={summary['skipped_task_count']}"
    )


if __name__ == "__main__":
    main()
