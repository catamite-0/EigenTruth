"""Roll up citation/search batch evidence workflow reports.

This workflow is intentionally local and dependency-free. It reads batch-level
citation/search evidence reports, optionally checks them against an unresolved
evidence queue's expected external-citation batches, verifies child artifact
manifests, and emits one release-auditable rollup report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    build_artifact_manifest,
    load_and_verify_artifact_manifest,
)

WORKFLOW = "citation_search_batch_evidence_rollup"
SUPPORTED_WORKFLOWS = {
    "citation_search_evidence_workflow",
    "external_citation_search_adapter_workflow",
    "source_family_citation_search_workflow",
}
DEFAULT_EXPECTED_REQUEST_TYPE = "external_citation"


def rollup_citation_search_batch_evidence(
    *,
    report_paths: Sequence[str | Path],
    report_json_path: str | Path,
    queue_report_path: str | Path | None = None,
    expected_batch_ids: Sequence[str] = (),
    expected_request_type: str = DEFAULT_EXPECTED_REQUEST_TYPE,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    require_child_manifests: bool = True,
    recursive_child_manifest_verification: bool = True,
    max_workers: int = 1,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Roll up batch-level citation/source-family evidence reports."""
    if not report_paths:
        raise ValueError("report_paths must contain at least one report.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    workers = _normalize_max_workers(max_workers)
    rollup_path = Path(report_json_path)
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else rollup_path.with_name("artifact-manifest.json")
    )
    expected_ids, expected_source = _resolve_expected_batch_ids(
        queue_report_path=queue_report_path,
        expected_batch_ids=expected_batch_ids,
        expected_request_type=expected_request_type,
    )
    rows = _normalize_child_reports(
        report_paths,
        require_child_manifest=require_child_manifests,
        recursive_child_manifest_verification=recursive_child_manifest_verification,
        max_workers=workers,
    )
    summary = _summary(rows, expected_batch_ids=expected_ids)
    gate = _gate(rows, summary=summary, require_child_manifests=require_child_manifests)
    status = "promote" if gate["promotion_ready"] else ("complete" if gate["passed"] else "blocked")
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "gate": gate,
        "summary": summary,
        "source": {
            "reports": tuple(str(path) for path in report_paths),
            "queue_report": None if queue_report_path is None else str(queue_report_path),
            "expected_batch_source": expected_source,
        },
        "config": {
            "expected_batch_ids": expected_ids,
            "expected_request_type": expected_request_type,
            "require_child_manifests": bool(require_child_manifests),
            "recursive_child_manifest_verification": bool(recursive_child_manifest_verification),
            "max_workers": workers,
        },
        "execution": {
            "max_workers": workers,
            "parallel_child_report_count": len(rows) if workers > 1 and len(rows) > 1 else 0,
        },
        "paths": {
            "report": str(rollup_path),
            "artifact_manifest": str(manifest_path),
        },
        "batch_reports": rows,
        "metadata": dict(metadata or {}),
    }
    _write_json(rollup_path, payload, compact=compact_json)
    manifest = build_artifact_manifest(
        _manifest_artifacts(
            rollup_path=rollup_path,
            report_paths=report_paths,
            rows=rows,
            queue_report_path=queue_report_path,
        ),
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "passed": gate["passed"],
            "promotion_ready": gate["promotion_ready"],
            "report_count": summary["report_count"],
            "expected_batch_count": summary["expected_batch_count"],
            "observed_batch_count": summary["observed_batch_count"],
            "missing_expected_batch_count": summary["missing_expected_batch_count"],
            "blocked_report_count": summary["blocked_report_count"],
            "child_manifest_failed_count": summary["child_manifest_failed_count"],
            "max_workers": workers,
            **dict(metadata or {}),
        },
        max_workers=workers,
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=rollup_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "passed": gate["passed"],
                "promotion_ready": gate["promotion_ready"],
                "report_count": summary["report_count"],
                "expected_batch_count": summary["expected_batch_count"],
                "observed_batch_count": summary["observed_batch_count"],
                "missing_expected_batch_count": summary["missing_expected_batch_count"],
                "blocked_report_count": summary["blocked_report_count"],
                "child_manifest_failed_count": summary["child_manifest_failed_count"],
                "artifact_manifest": str(manifest_path),
                "max_workers": workers,
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _normalize_child_reports(
    report_paths: Sequence[str | Path],
    *,
    require_child_manifest: bool,
    recursive_child_manifest_verification: bool,
    max_workers: int,
) -> tuple[dict[str, Any], ...]:
    if max_workers <= 1 or len(report_paths) <= 1:
        return tuple(
            _normalize_child_report(
                path,
                require_child_manifest=require_child_manifest,
                recursive_child_manifest_verification=recursive_child_manifest_verification,
                child_manifest_max_workers=max_workers,
            )
            for path in report_paths
        )

    results: list[dict[str, Any] | None] = [None] * len(report_paths)
    child_manifest_max_workers = 1
    with ThreadPoolExecutor(max_workers=min(max_workers, len(report_paths))) as executor:
        futures = {
            executor.submit(
                _normalize_child_report,
                path,
                require_child_manifest=require_child_manifest,
                recursive_child_manifest_verification=recursive_child_manifest_verification,
                child_manifest_max_workers=child_manifest_max_workers,
            ): index
            for index, path in enumerate(report_paths)
        }
        for future, index in futures.items():
            results[index] = future.result()
    return tuple(result for result in results if result is not None)


def _normalize_child_report(
    path: str | Path,
    *,
    require_child_manifest: bool,
    recursive_child_manifest_verification: bool,
    child_manifest_max_workers: int,
) -> dict[str, Any]:
    report_path = Path(path)
    report = _load_json_object(report_path)
    workflow = str(report.get("workflow") or "")
    status = str(report.get("status") or "")
    gate = _mapping(report.get("gate"))
    manifest_path = _nested(report, "paths", "artifact_manifest")
    manifest_verification = None
    manifest_error = None
    if manifest_path:
        try:
            manifest_verification = load_and_verify_artifact_manifest(
                manifest_path,
                recursive=recursive_child_manifest_verification,
                max_workers=child_manifest_max_workers,
            ).to_dict()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            manifest_error = str(exc)
    elif require_child_manifest:
        manifest_error = "child report does not expose paths.artifact_manifest"
    batch_ids = _batch_ids(report)
    return {
        "path": str(report_path),
        "workflow": workflow,
        "supported_workflow": workflow in SUPPORTED_WORKFLOWS,
        "status": status,
        "gate_passed": bool(gate.get("passed")),
        "promotion_ready": bool(gate.get("promotion_ready")),
        "blocking_reason_count": len(_sequence(gate.get("blocking_reasons", ()))),
        "selected_batch_ids": batch_ids,
        "selected_batch_count": len(batch_ids),
        "adapter_request_count": _count_value(
            report,
            ("request_summary", "adapter_request_count"),
            ("summary", "adapter_request_count"),
        ),
        "adapter_result_count": _count_value(report, ("adapter_summary", "result_count")),
        "source_document_count": _count_value(
            report,
            ("evidence_summary", "source_document_count"),
            ("summary", "source_document_count"),
        ),
        "corpus_document_count": _count_value(
            report,
            ("evidence_summary", "corpus_document_count"),
            ("summary", "corpus_document_count"),
        ),
        "child_artifact_manifest": None if manifest_path is None else str(manifest_path),
        "child_manifest_verification": manifest_verification,
        "child_manifest_passed": (
            None if manifest_verification is None else bool(manifest_verification.get("passed"))
        ),
        "child_manifest_error": manifest_error,
    }


def _summary(rows: Sequence[Mapping[str, Any]], *, expected_batch_ids: Sequence[str]) -> dict[str, Any]:
    observed_batch_ids = tuple(
        batch_id
        for row in rows
        for batch_id in _string_sequence(row.get("selected_batch_ids", ()))
    )
    observed_counts = Counter(observed_batch_ids)
    unique_observed = tuple(sorted(observed_counts))
    expected = tuple(dict.fromkeys(str(item) for item in expected_batch_ids if str(item)))
    expected_set = set(expected)
    observed_set = set(unique_observed)
    status_counts = Counter(str(row.get("status") or "") for row in rows)
    workflow_counts = Counter(str(row.get("workflow") or "") for row in rows)
    return {
        "report_count": len(rows),
        "workflow_counts": _sorted_counter(workflow_counts),
        "status_counts": _sorted_counter(status_counts),
        "unsupported_workflow_count": sum(1 for row in rows if not bool(row.get("supported_workflow"))),
        "blocked_report_count": sum(1 for row in rows if row.get("status") == "blocked"),
        "gate_passed_count": sum(1 for row in rows if bool(row.get("gate_passed"))),
        "promotion_ready_count": sum(1 for row in rows if bool(row.get("promotion_ready"))),
        "child_manifest_passed_count": sum(1 for row in rows if row.get("child_manifest_passed") is True),
        "child_manifest_failed_count": sum(
            1
            for row in rows
            if row.get("child_manifest_passed") is False or row.get("child_manifest_error")
        ),
        "child_manifest_missing_count": sum(1 for row in rows if not row.get("child_artifact_manifest")),
        "expected_batch_count": len(expected),
        "expected_batch_ids": expected,
        "observed_batch_count": len(unique_observed),
        "observed_batch_ids": unique_observed,
        "missing_expected_batch_count": len(expected_set - observed_set),
        "missing_expected_batch_ids": tuple(sorted(expected_set - observed_set)),
        "unexpected_batch_count": len(observed_set - expected_set) if expected else 0,
        "unexpected_batch_ids": tuple(sorted(observed_set - expected_set)) if expected else (),
        "duplicate_batch_count": sum(1 for count in observed_counts.values() if count > 1),
        "duplicate_batch_ids": tuple(sorted(batch_id for batch_id, count in observed_counts.items() if count > 1)),
        "unbatched_report_count": sum(1 for row in rows if not row.get("selected_batch_ids")),
        "adapter_request_count": sum(_int_or_zero(row.get("adapter_request_count")) for row in rows),
        "adapter_result_count": sum(_int_or_zero(row.get("adapter_result_count")) for row in rows),
        "source_document_count": sum(_int_or_zero(row.get("source_document_count")) for row in rows),
        "corpus_document_count": sum(_int_or_zero(row.get("corpus_document_count")) for row in rows),
    }


def _gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    require_child_manifests: bool,
) -> dict[str, Any]:
    blocking: list[dict[str, Any]] = []
    for row in rows:
        path = str(row.get("path") or "")
        if not bool(row.get("supported_workflow")):
            blocking.append({
                "gate": "workflow",
                "path": path,
                "reason": f"Unsupported child workflow {row.get('workflow')!r}.",
            })
        if not bool(row.get("gate_passed")):
            blocking.append({
                "gate": "child_gate",
                "path": path,
                "reason": f"Child workflow status is {row.get('status')!r} and gate_passed is false.",
            })
        if require_child_manifests:
            if row.get("child_manifest_error"):
                blocking.append({
                    "gate": "child_manifest",
                    "path": path,
                    "reason": str(row.get("child_manifest_error")),
                })
            elif row.get("child_manifest_passed") is not True:
                blocking.append({
                    "gate": "child_manifest",
                    "path": path,
                    "reason": "Child artifact manifest is missing or failed verification.",
                })
    for batch_id in _string_sequence(summary.get("missing_expected_batch_ids", ())):
        blocking.append({
            "gate": "batch_coverage",
            "batch_id": batch_id,
            "reason": "Expected unresolved evidence batch is missing from rollup inputs.",
        })
    for batch_id in _string_sequence(summary.get("duplicate_batch_ids", ())):
        blocking.append({
            "gate": "batch_coverage",
            "batch_id": batch_id,
            "reason": "Batch id appears in more than one child report.",
        })
    for batch_id in _string_sequence(summary.get("unexpected_batch_ids", ())):
        blocking.append({
            "gate": "batch_coverage",
            "batch_id": batch_id,
            "reason": "Child report contains a batch id outside the expected set.",
        })
    passed = not blocking
    promotion_ready = bool(passed and rows and all(bool(row.get("promotion_ready")) for row in rows))
    return {
        "passed": passed,
        "promotion_ready": promotion_ready,
        "blocking_reasons": tuple(blocking),
    }


def _resolve_expected_batch_ids(
    *,
    queue_report_path: str | Path | None,
    expected_batch_ids: Sequence[str],
    expected_request_type: str,
) -> tuple[tuple[str, ...], str]:
    explicit = tuple(dict.fromkeys(str(item).strip() for item in expected_batch_ids if str(item).strip()))
    if explicit:
        return explicit, "explicit"
    if queue_report_path is None:
        return (), "none"
    queue = _load_json_object(queue_report_path)
    expected = tuple(
        str(batch.get("batch_id"))
        for batch in _mapping_sequence(queue.get("execution_batches", ()))
        if str(batch.get("batch_id", "")).strip()
        and (
            not expected_request_type
            or expected_request_type == "any"
            or str(batch.get("request_type") or "") == expected_request_type
        )
    )
    return tuple(dict.fromkeys(expected)), "queue_report"


def _manifest_artifacts(
    *,
    rollup_path: Path,
    report_paths: Sequence[str | Path],
    rows: Sequence[Mapping[str, Any]],
    queue_report_path: str | Path | None,
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "rollup_report": rollup_path,
        "queue_report": None if queue_report_path is None else Path(queue_report_path),
    }
    for index, path in enumerate(report_paths, start=1):
        artifacts[f"batch_report_{index}"] = Path(path)
    for index, row in enumerate(rows, start=1):
        manifest_path = row.get("child_artifact_manifest")
        artifacts[f"batch_manifest_{index}"] = None if manifest_path is None else Path(str(manifest_path))
    return artifacts


def _batch_ids(report: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        _string_sequence(_nested(report, "request_summary", "selected_batch_ids"))
        or _string_sequence(_nested(report, "summary", "selected_batch_ids"))
        or _string_sequence(_nested(report, "config", "batch_ids"))
    ))


def _count_value(report: Mapping[str, Any], *paths: tuple[str, ...]) -> int:
    for path in paths:
        value = _nested(report, *path)
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return 0


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _nested(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    parsed = _optional_int(value)
    return 0 if parsed is None else parsed


def _normalize_max_workers(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_workers must be a positive integer.")
    return int(value)


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
    parser.add_argument("--batch-report", action="append", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--queue", default=None)
    parser.add_argument("--expected-batch-id", action="append", default=[])
    parser.add_argument("--expected-request-type", default=DEFAULT_EXPECTED_REQUEST_TYPE)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--allow-missing-child-manifest", action="store_true")
    parser.add_argument("--no-recursive-child-manifest-verification", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = rollup_citation_search_batch_evidence(
        report_paths=tuple(args.batch_report or ()),
        report_json_path=args.json,
        queue_report_path=args.queue,
        expected_batch_ids=tuple(args.expected_batch_id or ()),
        expected_request_type=args.expected_request_type,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        require_child_manifests=not bool(args.allow_missing_child_manifest),
        recursive_child_manifest_verification=not bool(args.no_recursive_child_manifest_verification),
        max_workers=args.max_workers,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "citation_search_batch_evidence_rollup_ok "
        f"status={payload['status']} "
        f"reports={summary['report_count']} "
        f"observed_batches={summary['observed_batch_count']} "
        f"missing_batches={summary['missing_expected_batch_count']} "
        f"promotion_ready={payload['gate']['promotion_ready']}"
    )


if __name__ == "__main__":
    main()
