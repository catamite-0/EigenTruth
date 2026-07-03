"""Build rerun queue items for blocked citation/search evidence batches."""

from __future__ import annotations

import argparse
import json
import sys
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

WORKFLOW = "citation_batch_evidence_rerun_queue"


def build_citation_batch_evidence_rerun_queue(
    *,
    source: str | Path,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    queue_report_path: str | Path | None = None,
    scores_path: str | Path | None = None,
    blind_spots_path: str | Path | None = None,
    source_catalog_paths: Sequence[str | Path] = (),
    search_command: str | None = None,
    controlled_sweep_paths: Sequence[str | Path] = (),
    query_mode: str = "claim_entity",
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Load frontier/gap evidence and build one queue row per blocked citation batch."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path = Path(source)
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    rerun_root = Path(output_dir) if output_dir is not None else source_path.parent / "citation-batch-reruns"
    payload = _load_json_object(source_path)
    batch_issues = _blocked_batches_from_payload(payload)
    entries = tuple(
        _queue_entry(
            issue,
            rerun_root=rerun_root,
            queue_report_path=queue_report_path,
            scores_path=scores_path,
            blind_spots_path=blind_spots_path,
            source_catalog_paths=source_catalog_paths,
            search_command=search_command,
            controlled_sweep_paths=controlled_sweep_paths,
            query_mode=query_mode,
            python_executable=python_executable,
        )
        for issue in batch_issues
    )
    command_count = sum(1 for entry in entries if entry["command_status"] == "ready")
    output = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready" if entries else "empty",
        "source": str(source_path),
        "summary": {
            "blocked_batch_count": len(entries),
            "missing_expected_batch_count": sum(
                1 for entry in entries if entry["issue_type"] == "missing_expected"
            ),
            "duplicate_batch_count": sum(1 for entry in entries if entry["issue_type"] == "duplicate"),
            "unexpected_batch_count": sum(1 for entry in entries if entry["issue_type"] == "unexpected"),
            "command_count": command_count,
            "missing_command_count": len(entries) - command_count,
        },
        "paths": {
            "rerun_queue": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
        },
        "config": {
            "queue_report": None if queue_report_path is None else str(queue_report_path),
            "scores": None if scores_path is None else str(scores_path),
            "blind_spots": None if blind_spots_path is None else str(blind_spots_path),
            "source_catalogs": tuple(str(path) for path in source_catalog_paths),
            "search_command": search_command,
            "controlled_sweeps": tuple(str(path) for path in controlled_sweep_paths),
            "query_mode": query_mode,
            "python_executable": python_executable,
        },
        "entries": entries,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(strict_json_dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = None
    if manifest_path is not None:
        manifest = _write_artifact_manifest(
            source_path=source_path,
            output_path=output_path,
            manifest_path=manifest_path,
            payload=output,
        )
    if registry_path is not None:
        assert name is not None and version is not None
        _record_registry(
            registry_path=Path(registry_path),
            name=name,
            version=version,
            report_path=output_path if output_path is not None else source_path,
            manifest_path=manifest_path,
            payload=output,
            manifest=manifest,
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_citation_batch_evidence_rerun_queue(
        source=args.source,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        output_dir=args.output_dir,
        queue_report_path=args.queue,
        scores_path=args.scores,
        blind_spots_path=args.blind_spots,
        source_catalog_paths=tuple(args.source_catalog or ()),
        search_command=args.search_command,
        controlled_sweep_paths=tuple(args.controlled_sweep or ()),
        query_mode=args.query_mode,
        python_executable=args.python,
    )
    summary = payload["summary"]
    print(
        "citation_batch_evidence_rerun_queue="
        f"{payload['status']} "
        f"blocked_batches={summary['blocked_batch_count']} "
        f"commands={summary['command_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="frontier release or evidence-gap JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--output-dir", default=None, help="root directory for per-batch rerun outputs")
    parser.add_argument("--queue", default=None, help="unresolved evidence queue for generated workflow commands")
    parser.add_argument("--scores", default=None, help="score dump for generated workflow commands")
    parser.add_argument("--blind-spots", default=None, help="blind-spot rows for generated workflow commands")
    parser.add_argument("--source-catalog", action="append", default=[], help="source-family catalog; repeatable")
    parser.add_argument("--search-command", default=None, help="external adapter command with {input}/{output}")
    parser.add_argument("--controlled-sweep", action="append", default=[], help="controlled sweep report; repeatable")
    parser.add_argument("--query-mode", default="claim_entity")
    parser.add_argument("--python", default=sys.executable, help="Python executable for generated commands")
    run(parser.parse_args(argv))


def _blocked_batches_from_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    workflow = payload.get("workflow")
    rows: list[dict[str, Any]] = []
    if workflow == "frontier_release_evidence_comparison":
        evidence_summary = _mapping(payload.get("evidence_summary"))
        rows.extend(_batch_rows_from_metadata(evidence_summary, source_workflow=str(workflow)))
        for decision in _mapping_sequence(payload.get("citation_batch_decisions", ())):
            metrics = _mapping(decision.get("metrics"))
            rollup = _optional_str(decision.get("name"))
            rows.extend(_batch_rows_from_metrics(metrics, source_workflow=str(workflow), rollup=rollup))
    elif workflow == "evidence_gap_plan":
        for gap in _mapping_sequence(payload.get("gaps", ())):
            metadata = _mapping(gap.get("metadata"))
            rows.extend(_batch_rows_from_metadata(metadata, source_workflow=str(workflow)))
    elif workflow == "citation_search_batch_evidence_rollup":
        summary = _mapping(payload.get("summary"))
        rows.extend(_batch_rows_from_metrics(summary, source_workflow=str(workflow), rollup=payload.get("name")))
    return tuple(_unique_batch_rows(rows))


def _batch_rows_from_metadata(
    metadata: Mapping[str, Any],
    *,
    source_workflow: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for key, issue_type in (
        ("citation_batch_missing_expected_batches", "missing_expected"),
        ("citation_batch_duplicate_batches", "duplicate"),
        ("citation_batch_unexpected_batches", "unexpected"),
    ):
        for item in _mapping_sequence(metadata.get(key, ())):
            batch_id = _optional_str(item.get("batch_id"))
            if batch_id:
                rows.append({
                    "issue_type": issue_type,
                    "rollup": _optional_str(item.get("rollup")),
                    "batch_id": batch_id,
                    "source_workflow": source_workflow,
                })
    return tuple(rows)


def _batch_rows_from_metrics(
    metrics: Mapping[str, Any],
    *,
    source_workflow: str,
    rollup: Any = None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for key, issue_type in (
        ("missing_expected_batch_ids", "missing_expected"),
        ("duplicate_batch_ids", "duplicate"),
        ("unexpected_batch_ids", "unexpected"),
    ):
        for batch_id in _string_tuple(metrics.get(key, ())):
            rows.append({
                "issue_type": issue_type,
                "rollup": _optional_str(rollup),
                "batch_id": batch_id,
                "source_workflow": source_workflow,
            })
    return tuple(rows)


def _queue_entry(
    issue: Mapping[str, Any],
    *,
    rerun_root: Path,
    queue_report_path: str | Path | None,
    scores_path: str | Path | None,
    blind_spots_path: str | Path | None,
    source_catalog_paths: Sequence[str | Path],
    search_command: str | None,
    controlled_sweep_paths: Sequence[str | Path],
    query_mode: str,
    python_executable: str,
) -> dict[str, Any]:
    batch_id = str(issue["batch_id"])
    output_dir = rerun_root / _slug(batch_id)
    command = _source_family_command(
        batch_id=batch_id,
        output_dir=output_dir,
        queue_report_path=queue_report_path,
        scores_path=scores_path,
        blind_spots_path=blind_spots_path,
        source_catalog_paths=source_catalog_paths,
        controlled_sweep_paths=controlled_sweep_paths,
        query_mode=query_mode,
        python_executable=python_executable,
    )
    command_kind = "source_family"
    if command is None:
        command = _external_command(
            batch_id=batch_id,
            output_dir=output_dir,
            queue_report_path=queue_report_path,
            scores_path=scores_path,
            blind_spots_path=blind_spots_path,
            search_command=search_command,
            controlled_sweep_paths=controlled_sweep_paths,
            query_mode=query_mode,
            python_executable=python_executable,
        )
        command_kind = "external"
    missing_inputs = _missing_command_inputs(
        queue_report_path=queue_report_path,
        scores_path=scores_path,
        blind_spots_path=blind_spots_path,
        source_catalog_paths=source_catalog_paths,
        search_command=search_command,
    )
    command_status = "ready" if command is not None else "missing_inputs"
    return {
        "issue_type": issue["issue_type"],
        "rollup": issue.get("rollup"),
        "batch_id": batch_id,
        "source_workflow": issue.get("source_workflow"),
        "rerun_output_dir": str(output_dir),
        "command_status": command_status,
        "command_kind": command_kind if command is not None else None,
        "missing_inputs": missing_inputs if command is None else (),
        "command": command,
        "dry_run_command": None,
    }


def _source_family_command(
    *,
    batch_id: str,
    output_dir: Path,
    queue_report_path: str | Path | None,
    scores_path: str | Path | None,
    blind_spots_path: str | Path | None,
    source_catalog_paths: Sequence[str | Path],
    controlled_sweep_paths: Sequence[str | Path],
    query_mode: str,
    python_executable: str,
) -> tuple[str, ...] | None:
    if queue_report_path is None or scores_path is None or blind_spots_path is None or not source_catalog_paths:
        return None
    command: list[str] = [
        python_executable,
        "benchmarks/run_source_family_citation_search_workflow.py",
        "--queue",
        str(queue_report_path),
        "--scores",
        str(scores_path),
        "--blind-spots",
        str(blind_spots_path),
        "--output-dir",
        str(output_dir / "source-family"),
        "--batch-id",
        batch_id,
        "--query-mode",
        query_mode,
    ]
    for path in source_catalog_paths:
        command.extend(("--source-catalog", str(path)))
    for path in controlled_sweep_paths:
        command.extend(("--controlled-sweep", str(path)))
    return tuple(command)


def _external_command(
    *,
    batch_id: str,
    output_dir: Path,
    queue_report_path: str | Path | None,
    scores_path: str | Path | None,
    blind_spots_path: str | Path | None,
    search_command: str | None,
    controlled_sweep_paths: Sequence[str | Path],
    query_mode: str,
    python_executable: str,
) -> tuple[str, ...] | None:
    if queue_report_path is None or scores_path is None or blind_spots_path is None or not search_command:
        return None
    command: list[str] = [
        python_executable,
        "benchmarks/run_external_citation_search_adapter_workflow.py",
        "--queue",
        str(queue_report_path),
        "--search-command",
        search_command,
        "--scores",
        str(scores_path),
        "--blind-spots",
        str(blind_spots_path),
        "--output-dir",
        str(output_dir / "external"),
        "--batch-id",
        batch_id,
        "--query-mode",
        query_mode,
    ]
    for path in controlled_sweep_paths:
        command.extend(("--controlled-sweep", str(path)))
    return tuple(command)


def _missing_command_inputs(
    *,
    queue_report_path: str | Path | None,
    scores_path: str | Path | None,
    blind_spots_path: str | Path | None,
    source_catalog_paths: Sequence[str | Path],
    search_command: str | None,
) -> tuple[str, ...]:
    missing = []
    if queue_report_path is None:
        missing.append("queue")
    if scores_path is None:
        missing.append("scores")
    if blind_spots_path is None:
        missing.append("blind_spots")
    if not source_catalog_paths and not search_command:
        missing.append("source_catalog_or_search_command")
    return tuple(missing)


def _write_artifact_manifest(
    *,
    source_path: Path,
    output_path: Path | None,
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(payload.get("summary"))
    manifest = build_artifact_manifest(
        {
            "source": source_path,
            "citation_batch_evidence_rerun_queue": output_path,
        },
        root=manifest_path.parent,
        metadata={
            "runner": "plan_citation_batch_evidence_reruns",
            "status": payload.get("status"),
            "source": str(source_path),
            "blocked_batch_count": summary.get("blocked_batch_count"),
            "command_count": summary.get("command_count"),
            "missing_command_count": summary.get("missing_command_count"),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _record_registry(
    *,
    registry_path: Path,
    name: str,
    version: str,
    report_path: Path,
    manifest_path: Path | None,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> None:
    summary = _mapping(payload.get("summary"))
    ArtifactRegistry.load_json(registry_path).record_report(
        name=name,
        version=version,
        path=report_path,
        metadata={
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "source": payload.get("source"),
            "blocked_batch_count": summary.get("blocked_batch_count"),
            "command_count": summary.get("command_count"),
            "missing_command_count": summary.get("missing_command_count"),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "manifest_summary": None if manifest is None else _mapping(manifest.get("summary")),
        },
    ).save_json()


def _unique_batch_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    seen: set[tuple[str, str | None, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("issue_type") or ""),
            _optional_str(row.get("rollup")),
            str(row.get("batch_id") or ""),
        )
        if not key[0] or not key[2] or key in seen:
            continue
        seen.add(key)
        unique.append(dict(row))
    return tuple(unique)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"source JSON must contain an object: {path}")
    return data


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "batch"


if __name__ == "__main__":
    main()
