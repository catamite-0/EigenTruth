"""Build rerun commands for source-family structured QA lane batches.

The lane execution queue is a scheduling artifact: it says which remaining
claim-mapping gaps need source-family collection, citation retrieval,
disambiguation, or world-model/calculator rule authoring next. This planner
turns those batches into explicit ``run_source_family_structured_qa_lane_batch``
commands and fail-closes command readiness when required local inputs are
missing.
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

WORKFLOW = "source_family_structured_qa_lane_rerun_queue"
LANE_QUEUE_WORKFLOW = "source_family_structured_qa_lane_execution_queue"
SOURCE_BACKED_REQUEST_TYPES = {
    "source_family_structured_fact",
    "entity_resolution",
    "external_citation",
    "source_family_fact_disambiguation",
}
RULE_REQUEST_TYPE = "world_model_or_calculator_rule"


def build_source_family_structured_qa_lane_rerun_queue(
    *,
    lane_queue_path: str | Path,
    collection_corpus_path: str | Path,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    source_catalog_paths: Sequence[str | Path] = (),
    batch_ids: Sequence[str] = (),
    max_batches: int | None = None,
    adapter_max_results: int = 3,
    adapter_max_query_variants: int = 3,
    adapter_min_text_overlap: float = 0.05,
    adapter_diversify_source_families: bool = True,
    default_source_family: str = "reference",
    keep_qid_values: bool = False,
    compact_json: bool = False,
    python_executable: str = sys.executable,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a lane queue and return command rows for executable batches."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when provided.")
    if int(adapter_max_results) <= 0:
        raise ValueError("adapter_max_results must be positive.")
    if int(adapter_max_query_variants) <= 0:
        raise ValueError("adapter_max_query_variants must be positive.")
    if not (0.0 <= float(adapter_min_text_overlap) <= 1.0):
        raise ValueError("adapter_min_text_overlap must be in [0, 1].")

    lane_queue = _load_lane_queue(lane_queue_path)
    selected_batches = _select_batches(
        lane_queue.get("execution_batches", ()),
        batch_ids=batch_ids,
        max_batches=max_batches,
    )
    lane_path = Path(lane_queue_path)
    collection_path = Path(collection_corpus_path)
    output_root = Path(output_dir) if output_dir is not None else lane_path.parent / "lane-reruns"
    source_catalogs = tuple(Path(path) for path in source_catalog_paths)
    entries = tuple(
        _queue_entry(
            batch,
            lane_queue_path=lane_path,
            collection_corpus_path=collection_path,
            output_root=output_root,
            source_catalog_paths=source_catalogs,
            adapter_max_results=int(adapter_max_results),
            adapter_max_query_variants=int(adapter_max_query_variants),
            adapter_min_text_overlap=float(adapter_min_text_overlap),
            adapter_diversify_source_families=bool(adapter_diversify_source_families),
            default_source_family=default_source_family,
            keep_qid_values=bool(keep_qid_values),
            compact_json=bool(compact_json),
            python_executable=python_executable,
        )
        for batch in selected_batches
    )
    summary = _summary(entries)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Executable rerun queue for source-family structured QA lane "
            "batches. Commands only rerun candidate collection or rule-stub "
            "handoffs; they do not promote weak matches or verifier evidence."
        ),
        "source": {
            "lane_queue": str(lane_path),
            "lane_queue_workflow": lane_queue.get("workflow"),
            "lane_queue_status": lane_queue.get("status"),
            "collection_corpus": str(collection_path),
            "source_catalogs": tuple(str(path) for path in source_catalogs),
        },
        "config": {
            "batch_ids": _string_tuple(batch_ids),
            "max_batches": max_batches,
            "adapter_max_results": int(adapter_max_results),
            "adapter_max_query_variants": int(adapter_max_query_variants),
            "adapter_min_text_overlap": float(adapter_min_text_overlap),
            "adapter_diversify_source_families": bool(adapter_diversify_source_families),
            "default_source_family": str(default_source_family),
            "keep_qid_values": bool(keep_qid_values),
            "compact_json": bool(compact_json),
            "python_executable": python_executable,
        },
        "label_usage": {
            "labels_used_for_rerun_planning": False,
            "answers_copied_to_commands": False,
            "model_answers_copied_to_commands": False,
            "commands_are_verifier_evidence": False,
        },
        "summary": summary,
        "entries": entries,
        "paths": {
            "rerun_queue": None if json_path is None else str(json_path),
            "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
        },
        "metadata": dict(metadata or {}),
    }
    output_path = None if json_path is None else Path(json_path)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(strict_json_dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = None
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "source_family_structured_qa_lane_rerun_queue": output_path,
                "source_family_structured_qa_lane_execution_queue": lane_path,
                "source_family_structured_qa_fact_collection_corpus": collection_path,
                **{f"source_catalog_{idx}": path for idx, path in enumerate(source_catalogs, start=1)},
            },
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "batch_count": summary["batch_count"],
                "ready_command_count": summary["ready_command_count"],
                "missing_command_count": summary["missing_command_count"],
                **dict(metadata or {}),
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else lane_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "batch_count": summary["batch_count"],
                "ready_command_count": summary["ready_command_count"],
                "missing_command_count": summary["missing_command_count"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_source_family_structured_qa_lane_rerun_queue(
        lane_queue_path=args.lane_queue,
        collection_corpus_path=args.collection_corpus,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        output_dir=args.output_dir,
        source_catalog_paths=tuple(args.source_catalog or ()),
        batch_ids=tuple(args.batch_id or ()),
        max_batches=args.max_batches,
        adapter_max_results=args.adapter_max_results,
        adapter_max_query_variants=args.adapter_max_query_variants,
        adapter_min_text_overlap=args.adapter_min_text_overlap,
        adapter_diversify_source_families=not bool(args.no_adapter_diversify_source_families),
        default_source_family=args.default_source_family,
        keep_qid_values=bool(args.keep_qid_values),
        compact_json=bool(args.compact_json),
        python_executable=args.python,
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "source_family_structured_qa_lane_rerun_queue="
        f"{payload['status']} "
        f"batches={summary['batch_count']} "
        f"ready={summary['ready_command_count']} "
        f"missing={summary['missing_command_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-queue", required=True)
    parser.add_argument("--collection-corpus", required=True)
    parser.add_argument("--source-catalog", action="append", default=[])
    parser.add_argument("--batch-id", action="append", default=[])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter-max-results", type=int, default=3)
    parser.add_argument("--adapter-max-query-variants", type=int, default=3)
    parser.add_argument("--adapter-min-text-overlap", type=float, default=0.05)
    parser.add_argument("--no-adapter-diversify-source-families", action="store_true")
    parser.add_argument("--default-source-family", default="reference")
    parser.add_argument("--keep-qid-values", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    run(parser.parse_args(argv))


def _queue_entry(
    batch: Mapping[str, Any],
    *,
    lane_queue_path: Path,
    collection_corpus_path: Path,
    output_root: Path,
    source_catalog_paths: Sequence[Path],
    adapter_max_results: int,
    adapter_max_query_variants: int,
    adapter_min_text_overlap: float,
    adapter_diversify_source_families: bool,
    default_source_family: str,
    keep_qid_values: bool,
    compact_json: bool,
    python_executable: str,
) -> dict[str, Any]:
    batch_id = str(batch.get("batch_id") or "")
    if not batch_id:
        raise ValueError("lane queue contains a batch without batch_id.")
    request_type = str(batch.get("request_type") or "")
    source_backed = request_type in SOURCE_BACKED_REQUEST_TYPES
    rule_only = request_type == RULE_REQUEST_TYPE
    output_dir = output_root / _slug(batch_id)
    missing_inputs = _missing_inputs(
        lane_queue_path=lane_queue_path,
        collection_corpus_path=collection_corpus_path,
        source_catalog_paths=source_catalog_paths if source_backed else (),
        require_source_catalog=source_backed,
    )
    command = _command(
        batch_id=batch_id,
        lane_queue_path=lane_queue_path,
        collection_corpus_path=collection_corpus_path,
        output_dir=output_dir,
        source_catalog_paths=source_catalog_paths if source_backed else (),
        adapter_max_results=adapter_max_results,
        adapter_max_query_variants=adapter_max_query_variants,
        adapter_min_text_overlap=adapter_min_text_overlap,
        adapter_diversify_source_families=adapter_diversify_source_families,
        default_source_family=default_source_family,
        keep_qid_values=keep_qid_values,
        compact_json=compact_json,
        python_executable=python_executable,
    )
    return {
        "batch_id": batch_id,
        "command_kind": "rule_authoring_lane_batch" if rule_only else "source_family_lane_batch",
        "command_status": "missing_inputs" if missing_inputs else "ready",
        "command": command,
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "lane-batch-workflow.json"),
        "artifact_manifest": str(output_dir / "artifact-manifest.json"),
        "missing_inputs": missing_inputs,
        "next_lane": batch.get("next_lane"),
        "lane_status": batch.get("lane_status"),
        "request_type": request_type,
        "adapter_family": batch.get("adapter_family"),
        "request_count": _int(batch.get("request_count")),
        "target_count": _int(batch.get("target_count")),
        "target_ids": _string_tuple(batch.get("target_ids", ())),
        "source_request_ids": _string_tuple(batch.get("source_request_ids", ())),
        "source_backed": source_backed,
        "rule_only": rule_only,
        "not_verifier_evidence": True,
    }


def _command(
    *,
    batch_id: str,
    lane_queue_path: Path,
    collection_corpus_path: Path,
    output_dir: Path,
    source_catalog_paths: Sequence[Path],
    adapter_max_results: int,
    adapter_max_query_variants: int,
    adapter_min_text_overlap: float,
    adapter_diversify_source_families: bool,
    default_source_family: str,
    keep_qid_values: bool,
    compact_json: bool,
    python_executable: str,
) -> tuple[str, ...]:
    command: list[str] = [
        python_executable,
        "benchmarks/run_source_family_structured_qa_lane_batch_workflow.py",
        "--lane-queue",
        str(lane_queue_path),
        "--collection-corpus",
        str(collection_corpus_path),
    ]
    for path in source_catalog_paths:
        command.extend(("--source-catalog", str(path)))
    command.extend((
        "--batch-id",
        batch_id,
        "--output-dir",
        str(output_dir),
        "--json",
        str(output_dir / "lane-batch-workflow.json"),
        "--artifact-manifest",
        str(output_dir / "artifact-manifest.json"),
        "--adapter-max-results",
        str(adapter_max_results),
        "--adapter-max-query-variants",
        str(adapter_max_query_variants),
        "--adapter-min-text-overlap",
        str(adapter_min_text_overlap),
        "--default-source-family",
        str(default_source_family),
        "--metadata",
        "source=source_family_structured_qa_lane_rerun_queue",
    ))
    if not adapter_diversify_source_families:
        command.append("--no-adapter-diversify-source-families")
    if keep_qid_values:
        command.append("--keep-qid-values")
    if compact_json:
        command.append("--compact-json")
    return tuple(command)


def _missing_inputs(
    *,
    lane_queue_path: Path,
    collection_corpus_path: Path,
    source_catalog_paths: Sequence[Path],
    require_source_catalog: bool,
) -> tuple[dict[str, str], ...]:
    missing: list[dict[str, str]] = []
    for role, path in (
        ("lane_queue", lane_queue_path),
        ("collection_corpus", collection_corpus_path),
    ):
        if not path.exists():
            missing.append({"role": role, "path": str(path), "reason": "missing_path"})
    if require_source_catalog and not source_catalog_paths:
        missing.append({"role": "source_catalog", "path": "", "reason": "no_source_catalog_configured"})
    for path in source_catalog_paths:
        if not path.exists():
            missing.append({"role": "source_catalog", "path": str(path), "reason": "missing_path"})
    return tuple(missing)


def _summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(entry.get("command_status") or "") for entry in entries)
    lanes = Counter(str(entry.get("next_lane") or "") for entry in entries)
    request_types = Counter(str(entry.get("request_type") or "") for entry in entries)
    missing_roles = Counter(
        str(missing.get("role") or "unknown")
        for entry in entries
        for missing in _mapping_sequence(entry.get("missing_inputs", ()))
    )
    return {
        "batch_count": len(entries),
        "ready_command_count": statuses.get("ready", 0),
        "missing_command_count": len(entries) - statuses.get("ready", 0),
        "source_backed_batch_count": sum(1 for entry in entries if entry.get("source_backed")),
        "rule_only_batch_count": sum(1 for entry in entries if entry.get("rule_only")),
        "command_status_counts": _sorted_counter(statuses),
        "lane_counts": _sorted_counter(lanes),
        "request_type_counts": _sorted_counter(request_types),
        "missing_input_role_counts": _sorted_counter(missing_roles),
        "missing_input_count": sum(len(_mapping_sequence(entry.get("missing_inputs", ()))) for entry in entries),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("batch_count") or 0) == 0:
        return "empty"
    ready = int(summary.get("ready_command_count") or 0)
    missing = int(summary.get("missing_command_count") or 0)
    if ready and not missing:
        return "ready"
    if ready:
        return "partial"
    return "blocked"


def _select_batches(
    batches: Any,
    *,
    batch_ids: Sequence[str],
    max_batches: int | None,
) -> tuple[dict[str, Any], ...]:
    selected_ids = set(_string_tuple(batch_ids))
    rows = []
    for item in _mapping_sequence(batches):
        batch_id = str(item.get("batch_id") or "")
        if selected_ids and batch_id not in selected_ids:
            continue
        rows.append(dict(item))
    if max_batches is not None:
        rows = rows[: int(max_batches)]
    return tuple(rows)


def _load_lane_queue(path: str | Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    if payload.get("workflow") != LANE_QUEUE_WORKFLOW:
        raise ValueError(f"lane_queue must be a {LANE_QUEUE_WORKFLOW} report.")
    return payload


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata entries must be key=value, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key cannot be empty in {value!r}")
        metadata[key] = raw.strip()
    return metadata


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _sorted_counter(counter: Counter[str] | Mapping[str, Any]) -> dict[str, int]:
    items = Counter({str(key): int(value) for key, value in dict(counter).items() if str(key)})
    return dict(sorted(items.items(), key=lambda item: (-item[1], item[0])))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _slug(value: str) -> str:
    text = "".join(ch if ch.isalnum() else "-" for ch in value.strip().lower())
    return "-".join(part for part in text.split("-") if part) or "batch"


if __name__ == "__main__":
    main()
