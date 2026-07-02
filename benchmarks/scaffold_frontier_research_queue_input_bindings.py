"""Scaffold source-backed binding sidecars for frontier input requests.

This workflow consumes ``frontier_research_queue_input_collection_plan`` output
and writes editable JSONL sidecars for numeric, temporal, subject, mechanism,
and entity-role bindings. It never fills evidence values, approves review, or
executes downstream rule-input fill commands.
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

from benchmarks.plan_frontier_research_queue_input_collection import (  # noqa: E402
    WORKFLOW as SOURCE_WORKFLOW,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_research_queue_input_binding_scaffold"

SIDECARES = {
    "source_backed_numeric_bindings": {
        "key": "numeric_bindings",
        "filename": "source-backed-numeric-bindings.jsonl",
        "task_collection_family": "numeric_rule_input_collection",
        "downstream_tool": "benchmarks/fill_world_model_rule_inputs_from_numeric_bindings.py",
        "flag": "--numeric-bindings",
    },
    "source_backed_subject_bindings": {
        "key": "subject_bindings",
        "filename": "source-backed-subject-bindings.jsonl",
        "task_collection_family": "numeric_subject_binding_collection",
        "downstream_tool": "benchmarks/fill_world_model_rule_inputs_from_numeric_bindings.py",
        "flag": "--subject-bindings",
    },
    "source_backed_temporal_bindings": {
        "key": "temporal_bindings",
        "filename": "source-backed-temporal-bindings.jsonl",
        "task_collection_family": "temporal_snapshot_rule_input_collection",
        "downstream_tool": "benchmarks/fill_world_model_rule_inputs_from_temporal_bindings.py",
        "flag": "--temporal-bindings",
    },
    "source_backed_mechanism_bindings": {
        "key": "mechanism_bindings",
        "filename": "source-backed-mechanism-bindings.jsonl",
        "task_collection_family": "mechanism_rule_input_collection",
        "downstream_tool": "benchmarks/fill_world_model_rule_inputs_from_mechanism_bindings.py",
        "flag": "--mechanism-bindings",
    },
    "source_backed_entity_bindings": {
        "key": "entity_bindings",
        "filename": "source-backed-entity-bindings.jsonl",
        "task_collection_family": "entity_role_rule_input_collection",
        "downstream_tool": "benchmarks/fill_world_model_rule_inputs_from_entity_bindings.py",
        "flag": "--entity-bindings",
    },
}


def scaffold_frontier_research_queue_input_bindings(
    *,
    input_collection_plan: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    expand_input_tasks: bool = True,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write editable source-backed binding skeleton sidecars."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path, plan = _load_mapping_source(input_collection_plan)
    if plan.get("workflow") != SOURCE_WORKFLOW:
        raise ValueError(f"input_collection_plan must have workflow={SOURCE_WORKFLOW!r}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(json_path or output / "frontier-input-binding-scaffold.json")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    requests = tuple(_mapping_sequence(plan.get("collection_requests", ())))
    review_requests = tuple(_mapping_sequence(plan.get("review_requests", ())))
    sidecar_rows: dict[str, list[dict[str, Any]]] = {
        str(config["key"]): [] for config in SIDECARES.values()
    }
    rows_by_request: dict[str, tuple[str, int]] = {}
    input_task_sources: dict[str, dict[str, Any]] = {}
    skipped_requests: list[dict[str, Any]] = []
    downstream_commands: list[dict[str, Any]] = []
    for request in requests:
        input_name = str(request.get("input_name") or "")
        config = SIDECARES.get(input_name)
        if config is None:
            skipped_requests.append(_skipped_request(request, reason="unsupported_input_name"))
            continue
        input_tasks_path = _input_tasks_path_hint(request)
        task_rows = _load_matching_tasks(
            input_tasks_path,
            task_collection_family=str(config["task_collection_family"]),
            expand_input_tasks=expand_input_tasks,
            source_root=None if source_path is None else source_path.parent,
        )
        input_task_sources[str(request.get("request_id") or "")] = {
            "input_tasks_path": None if input_tasks_path is None else str(input_tasks_path),
            "expanded_task_count": len(task_rows),
            "expand_input_tasks": bool(expand_input_tasks),
            "task_collection_family": str(config["task_collection_family"]),
        }
        rows = (
            tuple(
                _task_binding_skeleton(
                    request,
                    task,
                    input_name=input_name,
                    sidecar_key=str(config["key"]),
                )
                for task in task_rows
            )
            if task_rows
            else (_request_binding_skeleton(request, input_name=input_name, sidecar_key=str(config["key"])),)
        )
        sidecar_rows[str(config["key"])].extend(rows)
        rows_by_request[str(request.get("request_id") or "")] = (str(config["key"]), len(rows))
        downstream_commands.append(
            _downstream_command(
                request,
                config=config,
                output_dir=output,
                sidecar_path=output / str(config["filename"]),
                input_tasks_path=input_tasks_path,
            )
        )

    paths = {
        "input_binding_scaffold": str(report_path),
        "artifact_manifest": str(manifest_path),
        "review_requests": str(output / "frontier-input-review-requests.jsonl"),
        **{
            key: str(output / str(config["filename"]))
            for key, config in ((str(item["key"]), item) for item in SIDECARES.values())
        },
    }
    summary = _summary(
        sidecar_rows=sidecar_rows,
        review_requests=review_requests,
        skipped_requests=skipped_requests,
        downstream_commands=downstream_commands,
    )
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Editable sidecar scaffold for source-backed frontier input "
            "bindings. Rows are empty non-evidence skeletons and remain "
            "review-gated until a human or external adapter fills values."
        ),
        "source": {
            "input_collection_plan": None if source_path is None else str(source_path),
            "input_collection_workflow": plan.get("workflow"),
            "input_collection_status": plan.get("status"),
            "collection_request_count": len(requests),
            "review_request_count": len(review_requests),
        },
        "label_usage": {
            "labels_used_for_binding_scaffold": False,
            "labels_copied_to_binding_sidecars": False,
            "model_answers_copied_to_binding_sidecars": False,
            "scaffold_rows_are_verifier_evidence": False,
            "scaffold_approves_review": False,
            "scaffold_executes_commands": False,
        },
        "config": {
            "expand_input_tasks": bool(expand_input_tasks),
            "default_review_status": "needs_review",
            "writes_empty_values_only": True,
        },
        "summary": summary,
        "paths": paths,
        "rows_by_request": rows_by_request,
        "input_task_sources": input_task_sources,
        "downstream_commands": tuple(downstream_commands),
        "skipped_requests": tuple(skipped_requests),
        "metadata": dict(metadata or {}),
    }

    _write_json(report_path, payload, compact=compact_json)
    for key, rows in sidecar_rows.items():
        _write_jsonl(paths[key], rows, compact=compact_json)
    _write_jsonl(paths["review_requests"], review_requests, compact=compact_json)
    manifest = _write_manifest(
        manifest_path=manifest_path,
        output_path=report_path,
        sidecar_paths={key: Path(paths[key]) for key in sidecar_rows},
        review_requests_path=Path(paths["review_requests"]),
        source_path=source_path,
        payload=payload,
        metadata=metadata or {},
        compact=compact_json,
    )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "input_collection_plan": None if source_path is None else str(source_path),
                "artifact_manifest": str(manifest_path),
                "binding_skeleton_count": summary["binding_skeleton_count"],
                "expanded_task_count": summary["expanded_task_count"],
                "review_request_count": summary["review_request_count"],
                "downstream_command_count": summary["downstream_command_count"],
                "manifest_summary": manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _task_binding_skeleton(
    request: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    input_name: str,
    sidecar_key: str,
) -> dict[str, Any]:
    row = _base_skeleton(request, input_name=input_name, sidecar_key=sidecar_key)
    request_id = str(task.get("source_request_id") or task.get("request_id") or "")
    row.update({
        "binding_id": f"{_slug(str(request.get('request_id') or input_name))}:{_slug(request_id)}",
        "request_id": request_id,
        "source_request_id": request_id,
        "target_id": str(task.get("target_id") or ""),
        "task_id": str(task.get("task_id") or ""),
        "question": str(task.get("question") or ""),
        "rule_family": str(task.get("rule_family") or ""),
        "collection_family": str(task.get("collection_family") or ""),
        "missing_inputs": _string_tuple(task.get("missing_inputs", ())),
        "field_tasks": tuple(_mapping_sequence(task.get("field_tasks", ()))),
        "scaffold_status": "needs_source_backed_values",
    })
    return row


def _request_binding_skeleton(
    request: Mapping[str, Any],
    *,
    input_name: str,
    sidecar_key: str,
) -> dict[str, Any]:
    row = _base_skeleton(request, input_name=input_name, sidecar_key=sidecar_key)
    row.update({
        "binding_id": f"{_slug(str(request.get('request_id') or input_name))}:request-level",
        "request_id": "",
        "source_request_id": "",
        "target_id": "",
        "scaffold_status": "needs_input_task_expansion_or_manual_request_id",
    })
    return row


def _base_skeleton(
    request: Mapping[str, Any],
    *,
    input_name: str,
    sidecar_key: str,
) -> dict[str, Any]:
    skeleton = dict(_mapping(request.get("recommended_binding_skeleton")))
    for key in _string_tuple(request.get("required_binding_fields", ())):
        skeleton.setdefault(key, "")
    skeleton["review_status"] = "needs_review"
    skeleton["not_verifier_evidence"] = True
    skeleton["collection_request_id"] = str(request.get("request_id") or "")
    skeleton["action_id"] = str(request.get("action_id") or "")
    skeleton["input_name"] = input_name
    skeleton["sidecar_key"] = sidecar_key
    skeleton["required_binding_fields"] = _string_tuple(request.get("required_binding_fields", ()))
    skeleton["source_gap_ids"] = _string_tuple(request.get("source_gap_ids", ()))
    skeleton["source_note"] = ""
    return skeleton


def _downstream_command(
    request: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    sidecar_path: Path,
    input_tasks_path: str | Path | None,
) -> dict[str, Any]:
    command_dir = output_dir / "downstream" / _slug(str(request.get("request_id") or "request"))
    command = (
        f"{config['downstream_tool']} --input-tasks "
        f"{input_tasks_path if input_tasks_path is not None else '...'} "
        f"{config['flag']} {sidecar_path} --output-dir {command_dir} "
        f"--json {command_dir / 'rule-input-fill.json'} "
        f"--artifact-manifest {command_dir / 'artifact-manifest.json'}"
    )
    return {
        "request_id": str(request.get("request_id") or ""),
        "input_name": str(request.get("input_name") or ""),
        "sidecar_key": str(config["key"]),
        "sidecar_path": str(sidecar_path),
        "input_tasks_path": None if input_tasks_path is None else str(input_tasks_path),
        "command": command,
        "review_required": True,
        "executes_commands": False,
    }


def _input_tasks_path_hint(request: Mapping[str, Any]) -> str | None:
    for placeholder in _mapping_sequence(request.get("blocking_placeholders", ())):
        context = _mapping(placeholder.get("context"))
        before = _string_tuple(context.get("before", ()))
        flag = str(placeholder.get("flag") or "")
        if len(before) >= 2 and before[-1] == flag and before[-2] and not before[-2].startswith("--"):
            return before[-2]
    return None


def _load_matching_tasks(
    path: str | Path | None,
    *,
    task_collection_family: str,
    expand_input_tasks: bool,
    source_root: Path | None,
) -> tuple[Mapping[str, Any], ...]:
    if not expand_input_tasks or path is None:
        return ()
    task_path = _resolve_path(path, source_root=source_root)
    if not task_path.exists():
        return ()
    rows = _load_jsonl_mappings(task_path)
    return tuple(
        row
        for row in rows
        if str(row.get("collection_family") or "") == task_collection_family
    )


def _skipped_request(request: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "request_id": str(request.get("request_id") or ""),
        "action_id": str(request.get("action_id") or ""),
        "input_name": str(request.get("input_name") or ""),
        "collection_family": str(request.get("collection_family") or ""),
        "reason": reason,
        "not_verifier_evidence": True,
    }


def _summary(
    *,
    sidecar_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    review_requests: Sequence[Mapping[str, Any]],
    skipped_requests: Sequence[Mapping[str, Any]],
    downstream_commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sidecar_counts = {key: len(tuple(rows)) for key, rows in sorted(sidecar_rows.items())}
    expanded_count = sum(
        1
        for rows in sidecar_rows.values()
        for row in rows
        if str(row.get("scaffold_status") or "") == "needs_source_backed_values"
    )
    request_level_count = sum(
        1
        for rows in sidecar_rows.values()
        for row in rows
        if str(row.get("scaffold_status") or "") == "needs_input_task_expansion_or_manual_request_id"
    )
    review_family_counts = Counter(str(item.get("review_family") or "") for item in review_requests)
    return {
        "binding_skeleton_count": sum(sidecar_counts.values()),
        "expanded_task_count": expanded_count,
        "request_level_skeleton_count": request_level_count,
        "review_request_count": len(review_requests),
        "skipped_request_count": len(skipped_requests),
        "downstream_command_count": len(downstream_commands),
        "sidecar_counts": sidecar_counts,
        "review_family_counts": _sorted_counter(review_family_counts),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("binding_skeleton_count", 0)) > 0:
        return "needs_binding_values"
    if int(summary.get("review_request_count", 0)) > 0:
        return "needs_review"
    return "empty"


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    sidecar_paths: Mapping[str, Path],
    review_requests_path: Path,
    source_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "frontier_research_queue_input_binding_scaffold": output_path,
        "input_collection_plan": source_path,
        "review_requests": review_requests_path,
        **{key: value for key, value in sidecar_paths.items()},
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "scaffold_frontier_research_queue_input_bindings",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "binding_skeleton_count": _nested(payload, "summary", "binding_skeleton_count"),
            "expanded_task_count": _nested(payload, "summary", "expanded_task_count"),
            "review_request_count": _nested(payload, "summary", "review_request_count"),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def _load_mapping_source(source: str | Path | Mapping[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    if isinstance(source, Mapping):
        return None, dict(source)
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return path, dict(payload)


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
    reserved = {"answer", "answers", "is_false", "label", "labels", "model_answer", "score_label"}
    return {str(key): value for key, value in row.items() if str(key) not in reserved}


def _resolve_path(path: str | Path, *, source_root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if source_root is not None and (source_root / candidate).exists():
        return source_root / candidate
    return ROOT / candidate


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if compact
        else strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool = False) -> None:
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
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((str(key), int(value)) for key, value in counter.items() if str(key)))


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    return "-".join(part for part in text.split("-") if part) or "item"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-collection-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--no-expand-input-tasks",
        action="store_true",
        help="write one request-level skeleton per collection request instead of reading input task JSONL",
    )
    parser.add_argument("--compact-json", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = scaffold_frontier_research_queue_input_bindings(
        input_collection_plan=args.input_collection_plan,
        output_dir=args.output_dir,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        expand_input_tasks=not bool(args.no_expand_input_tasks),
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
