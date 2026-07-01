"""Build a command plan from a frontier status research queue."""

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

WORKFLOW = "frontier_research_queue_command_plan"
SUPPORTED_SOURCE_WORKFLOWS = frozenset({"frontier_status_report", "evidence_gap_plan"})


def build_frontier_research_queue_command_plan(
    *,
    source: str | Path | Mapping[str, Any],
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    include_action_ids: Sequence[str] = (),
    exclude_action_ids: Sequence[str] = (),
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn research-queue actions into a reviewable command plan.

    This is a planning artifact only. It does not execute commands, bind
    placeholders, or create verifier/release evidence.
    """
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path, source_payload = _load_mapping_source(source)
    workflow = source_payload.get("workflow")
    if workflow not in SUPPORTED_SOURCE_WORKFLOWS:
        raise ValueError(
            "source must have workflow 'frontier_status_report' or 'evidence_gap_plan'."
        )

    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    plan_root = _plan_root(source_path=source_path, output_path=output_path, output_dir=output_dir)
    actions = _filtered_actions(
        _source_actions(source_payload),
        include_action_ids=include_action_ids,
        exclude_action_ids=exclude_action_ids,
    )
    entries = tuple(
        _command_entry(action, index=index, plan_root=plan_root)
        for index, action in enumerate(actions, start=1)
    )
    summary = _summary(entries)
    status = _status(summary)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "source": {
            "path": None if source_path is None else str(source_path),
            "workflow": workflow,
            "status": source_payload.get("status"),
            "research_refresh_status": _nested(
                source_payload, "research_queue", "refresh_status"
            ),
        },
        "summary": summary,
        "paths": {
            "command_plan": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "output_dir": str(plan_root),
        },
        "config": {
            "executes_commands": False,
            "include_action_ids": tuple(str(item) for item in include_action_ids if str(item)),
            "exclude_action_ids": tuple(str(item) for item in exclude_action_ids if str(item)),
        },
        "entries": entries,
        "metadata": dict(metadata or {}),
    }
    if output_path is not None:
        _write_json(output_path, payload, compact=compact_json)
    manifest = None
    if manifest_path is not None:
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            source_path=source_path,
            payload=payload,
            metadata=metadata or {},
            compact=compact_json,
        )
    if registry_path is not None and output_path is None and source_path is None:
        raise ValueError("registry_path requires json_path when source is an in-memory payload.")
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else source_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "source_workflow": workflow,
                "source_path": None if source_path is None else str(source_path),
                "entry_count": summary["entry_count"],
                "ready_entry_count": summary["ready_entry_count"],
                "needs_input_entry_count": summary["needs_input_entry_count"],
                "missing_command_template_count": summary["missing_command_template_count"],
                "command_count": summary["command_count"],
                "placeholder_count": summary["placeholder_count"],
                "missing_input_count": summary["missing_input_count"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _source_actions(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if payload.get("workflow") == "frontier_status_report":
        return tuple(_mapping_sequence(_nested(payload, "research_queue", "actions")))
    return tuple(_mapping_sequence(payload.get("actions", ())))


def _filtered_actions(
    actions: Sequence[Mapping[str, Any]],
    *,
    include_action_ids: Sequence[str],
    exclude_action_ids: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    include = {str(item) for item in include_action_ids if str(item)}
    exclude = {str(item) for item in exclude_action_ids if str(item)}
    filtered = []
    for action in actions:
        action_id = str(action.get("action_id") or "")
        if include and action_id not in include:
            continue
        if action_id in exclude:
            continue
        filtered.append(action)
    return tuple(
        sorted(
            filtered,
            key=lambda item: (-_int_or_zero(item.get("priority")), str(item.get("action_id") or "")),
        )
    )


def _command_entry(action: Mapping[str, Any], *, index: int, plan_root: Path) -> dict[str, Any]:
    action_id = str(action.get("action_id") or f"frontier-research-action-{index:04d}")
    metadata = _mapping(action.get("metadata"))
    command_templates = _string_tuple(action.get("suggested_commands", ()))
    placeholder_count = sum(command.count("...") for command in command_templates)
    required_inputs = _string_tuple(metadata.get("required_inputs", ()))
    closure_outputs = _string_tuple(metadata.get("closure_outputs", ()))
    missing_inputs = _missing_inputs(
        required_inputs=required_inputs,
        placeholder_count=placeholder_count,
        command_templates=command_templates,
    )
    if not command_templates:
        command_status = "missing_command_templates"
    elif missing_inputs:
        command_status = "needs_inputs"
    else:
        command_status = "ready"
    bound_output_dir = plan_root / _slug(action_id)
    planned_outputs = tuple(
        {
            "name": output,
            "path": str(bound_output_dir / f"{_slug(output)}.json"),
            "status": "planned",
        }
        for output in closure_outputs
    )
    return {
        "entry_id": f"frontier-research-{index:04d}",
        "action_id": action_id,
        "title": str(action.get("title") or action_id),
        "action_type": str(action.get("action_type") or "workflow"),
        "priority": _int_or_zero(action.get("priority")),
        "command_status": command_status,
        "evidence_routes": _string_tuple(action.get("evidence_routes", ())),
        "source_gap_ids": _string_tuple(action.get("source_gap_ids", ())),
        "command_templates": command_templates,
        "required_inputs": required_inputs,
        "missing_inputs": missing_inputs,
        "planned_outputs": planned_outputs,
        "binding_hints": {
            "action_id": action_id,
            "bound_output_dir": str(bound_output_dir),
            "command_templates_need_binding": placeholder_count > 0,
            "input_bindings": tuple(
                {
                    "name": name,
                    "placeholder": "..." if name == "bound_command_template_values" else f"<{name}>",
                    "required": True,
                    "status": "unbound",
                }
                for name in tuple(dict.fromkeys((*required_inputs, *missing_inputs)))
            ),
            "output_bindings": planned_outputs,
        },
        "command_summary": {
            "command_template_count": len(command_templates),
            "placeholder_count": placeholder_count,
            "missing_input_count": len(missing_inputs),
            "planned_output_count": len(planned_outputs),
        },
        "metadata": {
            "workflow_keys": _workflow_keys(metadata),
            "required_input_count": len(required_inputs),
            "closure_output_count": len(closure_outputs),
        },
    }


def _missing_inputs(
    *,
    required_inputs: Sequence[str],
    placeholder_count: int,
    command_templates: Sequence[str],
) -> tuple[str, ...]:
    missing = list(required_inputs)
    if placeholder_count > 0:
        missing.append("bound_command_template_values")
    if not command_templates:
        return ()
    return tuple(dict.fromkeys(str(item) for item in missing if str(item)))


def _summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("command_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    action_ids = tuple(str(entry.get("action_id") or "") for entry in entries)
    route_counts: dict[str, int] = {}
    for entry in entries:
        for route in _string_tuple(entry.get("evidence_routes", ())):
            route_counts[route] = route_counts.get(route, 0) + 1
    return {
        "entry_count": len(entries),
        "ready_entry_count": status_counts.get("ready", 0),
        "needs_input_entry_count": status_counts.get("needs_inputs", 0),
        "missing_command_template_count": status_counts.get("missing_command_templates", 0),
        "command_count": sum(
            len(_string_tuple(entry.get("command_templates", ()))) for entry in entries
        ),
        "placeholder_count": sum(
            int(_mapping(entry.get("command_summary")).get("placeholder_count", 0))
            for entry in entries
        ),
        "missing_input_count": sum(
            len(_string_tuple(entry.get("missing_inputs", ()))) for entry in entries
        ),
        "planned_output_count": sum(
            len(_mapping_sequence(entry.get("planned_outputs", ()))) for entry in entries
        ),
        "action_ids": action_ids,
        "evidence_route_counts": dict(sorted(route_counts.items())),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if _int_or_zero(summary.get("entry_count")) == 0:
        return "empty"
    if _int_or_zero(summary.get("missing_command_template_count")) > 0:
        return "needs_commands"
    if _int_or_zero(summary.get("needs_input_entry_count")) > 0:
        return "needs_inputs"
    return "ready"


def _plan_root(
    *,
    source_path: Path | None,
    output_path: Path | None,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if output_path is not None:
        return output_path.parent / "frontier-research-queue-commands"
    if source_path is not None:
        return source_path.parent / "frontier-research-queue-commands"
    return ROOT / "artifacts" / "frontier-research-queue-commands"


def _workflow_keys(metadata: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in metadata.items()
        if key.endswith("_workflow") and isinstance(value, str) and value
    }


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    source_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts = {
        name: path
        for name, path in {
            "frontier_research_queue_command_plan": output_path,
            "source": source_path,
        }.items()
        if path is not None
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "plan_frontier_research_queue_commands",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "entry_count": _nested(payload, "summary", "entry_count"),
            "command_count": _nested(payload, "summary", "command_count"),
            "missing_input_count": _nested(payload, "summary", "missing_input_count"),
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


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value)).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "item"


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="frontier status report or evidence-gap plan")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--output-dir", default=None, help="planned output directory root")
    parser.add_argument("--include-action-id", action="append", default=())
    parser.add_argument("--exclude-action-id", action="append", default=())
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_frontier_research_queue_command_plan(
        source=args.source,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        output_dir=args.output_dir,
        include_action_ids=tuple(args.include_action_id or ()),
        exclude_action_ids=tuple(args.exclude_action_id or ()),
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
