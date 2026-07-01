"""Scaffold reviewed bindings for a frontier research-queue command plan."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.frontier_research_command_requirements import (  # noqa: E402
    frontier_command_requirement_summary,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

COMMAND_PLAN_WORKFLOW = "frontier_research_queue_command_plan"
WORKFLOW = "frontier_research_queue_binding_scaffold"
BINDINGS_WORKFLOW = "frontier_research_queue_command_bindings"


def scaffold_frontier_research_queue_bindings(
    *,
    command_plan: str | Path | Mapping[str, Any],
    json_path: str | Path | None = None,
    bindings_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    registry_output_path: str | Path | None = None,
    default_version: str = "0.1",
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a review scaffold for frontier command-plan bindings.

    The optional bindings JSON is intentionally unbound by default. It is a
    skeleton that can be edited after review and then passed to
    ``bind_frontier_research_queue_command_plan.py``.
    """
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    command_plan_path, command_plan_payload = _load_mapping_source(command_plan)
    if command_plan_payload.get("workflow") != COMMAND_PLAN_WORKFLOW:
        raise ValueError(f"command_plan must have workflow={COMMAND_PLAN_WORKFLOW!r}.")
    output_path = None if json_path is None else Path(json_path)
    bindings_path = None if bindings_json_path is None else Path(bindings_json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    if registry_path is not None and output_path is None and command_plan_path is None:
        raise ValueError("registry_path requires json_path when command_plan is in-memory.")
    entries = tuple(
        _scaffold_entry(
            entry,
            registry_output_path=registry_output_path,
            default_version=default_version,
        )
        for entry in _mapping_sequence(command_plan_payload.get("entries", ()))
    )
    bindings_sidecar = _bindings_sidecar(
        command_plan_path=command_plan_path,
        command_plan_payload=command_plan_payload,
        entries=entries,
    )
    summary = _summary(entries)
    status = _status(summary)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "source": {
            "command_plan": None if command_plan_path is None else str(command_plan_path),
            "command_plan_workflow": command_plan_payload.get("workflow"),
            "command_plan_status": command_plan_payload.get("status"),
        },
        "summary": summary,
        "paths": {
            "binding_scaffold": None if output_path is None else str(output_path),
            "bindings_json": None if bindings_path is None else str(bindings_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
        },
        "config": {
            "auto_binds_values": False,
            "default_version": str(default_version),
            "registry_output_path": None
            if registry_output_path is None
            else str(registry_output_path),
        },
        "entries": entries,
        "bindings_sidecar": bindings_sidecar,
        "metadata": dict(metadata or {}),
    }
    if output_path is not None:
        _write_json(output_path, payload, compact=compact_json)
    if bindings_path is not None:
        _write_json(bindings_path, bindings_sidecar, compact=compact_json)
    manifest = None
    if manifest_path is not None:
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            bindings_path=bindings_path,
            command_plan_path=command_plan_path,
            payload=payload,
            metadata=metadata or {},
            compact=compact_json,
        )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else command_plan_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "command_plan": None if command_plan_path is None else str(command_plan_path),
                "bindings_json": None if bindings_path is None else str(bindings_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "entry_count": summary["entry_count"],
                "required_input_count": summary["required_input_count"],
                "placeholder_count": summary["placeholder_count"],
                "suggested_binding_count": summary["suggested_binding_count"],
                "command_requirement_issue_count": summary["command_requirement_issue_count"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _scaffold_entry(
    entry: Mapping[str, Any],
    *,
    registry_output_path: str | Path | None,
    default_version: str,
) -> dict[str, Any]:
    action_id = str(entry.get("action_id") or entry.get("entry_id") or "frontier-action")
    planned_outputs = tuple(_mapping_sequence(entry.get("planned_outputs", ())))
    command_templates = _string_tuple(entry.get("command_templates", ()))
    required_input_names = _string_tuple(entry.get("required_inputs", ()))
    command_requirements = tuple(
        frontier_command_requirement_summary(
            command,
            index=command_index,
            required_inputs=required_input_names,
        )
        for command_index, command in enumerate(command_templates, start=1)
    )
    placeholder_records = []
    output_index = 0
    for command_index, command in enumerate(command_templates, start=1):
        records, output_index = _placeholder_records(
            command,
            action_id=action_id,
            command_index=command_index,
            planned_outputs=planned_outputs,
            output_index=output_index,
            registry_output_path=registry_output_path,
            default_version=default_version,
        )
        placeholder_records.extend(records)
    required_inputs = tuple(
        {
            "name": name,
            "status": "unbound",
            "suggested_binding": _input_suggestion(name),
        }
        for name in required_input_names
    )
    command_requirement_issue_count = _command_requirement_issue_count(command_requirements)
    return {
        "entry_id": str(entry.get("entry_id") or action_id),
        "action_id": action_id,
        "title": str(entry.get("title") or action_id),
        "priority": _int_or_zero(entry.get("priority")),
        "command_status": str(entry.get("command_status") or "unknown"),
        "evidence_routes": _string_tuple(entry.get("evidence_routes", ())),
        "source_gap_ids": _string_tuple(entry.get("source_gap_ids", ())),
        "required_inputs": required_inputs,
        "command_requirements": command_requirements,
        "placeholder_records": tuple(placeholder_records),
        "binding_summary": {
            "required_input_count": len(required_inputs),
            "command_template_count": len(command_templates),
            "placeholder_count": len(placeholder_records),
            "suggested_binding_count": sum(
                1 for record in placeholder_records if record.get("suggested_binding")
            ),
            "command_requirement_issue_count": command_requirement_issue_count,
        },
    }


def _placeholder_records(
    command: str,
    *,
    action_id: str,
    command_index: int,
    planned_outputs: Sequence[Mapping[str, Any]],
    output_index: int,
    registry_output_path: str | Path | None,
    default_version: str,
) -> tuple[tuple[dict[str, Any], ...], int]:
    tokens = _command_tokens(command)
    records = []
    placeholder_seen = 0
    previous_report_path: str | None = None
    for token_index, token in enumerate(tokens):
        if token != "...":
            continue
        placeholder_seen += 1
        flag = _previous_flag(tokens, token_index)
        suggestion, output_index, previous_report_path = _placeholder_suggestion(
            flag=flag,
            action_id=action_id,
            command_index=command_index,
            placeholder_index=placeholder_seen,
            planned_outputs=planned_outputs,
            output_index=output_index,
            previous_report_path=previous_report_path,
            registry_output_path=registry_output_path,
            default_version=default_version,
        )
        records.append({
            "command_index": command_index,
            "placeholder_index": placeholder_seen,
            "token_index": token_index,
            "flag": flag,
            "status": "unbound",
            "suggested_binding": suggestion,
            "context": _context(tokens, token_index),
        })
    return tuple(records), output_index


def _placeholder_suggestion(
    *,
    flag: str | None,
    action_id: str,
    command_index: int,
    placeholder_index: int,
    planned_outputs: Sequence[Mapping[str, Any]],
    output_index: int,
    previous_report_path: str | None,
    registry_output_path: str | Path | None,
    default_version: str,
) -> tuple[Mapping[str, Any] | None, int, str | None]:
    normalized = "" if flag is None else flag.lstrip("-").replace("-", "_")
    if normalized in {"json", "audit_json"}:
        if output_index < len(planned_outputs):
            path = _optional_str(planned_outputs[output_index].get("path"))
            output_index += 1
            if path:
                return {"path": path, "source": "planned_output"}, output_index, path
        return None, output_index, previous_report_path
    if normalized == "artifact_manifest":
        path = _manifest_suggestion(previous_report_path, action_id, command_index)
        return {"path": path, "source": "derived_manifest_path"}, output_index, previous_report_path
    if normalized == "registry" and registry_output_path is not None:
        return {"path": str(registry_output_path), "source": "registry_output_path"}, output_index, previous_report_path
    if normalized == "name":
        return {
            "value": f"{_slug(action_id)}-command-{command_index}",
            "source": "action_id_command_index",
        }, output_index, previous_report_path
    if normalized == "version":
        return {"value": str(default_version), "source": "default_version"}, output_index, previous_report_path
    if normalized.startswith("min_") or normalized.endswith("_rate"):
        return {"review_required": True, "reason": "metric_gate_threshold"}, output_index, previous_report_path
    if normalized.startswith("max_"):
        return {"review_required": True, "reason": "metric_gate_threshold"}, output_index, previous_report_path
    if normalized:
        return {
            "review_required": True,
            "reason": "input_or_report_path",
            "input_name_hint": normalized,
        }, output_index, previous_report_path
    return {
        "review_required": True,
        "reason": "unlabeled_placeholder",
        "placeholder_name": f"command_{command_index}_placeholder_{placeholder_index}",
    }, output_index, previous_report_path


def _manifest_suggestion(previous_report_path: str | None, action_id: str, command_index: int) -> str:
    if previous_report_path:
        return str(Path(previous_report_path).with_name("artifact-manifest.json"))
    return str(Path("artifacts") / _slug(action_id) / f"command-{command_index}-manifest.json")


def _input_suggestion(name: str) -> Mapping[str, Any]:
    return {"review_required": True, "input_name_hint": str(name)}


def _bindings_sidecar(
    *,
    command_plan_path: Path | None,
    command_plan_payload: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": BINDINGS_WORKFLOW,
        "status": "needs_review",
        "source": {
            "command_plan": None if command_plan_path is None else str(command_plan_path),
            "command_plan_status": command_plan_payload.get("status"),
        },
        "inputs": {},
        "bindings": {
            str(entry.get("action_id")): {
                "command_template_values": (),
                "review_status": "needs_review",
                "required_inputs": tuple(
                    item["name"] for item in _mapping_sequence(entry.get("required_inputs", ()))
                ),
                "command_requirements": tuple(
                    _mapping_sequence(entry.get("command_requirements", ()))
                ),
                "command_requirement_issue_count": _int_or_zero(
                    _nested(entry, "binding_summary", "command_requirement_issue_count")
                ),
                "placeholder_count": _int_or_zero(
                    _nested(entry, "binding_summary", "placeholder_count")
                ),
            }
            for entry in entries
            if str(entry.get("action_id") or "")
        },
    }


def _summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "entry_count": len(entries),
        "required_input_count": sum(
            len(_mapping_sequence(entry.get("required_inputs", ()))) for entry in entries
        ),
        "placeholder_count": sum(
            _int_or_zero(_nested(entry, "binding_summary", "placeholder_count"))
            for entry in entries
        ),
        "suggested_binding_count": sum(
            _int_or_zero(_nested(entry, "binding_summary", "suggested_binding_count"))
            for entry in entries
        ),
        "command_requirement_issue_count": sum(
            _int_or_zero(_nested(entry, "binding_summary", "command_requirement_issue_count"))
            for entry in entries
        ),
        "action_ids": tuple(str(entry.get("action_id") or "") for entry in entries),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if _int_or_zero(summary.get("entry_count")) == 0:
        return "empty"
    if _int_or_zero(summary.get("required_input_count")) or _int_or_zero(
        summary.get("placeholder_count")
    ):
        return "needs_review"
    return "ready"


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    bindings_path: Path | None,
    command_plan_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "frontier_research_queue_binding_scaffold": output_path,
        "bindings_json": bindings_path,
        "command_plan": command_plan_path,
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "scaffold_frontier_research_queue_bindings",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "entry_count": _nested(payload, "summary", "entry_count"),
            "required_input_count": _nested(payload, "summary", "required_input_count"),
            "placeholder_count": _nested(payload, "summary", "placeholder_count"),
            "suggested_binding_count": _nested(payload, "summary", "suggested_binding_count"),
            "command_requirement_issue_count": _nested(
                payload,
                "summary",
                "command_requirement_issue_count",
            ),
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


def _command_requirement_issue_count(requirements: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for requirement in requirements:
        count += len(_string_tuple(requirement.get("missing_required_flags")))
        count += len(_mapping_sequence(requirement.get("missing_required_input_flags")))
    return count


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _command_tokens(command: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(command))
    except ValueError:
        return tuple(str(command).split())


def _previous_flag(tokens: Sequence[str], index: int) -> str | None:
    if index <= 0:
        return None
    previous = tokens[index - 1]
    return previous if previous.startswith("--") else None


def _context(tokens: Sequence[str], index: int) -> dict[str, Any]:
    start = max(0, index - 2)
    end = min(len(tokens), index + 3)
    return {
        "before": tuple(tokens[start:index]),
        "after": tuple(tokens[index + 1 : end]),
    }


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


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
    parser.add_argument("--command-plan", required=True, help="frontier command plan JSON")
    parser.add_argument("--json", default=None, help="optional scaffold JSON path")
    parser.add_argument("--bindings-json", default=None, help="optional binding sidecar skeleton path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--registry-output-path", default=None, help="suggested registry path")
    parser.add_argument("--default-version", default="0.1", help="suggested artifact version")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = scaffold_frontier_research_queue_bindings(
        command_plan=args.command_plan,
        json_path=args.json,
        bindings_json_path=args.bindings_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        registry_output_path=args.registry_output_path,
        default_version=args.default_version,
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
