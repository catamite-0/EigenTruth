"""Bind runtime-drift completion command templates without executing them."""

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

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

COMPLETION_WORKFLOW = "runtime_drift_evidence_completion_plan"
WORKFLOW = "runtime_drift_evidence_bound_command_plan"


def build_runtime_drift_bound_command_plan(
    *,
    completion_plan: str | Path | Mapping[str, Any],
    bindings: str | Path | Mapping[str, Any],
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind command-template placeholders in a runtime-drift completion plan.

    This is still a planning artifact. It does not execute commands and it does
    not turn missing evidence into verifier or release evidence.
    """
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    completion_path, completion_payload = _load_mapping_source(completion_plan)
    bindings_path, bindings_payload = _load_mapping_source(bindings)
    if completion_payload.get("workflow") != COMPLETION_WORKFLOW:
        raise ValueError(f"completion_plan must have workflow={COMPLETION_WORKFLOW!r}.")
    entries = tuple(_mapping_sequence(completion_payload.get("entries", ())))
    bindings_by_action = _bindings_by_action(bindings_payload)
    global_inputs = _input_bindings(bindings_payload)
    bound_entries = tuple(
        _bind_entry(entry, bindings_by_action.get(str(entry.get("action_id")), {}), global_inputs)
        for entry in entries
    )
    summary = _bound_summary(bound_entries)
    status = _bound_status(summary)
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "source": {
            "completion_plan": None if completion_path is None else str(completion_path),
            "bindings": None if bindings_path is None else str(bindings_path),
            "completion_workflow": completion_payload.get("workflow"),
            "completion_status": completion_payload.get("status"),
        },
        "summary": summary,
        "paths": {
            "bound_command_plan": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
        },
        "config": {
            "binding_mode": "ordered_template_values_or_bound_commands",
            "executes_commands": False,
        },
        "entries": bound_entries,
        "metadata": dict(metadata or {}),
    }
    if output_path is not None:
        _write_json(output_path, payload, compact=compact_json)
    manifest = None
    if manifest_path is not None:
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            completion_path=completion_path,
            bindings_path=bindings_path,
            payload=payload,
            metadata=metadata or {},
            compact=compact_json,
        )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else completion_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "completion_plan": None if completion_path is None else str(completion_path),
                "bindings": None if bindings_path is None else str(bindings_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "entry_count": summary["entry_count"],
                "ready_entry_count": summary["ready_entry_count"],
                "command_count": summary["command_count"],
                "unbound_placeholder_count": summary["unbound_placeholder_count"],
                "missing_input_count": summary["missing_input_count"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _bind_entry(
    entry: Mapping[str, Any],
    entry_bindings: Mapping[str, Any],
    global_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    action_id = str(entry.get("action_id") or "")
    binding_hints = _mapping(entry.get("binding_hints"))
    input_hints = tuple(_mapping_sequence(binding_hints.get("input_bindings", ())))
    planned_outputs = tuple(_mapping_sequence(binding_hints.get("output_bindings", ())))
    command_templates = _string_tuple(entry.get("command_templates", ()))
    entry_inputs = {
        **global_inputs,
        **_input_bindings(entry_bindings),
    }
    supplied_bound_commands = _string_tuple(entry_bindings.get("bound_commands", ()))
    if supplied_bound_commands:
        command_result = _use_supplied_bound_commands(supplied_bound_commands)
    else:
        command_result = _bind_command_templates(
            command_templates,
            _command_template_values(entry_bindings),
        )
    unbound_inputs = _unbound_inputs(
        input_hints=input_hints,
        entry_inputs=entry_inputs,
        command_result=command_result,
    )
    if not command_templates and not supplied_bound_commands:
        command_status = "missing_command_templates"
    elif unbound_inputs or command_result["unbound_placeholder_count"]:
        command_status = "needs_inputs"
    else:
        command_status = "ready"
    return {
        "entry_id": str(entry.get("entry_id") or action_id),
        "action_id": action_id,
        "title": str(entry.get("title") or action_id),
        "source_command_status": str(entry.get("command_status") or "unknown"),
        "command_status": command_status,
        "evidence_routes": _string_tuple(entry.get("evidence_routes", ())),
        "missing_metrics": _string_tuple(entry.get("missing_metrics", ())),
        "required_inputs": _string_tuple(entry.get("required_inputs", ())),
        "bound_inputs": _json_ready_inputs(entry_inputs),
        "unbound_inputs": tuple(unbound_inputs),
        "bound_commands": tuple(command_result["commands"]),
        "planned_outputs": planned_outputs,
        "binding_summary": {
            "command_template_count": len(command_templates),
            "command_count": len(command_result["commands"]),
            "placeholder_count": command_result["placeholder_count"],
            "bound_placeholder_count": command_result["bound_placeholder_count"],
            "unbound_placeholder_count": command_result["unbound_placeholder_count"],
            "unused_binding_value_count": command_result["unused_value_count"],
            "planned_output_count": len(planned_outputs),
        },
    }


def _bind_command_templates(
    command_templates: Sequence[str],
    values: Sequence[Any],
) -> dict[str, Any]:
    rendered_values = [_render_binding_value(value) for value in values]
    value_index = 0
    commands: list[str] = []
    placeholder_count = sum(str(template).count("...") for template in command_templates)
    bound_placeholder_count = 0
    for template in command_templates:
        command = str(template)
        while "..." in command:
            if value_index >= len(rendered_values):
                break
            command = command.replace("...", rendered_values[value_index], 1)
            value_index += 1
            bound_placeholder_count += 1
        commands.append(command)
    return {
        "commands": tuple(commands),
        "placeholder_count": placeholder_count,
        "bound_placeholder_count": bound_placeholder_count,
        "unbound_placeholder_count": placeholder_count - bound_placeholder_count,
        "unused_value_count": max(0, len(rendered_values) - value_index),
    }


def _use_supplied_bound_commands(commands: Sequence[str]) -> dict[str, Any]:
    placeholder_count = sum(str(command).count("...") for command in commands)
    return {
        "commands": tuple(str(command) for command in commands),
        "placeholder_count": placeholder_count,
        "bound_placeholder_count": 0,
        "unbound_placeholder_count": placeholder_count,
        "unused_value_count": 0,
    }


def _unbound_inputs(
    *,
    input_hints: Sequence[Mapping[str, Any]],
    entry_inputs: Mapping[str, Any],
    command_result: Mapping[str, Any],
) -> tuple[str, ...]:
    missing: list[str] = []
    for hint in input_hints:
        if hint.get("required") is False:
            continue
        name = str(hint.get("name") or "")
        if not name:
            continue
        if name == "bound_command_template_values":
            if int(command_result["unbound_placeholder_count"]) > 0:
                missing.append(name)
            continue
        if name not in entry_inputs or _is_empty_binding(entry_inputs[name]):
            missing.append(name)
    return tuple(dict.fromkeys(missing))


def _bound_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("command_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "entry_count": len(entries),
        "ready_entry_count": status_counts.get("ready", 0),
        "needs_input_entry_count": status_counts.get("needs_inputs", 0),
        "missing_command_template_count": status_counts.get("missing_command_templates", 0),
        "command_count": sum(
            len(_string_tuple(entry.get("bound_commands", ()))) for entry in entries
        ),
        "placeholder_count": sum(
            int(_mapping(entry.get("binding_summary")).get("placeholder_count", 0))
            for entry in entries
        ),
        "bound_placeholder_count": sum(
            int(_mapping(entry.get("binding_summary")).get("bound_placeholder_count", 0))
            for entry in entries
        ),
        "unbound_placeholder_count": sum(
            int(_mapping(entry.get("binding_summary")).get("unbound_placeholder_count", 0))
            for entry in entries
        ),
        "missing_input_count": sum(
            len(_string_tuple(entry.get("unbound_inputs", ()))) for entry in entries
        ),
        "expected_output_count": sum(
            len(_mapping_sequence(entry.get("planned_outputs", ()))) for entry in entries
        ),
        "command_status_counts": dict(sorted(status_counts.items())),
    }


def _bound_status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("entry_count", 0)) == 0:
        return "empty"
    if int(summary.get("missing_command_template_count", 0)) > 0:
        return "needs_inputs"
    if int(summary.get("missing_input_count", 0)) > 0:
        return "needs_inputs"
    if int(summary.get("unbound_placeholder_count", 0)) > 0:
        return "needs_inputs"
    return "ready"


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    completion_path: Path | None,
    bindings_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "runtime_drift_bound_command_plan": output_path,
        "completion_plan": completion_path,
        "bindings": bindings_path,
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "bind_runtime_drift_completion_plan",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "entry_count": _nested_value(payload, "summary", "entry_count"),
            "ready_entry_count": _nested_value(payload, "summary", "ready_entry_count"),
            "command_count": _nested_value(payload, "summary", "command_count"),
            "unbound_placeholder_count": _nested_value(
                payload,
                "summary",
                "unbound_placeholder_count",
            ),
            "missing_input_count": _nested_value(payload, "summary", "missing_input_count"),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def _bindings_by_action(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = payload.get("entries", payload.get("action_bindings", payload.get("bindings", {})))
    if isinstance(raw, Mapping):
        return {str(key): _mapping(value) for key, value in raw.items()}
    return {
        str(item.get("action_id")): item
        for item in _mapping_sequence(raw)
        if str(item.get("action_id", ""))
    }


def _input_bindings(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("inputs", payload.get("input_bindings", {}))
    if not isinstance(raw, Mapping):
        raise ValueError("inputs/input_bindings must be a JSON object when supplied.")
    return {str(key): value for key, value in raw.items() if str(key)}


def _command_template_values(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = payload.get("command_template_values", payload.get("bound_command_template_values", ()))
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise ValueError("command_template_values must be a JSON array when supplied.")
    return tuple(raw)


def _render_binding_value(value: Any) -> str:
    if isinstance(value, Mapping):
        if "raw" in value:
            return str(value["raw"])
        if "shell" in value:
            return str(value["shell"])
        if "path" in value:
            return shlex.quote(str(value["path"]))
        if "value" in value:
            return shlex.quote(str(value["value"]))
        raise ValueError("binding value objects must contain raw, shell, path, or value.")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_render_binding_value(item) for item in value)
    return shlex.quote(str(value))


def _json_ready_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready_value(value) for key, value in sorted(inputs.items())}


def _json_ready_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready_value(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_json_ready_value(item) for item in value)
    return value


def _is_empty_binding(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) == 0
    if isinstance(value, Mapping):
        return len(value) == 0
    return False


def _load_mapping_source(source: str | Path | Mapping[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    if isinstance(source, Mapping):
        return None, dict(source)
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return path, dict(payload)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _nested_value(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


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
    parser.add_argument("--completion-plan", required=True, help="runtime-drift completion plan JSON")
    parser.add_argument("--bindings", required=True, help="binding sidecar JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_runtime_drift_bound_command_plan(
        completion_plan=args.completion_plan,
        bindings=args.bindings,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
