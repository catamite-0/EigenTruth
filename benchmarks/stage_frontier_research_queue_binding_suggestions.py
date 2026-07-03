"""Stage safe frontier command binding suggestions for review.

This script consumes ``frontier_research_queue_binding_scaffold`` output and
materializes only non-review-required placeholder suggestions into
``bound_commands``. It does not fill source-backed inputs, does not mark entries
as reviewed, and does not execute commands.
"""

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

SCAFFOLD_WORKFLOW = "frontier_research_queue_binding_scaffold"
BINDINGS_WORKFLOW = "frontier_research_queue_command_bindings"
WORKFLOW = "frontier_research_queue_binding_suggestion_staging"


def stage_frontier_research_queue_binding_suggestions(
    *,
    scaffold: str | Path | Mapping[str, Any],
    bindings_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    stage_upstream_outputs: bool = False,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bindings sidecar with safe suggestions staged for review."""
    if artifact_manifest_path is not None and bindings_json_path is None:
        raise ValueError("artifact_manifest_path requires bindings_json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if registry_path is not None and bindings_json_path is None:
        raise ValueError("registry_path requires bindings_json_path.")

    scaffold_path, scaffold_payload = _load_mapping_source(scaffold)
    if scaffold_payload.get("workflow") != SCAFFOLD_WORKFLOW:
        raise ValueError(f"scaffold must have workflow={SCAFFOLD_WORKFLOW!r}.")

    output_path = None if bindings_json_path is None else Path(bindings_json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    entries = tuple(
        _stage_entry(entry, stage_upstream_outputs=stage_upstream_outputs)
        for entry in _mapping_sequence(scaffold_payload.get("entries", ()))
    )
    summary = _summary(entries)
    status = "empty" if summary["entry_count"] == 0 else "needs_review"
    payload = {
        "schema_version": 1,
        "workflow": BINDINGS_WORKFLOW,
        "status": status,
        "source": {
            "binding_scaffold": None if scaffold_path is None else str(scaffold_path),
            "binding_scaffold_workflow": scaffold_payload.get("workflow"),
            "command_plan": _nested(scaffold_payload, "source", "command_plan"),
        },
        "config": {
            "stage_upstream_outputs": bool(stage_upstream_outputs),
            "auto_approves_review": False,
            "auto_binds_source_backed_inputs": False,
        },
        "generated_by": WORKFLOW,
        "review_summary": {
            "entry_count": summary["entry_count"],
            "required_input_count": summary["required_input_count"],
            "placeholder_count": summary["placeholder_count"],
            "staged_placeholder_count": summary["staged_placeholder_count"],
            "staged_upstream_output_count": summary["staged_upstream_output_count"],
            "remaining_placeholder_count": summary["remaining_placeholder_count"],
            "review_required_placeholder_count": summary["review_required_placeholder_count"],
            "missing_suggestion_placeholder_count": summary[
                "missing_suggestion_placeholder_count"
            ],
        },
        "staging_summary": summary,
        "inputs": {},
        "bindings": {
            str(entry["action_id"]): {
                "bound_commands": entry["bound_commands"],
                "command_template_values": (),
                "review_status": "needs_review",
                "required_inputs": entry["required_inputs"],
                "command_requirements": entry["command_requirements"],
                "command_requirement_issue_count": entry["command_requirement_issue_count"],
                "placeholder_count": entry["placeholder_count"],
                "staged_placeholder_count": entry["staged_placeholder_count"],
                "staged_upstream_output_count": entry["staged_upstream_output_count"],
                "remaining_placeholder_count": entry["remaining_placeholder_count"],
                "input_reviews": entry["input_reviews"],
                "placeholder_reviews": entry["placeholder_reviews"],
            }
            for entry in entries
            if str(entry.get("action_id") or "")
        },
        "metadata": dict(metadata or {}),
    }
    if output_path is not None:
        _write_json(output_path, payload, compact=compact_json)
    manifest = None
    if manifest_path is not None:
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            scaffold_path=scaffold_path,
            payload=payload,
            metadata=metadata or {},
            compact=compact_json,
        )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path,
            metadata={
                "workflow": WORKFLOW,
                "bindings_workflow": BINDINGS_WORKFLOW,
                "status": status,
                "binding_scaffold": None if scaffold_path is None else str(scaffold_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "entry_count": summary["entry_count"],
                "placeholder_count": summary["placeholder_count"],
                "staged_placeholder_count": summary["staged_placeholder_count"],
                "staged_upstream_output_count": summary["staged_upstream_output_count"],
                "remaining_placeholder_count": summary["remaining_placeholder_count"],
                "review_required_placeholder_count": summary[
                    "review_required_placeholder_count"
                ],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _stage_entry(
    entry: Mapping[str, Any],
    *,
    stage_upstream_outputs: bool,
) -> dict[str, Any]:
    action_id = str(entry.get("action_id") or entry.get("entry_id") or "frontier-action")
    templates = _string_tuple(entry.get("command_templates", ()))
    placeholder_records = tuple(
        sorted(
            _mapping_sequence(entry.get("placeholder_records", ())),
            key=lambda item: (
                _int_or_zero(item.get("command_index")),
                _int_or_zero(item.get("placeholder_index")),
            ),
        )
    )
    records_by_command: dict[int, list[Mapping[str, Any]]] = {}
    for record in placeholder_records:
        records_by_command.setdefault(_int_or_zero(record.get("command_index")), []).append(record)

    staged_count = 0
    staged_upstream_count = 0
    review_required_count = 0
    missing_suggestion_count = 0
    bound_commands = []
    staged_placeholder_reviews = []
    available_outputs: dict[str, str] = {}
    for command_index, template in enumerate(templates, start=1):
        command = str(template)
        script = _command_script(command)
        for record in records_by_command.get(command_index, []):
            flag = str(record.get("flag") or "")
            normalized_flag = _normalized_flag(flag)
            suggestion = _mapping(record.get("suggested_binding"))
            placeholder_index = _int_or_zero(record.get("placeholder_index"))
            replacement = _placeholder_sentinel(command_index, placeholder_index)
            stage_status = "needs_review"
            upstream_input = _suggestion_upstream_input(suggestion)
            if stage_upstream_outputs and upstream_input in available_outputs:
                replacement = available_outputs[upstream_input]
                staged_count += 1
                staged_upstream_count += 1
                stage_status = "staged_upstream_output"
            elif suggestion.get("review_required") is True:
                review_required_count += 1
            else:
                rendered = _render_suggestion(suggestion)
                if rendered is None:
                    missing_suggestion_count += 1
                else:
                    replacement = rendered
                    staged_count += 1
                    stage_status = "staged"
            command = command.replace("...", replacement, 1)
            _remember_output(
                available_outputs,
                flag=normalized_flag,
                script=script,
                replacement=replacement,
            )
            staged_placeholder_reviews.append({
                **dict(record),
                "stage_status": stage_status,
            })
        bound_commands.append(_restore_placeholder_sentinels(command))

    placeholder_count = len(placeholder_records)
    remaining_count = sum(command.count("...") for command in bound_commands)
    input_reviews = tuple(_mapping_sequence(entry.get("required_inputs", ())))
    command_requirements = tuple(_mapping_sequence(entry.get("command_requirements", ())))
    return {
        "action_id": action_id,
        "required_inputs": tuple(
            str(item.get("name") or "") for item in input_reviews if str(item.get("name") or "")
        ),
        "command_requirements": command_requirements,
        "command_requirement_issue_count": _int_or_zero(
            _nested(entry, "binding_summary", "command_requirement_issue_count")
        ),
        "bound_commands": tuple(bound_commands),
        "input_reviews": input_reviews,
        "placeholder_reviews": tuple(staged_placeholder_reviews),
        "placeholder_count": placeholder_count,
        "staged_placeholder_count": staged_count,
        "staged_upstream_output_count": staged_upstream_count,
        "remaining_placeholder_count": remaining_count,
        "review_required_placeholder_count": review_required_count,
        "missing_suggestion_placeholder_count": missing_suggestion_count,
    }


def _summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "entry_count": len(entries),
        "required_input_count": sum(
            len(_string_tuple(entry.get("required_inputs", ()))) for entry in entries
        ),
        "command_count": sum(
            len(_string_tuple(entry.get("bound_commands", ()))) for entry in entries
        ),
        "placeholder_count": sum(_int_or_zero(entry.get("placeholder_count")) for entry in entries),
        "staged_placeholder_count": sum(
            _int_or_zero(entry.get("staged_placeholder_count")) for entry in entries
        ),
        "staged_upstream_output_count": sum(
            _int_or_zero(entry.get("staged_upstream_output_count")) for entry in entries
        ),
        "remaining_placeholder_count": sum(
            _int_or_zero(entry.get("remaining_placeholder_count")) for entry in entries
        ),
        "review_required_placeholder_count": sum(
            _int_or_zero(entry.get("review_required_placeholder_count")) for entry in entries
        ),
        "missing_suggestion_placeholder_count": sum(
            _int_or_zero(entry.get("missing_suggestion_placeholder_count")) for entry in entries
        ),
        "action_ids": tuple(str(entry.get("action_id") or "") for entry in entries),
    }


def _render_suggestion(suggestion: Mapping[str, Any]) -> str | None:
    if not suggestion:
        return None
    if "raw" in suggestion:
        return str(suggestion["raw"])
    if "shell" in suggestion:
        return str(suggestion["shell"])
    if "path" in suggestion:
        return shlex.quote(str(suggestion["path"]))
    if "value" in suggestion:
        return shlex.quote(str(suggestion["value"]))
    return None


def _suggestion_upstream_input(suggestion: Mapping[str, Any]) -> str:
    if not suggestion:
        return ""
    if suggestion.get("reason") in {"upstream_command_output", "input_or_report_path"}:
        return str(suggestion.get("input_name_hint") or "")
    if suggestion.get("reason") == "required_input_flag":
        return str(suggestion.get("input_name") or "")
    return ""


def _remember_output(
    available_outputs: dict[str, str],
    *,
    flag: str,
    script: str | None,
    replacement: str,
) -> None:
    if replacement.startswith("__EIGENTRUTH_UNBOUND_PLACEHOLDER_"):
        return
    if flag == "acquisition_plan_jsonl":
        available_outputs["source_family_acquisition_plan"] = replacement
    elif flag == "tasks_jsonl":
        available_outputs["source_family_collection_tasks"] = replacement
        available_outputs["tasks"] = replacement
    elif flag == "rule_inputs_jsonl":
        available_outputs["rule_inputs"] = replacement
    elif flag == "rule_results_jsonl":
        available_outputs["rule_results"] = replacement
    elif flag == "json" and script == "benchmarks/run_world_model_rule_authoring_adapter.py":
        available_outputs["adapter_report"] = replacement
    elif flag == "json" and script == "benchmarks/bind_frontier_research_queue_command_plan.py":
        available_outputs["bound_command_plan"] = replacement
        available_outputs["reviewed_frontier_bound_command_plan"] = replacement


def _command_script(command: str) -> str | None:
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = str(command).split()
    for item in argv:
        if str(item).endswith(".py"):
            return _normalize_script_path(str(item))
    return None


def _normalize_script_path(path: str) -> str:
    text = str(path).replace("\\", "/")
    if "/benchmarks/" in text:
        return "benchmarks/" + text.rsplit("/benchmarks/", 1)[1]
    while text.startswith("./"):
        text = text[2:]
    return text


def _normalized_flag(flag: str | None) -> str:
    return "" if flag is None else str(flag).lstrip("-").replace("-", "_")


def _placeholder_sentinel(command_index: int, placeholder_index: int) -> str:
    return f"__EIGENTRUTH_UNBOUND_PLACEHOLDER_{command_index}_{placeholder_index}__"


def _restore_placeholder_sentinels(command: str) -> str:
    parts = []
    cursor = 0
    marker = "__EIGENTRUTH_UNBOUND_PLACEHOLDER_"
    while True:
        start = command.find(marker, cursor)
        if start < 0:
            parts.append(command[cursor:])
            break
        end = command.find("__", start + len(marker))
        if end < 0:
            parts.append(command[cursor:])
            break
        parts.append(command[cursor:start])
        parts.append("...")
        cursor = end + 2
    return "".join(parts)


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    scaffold_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "frontier_research_queue_staged_command_bindings": output_path,
        "binding_scaffold": scaffold_path,
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "stage_frontier_research_queue_binding_suggestions",
            "workflow": WORKFLOW,
            "bindings_workflow": BINDINGS_WORKFLOW,
            "status": payload.get("status"),
            "entry_count": _nested(payload, "staging_summary", "entry_count"),
            "staged_placeholder_count": _nested(
                payload,
                "staging_summary",
                "staged_placeholder_count",
            ),
            "staged_upstream_output_count": _nested(
                payload,
                "staging_summary",
                "staged_upstream_output_count",
            ),
            "remaining_placeholder_count": _nested(
                payload,
                "staging_summary",
                "remaining_placeholder_count",
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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
    parser.add_argument("--scaffold", required=True, help="frontier binding scaffold JSON")
    parser.add_argument("--bindings-json", default=None, help="output staged bindings sidecar")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument(
        "--stage-upstream-outputs",
        action="store_true",
        help="also bind upstream child-output placeholders when the same staged action produced them",
    )
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = stage_frontier_research_queue_binding_suggestions(
        scaffold=args.scaffold,
        bindings_json_path=args.bindings_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        stage_upstream_outputs=bool(args.stage_upstream_outputs),
        compact_json=bool(args.compact_json),
    )
    if args.bindings_json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
