"""Bind frontier research-queue command templates without executing them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bind_runtime_drift_completion_plan import (  # noqa: E402
    _bind_entry as _bind_template_entry,
)
from benchmarks.bind_runtime_drift_completion_plan import (  # noqa: E402
    _bindings_by_action,
    _bound_status,
    _input_bindings,
    _load_mapping_source,
    _mapping,
    _mapping_sequence,
    _nested_value,
    _string_tuple,
    _write_json,
)
from benchmarks.bind_runtime_drift_completion_plan import (  # noqa: E402
    _bound_summary as _base_bound_summary,
)
from benchmarks.frontier_research_command_requirements import (  # noqa: E402
    validate_frontier_bound_commands,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

COMMAND_PLAN_WORKFLOW = "frontier_research_queue_command_plan"
WORKFLOW = "frontier_research_queue_bound_command_plan"
APPROVED_REVIEW_STATUSES = ("approved", "reviewed")


def build_frontier_research_queue_bound_command_plan(
    *,
    command_plan: str | Path | Mapping[str, Any],
    bindings: str | Path | Mapping[str, Any],
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind frontier research command-template placeholders.

    This remains a planning artifact. It never executes commands and it does
    not convert a research queue into verifier or release evidence.
    """
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    command_plan_path, command_plan_payload = _load_mapping_source(command_plan)
    bindings_path, bindings_payload = _load_mapping_source(bindings)
    if command_plan_payload.get("workflow") != COMMAND_PLAN_WORKFLOW:
        raise ValueError(f"command_plan must have workflow={COMMAND_PLAN_WORKFLOW!r}.")
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    if registry_path is not None and output_path is None and command_plan_path is None:
        raise ValueError("registry_path requires json_path when command_plan is in-memory.")
    entries = tuple(_mapping_sequence(command_plan_payload.get("entries", ())))
    bindings_by_action = _bindings_by_action(bindings_payload)
    global_inputs = _input_bindings(bindings_payload)
    bound_entries = tuple(
        _bind_frontier_entry(
            entry,
            bindings_by_action.get(str(entry.get("action_id")), {}),
            global_inputs,
        )
        for entry in entries
    )
    summary = _frontier_bound_summary(bound_entries)
    status = _bound_status(summary)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "source": {
            "command_plan": None if command_plan_path is None else str(command_plan_path),
            "bindings": None if bindings_path is None else str(bindings_path),
            "command_plan_workflow": command_plan_payload.get("workflow"),
            "command_plan_status": command_plan_payload.get("status"),
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
            command_plan_path=command_plan_path,
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
            path=output_path if output_path is not None else command_plan_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "command_plan": None if command_plan_path is None else str(command_plan_path),
                "bindings": None if bindings_path is None else str(bindings_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "entry_count": summary["entry_count"],
                "ready_entry_count": summary["ready_entry_count"],
                "command_count": summary["command_count"],
                "unbound_placeholder_count": summary["unbound_placeholder_count"],
                "missing_input_count": summary["missing_input_count"],
                "review_required_entry_count": summary["review_required_entry_count"],
                "binding_review_status_counts": summary["binding_review_status_counts"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _bind_frontier_entry(
    entry: Mapping[str, Any],
    entry_bindings: Mapping[str, Any],
    global_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    bound = dict(_bind_template_entry(entry, entry_bindings, global_inputs))
    metadata = _mapping(entry.get("metadata"))
    planned_outputs = tuple(_mapping_sequence(entry.get("planned_outputs", ())))
    bound["action_type"] = str(entry.get("action_type") or "workflow")
    bound["priority"] = _int_or_zero(entry.get("priority"))
    bound["source_gap_ids"] = _string_tuple(entry.get("source_gap_ids", ()))
    if planned_outputs:
        bound["planned_outputs"] = planned_outputs
    validation = validate_frontier_bound_commands(
        _string_tuple(bound.get("bound_commands", ())),
        required_inputs=_string_tuple(entry.get("required_inputs", ())),
    )
    if validation["issue_count"]:
        bound["command_status"] = "needs_inputs"
        bound["unbound_inputs"] = tuple(
            dict.fromkeys((*_string_tuple(bound.get("unbound_inputs", ())), "valid_bound_commands"))
        )
    bound["metadata"] = {
        **dict(metadata),
        "workflow_keys": _workflow_keys(metadata) or dict(_mapping(metadata.get("workflow_keys"))),
        "source_required_input_count": len(_string_tuple(entry.get("required_inputs", ()))),
        "source_planned_output_count": len(_mapping_sequence(entry.get("planned_outputs", ()))),
    }
    review_status = _binding_review_status(entry_bindings)
    bound["binding_review_status"] = review_status
    bound["binding_review_required"] = review_status not in APPROVED_REVIEW_STATUSES
    bound["command_validation"] = validation
    return bound


def _frontier_bound_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = dict(_base_bound_summary(entries))
    route_counts: dict[str, int] = {}
    review_status_counts: dict[str, int] = {}
    source_gap_ids: list[str] = []
    for entry in entries:
        for route in _string_tuple(entry.get("evidence_routes", ())):
            route_counts[route] = route_counts.get(route, 0) + 1
        review_status = str(entry.get("binding_review_status") or "unknown")
        review_status_counts[review_status] = review_status_counts.get(review_status, 0) + 1
        source_gap_ids.extend(_string_tuple(entry.get("source_gap_ids", ())))
    summary["action_ids"] = tuple(str(entry.get("action_id") or "") for entry in entries)
    summary["evidence_route_counts"] = dict(sorted(route_counts.items()))
    summary["source_gap_ids"] = tuple(dict.fromkeys(source_gap_ids))
    summary["binding_review_status_counts"] = dict(sorted(review_status_counts.items()))
    summary["review_required_entry_count"] = sum(
        1 for entry in entries if entry.get("binding_review_required") is True
    )
    summary["command_validation_issue_count"] = sum(
        _int_or_zero(_mapping(entry.get("command_validation")).get("issue_count"))
        for entry in entries
    )
    return summary


def _binding_review_status(entry_bindings: Mapping[str, Any]) -> str:
    status = str(entry_bindings.get("review_status") or "").strip().lower()
    return status or "untracked"


def _workflow_keys(metadata: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in metadata.items()
        if key.endswith("_workflow") and isinstance(value, str) and value
    }


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    command_plan_path: Path | None,
    bindings_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "frontier_research_queue_bound_command_plan": output_path,
        "command_plan": command_plan_path,
        "bindings": bindings_path,
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "bind_frontier_research_queue_command_plan",
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
            "review_required_entry_count": _nested_value(
                payload,
                "summary",
                "review_required_entry_count",
            ),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-plan", required=True, help="frontier command plan JSON")
    parser.add_argument("--bindings", required=True, help="binding sidecar JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_frontier_research_queue_bound_command_plan(
        command_plan=args.command_plan,
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
