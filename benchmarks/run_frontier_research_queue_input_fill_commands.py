"""Dry-run or execute audited frontier input-fill commands.

This workflow consumes ``frontier_research_queue_input_binding_audit`` reports
and runs only the downstream fill commands that passed the sidecar audit. It is
an execution report, not verifier evidence or release evidence.
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

from benchmarks.audit_frontier_research_queue_input_bindings import (  # noqa: E402
    WORKFLOW as AUDIT_WORKFLOW,
)
from benchmarks.run_runtime_drift_bound_command_plan import (  # noqa: E402
    _run_entries,
    _status,
    _summary,
    _timeout,
    _worker_count,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_research_queue_input_fill_command_run_report"
OUTPUT_FLAGS = (
    ("report", "--json"),
    ("artifact_manifest", "--artifact-manifest"),
    ("rule_inputs", "--rule-inputs-jsonl"),
    ("unfilled_tasks", "--unfilled-tasks-jsonl"),
    ("skipped_bindings", "--skipped-bindings-jsonl"),
    ("skipped_subject_bindings", "--skipped-subject-bindings-jsonl"),
)


def run_frontier_research_queue_input_fill_commands(
    *,
    input_binding_audit: str | Path | Mapping[str, Any],
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    dry_run: bool = True,
    allow_partial_ready: bool = False,
    cwd: str | Path | None = None,
    python_executable: str = sys.executable,
    command_timeout_seconds: float | None = None,
    stop_on_failure: bool = True,
    workers: int = 1,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run report from audited frontier input-fill commands."""
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a bool.")
    if not isinstance(allow_partial_ready, bool):
        raise ValueError("allow_partial_ready must be a bool.")
    if not isinstance(stop_on_failure, bool):
        raise ValueError("stop_on_failure must be a bool.")
    worker_count = _worker_count(workers)
    if worker_count > 1 and stop_on_failure:
        raise ValueError("workers > 1 requires stop_on_failure=False.")
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    timeout = _timeout(command_timeout_seconds)
    source_path, audit = _load_mapping_source(input_binding_audit)
    if audit.get("workflow") != AUDIT_WORKFLOW:
        raise ValueError(f"input_binding_audit must have workflow={AUDIT_WORKFLOW!r}.")
    audit_ready = str(audit.get("status") or "") == "ready"
    working_dir = Path(cwd) if cwd is not None else ROOT
    entries = _entries_from_audit(
        audit,
        dry_run=dry_run,
        audit_ready=audit_ready,
        allow_partial_ready=allow_partial_ready,
    )
    executed_entries = _run_entries(
        entries,
        dry_run=dry_run,
        cwd=working_dir,
        python_executable=python_executable,
        timeout=timeout,
        stop_on_failure=stop_on_failure,
        workers=worker_count,
    )
    summary = _frontier_fill_summary(
        _summary(executed_entries),
        entries=entries,
        audit=audit,
        audit_ready=audit_ready,
    )
    status = _status(summary=summary, dry_run=dry_run)
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Dry-run or execution report for audited source-backed frontier "
            "input-fill commands. This report is not verifier evidence and "
            "does not promote deterministic rule candidates."
        ),
        "source": {
            "input_binding_audit": None if source_path is None else str(source_path),
            "input_binding_audit_status": audit.get("status"),
            "ready_binding_count": _nested(audit, "summary", "ready_binding_count"),
            "blocked_binding_count": _nested(audit, "summary", "blocked_binding_count"),
        },
        "label_usage": {
            "labels_used_for_fill_execution": False,
            "fill_run_report_is_verifier_evidence": False,
            "fill_run_report_promotes_rule_candidates": False,
        },
        "summary": summary,
        "paths": {
            "run_report": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
        },
        "config": {
            "dry_run": bool(dry_run),
            "executes_commands": not dry_run,
            "allow_partial_ready": bool(allow_partial_ready),
            "requires_audit_ready_for_execute": not allow_partial_ready,
            "cwd": str(working_dir),
            "python_executable": str(python_executable),
            "command_timeout_seconds": timeout,
            "stop_on_failure": bool(stop_on_failure),
            "workers": worker_count,
            "execution_mode": "parallel" if worker_count > 1 else "sequential",
        },
        "entries": tuple(executed_entries),
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
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output_path if output_path is not None else source_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "input_binding_audit": None if source_path is None else str(source_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "dry_run": bool(dry_run),
                "audit_ready": audit_ready,
                "ready_for_fill_command_count": summary["ready_for_fill_command_count"],
                "blocked_fill_command_count": summary["blocked_fill_command_count"],
                "command_count": summary["command_count"],
                "dry_run_count": summary["dry_run_count"],
                "succeeded_count": summary["succeeded_count"],
                "failed_count": summary["failed_count"],
                "skipped_count": summary["skipped_count"],
                "missing_output_count": summary["missing_output_count"],
                "workers": worker_count,
                "execution_mode": "parallel" if worker_count > 1 else "sequential",
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _entries_from_audit(
    audit: Mapping[str, Any],
    *,
    dry_run: bool,
    audit_ready: bool,
    allow_partial_ready: bool,
) -> tuple[dict[str, Any], ...]:
    entries = []
    audit_blocks_execute = (not dry_run) and (not audit_ready) and (not allow_partial_ready)
    for index, command in enumerate(_mapping_sequence(audit.get("downstream_commands", ())), start=1):
        ready_for_fill = command.get("ready_for_fill") is True
        command_text = str(command.get("command") or "")
        if not ready_for_fill:
            command_status = "needs_inputs"
            block_reason = "input_binding_audit_not_ready_for_fill"
        elif audit_blocks_execute:
            command_status = "needs_inputs"
            block_reason = "input_binding_audit_not_ready"
        else:
            command_status = "ready"
            block_reason = ""
        entries.append({
            "entry_id": str(command.get("request_id") or f"input-fill-command-{index}"),
            "action_id": str(command.get("request_id") or f"input-fill-command-{index}"),
            "title": str(command.get("input_name") or command.get("sidecar_key") or f"input fill {index}"),
            "command_status": command_status,
            "execution_block_reason": block_reason,
            "bound_commands": (command_text,),
            "planned_outputs": _planned_outputs_from_command(command_text),
            "evidence_routes": ("world_model_rule_input_fill",),
            "input_name": str(command.get("input_name") or ""),
            "sidecar_key": str(command.get("sidecar_key") or ""),
            "ready_for_fill": ready_for_fill,
        })
    return tuple(entries)


def _planned_outputs_from_command(command: str) -> tuple[dict[str, str], ...]:
    try:
        argv = tuple(shlex.split(command))
    except ValueError:
        return ()
    outputs = []
    for name, flag in OUTPUT_FLAGS:
        value = _flag_value(argv, flag)
        if value:
            outputs.append({"name": name, "path": value})
    return tuple(outputs)


def _flag_value(argv: Sequence[str], flag: str) -> str:
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return str(argv[index + 1])
        prefix = f"{flag}="
        if item.startswith(prefix):
            return item[len(prefix):]
    return ""


def _frontier_fill_summary(
    summary: Mapping[str, Any],
    *,
    entries: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    audit_ready: bool,
) -> dict[str, Any]:
    result = dict(summary)
    ready_commands = sum(1 for entry in entries if entry.get("ready_for_fill") is True)
    blocked_commands = sum(1 for entry in entries if entry.get("ready_for_fill") is not True)
    audit_gate_block_count = sum(
        1 for entry in entries if entry.get("execution_block_reason") == "input_binding_audit_not_ready"
    )
    result.update({
        "audit_ready": audit_ready,
        "audit_status": str(audit.get("status") or ""),
        "ready_binding_count": _nested(audit, "summary", "ready_binding_count"),
        "blocked_binding_count": _nested(audit, "summary", "blocked_binding_count"),
        "ready_for_fill_command_count": ready_commands,
        "blocked_fill_command_count": blocked_commands,
        "audit_gate_blocked_command_count": audit_gate_block_count,
    })
    return result


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    source_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    manifest = build_artifact_manifest(
        {
            "frontier_research_queue_input_fill_command_run_report": output_path,
            "frontier_research_queue_input_binding_audit": source_path,
        },
        root=manifest_path.parent,
        metadata={
            "runner": "run_frontier_research_queue_input_fill_commands",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "dry_run": _nested(payload, "config", "dry_run"),
            "audit_ready": _nested(payload, "summary", "audit_ready"),
            "ready_for_fill_command_count": _nested(
                payload,
                "summary",
                "ready_for_fill_command_count",
            ),
            "blocked_fill_command_count": _nested(
                payload,
                "summary",
                "blocked_fill_command_count",
            ),
            "command_count": _nested(payload, "summary", "command_count"),
            "dry_run_count": _nested(payload, "summary", "dry_run_count"),
            "executed_count": _nested(payload, "summary", "executed_count"),
            "failed_count": _nested(payload, "summary", "failed_count"),
            "missing_output_count": _nested(payload, "summary", "missing_output_count"),
            "workers": _nested(payload, "config", "workers"),
            "execution_mode": _nested(payload, "config", "execution_mode"),
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


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if compact
        else strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    output.write_text(text, encoding="utf-8")


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata value must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        metadata[key] = item
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-binding-audit", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--execute", action="store_true", help="execute ready fill commands")
    parser.add_argument(
        "--allow-partial-ready",
        action="store_true",
        help="allow --execute to run ready commands even when the audit report is not fully ready",
    )
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--command-timeout-seconds", type=float, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="continue after a command failure instead of skipping remaining commands",
    )
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_frontier_research_queue_input_fill_commands(
        input_binding_audit=args.input_binding_audit,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        dry_run=not args.execute,
        allow_partial_ready=args.allow_partial_ready,
        cwd=args.cwd,
        python_executable=args.python,
        command_timeout_seconds=args.command_timeout_seconds,
        stop_on_failure=not args.continue_on_failure,
        workers=args.workers,
        metadata=_parse_metadata(args.metadata),
        compact_json=args.compact_json,
    )
    print(strict_json_dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
