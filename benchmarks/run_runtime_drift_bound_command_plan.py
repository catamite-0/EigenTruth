"""Dry-run or execute a bound runtime-drift command plan."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
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

BOUND_PLAN_WORKFLOW = "runtime_drift_evidence_bound_command_plan"
WORKFLOW = "runtime_drift_bound_command_run_report"


def run_runtime_drift_bound_command_plan(
    *,
    bound_command_plan: str | Path | Mapping[str, Any],
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    dry_run: bool = True,
    cwd: str | Path | None = None,
    python_executable: str = sys.executable,
    command_timeout_seconds: float | None = None,
    stop_on_failure: bool = True,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run report for a bound runtime-drift command plan.

    By default commands are not executed. The dry-run report is only an
    execution plan, not release evidence.
    """
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a bool.")
    if not isinstance(stop_on_failure, bool):
        raise ValueError("stop_on_failure must be a bool.")
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    timeout = _timeout(command_timeout_seconds)
    source_path, plan = _load_mapping_source(bound_command_plan)
    if plan.get("workflow") != BOUND_PLAN_WORKFLOW:
        raise ValueError(f"bound_command_plan must have workflow={BOUND_PLAN_WORKFLOW!r}.")
    working_dir = Path(cwd) if cwd is not None else ROOT
    entries = tuple(_mapping_sequence(plan.get("entries", ())))
    executed_entries = []
    stop_requested = False
    for entry in entries:
        report_entry, stop_requested = _run_entry(
            entry,
            dry_run=dry_run,
            cwd=working_dir,
            python_executable=python_executable,
            timeout=timeout,
            stop_on_failure=stop_on_failure,
            skip_remaining=stop_requested,
        )
        executed_entries.append(report_entry)
    summary = _summary(executed_entries)
    status = _status(summary=summary, dry_run=dry_run)
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "source": {
            "bound_command_plan": None if source_path is None else str(source_path),
            "bound_plan_status": plan.get("status"),
        },
        "summary": summary,
        "paths": {
            "run_report": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
        },
        "config": {
            "dry_run": bool(dry_run),
            "executes_commands": not dry_run,
            "cwd": str(working_dir),
            "python_executable": str(python_executable),
            "command_timeout_seconds": timeout,
            "stop_on_failure": bool(stop_on_failure),
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
                "bound_command_plan": None if source_path is None else str(source_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "dry_run": bool(dry_run),
                "entry_count": summary["entry_count"],
                "command_count": summary["command_count"],
                "dry_run_count": summary["dry_run_count"],
                "succeeded_count": summary["succeeded_count"],
                "failed_count": summary["failed_count"],
                "skipped_count": summary["skipped_count"],
                "invalid_command_count": summary["invalid_command_count"],
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _run_entry(
    entry: Mapping[str, Any],
    *,
    dry_run: bool,
    cwd: Path,
    python_executable: str,
    timeout: float | None,
    stop_on_failure: bool,
    skip_remaining: bool,
) -> tuple[dict[str, Any], bool]:
    action_id = str(entry.get("action_id") or entry.get("entry_id") or "runtime-drift-action")
    entry_ready = str(entry.get("command_status") or "unknown") == "ready"
    command_reports = []
    stop_requested = skip_remaining
    for index, command in enumerate(_string_tuple(entry.get("bound_commands", ())), start=1):
        if stop_requested:
            command_reports.append(_skipped_command(command, index=index, reason="prior_command_failed"))
            continue
        if not entry_ready:
            command_reports.append(_skipped_command(command, index=index, reason="entry_not_ready"))
            continue
        report = _run_command(
            command,
            index=index,
            dry_run=dry_run,
            cwd=cwd,
            python_executable=python_executable,
            timeout=timeout,
        )
        command_reports.append(report)
        if stop_on_failure and report["status"] in {"failed", "timed_out", "invalid_command"}:
            stop_requested = True
    entry_status = _entry_status(entry_ready=entry_ready, command_reports=command_reports)
    return {
        "entry_id": str(entry.get("entry_id") or action_id),
        "action_id": action_id,
        "title": str(entry.get("title") or action_id),
        "source_command_status": str(entry.get("command_status") or "unknown"),
        "execution_status": entry_status,
        "evidence_routes": _string_tuple(entry.get("evidence_routes", ())),
        "planned_outputs": tuple(_mapping_sequence(entry.get("planned_outputs", ()))),
        "commands": tuple(command_reports),
    }, stop_requested


def _run_command(
    command: str,
    *,
    index: int,
    dry_run: bool,
    cwd: Path,
    python_executable: str,
    timeout: float | None,
) -> dict[str, Any]:
    parsed = _parse_command(command, python_executable=python_executable)
    base = {
        "index": int(index),
        "command": str(command),
        "argv": parsed.get("argv", ()),
        "cwd": str(cwd),
    }
    if parsed["status"] != "ready":
        return {**base, "status": parsed["status"], "error": parsed["error"]}
    if dry_run:
        return {**base, "status": "dry_run", "returncode": None}
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(parsed["argv"]),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            **base,
            "status": "timed_out",
            "returncode": None,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
            "timeout_seconds": timeout,
        }
    status = "succeeded" if int(result.returncode) == 0 else "failed"
    return {
        **base,
        "status": status,
        "returncode": int(result.returncode),
        "elapsed_seconds": time.monotonic() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _parse_command(command: str, *, python_executable: str) -> dict[str, Any]:
    if "..." in command:
        return {"status": "invalid_command", "argv": (), "error": "unbound_placeholder"}
    try:
        argv = tuple(shlex.split(command))
    except ValueError as exc:
        return {"status": "invalid_command", "argv": (), "error": str(exc)}
    if not argv:
        return {"status": "invalid_command", "argv": (), "error": "empty_command"}
    if argv[0].endswith(".py"):
        argv = (str(python_executable), *argv)
    return {"status": "ready", "argv": argv, "error": None}


def _skipped_command(command: str, *, index: int, reason: str) -> dict[str, Any]:
    return {
        "index": int(index),
        "command": str(command),
        "argv": (),
        "status": "skipped",
        "skip_reason": str(reason),
        "returncode": None,
    }


def _entry_status(*, entry_ready: bool, command_reports: Sequence[Mapping[str, Any]]) -> str:
    statuses = tuple(str(command.get("status") or "unknown") for command in command_reports)
    if not command_reports:
        return "missing_commands" if entry_ready else "skipped_not_ready"
    if not entry_ready:
        return "skipped_not_ready"
    if any(status in {"failed", "timed_out", "invalid_command"} for status in statuses):
        return "failed"
    if all(status == "dry_run" for status in statuses):
        return "dry_run"
    if all(status == "succeeded" for status in statuses):
        return "succeeded"
    if any(status == "skipped" for status in statuses):
        return "partially_skipped"
    return "unknown"


def _summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entry_status_counts: dict[str, int] = {}
    command_status_counts: dict[str, int] = {}
    expected_output_count = 0
    for entry in entries:
        entry_status = str(entry.get("execution_status") or "unknown")
        entry_status_counts[entry_status] = entry_status_counts.get(entry_status, 0) + 1
        expected_output_count += len(_mapping_sequence(entry.get("planned_outputs", ())))
        for command in _mapping_sequence(entry.get("commands", ())):
            command_status = str(command.get("status") or "unknown")
            command_status_counts[command_status] = command_status_counts.get(command_status, 0) + 1
    return {
        "entry_count": len(entries),
        "ready_entry_count": entry_status_counts.get("dry_run", 0)
        + entry_status_counts.get("succeeded", 0),
        "command_count": sum(command_status_counts.values()),
        "dry_run_count": command_status_counts.get("dry_run", 0),
        "executed_count": command_status_counts.get("succeeded", 0)
        + command_status_counts.get("failed", 0)
        + command_status_counts.get("timed_out", 0),
        "succeeded_count": command_status_counts.get("succeeded", 0),
        "failed_count": command_status_counts.get("failed", 0),
        "timed_out_count": command_status_counts.get("timed_out", 0),
        "skipped_count": command_status_counts.get("skipped", 0),
        "invalid_command_count": command_status_counts.get("invalid_command", 0),
        "expected_output_count": expected_output_count,
        "entry_status_counts": dict(sorted(entry_status_counts.items())),
        "command_status_counts": dict(sorted(command_status_counts.items())),
    }


def _status(*, summary: Mapping[str, Any], dry_run: bool) -> str:
    if int(summary.get("entry_count", 0)) == 0:
        return "empty"
    if int(summary.get("invalid_command_count", 0)) > 0:
        return "blocked"
    if int(summary.get("failed_count", 0)) > 0 or int(summary.get("timed_out_count", 0)) > 0:
        return "blocked"
    if int(summary.get("skipped_count", 0)) > 0:
        return "needs_inputs"
    if dry_run:
        return "dry_run"
    return "succeeded"


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path | None,
    source_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "runtime_drift_bound_command_run_report": output_path,
        "runtime_drift_bound_command_plan": source_path,
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "run_runtime_drift_bound_command_plan",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "dry_run": _nested_value(payload, "config", "dry_run"),
            "entry_count": _nested_value(payload, "summary", "entry_count"),
            "command_count": _nested_value(payload, "summary", "command_count"),
            "dry_run_count": _nested_value(payload, "summary", "dry_run_count"),
            "executed_count": _nested_value(payload, "summary", "executed_count"),
            "failed_count": _nested_value(payload, "summary", "failed_count"),
            "invalid_command_count": _nested_value(payload, "summary", "invalid_command_count"),
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


def _timeout(value: float | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("command_timeout_seconds must be positive and finite when set.")
    return parsed


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
    parser.add_argument("--bound-command-plan", required=True, help="bound command plan JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional local artifact registry JSON")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--execute", action="store_true", help="execute commands; default is dry-run")
    parser.add_argument("--cwd", default=None, help="working directory for command execution")
    parser.add_argument("--python", default=sys.executable, help="Python executable for .py commands")
    parser.add_argument("--command-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="continue after a command failure instead of skipping remaining commands",
    )
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = run_runtime_drift_bound_command_plan(
        bound_command_plan=args.bound_command_plan,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        dry_run=not bool(args.execute),
        cwd=args.cwd,
        python_executable=args.python,
        command_timeout_seconds=args.command_timeout_seconds,
        stop_on_failure=not bool(args.continue_on_failure),
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
