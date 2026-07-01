"""Dry-run or execute a bound frontier research-queue command plan."""

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

from benchmarks.run_runtime_drift_bound_command_plan import (  # noqa: E402
    _load_mapping_source,
    _mapping_sequence,
    _nested_value,
    _run_entries,
    _status,
    _summary,
    _timeout,
    _worker_count,
    _write_json,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

BOUND_PLAN_WORKFLOW = "frontier_research_queue_bound_command_plan"
WORKFLOW = "frontier_research_queue_bound_command_run_report"


def run_frontier_research_queue_bound_command_plan(
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
    workers: int = 1,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run report for a bound frontier research queue.

    The default dry-run mode parses commands and records expected execution,
    but does not run child workflows or create verifier/release evidence.
    """
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a bool.")
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
    source_path, plan = _load_mapping_source(bound_command_plan)
    if plan.get("workflow") != BOUND_PLAN_WORKFLOW:
        raise ValueError(f"bound_command_plan must have workflow={BOUND_PLAN_WORKFLOW!r}.")
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    if registry_path is not None and output_path is None and source_path is None:
        raise ValueError("registry_path requires json_path when bound_command_plan is in-memory.")
    working_dir = Path(cwd) if cwd is not None else ROOT
    entries = tuple(_mapping_sequence(plan.get("entries", ())))
    executed_entries = _run_entries(
        entries,
        dry_run=dry_run,
        cwd=working_dir,
        python_executable=python_executable,
        timeout=timeout,
        stop_on_failure=stop_on_failure,
        workers=worker_count,
    )
    summary = _summary(executed_entries)
    status = _status(summary=summary, dry_run=dry_run)
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
                "workers": worker_count,
                "execution_mode": "parallel" if worker_count > 1 else "sequential",
                "manifest_summary": {} if manifest is None else manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


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
        "frontier_research_queue_bound_command_run_report": output_path,
        "frontier_research_queue_bound_command_plan": source_path,
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "run_frontier_research_queue_bound_command_plan",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "dry_run": _nested_value(payload, "config", "dry_run"),
            "entry_count": _nested_value(payload, "summary", "entry_count"),
            "command_count": _nested_value(payload, "summary", "command_count"),
            "dry_run_count": _nested_value(payload, "summary", "dry_run_count"),
            "executed_count": _nested_value(payload, "summary", "executed_count"),
            "failed_count": _nested_value(payload, "summary", "failed_count"),
            "invalid_command_count": _nested_value(payload, "summary", "invalid_command_count"),
            "workers": _nested_value(payload, "config", "workers"),
            "execution_mode": _nested_value(payload, "config", "execution_mode"),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-command-plan", required=True, help="bound frontier command plan JSON")
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
        "--workers",
        type=int,
        default=1,
        help="bounded parallel workers for independent entries; requires --continue-on-failure",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="continue after a command failure instead of skipping remaining commands",
    )
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = run_frontier_research_queue_bound_command_plan(
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
        workers=args.workers,
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
