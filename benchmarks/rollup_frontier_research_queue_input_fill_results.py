"""Roll up audited frontier input-fill outputs for rule-adapter execution.

This workflow consumes ``frontier_research_queue_input_fill_command_run_report``
artifacts, combines materialized rule-input JSONL sidecars, and writes a single
adapter-ready rule-input file. It does not execute the rule adapter or promote
deterministic candidates.
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

from benchmarks.run_frontier_research_queue_input_fill_commands import (  # noqa: E402
    WORKFLOW as FILL_RUN_WORKFLOW,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_research_queue_input_fill_result_rollup"


def rollup_frontier_research_queue_input_fill_results(
    *,
    input_fill_command_run: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    rule_stubs_path: str | Path | None = None,
    json_path: str | Path | None = None,
    combined_rule_inputs_path: str | Path | None = None,
    combined_unfilled_tasks_path: str | Path | None = None,
    fill_report_rows_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine materialized fill outputs into one adapter-ready sidecar."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path, run_report = _load_mapping_source(input_fill_command_run)
    if run_report.get("workflow") != FILL_RUN_WORKFLOW:
        raise ValueError(f"input_fill_command_run must have workflow={FILL_RUN_WORKFLOW!r}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(json_path or output / "frontier-input-fill-result-rollup.json")
    rule_inputs_path = Path(combined_rule_inputs_path or output / "combined-rule-inputs.jsonl")
    unfilled_path = Path(combined_unfilled_tasks_path or output / "combined-unfilled-rule-input-tasks.jsonl")
    fill_rows_path = Path(fill_report_rows_path or output / "fill-report-rows.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    source_root = None if source_path is None else source_path.parent
    fill_reports = tuple(_fill_report_rows(run_report, source_root=source_root))
    combined_rule_inputs: list[dict[str, Any]] = []
    combined_unfilled: list[dict[str, Any]] = []
    for row in fill_reports:
        combined_rule_inputs.extend(row["rule_inputs"])
        combined_unfilled.extend(row["unfilled_tasks"])
    duplicate_request_ids = _duplicate_request_ids(combined_rule_inputs)
    blocked_rows = tuple(row for row in fill_reports if row["status"] == "blocked")
    summary = _summary(
        fill_reports=fill_reports,
        combined_rule_inputs=combined_rule_inputs,
        combined_unfilled=combined_unfilled,
        duplicate_request_ids=duplicate_request_ids,
        blocked_rows=blocked_rows,
    )
    downstream_adapter_command = _downstream_adapter_command(
        rule_stubs_path=rule_stubs_path,
        rule_inputs_path=rule_inputs_path,
        output_dir=output,
    )
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Rolls up materialized frontier rule-input fill outputs into one "
            "adapter-ready rule-input sidecar. The rollup is not verifier "
            "evidence and does not execute adapter or promotion gates."
        ),
        "source": {
            "input_fill_command_run": None if source_path is None else str(source_path),
            "input_fill_command_run_workflow": run_report.get("workflow"),
            "input_fill_command_run_status": run_report.get("status"),
            "rule_stubs": None if rule_stubs_path is None else str(rule_stubs_path),
        },
        "label_usage": {
            "labels_used_for_rollup": False,
            "combined_rule_inputs_are_verifier_evidence": False,
            "rollup_executes_adapter": False,
            "rollup_promotes_rule_candidates": False,
        },
        "summary": summary,
        "paths": {
            "report": str(report_path),
            "combined_rule_inputs": str(rule_inputs_path),
            "combined_unfilled_tasks": str(unfilled_path),
            "fill_report_rows": str(fill_rows_path),
            "artifact_manifest": str(manifest_path),
            "rule_stubs": None if rule_stubs_path is None else str(rule_stubs_path),
        },
        "fill_report_rows": tuple(_public_fill_report_row(row) for row in fill_reports),
        "downstream_adapter_command": downstream_adapter_command,
        "metadata": dict(metadata or {}),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(rule_inputs_path, combined_rule_inputs, compact=compact_json)
    _write_jsonl(unfilled_path, combined_unfilled, compact=compact_json)
    _write_jsonl(
        fill_rows_path,
        tuple(_public_fill_report_row(row) for row in fill_reports),
        compact=compact_json,
    )
    manifest = build_artifact_manifest(
        {
            "frontier_research_queue_input_fill_result_rollup": report_path,
            "combined_rule_inputs": rule_inputs_path,
            "combined_unfilled_rule_input_tasks": unfilled_path,
            "fill_report_rows": fill_rows_path,
            "input_fill_command_run": source_path,
            "rule_stubs": None if rule_stubs_path is None else Path(rule_stubs_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "fill_report_count": summary["fill_report_count"],
            "combined_rule_input_count": summary["combined_rule_input_count"],
            "combined_unfilled_task_count": summary["combined_unfilled_task_count"],
            "blocked_fill_report_count": summary["blocked_fill_report_count"],
            "duplicate_request_id_count": summary["duplicate_request_id_count"],
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "artifact_manifest": str(manifest_path),
                "fill_report_count": summary["fill_report_count"],
                "combined_rule_input_count": summary["combined_rule_input_count"],
                "combined_unfilled_task_count": summary["combined_unfilled_task_count"],
                "blocked_fill_report_count": summary["blocked_fill_report_count"],
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _fill_report_rows(
    run_report: Mapping[str, Any],
    *,
    source_root: Path | None,
) -> list[dict[str, Any]]:
    rows = []
    for entry in _mapping_sequence(run_report.get("entries", ())):
        execution_status = str(entry.get("execution_status") or "")
        for command in _mapping_sequence(entry.get("commands", ())):
            command_status = str(command.get("status") or "")
            expected_outputs = tuple(_mapping_sequence(entry.get("expected_outputs", ())))
            report_path = _expected_output_path(expected_outputs, "report", source_root=source_root)
            rule_inputs_path = _expected_output_path(
                expected_outputs,
                "rule_inputs",
                source_root=source_root,
            )
            unfilled_path = _expected_output_path(
                expected_outputs,
                "unfilled_tasks",
                source_root=source_root,
            )
            rows.append(
                _fill_report_row(
                    entry=entry,
                    command=command,
                    execution_status=execution_status,
                    command_status=command_status,
                    report_path=report_path,
                    rule_inputs_path=rule_inputs_path,
                    unfilled_path=unfilled_path,
                )
            )
    return rows


def _fill_report_row(
    *,
    entry: Mapping[str, Any],
    command: Mapping[str, Any],
    execution_status: str,
    command_status: str,
    report_path: Path | None,
    rule_inputs_path: Path | None,
    unfilled_path: Path | None,
) -> dict[str, Any]:
    failures = []
    report_payload: Mapping[str, Any] | None = None
    rule_inputs: tuple[Mapping[str, Any], ...] = ()
    unfilled_tasks: tuple[Mapping[str, Any], ...] = ()
    if execution_status not in {"succeeded", "dry_run"}:
        failures.append(f"entry_{execution_status or 'unknown'}")
    if command_status != "succeeded":
        failures.append(f"command_{command_status or 'unknown'}")
    if report_path is None:
        failures.append("missing_report_path")
    elif not report_path.exists():
        failures.append("report_not_materialized")
    else:
        report_payload = _load_json_object(report_path)
        if str(report_payload.get("status") or "") == "blocked":
            failures.append("fill_report_blocked")
    if rule_inputs_path is None:
        failures.append("missing_rule_inputs_path")
    elif not rule_inputs_path.exists():
        failures.append("rule_inputs_not_materialized")
    else:
        rule_inputs = _load_jsonl_mappings(rule_inputs_path)
        for rule_input in rule_inputs:
            if rule_input.get("not_verifier_evidence") is not True:
                failures.append("rule_input_missing_non_evidence_marker")
            if rule_input.get("candidate_results_require_promotion_gate") is not True:
                failures.append("rule_input_missing_promotion_gate_marker")
    if unfilled_path is not None and unfilled_path.exists():
        unfilled_tasks = _load_jsonl_mappings(unfilled_path)
    status = "ready" if not failures else "blocked"
    return {
        "entry_id": str(entry.get("entry_id") or ""),
        "action_id": str(entry.get("action_id") or ""),
        "execution_status": execution_status,
        "command_status": command_status,
        "status": status,
        "failures": tuple(dict.fromkeys(failures)),
        "report_path": None if report_path is None else str(report_path),
        "rule_inputs_path": None if rule_inputs_path is None else str(rule_inputs_path),
        "unfilled_tasks_path": None if unfilled_path is None else str(unfilled_path),
        "fill_report_status": None if report_payload is None else report_payload.get("status"),
        "fill_report_workflow": None if report_payload is None else report_payload.get("workflow"),
        "fill_report_summary": None if report_payload is None else report_payload.get("summary"),
        "rule_inputs": tuple(dict(row) for row in rule_inputs),
        "unfilled_tasks": tuple(dict(row) for row in unfilled_tasks),
    }


def _public_fill_report_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_id": row["entry_id"],
        "action_id": row["action_id"],
        "execution_status": row["execution_status"],
        "command_status": row["command_status"],
        "status": row["status"],
        "failures": row["failures"],
        "report_path": row["report_path"],
        "rule_inputs_path": row["rule_inputs_path"],
        "unfilled_tasks_path": row["unfilled_tasks_path"],
        "fill_report_status": row["fill_report_status"],
        "fill_report_workflow": row["fill_report_workflow"],
        "fill_report_summary": row["fill_report_summary"],
        "rule_input_count": len(tuple(row["rule_inputs"])),
        "unfilled_task_count": len(tuple(row["unfilled_tasks"])),
        "not_verifier_evidence": True,
    }


def _summary(
    *,
    fill_reports: Sequence[Mapping[str, Any]],
    combined_rule_inputs: Sequence[Mapping[str, Any]],
    combined_unfilled: Sequence[Mapping[str, Any]],
    duplicate_request_ids: Sequence[str],
    blocked_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    family_counts = Counter(str(row.get("rule_family") or "") for row in combined_rule_inputs)
    workflow_counts = Counter(str(row.get("fill_report_workflow") or "") for row in fill_reports)
    status_counts = Counter(str(row.get("status") or "") for row in fill_reports)
    failure_counts: Counter[str] = Counter()
    for row in fill_reports:
        failure_counts.update(str(item) for item in _sequence(row.get("failures", ())))
    return {
        "fill_report_count": len(fill_reports),
        "ready_fill_report_count": status_counts.get("ready", 0),
        "blocked_fill_report_count": len(blocked_rows),
        "combined_rule_input_count": len(combined_rule_inputs),
        "combined_unfilled_task_count": len(combined_unfilled),
        "duplicate_request_id_count": len(tuple(duplicate_request_ids)),
        "duplicate_request_ids": tuple(duplicate_request_ids),
        "rule_family_counts": _sorted_counter(family_counts),
        "fill_report_workflow_counts": _sorted_counter(workflow_counts),
        "fill_report_status_counts": _sorted_counter(status_counts),
        "failure_counts": _sorted_counter(failure_counts),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("blocked_fill_report_count", 0)) > 0:
        return "blocked"
    if int(summary.get("duplicate_request_id_count", 0)) > 0:
        return "blocked"
    if int(summary.get("combined_rule_input_count", 0)) > 0:
        if int(summary.get("combined_unfilled_task_count", 0)) > 0:
            return "partial"
        return "ready_for_adapter"
    if int(summary.get("combined_unfilled_task_count", 0)) > 0:
        return "needs_inputs"
    return "empty"


def _downstream_adapter_command(
    *,
    rule_stubs_path: str | Path | None,
    rule_inputs_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    adapter_dir = output_dir / "downstream-rule-adapter"
    command = (
        "benchmarks/run_world_model_rule_authoring_adapter.py "
        f"--rule-stubs {rule_stubs_path if rule_stubs_path is not None else '...'} "
        f"--rule-inputs {rule_inputs_path} "
        f"--output-dir {adapter_dir} "
        f"--json {adapter_dir / 'world-model-rule-authoring-adapter.json'} "
        f"--rule-results-jsonl {adapter_dir / 'world-model-rule-results.jsonl'} "
        f"--artifact-manifest {adapter_dir / 'artifact-manifest.json'}"
    )
    return {
        "command": command,
        "rule_stubs_path": None if rule_stubs_path is None else str(rule_stubs_path),
        "rule_inputs_path": str(rule_inputs_path),
        "output_dir": str(adapter_dir),
        "ready_for_adapter": rule_stubs_path is not None,
        "executes_commands": False,
    }


def _expected_output_path(
    outputs: Sequence[Mapping[str, Any]],
    name: str,
    *,
    source_root: Path | None,
) -> Path | None:
    for output in outputs:
        if str(output.get("name") or "") == name:
            path = str(output.get("path") or "")
            return None if not path else _resolve_path(path, source_root=source_root)
    return None


def _duplicate_request_ids(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    counts = Counter(str(row.get("request_id") or "") for row in rows if str(row.get("request_id") or ""))
    return tuple(sorted(request_id for request_id, count in counts.items() if count > 1))


def _load_mapping_source(source: str | Path | Mapping[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    if isinstance(source, Mapping):
        return None, dict(source)
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return path, dict(payload)


def _load_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(row)
    return tuple(rows)


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


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (value,)
    return tuple(value)


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


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
    parser.add_argument("--input-fill-command-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rule-stubs", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--combined-rule-inputs-jsonl", default=None)
    parser.add_argument("--combined-unfilled-tasks-jsonl", default=None)
    parser.add_argument("--fill-report-rows-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = rollup_frontier_research_queue_input_fill_results(
        input_fill_command_run=args.input_fill_command_run,
        output_dir=args.output_dir,
        rule_stubs_path=args.rule_stubs,
        json_path=args.json,
        combined_rule_inputs_path=args.combined_rule_inputs_jsonl,
        combined_unfilled_tasks_path=args.combined_unfilled_tasks_jsonl,
        fill_report_rows_path=args.fill_report_rows_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata),
        compact_json=args.compact_json,
    )
    print(strict_json_dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
