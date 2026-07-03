"""Roll up frontier multiple-testing reruns into release evidence.

The planner emits one ``run_truthfulqa_frontier_workflow.py`` command per
blocked cell. This workflow reads the completed child frontier reports and
checks both the top-level multiple-testing gate and the specific queued cell.
"""

from __future__ import annotations

import argparse
import json
import math
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

WORKFLOW = "frontier_multiple_testing_rerun_rollup"
QUEUE_WORKFLOW = "frontier_multiple_testing_rerun_queue"
CHILD_WORKFLOW = "truthfulqa_frontier_workflow"
DEFAULT_CHILD_REPORT_NAME = "truthfulqa-frontier-workflow.json"


def rollup_frontier_multiple_testing_reruns(
    *,
    queue_path: str | Path,
    report_json_path: str | Path,
    report_paths: Sequence[str | Path] = (),
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    require_all_reports: bool = False,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Summarize completed multiple-testing reruns and recommend release status."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    queue_file = Path(queue_path)
    rollup_path = Path(report_json_path)
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else rollup_path.with_name("artifact-manifest.json")
    )
    queue = _load_json_object(queue_file)
    entries = _queue_entries(queue)
    explicit_reports = _load_explicit_reports(report_paths, queue_dir=queue_file.parent)
    candidates = tuple(
        _candidate_for_entry(entry, queue_dir=queue_file.parent, explicit_reports=explicit_reports)
        for entry in entries
    )
    summary = _summary(queue=queue, entries=entries, candidates=candidates)
    gate = _gate(candidates, summary=summary, require_all_reports=require_all_reports)
    recommended = _recommended_candidate(candidates)
    status = _status(gate=gate, summary=summary)
    observed_report_paths = tuple(dict.fromkeys(
        str(candidate["observed_report_path"])
        for candidate in candidates
        if candidate.get("observed_report_path")
    ))
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "gate": gate,
        "summary": summary,
        "recommended_candidate": recommended,
        "source": {
            "queue": str(queue_file),
            "queue_workflow": queue.get("workflow"),
            "explicit_reports": tuple(str(path) for path in report_paths),
            "observed_reports": observed_report_paths,
        },
        "config": {"require_all_reports": bool(require_all_reports)},
        "paths": {
            "report": str(rollup_path),
            "artifact_manifest": str(manifest_path),
        },
        "candidates": candidates,
        "metadata": dict(metadata or {}),
    }
    _write_json(rollup_path, payload, compact=compact_json)
    manifest = _write_artifact_manifest(
        rollup_path=rollup_path,
        manifest_path=manifest_path,
        queue_path=queue_file,
        observed_report_paths=observed_report_paths,
        payload=payload,
        metadata=metadata or {},
        compact=compact_json,
    )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=rollup_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "passed": gate["passed"],
                "promotion_ready": gate["promotion_ready"],
                "queue": str(queue_file),
                "artifact_manifest": str(manifest_path),
                "candidate_count": summary["candidate_count"],
                "promotion_ready_count": summary["promotion_ready_count"],
                "missing_report_count": summary["missing_report_count"],
                "best_cell": None if recommended is None else recommended.get("cell"),
                "best_run": None if recommended is None else recommended.get("run"),
                "manifest_summary": _mapping(manifest.get("summary")),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _candidate_for_entry(
    entry: Mapping[str, Any],
    *,
    queue_dir: Path,
    explicit_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_path = _expected_report_path(entry)
    observed = _load_expected_report(expected_path, queue_dir=queue_dir)
    report_source = "expected_path"
    if observed is None:
        observed = _match_explicit_report(entry, explicit_reports)
        report_source = "explicit_report" if observed is not None else "missing"
    base = _candidate_base(entry, expected_path=expected_path)
    if observed is None:
        return {
            **base,
            "candidate_status": "missing_report",
            "observed_report_path": None,
            "report_source": report_source,
            "metrics": {},
            "promotion_ready": False,
            "blocking_reasons": (
                {"gate": "report", "reason": "Expected frontier workflow rerun report is missing."},
            ),
        }
    report = _mapping(observed.get("report"))
    report_path = str(observed.get("path") or "")
    if observed.get("error"):
        return {
            **base,
            "candidate_status": "invalid_report",
            "observed_report_path": report_path,
            "report_source": report_source,
            "metrics": {},
            "promotion_ready": False,
            "blocking_reasons": (
                {"gate": "report", "path": report_path, "reason": str(observed["error"])},
            ),
        }
    metrics = _candidate_metrics(entry=entry, report=report)
    reasons = _blocking_reasons(report=report, metrics=metrics)
    promotion_ready = not reasons
    return {
        **base,
        "candidate_status": "promotion_ready" if promotion_ready else "blocked",
        "observed_report_path": report_path,
        "report_source": report_source,
        "report_workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "metrics": metrics,
        "promotion_ready": promotion_ready,
        "blocking_reasons": tuple(reasons),
    }


def _candidate_base(entry: Mapping[str, Any], *, expected_path: str | None) -> dict[str, Any]:
    return {
        "run": entry.get("run"),
        "cell": entry.get("cell"),
        "source_status": entry.get("status"),
        "source_false_alarm": entry.get("false_alarm"),
        "source_detection": entry.get("detection"),
        "workflow_report": entry.get("workflow_report"),
        "source_report": entry.get("source_report"),
        "source_calibration": entry.get("source_calibration"),
        "expected_report_path": expected_path,
    }


def _candidate_metrics(*, entry: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    gate = _mapping(report.get("multiple_testing_gate"))
    cells = _cell_summaries(gate.get("cells", ()))
    cell_name = str(entry.get("cell") or "")
    cell = _matching_cell(cells, cell_name)
    return {
        "gate_enabled": gate.get("enabled"),
        "gate_all_pass": gate.get("all_pass"),
        "gate_cell_count": _optional_int(gate.get("cell_count")),
        "gate_pass_count": _optional_int(gate.get("pass_count")),
        "gate_fail_count": _optional_int(gate.get("fail_count")),
        "gate_unknown_count": _optional_int(gate.get("unknown_count")),
        "gate_reported_cell_count": len(cells),
        "gate_failed_cells": tuple(item["cell"] for item in cells if item["status"] == "failed"),
        "gate_unknown_cells": tuple(item["cell"] for item in cells if item["status"] == "unknown"),
        "cell_found": cell is not None,
        "cell": None if cell is None else cell["cell"],
        "cell_status": None if cell is None else cell["status"],
        "cell_false_alarm": None if cell is None else cell.get("false_alarm"),
        "cell_detection": None if cell is None else cell.get("detection"),
        "cell_report": None if cell is None else cell.get("report"),
        "cell_calibration": None if cell is None else cell.get("calibration"),
    }


def _blocking_reasons(
    *,
    report: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    reasons: list[dict[str, Any]] = []
    if report.get("workflow") != CHILD_WORKFLOW:
        reasons.append({
            "gate": "workflow",
            "reason": f"Expected workflow {CHILD_WORKFLOW!r}, got {report.get('workflow')!r}.",
        })
    if report.get("status") not in {"complete", "promote"}:
        reasons.append({
            "gate": "report_status",
            "reason": f"Report status is {report.get('status')!r}.",
        })
    if metrics.get("gate_enabled") is not True:
        reasons.append({"gate": "multiple_testing_gate", "reason": "multiple_testing_gate.enabled is not true."})
    if metrics.get("gate_all_pass") is not True:
        reasons.append({"gate": "multiple_testing_gate", "reason": "multiple_testing_gate.all_pass is not true."})
    cell_count = _optional_int(metrics.get("gate_cell_count"))
    reported_cell_count = _optional_int(metrics.get("gate_reported_cell_count"))
    pass_count = _optional_int(metrics.get("gate_pass_count"))
    fail_count = _optional_int(metrics.get("gate_fail_count"))
    unknown_count = _optional_int(metrics.get("gate_unknown_count"))
    if cell_count is None or cell_count < 1:
        reasons.append({"gate": "cell_count", "reason": "multiple_testing_gate.cell_count is missing or zero."})
    elif reported_cell_count != cell_count:
        reasons.append({
            "gate": "cell_count",
            "reason": (
                f"multiple_testing_gate.cells length {reported_cell_count} "
                f"does not match cell_count {cell_count}."
            ),
        })
    if None in {pass_count, fail_count, unknown_count}:
        reasons.append({
            "gate": "cell_counts",
            "reason": "multiple_testing_gate pass/fail/unknown counts are incomplete.",
        })
    elif cell_count is not None and pass_count + fail_count + unknown_count != cell_count:
        reasons.append({
            "gate": "cell_counts",
            "reason": "multiple_testing_gate pass/fail/unknown counts do not sum to cell_count.",
        })
    if fail_count not in {0, None}:
        reasons.append({"gate": "fail_count", "reason": f"multiple_testing_gate.fail_count {fail_count} is non-zero."})
    if unknown_count not in {0, None}:
        reasons.append({
            "gate": "unknown_count",
            "reason": f"multiple_testing_gate.unknown_count {unknown_count} is non-zero.",
        })
    if metrics.get("cell_found") is not True:
        reasons.append({"gate": "cell", "reason": "Queued cell is missing from child multiple_testing_gate.cells."})
    elif metrics.get("cell_status") != "passed":
        reasons.append({
            "gate": "cell",
            "reason": f"Queued cell status is {metrics.get('cell_status')!r}.",
        })
    if metrics.get("cell_found") is True and (not metrics.get("cell_report") or not metrics.get("cell_calibration")):
        reasons.append({
            "gate": "cell_artifacts",
            "reason": "Queued cell is missing report or calibration artifact path.",
        })
    return tuple(reasons)


def _summary(
    *,
    queue: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed_paths = {
        str(candidate.get("observed_report_path"))
        for candidate in candidates
        if candidate.get("observed_report_path")
    }
    return {
        "queue_workflow": queue.get("workflow"),
        "queue_status": queue.get("status"),
        "queue_entry_count": len(_mapping_sequence(queue.get("entries", ()))),
        "expected_entry_count": len(entries),
        "candidate_count": len(candidates),
        "observed_report_count": len(observed_paths),
        "missing_report_count": sum(1 for item in candidates if item["candidate_status"] == "missing_report"),
        "invalid_report_count": sum(1 for item in candidates if item["candidate_status"] == "invalid_report"),
        "promotion_ready_count": sum(1 for item in candidates if bool(item.get("promotion_ready"))),
        "blocked_candidate_count": sum(1 for item in candidates if item["candidate_status"] == "blocked"),
        "cells": tuple(sorted(str(entry.get("cell")) for entry in entries if entry.get("cell"))),
        "runs": tuple(sorted({str(entry.get("run")) for entry in entries if entry.get("run")})),
    }


def _gate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    require_all_reports: bool,
) -> dict[str, Any]:
    blocking: list[dict[str, Any]] = []
    if not candidates:
        blocking.append({"gate": "queue", "reason": "Queue does not contain ready multiple-testing entries."})
    for candidate in candidates:
        if candidate["candidate_status"] in {"missing_report", "invalid_report"}:
            blocking.append({
                "gate": "report_coverage",
                "run": candidate.get("run"),
                "cell": candidate.get("cell"),
                "expected_report_path": candidate.get("expected_report_path"),
                "reason": f"Candidate report status is {candidate['candidate_status']}.",
            })
    for candidate in candidates:
        if candidate["candidate_status"] == "blocked":
            for reason in _mapping_sequence(candidate.get("blocking_reasons", ())):
                blocking.append({
                    "gate": reason.get("gate"),
                    "run": candidate.get("run"),
                    "cell": candidate.get("cell"),
                    "reason": reason.get("reason"),
                })
    passed = not blocking and bool(candidates)
    promotion_ready = bool(passed and summary.get("promotion_ready_count") == summary.get("candidate_count"))
    return {
        "passed": passed,
        "promotion_ready": promotion_ready,
        "require_all_reports": bool(require_all_reports),
        "blocking_reasons": tuple(blocking),
    }


def _recommended_candidate(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    best = sorted(candidates, key=_candidate_rank, reverse=True)[0]
    return {
        "run": best.get("run"),
        "cell": best.get("cell"),
        "candidate_status": best.get("candidate_status"),
        "promotion_ready": best.get("promotion_ready"),
        "expected_report_path": best.get("expected_report_path"),
        "observed_report_path": best.get("observed_report_path"),
        "metrics": best.get("metrics"),
        "blocking_reasons": best.get("blocking_reasons"),
    }


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = _mapping(candidate.get("metrics"))
    return (
        bool(candidate.get("promotion_ready")),
        _rank_float(metrics.get("cell_detection")),
        -_rank_float(metrics.get("cell_false_alarm"), missing=1.0),
        str(candidate.get("run") or ""),
        str(candidate.get("cell") or ""),
    )


def _status(*, gate: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    if summary.get("candidate_count", 0) == 0:
        return "empty"
    return "promote" if gate.get("promotion_ready") else "blocked"


def _queue_entries(queue: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if queue.get("workflow") != QUEUE_WORKFLOW:
        raise ValueError(f"queue workflow must be {QUEUE_WORKFLOW!r}.")
    return tuple(
        entry
        for entry in _mapping_sequence(queue.get("entries", ()))
        if entry.get("command_status") == "ready" and entry.get("command")
    )


def _load_explicit_reports(
    report_paths: Sequence[str | Path],
    *,
    queue_dir: Path,
) -> tuple[dict[str, Any], ...]:
    return tuple(_load_report_record(_resolve_path(Path(path), base=queue_dir)) for path in report_paths)


def _load_expected_report(path: str | None, *, queue_dir: Path) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = _resolve_existing_path(Path(path), base=queue_dir)
    return None if resolved is None else _load_report_record(resolved)


def _load_report_record(path: Path) -> dict[str, Any]:
    try:
        return {"path": str(path), "report": _load_json_object(path), "error": None}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"path": str(path), "report": {}, "error": str(exc)}


def _match_explicit_report(
    entry: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    cell_name = str(entry.get("cell") or "")
    for record in reports:
        report = _mapping(record.get("report"))
        if record.get("error") or report.get("workflow") != CHILD_WORKFLOW:
            continue
        cells = _cell_summaries(_nested(report, "multiple_testing_gate", "cells"))
        if any(item["cell"] == cell_name for item in cells):
            return record
    return None


def _expected_report_path(entry: Mapping[str, Any]) -> str | None:
    command = _sequence(entry.get("command"))
    for index, part in enumerate(command):
        if str(part) == "--output-dir" and index + 1 < len(command):
            output_dir = Path(str(command[index + 1]))
            return str(output_dir / DEFAULT_CHILD_REPORT_NAME)
    output_dir = entry.get("rerun_output_dir")
    return None if output_dir is None else str(Path(str(output_dir)) / DEFAULT_CHILD_REPORT_NAME)


def _cell_summaries(value: Any) -> tuple[dict[str, Any], ...]:
    cells = []
    for index, item in enumerate(_mapping_sequence(value)):
        cell_name = item.get("cell") or item.get("name") or f"cell_{index}"
        passed = item.get("pass")
        if passed is True:
            status = "passed"
        elif passed is False:
            status = "failed"
        else:
            status = "unknown"
        cells.append({
            "cell": str(cell_name),
            "status": status,
            "false_alarm": _optional_float(item.get("false_alarm")),
            "detection": _optional_float(item.get("detection")),
            "report": item.get("report"),
            "calibration": item.get("calibration"),
        })
    return tuple(cells)


def _matching_cell(cells: Sequence[Mapping[str, Any]], cell_name: str) -> Mapping[str, Any] | None:
    for cell in cells:
        if cell.get("cell") == cell_name:
            return cell
    return cells[0] if len(cells) == 1 and not cell_name else None


def _write_artifact_manifest(
    *,
    rollup_path: Path,
    manifest_path: Path,
    queue_path: Path,
    observed_report_paths: Sequence[str],
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> dict[str, Any]:
    summary = _mapping(payload.get("summary"))
    artifacts: dict[str, str | Path | None] = {
        "frontier_multiple_testing_rerun_rollup": rollup_path,
        "frontier_multiple_testing_rerun_queue": queue_path,
    }
    for index, path in enumerate(observed_report_paths, start=1):
        artifacts[f"frontier_workflow_rerun_report_{index}"] = Path(path)
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "rollup_frontier_multiple_testing_reruns",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "passed": _mapping(payload.get("gate")).get("passed"),
            "promotion_ready": _mapping(payload.get("gate")).get("promotion_ready"),
            "candidate_count": summary.get("candidate_count"),
            "promotion_ready_count": summary.get("promotion_ready_count"),
            "missing_report_count": summary.get("missing_report_count"),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _resolve_existing_path(path: Path, *, base: Path) -> Path | None:
    candidates = (path,) if path.is_absolute() else (path, base / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_path(path: Path, *, base: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    candidate = base / path
    return candidate if candidate.exists() else path


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _rank_float(value: Any, *, missing: float = 0.0) -> float:
    parsed = _optional_float(value)
    return missing if parsed is None else parsed


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = rollup_frontier_multiple_testing_reruns(
        queue_path=args.queue,
        report_json_path=args.json,
        report_paths=tuple(args.report or ()),
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        require_all_reports=bool(args.require_all_reports),
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "frontier_multiple_testing_rerun_rollup="
        f"{payload['status']} "
        f"candidates={summary['candidate_count']} "
        f"promotion_ready={payload['gate']['promotion_ready']} "
        f"missing_reports={summary['missing_report_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, help="frontier multiple-testing rerun queue JSON")
    parser.add_argument("--report", action="append", default=[], help="completed child frontier report; repeatable")
    parser.add_argument("--json", required=True, help="rollup output JSON")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--require-all-reports", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
