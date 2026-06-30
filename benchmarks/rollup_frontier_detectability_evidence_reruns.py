"""Roll up frontier detectability rerun reports into release evidence.

This workflow consumes ``plan_frontier_detectability_evidence_reruns.py`` queue
output plus completed child reports. Taxonomy reruns can promote the
detectability gate when the entrenched-false rate falls below threshold.
Blind-spot analyses are treated as completed diagnostic evidence, not as a
release promotion by themselves.
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

WORKFLOW = "frontier_detectability_evidence_rerun_rollup"
QUEUE_WORKFLOW = "frontier_detectability_evidence_rerun_queue"
BLIND_SPOT_WORKFLOW = "detectability_blind_spot_analysis"
TAXONOMY_WORKFLOW = "detectability_taxonomy"
DEFAULT_MAX_ENTRENCHED_FALSE_RATE = 0.25


def rollup_frontier_detectability_evidence_reruns(
    *,
    queue_path: str | Path,
    report_json_path: str | Path,
    report_paths: Sequence[str | Path] = (),
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    max_entrenched_false_rate: float = DEFAULT_MAX_ENTRENCHED_FALSE_RATE,
    require_all_reports: bool = False,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Summarize detectability rerun reports and gate release readiness."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    threshold = _finite_float(max_entrenched_false_rate, name="max_entrenched_false_rate")
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
        _candidate_for_entry(
            entry,
            queue_dir=queue_file.parent,
            explicit_reports=explicit_reports,
            max_entrenched_false_rate=threshold,
        )
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
        "config": {
            "max_entrenched_false_rate": threshold,
            "require_all_reports": bool(require_all_reports),
        },
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
                "audit_ready": gate["audit_ready"],
                "queue": str(queue_file),
                "artifact_manifest": str(manifest_path),
                "candidate_count": summary["candidate_count"],
                "promotion_ready_count": summary["promotion_ready_count"],
                "audit_ready_count": summary["audit_ready_count"],
                "missing_report_count": summary["missing_report_count"],
                "best_run": None if recommended is None else recommended.get("run"),
                "best_command_kind": None if recommended is None else recommended.get("command_kind"),
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
    max_entrenched_false_rate: float,
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
            "audit_ready": False,
            "promotion_ready": False,
            "blocking_reasons": (
                {"gate": "report", "reason": "Expected detectability rerun report is missing."},
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
            "audit_ready": False,
            "promotion_ready": False,
            "blocking_reasons": (
                {"gate": "report", "path": report_path, "reason": str(observed["error"])},
            ),
        }
    command_kind = str(entry.get("command_kind") or "")
    metrics = _candidate_metrics(command_kind=command_kind, report=report)
    reasons = _blocking_reasons(
        command_kind=command_kind,
        report=report,
        metrics=metrics,
        max_entrenched_false_rate=max_entrenched_false_rate,
    )
    audit_ready = command_kind == "blind_spot_analysis" and not reasons
    promotion_ready = command_kind == "taxonomy_report" and not reasons
    status = "promotion_ready" if promotion_ready else ("audit_ready" if audit_ready else "blocked")
    return {
        **base,
        "candidate_status": status,
        "observed_report_path": report_path,
        "report_source": report_source,
        "report_workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "metrics": metrics,
        "audit_ready": audit_ready,
        "promotion_ready": promotion_ready,
        "blocking_reasons": tuple(reasons),
    }


def _candidate_base(entry: Mapping[str, Any], *, expected_path: str | None) -> dict[str, Any]:
    return {
        "run": entry.get("run"),
        "command_kind": entry.get("command_kind"),
        "command_status": entry.get("command_status"),
        "source_report": entry.get("source_report"),
        "source_score_dump": entry.get("source_score_dump"),
        "taxonomy_config": _mapping(entry.get("taxonomy_config")),
        "expected_report_path": expected_path,
    }


def _candidate_metrics(*, command_kind: str, report: Mapping[str, Any]) -> dict[str, Any]:
    if command_kind == "taxonomy_report":
        taxonomy = _mapping(report.get("report"))
        entrenched = _mapping(_nested(taxonomy, "cells", "entrenched"))
        false_distribution = _mapping(_nested(taxonomy, "false_distribution", "entrenched"))
        blind_spot = _mapping(taxonomy.get("blind_spot"))
        share_of_false = _mapping(entrenched.get("share_of_false"))
        return {
            "n_total": _optional_int(taxonomy.get("n_total")),
            "n_false": _optional_int(taxonomy.get("n_false")),
            "entrenched_false_rate": _first_float(
                false_distribution.get("rate"),
                share_of_false.get("estimate"),
                entrenched.get("rate"),
            ),
            "entrenched_false_count": _first_int(
                false_distribution.get("count"),
                entrenched.get("n_false"),
                entrenched.get("count"),
            ),
            "blind_spot_false_count": _optional_int(blind_spot.get("n_false")),
        }
    if command_kind == "blind_spot_analysis":
        summary = _mapping(report.get("summary"))
        return {
            "cell": summary.get("cell"),
            "false_only": summary.get("false_only"),
            "selected_record_count": _optional_int(summary.get("selected_record_count")),
            "emitted_record_count": _optional_int(summary.get("emitted_record_count")),
            "expected_selected_record_count": _optional_int(summary.get("expected_selected_record_count")),
            "assignment_check_passed": summary.get("assignment_check_passed"),
            "truncated": summary.get("truncated"),
            "question_type_counts": _mapping(summary.get("question_type_counts")),
            "feature_counts": _mapping(summary.get("feature_counts")),
            "cell_false_counts": _mapping(summary.get("cell_false_counts")),
        }
    return {}


def _blocking_reasons(
    *,
    command_kind: str,
    report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    max_entrenched_false_rate: float,
) -> tuple[dict[str, Any], ...]:
    reasons: list[dict[str, Any]] = []
    expected_workflow = _expected_workflow(command_kind)
    if expected_workflow is None:
        reasons.append({
            "gate": "command_kind",
            "reason": f"Unsupported detectability rerun command_kind {command_kind!r}.",
        })
        return tuple(reasons)
    if report.get("workflow") != expected_workflow:
        reasons.append({
            "gate": "workflow",
            "reason": f"Expected workflow {expected_workflow!r}, got {report.get('workflow')!r}.",
        })
    if report.get("status") not in {"complete", "promote"}:
        reasons.append({
            "gate": "report_status",
            "reason": f"Report status is {report.get('status')!r}.",
        })
    if command_kind == "taxonomy_report":
        rate = _optional_float(metrics.get("entrenched_false_rate"))
        if rate is None:
            reasons.append({
                "gate": "entrenched_false_rate",
                "reason": "entrenched_false_rate is missing or non-finite.",
            })
        elif rate > max_entrenched_false_rate:
            reasons.append({
                "gate": "entrenched_false_rate",
                "reason": f"entrenched_false_rate {rate} exceeds {max_entrenched_false_rate}.",
            })
    elif command_kind == "blind_spot_analysis":
        if metrics.get("assignment_check_passed") is not True:
            reasons.append({
                "gate": "assignment_check",
                "reason": "Blind-spot assignment check did not pass.",
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
        "taxonomy_candidate_count": sum(1 for item in candidates if item.get("command_kind") == "taxonomy_report"),
        "blind_spot_analysis_count": sum(
            1 for item in candidates if item.get("command_kind") == "blind_spot_analysis"
        ),
        "promotion_ready_count": sum(1 for item in candidates if bool(item.get("promotion_ready"))),
        "audit_ready_count": sum(1 for item in candidates if bool(item.get("audit_ready"))),
        "blocked_candidate_count": sum(1 for item in candidates if item["candidate_status"] == "blocked"),
        "runs": tuple(sorted(str(entry.get("run")) for entry in entries if entry.get("run"))),
    }


def _gate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    require_all_reports: bool,
) -> dict[str, Any]:
    blocking: list[dict[str, Any]] = []
    promotion_ready = bool(summary.get("promotion_ready_count"))
    audit_ready = bool(summary.get("audit_ready_count"))
    if not candidates:
        blocking.append({
            "gate": "queue",
            "reason": "Queue does not contain ready detectability rerun entries.",
        })
    if require_all_reports:
        for candidate in candidates:
            if candidate["candidate_status"] in {"missing_report", "invalid_report"}:
                blocking.append({
                    "gate": "report_coverage",
                    "run": candidate.get("run"),
                    "command_kind": candidate.get("command_kind"),
                    "expected_report_path": candidate.get("expected_report_path"),
                    "reason": f"Candidate report status is {candidate['candidate_status']}.",
                })
    if not promotion_ready:
        for candidate in candidates:
            if candidate["candidate_status"] == "blocked":
                for reason in _mapping_sequence(candidate.get("blocking_reasons", ())):
                    blocking.append({
                        "gate": reason.get("gate"),
                        "run": candidate.get("run"),
                        "command_kind": candidate.get("command_kind"),
                        "reason": reason.get("reason"),
                    })
    passed = not blocking and bool(candidates)
    return {
        "passed": passed,
        "promotion_ready": promotion_ready,
        "audit_ready": audit_ready,
        "require_all_reports": bool(require_all_reports),
        "blocking_reasons": tuple(blocking),
    }


def _recommended_candidate(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    best = sorted(candidates, key=_candidate_rank, reverse=True)[0]
    return {
        "run": best.get("run"),
        "command_kind": best.get("command_kind"),
        "candidate_status": best.get("candidate_status"),
        "promotion_ready": best.get("promotion_ready"),
        "audit_ready": best.get("audit_ready"),
        "expected_report_path": best.get("expected_report_path"),
        "observed_report_path": best.get("observed_report_path"),
        "taxonomy_config": best.get("taxonomy_config"),
        "metrics": best.get("metrics"),
        "blocking_reasons": best.get("blocking_reasons"),
    }


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = _mapping(candidate.get("metrics"))
    rate = _optional_float(metrics.get("entrenched_false_rate"))
    return (
        bool(candidate.get("promotion_ready")),
        bool(candidate.get("audit_ready")),
        -_rank_float(rate, missing=1.0),
        _rank_float(metrics.get("selected_record_count")),
        str(candidate.get("run") or ""),
        str(candidate.get("command_kind") or ""),
    )


def _status(*, gate: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    if summary.get("candidate_count", 0) == 0:
        return "empty"
    if gate.get("promotion_ready"):
        return "promote"
    return "complete" if gate.get("passed") else "blocked"


def _queue_entries(queue: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if queue.get("workflow") != QUEUE_WORKFLOW:
        raise ValueError(f"queue workflow must be {QUEUE_WORKFLOW!r}.")
    return tuple(
        entry
        for entry in _mapping_sequence(queue.get("entries", ()))
        if entry.get("command_status") == "ready"
        and entry.get("command_kind") in {"blind_spot_analysis", "taxonomy_report"}
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
    expected_workflow = _expected_workflow(str(entry.get("command_kind") or ""))
    run_name = str(entry.get("run") or "")
    source_report = str(entry.get("source_report") or "")
    for record in reports:
        report = _mapping(record.get("report"))
        if record.get("error") or report.get("workflow") != expected_workflow:
            continue
        if expected_workflow == TAXONOMY_WORKFLOW and run_name and _report_run_name(report) != run_name:
            continue
        if expected_workflow == BLIND_SPOT_WORKFLOW and source_report:
            observed_source = str(_nested(report, "source", "taxonomy_report_path") or "")
            if observed_source and observed_source != source_report:
                continue
        return record
    return None


def _expected_workflow(command_kind: str) -> str | None:
    if command_kind == "blind_spot_analysis":
        return BLIND_SPOT_WORKFLOW
    if command_kind == "taxonomy_report":
        return TAXONOMY_WORKFLOW
    return None


def _report_run_name(report: Mapping[str, Any]) -> str:
    metadata = _mapping(report.get("metadata"))
    for key in ("run_name", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source_summary = _mapping(_nested(report, "source", "score_dump_summary"))
    for key in ("name", "run_name", "model"):
        value = source_summary.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _expected_report_path(entry: Mapping[str, Any]) -> str | None:
    command = _sequence(entry.get("command"))
    for index, part in enumerate(command):
        if str(part) == "--json" and index + 1 < len(command):
            value = str(command[index + 1])
            return value or None
    return None


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
        "frontier_detectability_evidence_rerun_rollup": rollup_path,
        "frontier_detectability_evidence_rerun_queue": queue_path,
    }
    for index, path in enumerate(observed_report_paths, start=1):
        artifacts[f"detectability_rerun_report_{index}"] = Path(path)
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "rollup_frontier_detectability_evidence_reruns",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "passed": _mapping(payload.get("gate")).get("passed"),
            "promotion_ready": _mapping(payload.get("gate")).get("promotion_ready"),
            "audit_ready": _mapping(payload.get("gate")).get("audit_ready"),
            "candidate_count": summary.get("candidate_count"),
            "promotion_ready_count": summary.get("promotion_ready_count"),
            "audit_ready_count": summary.get("audit_ready_count"),
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


def _finite_float(value: Any, *, name: str) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        raise ValueError(f"{name} must be finite.")
    return parsed


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
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
    payload = rollup_frontier_detectability_evidence_reruns(
        queue_path=args.queue,
        report_json_path=args.json,
        report_paths=tuple(args.report or ()),
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        max_entrenched_false_rate=args.max_entrenched_false_rate,
        require_all_reports=bool(args.require_all_reports),
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "frontier_detectability_evidence_rerun_rollup="
        f"{payload['status']} "
        f"candidates={summary['candidate_count']} "
        f"promotion_ready={payload['gate']['promotion_ready']} "
        f"audit_ready={payload['gate']['audit_ready']} "
        f"missing_reports={summary['missing_report_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, help="frontier detectability rerun queue JSON")
    parser.add_argument("--report", action="append", default=[], help="completed rerun report; repeatable")
    parser.add_argument("--json", required=True, help="rollup output JSON")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument(
        "--max-entrenched-false-rate",
        type=float,
        default=DEFAULT_MAX_ENTRENCHED_FALSE_RATE,
    )
    parser.add_argument("--require-all-reports", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
