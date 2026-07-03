"""Roll up frontier stability reruns into release evidence.

The companion planner emits one ``eval_verifier_stability.py`` or
``eval_abstention_stability.py`` command per blocked stability track. This
workflow reads completed child reports and applies the same release thresholds
used by ``compare_frontier_release_evidence.py`` without rerunning model code.
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

from benchmarks.compare_frontier_release_evidence import (  # noqa: E402
    DEFAULT_MAX_ABSTENTION_RATE_MEAN,
    DEFAULT_MAX_VERIFIED_FALSE_ALARM_MEAN,
    DEFAULT_MIN_ABSTENTION_CONDITIONAL_CORRECTNESS_LOWER_BOUND_MEAN,
    DEFAULT_MIN_ABSTENTION_PASS_SEED_RATE,
    DEFAULT_MIN_VERIFIED_DETECTION_MEAN,
    DEFAULT_MIN_VERIFIER_BEATS_INTERNAL_SEED_RATE,
    DEFAULT_MIN_VERIFIER_DELTA_DETECTION_MEAN,
    DEFAULT_MIN_VERIFIER_PASS_SEED_RATE,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_stability_evidence_rerun_rollup"
QUEUE_WORKFLOW = "frontier_stability_evidence_rerun_queue"
TRACK_WORKFLOWS = {
    "verifier_stability": "verifier_stability",
    "abstention_stability": "abstention_stability",
}
DEFAULT_REPORT_NAMES = {
    "verifier_stability": "verifier-stability-report.json",
    "abstention_stability": "abstention-stability-report.json",
}


def rollup_frontier_stability_evidence_reruns(
    *,
    queue_path: str | Path,
    report_json_path: str | Path,
    report_paths: Sequence[str | Path] = (),
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    max_verified_false_alarm_mean: float = DEFAULT_MAX_VERIFIED_FALSE_ALARM_MEAN,
    min_verified_detection_mean: float = DEFAULT_MIN_VERIFIED_DETECTION_MEAN,
    min_verifier_delta_detection_mean: float = DEFAULT_MIN_VERIFIER_DELTA_DETECTION_MEAN,
    min_verifier_pass_seed_rate: float = DEFAULT_MIN_VERIFIER_PASS_SEED_RATE,
    min_verifier_beats_internal_seed_rate: float = DEFAULT_MIN_VERIFIER_BEATS_INTERNAL_SEED_RATE,
    min_abstention_pass_seed_rate: float = DEFAULT_MIN_ABSTENTION_PASS_SEED_RATE,
    min_abstention_conditional_correctness_lower_bound_mean: float = (
        DEFAULT_MIN_ABSTENTION_CONDITIONAL_CORRECTNESS_LOWER_BOUND_MEAN
    ),
    max_abstention_rate_mean: float = DEFAULT_MAX_ABSTENTION_RATE_MEAN,
    require_all_reports: bool = False,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Summarize completed frontier stability reruns and recommend release status."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    _validate_config(
        max_verified_false_alarm_mean=max_verified_false_alarm_mean,
        min_verified_detection_mean=min_verified_detection_mean,
        min_verifier_delta_detection_mean=min_verifier_delta_detection_mean,
        min_verifier_pass_seed_rate=min_verifier_pass_seed_rate,
        min_verifier_beats_internal_seed_rate=min_verifier_beats_internal_seed_rate,
        min_abstention_pass_seed_rate=min_abstention_pass_seed_rate,
        min_abstention_conditional_correctness_lower_bound_mean=(
            min_abstention_conditional_correctness_lower_bound_mean
        ),
        max_abstention_rate_mean=max_abstention_rate_mean,
    )
    config = {
        "max_verified_false_alarm_mean": float(max_verified_false_alarm_mean),
        "min_verified_detection_mean": float(min_verified_detection_mean),
        "min_verifier_delta_detection_mean": float(min_verifier_delta_detection_mean),
        "min_verifier_pass_seed_rate": float(min_verifier_pass_seed_rate),
        "min_verifier_beats_internal_seed_rate": float(min_verifier_beats_internal_seed_rate),
        "min_abstention_pass_seed_rate": float(min_abstention_pass_seed_rate),
        "min_abstention_conditional_correctness_lower_bound_mean": float(
            min_abstention_conditional_correctness_lower_bound_mean
        ),
        "max_abstention_rate_mean": float(max_abstention_rate_mean),
        "require_all_reports": bool(require_all_reports),
    }

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
            config=config,
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
        "config": config,
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
                "verifier_track_status": summary["track_statuses"].get("verifier_stability"),
                "abstention_track_status": summary["track_statuses"].get("abstention_stability"),
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
    config: Mapping[str, float],
) -> dict[str, Any]:
    track = str(entry.get("track") or "")
    expected_path = _expected_report_path(entry)
    observed = _load_expected_report(expected_path, queue_dir=queue_dir)
    report_source = "expected_path"
    if observed is None:
        observed = _match_explicit_report(track, explicit_reports)
        report_source = "explicit_report" if observed is not None else "missing"
    base = _candidate_base(entry, expected_path=expected_path)
    if observed is None:
        return {
            **base,
            "candidate_status": "missing_report",
            "observed_report_path": None,
            "report_source": report_source,
            "metrics": {},
            "run_decisions": (),
            "promotion_ready": False,
            "blocking_reasons": (
                {"gate": "report", "reason": "Expected stability rerun report is missing."},
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
            "run_decisions": (),
            "promotion_ready": False,
            "blocking_reasons": (
                {"gate": "report", "path": report_path, "reason": str(observed["error"])},
            ),
        }
    metrics, run_decisions = _candidate_metrics(track=track, report=report, config=config)
    reasons = _blocking_reasons(track=track, report=report, metrics=metrics, run_decisions=run_decisions)
    promotion_ready = not reasons
    return {
        **base,
        "candidate_status": "promotion_ready" if promotion_ready else "blocked",
        "observed_report_path": report_path,
        "report_source": report_source,
        "report_workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "metrics": metrics,
        "run_decisions": run_decisions,
        "promotion_ready": promotion_ready,
        "blocking_reasons": tuple(reasons),
    }


def _candidate_base(entry: Mapping[str, Any], *, expected_path: str | None) -> dict[str, Any]:
    return {
        "track": entry.get("track"),
        "source_workflow": entry.get("source_workflow"),
        "source_report": entry.get("source_report"),
        "command_status": entry.get("command_status"),
        "expected_report_path": expected_path,
        "rerun_output_dir": entry.get("rerun_output_dir"),
    }


def _candidate_metrics(
    *,
    track: str,
    report: Mapping[str, Any],
    config: Mapping[str, float],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    runs = _mapping_sequence(report.get("runs", ()))
    if track == "verifier_stability":
        run_decisions = tuple(_verifier_run_decision(run, config=config) for run in runs)
    elif track == "abstention_stability":
        run_decisions = tuple(_abstention_run_decision(run, config=config) for run in runs)
    else:
        run_decisions = ()
    passed_count = sum(1 for item in run_decisions if item["status"] == "promote")
    blocked_count = sum(1 for item in run_decisions if item["status"] == "blocked")
    return (
        {
            "run_count": len(runs),
            "promoted_run_count": passed_count,
            "blocked_run_count": blocked_count,
            "run_names": tuple(item["name"] for item in run_decisions),
        },
        run_decisions,
    )


def _verifier_run_decision(
    run: Mapping[str, Any],
    *,
    config: Mapping[str, float],
) -> dict[str, Any]:
    name = str(run.get("name") or "")
    stability = _mapping(run.get("stability"))
    seed_count = _positive_int(stability.get("seed_count"))
    pass_count = _non_negative_int(stability.get("verified_pass_seed_count"))
    beats_count = _non_negative_int(stability.get("verified_beats_internal_detection_seed_count"))
    metrics = {
        "verified_false_alarm_mean": _metric_mean(stability, "verified_false_alarm"),
        "verified_detection_mean": _metric_mean(stability, "verified_detection"),
        "delta_detection_mean": _metric_mean(stability, "delta_detection"),
        "verified_pass_seed_count": pass_count,
        "verified_beats_internal_detection_seed_count": beats_count,
        "seed_count": seed_count,
        "verified_pass_seed_rate": _ratio(pass_count, seed_count),
        "verified_beats_internal_detection_seed_rate": _ratio(beats_count, seed_count),
    }
    checks = (
        _max_check(
            "verified_false_alarm_mean",
            metrics["verified_false_alarm_mean"],
            config["max_verified_false_alarm_mean"],
        ),
        _min_check(
            "verified_detection_mean",
            metrics["verified_detection_mean"],
            config["min_verified_detection_mean"],
        ),
        _min_check(
            "delta_detection_mean",
            metrics["delta_detection_mean"],
            config["min_verifier_delta_detection_mean"],
        ),
        _min_check(
            "verified_pass_seed_rate",
            metrics["verified_pass_seed_rate"],
            config["min_verifier_pass_seed_rate"],
        ),
        _min_check(
            "verified_beats_internal_detection_seed_rate",
            metrics["verified_beats_internal_detection_seed_rate"],
            config["min_verifier_beats_internal_seed_rate"],
        ),
    )
    reasons = []
    if not name:
        reasons.append("run name is missing")
    if seed_count is None:
        reasons.append("seed_count is missing or non-positive")
    reasons.extend(_failed_reasons(checks, prefix=f"verifier_stability.{name}"))
    return {
        "name": name,
        "track": "verifier_stability",
        "status": "promote" if not reasons else "blocked",
        "metrics": metrics,
        "checks": checks,
        "blocking_reasons": tuple(reasons),
    }


def _abstention_run_decision(
    run: Mapping[str, Any],
    *,
    config: Mapping[str, float],
) -> dict[str, Any]:
    name = str(run.get("name") or "")
    stability = _mapping(run.get("stability"))
    seed_count = _positive_int(stability.get("seed_count"))
    pass_count = _non_negative_int(stability.get("release_gate_pass_seed_count"))
    feasibility = _mapping(run.get("supervised_feasibility_frontier"))
    feasible_best = _mapping(feasibility.get("best"))
    metrics = {
        "conditional_correctness_lower_bound_mean": _metric_mean(
            stability,
            "conditional_correctness_lower_bound",
        ),
        "empirical_abstention_rate_mean": _metric_mean(stability, "empirical_abstention_rate"),
        "release_gate_pass_seed_count": pass_count,
        "release_gate_block_seed_count": _non_negative_int(stability.get("release_gate_block_seed_count")),
        "seed_count": seed_count,
        "release_gate_pass_seed_rate": _ratio(pass_count, seed_count),
        "stable_recommended_score_name": stability.get("stable_recommended_score_name"),
        "recommended_score_name_counts": _mapping(stability.get("recommended_score_name_counts")),
        "supervised_feasibility_target_passed": feasibility.get("target_passed"),
        "supervised_feasibility_score_name": feasible_best.get("score_name"),
        "supervised_feasibility_conditional_correctness_lower_bound": feasible_best.get(
            "conditional_correctness_lower_bound"
        ),
        "supervised_feasibility_empirical_abstention_rate": feasible_best.get(
            "empirical_abstention_rate"
        ),
    }
    checks = (
        _min_check(
            "conditional_correctness_lower_bound_mean",
            metrics["conditional_correctness_lower_bound_mean"],
            config["min_abstention_conditional_correctness_lower_bound_mean"],
        ),
        _max_check(
            "empirical_abstention_rate_mean",
            metrics["empirical_abstention_rate_mean"],
            config["max_abstention_rate_mean"],
        ),
        _min_check(
            "release_gate_pass_seed_rate",
            metrics["release_gate_pass_seed_rate"],
            config["min_abstention_pass_seed_rate"],
        ),
    )
    reasons = []
    if not name:
        reasons.append("run name is missing")
    if seed_count is None:
        reasons.append("seed_count is missing or non-positive")
    reasons.extend(_failed_reasons(checks, prefix=f"abstention_stability.{name}"))
    return {
        "name": name,
        "track": "abstention_stability",
        "status": "promote" if not reasons else "blocked",
        "metrics": metrics,
        "checks": checks,
        "blocking_reasons": tuple(reasons),
    }


def _blocking_reasons(
    *,
    track: str,
    report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    run_decisions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    reasons: list[dict[str, Any]] = []
    expected_workflow = TRACK_WORKFLOWS.get(track)
    if expected_workflow is None:
        reasons.append({"gate": "track", "reason": f"Unsupported stability track {track!r}."})
    elif report.get("workflow") != expected_workflow:
        reasons.append({
            "gate": "workflow",
            "reason": f"Expected workflow {expected_workflow!r}, got {report.get('workflow')!r}.",
        })
    if report.get("status") not in {"complete", "promote"}:
        reasons.append({
            "gate": "report_status",
            "reason": f"Report status is {report.get('status')!r}.",
        })
    if metrics.get("run_count") in {0, None}:
        reasons.append({"gate": "runs", "reason": "Stability report contains no runs."})
    for decision in run_decisions:
        if decision.get("status") != "promote":
            reasons.append({
                "gate": "run",
                "run": decision.get("name"),
                "reason": "Stability run does not satisfy release thresholds.",
                "blocking_reasons": decision.get("blocking_reasons"),
            })
    return tuple(reasons)


def _summary(
    *,
    queue: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    track_statuses = {
        str(candidate.get("track")): (
            "promote" if bool(candidate.get("promotion_ready")) else "blocked"
        )
        for candidate in candidates
        if candidate.get("track")
    }
    return {
        "queue_workflow": queue.get("workflow"),
        "queue_status": queue.get("status"),
        "queue_entry_count": len(_mapping_sequence(queue.get("entries", ()))),
        "expected_entry_count": len(entries),
        "candidate_count": len(candidates),
        "observed_report_count": len({
            str(candidate.get("observed_report_path"))
            for candidate in candidates
            if candidate.get("observed_report_path")
        }),
        "missing_report_count": sum(1 for item in candidates if item["candidate_status"] == "missing_report"),
        "invalid_report_count": sum(1 for item in candidates if item["candidate_status"] == "invalid_report"),
        "promotion_ready_count": sum(1 for item in candidates if bool(item.get("promotion_ready"))),
        "blocked_candidate_count": sum(1 for item in candidates if item["candidate_status"] == "blocked"),
        "tracks": tuple(sorted(str(entry.get("track")) for entry in entries if entry.get("track"))),
        "track_statuses": track_statuses,
    }


def _gate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
    require_all_reports: bool,
) -> dict[str, Any]:
    blocking: list[dict[str, Any]] = []
    if not candidates:
        blocking.append({"gate": "queue", "reason": "Queue does not contain ready stability rerun entries."})
    for candidate in candidates:
        if candidate["candidate_status"] in {"missing_report", "invalid_report"}:
            blocking.append({
                "gate": "report_coverage",
                "track": candidate.get("track"),
                "expected_report_path": candidate.get("expected_report_path"),
                "reason": f"Candidate report status is {candidate['candidate_status']}.",
            })
    for candidate in candidates:
        if candidate["candidate_status"] == "blocked":
            for reason in _mapping_sequence(candidate.get("blocking_reasons", ())):
                blocking.append({
                    "gate": reason.get("gate"),
                    "track": candidate.get("track"),
                    "run": reason.get("run"),
                    "reason": reason.get("reason"),
                    "blocking_reasons": reason.get("blocking_reasons"),
                })
    passed = not blocking and bool(candidates)
    promotion_ready = bool(
        passed and summary.get("promotion_ready_count") == summary.get("candidate_count")
    )
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
        "track": best.get("track"),
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
        _rank_float(metrics.get("promoted_run_count")),
        -_rank_float(metrics.get("blocked_run_count")),
        str(candidate.get("track") or ""),
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
        if entry.get("command_status") == "ready"
        and entry.get("track") in TRACK_WORKFLOWS
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
    track: str,
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    expected_workflow = TRACK_WORKFLOWS.get(track)
    for record in reports:
        report = _mapping(record.get("report"))
        if record.get("error") or report.get("workflow") != expected_workflow:
            continue
        return record
    return None


def _expected_report_path(entry: Mapping[str, Any]) -> str | None:
    command = _sequence(entry.get("command"))
    for index, part in enumerate(command):
        if str(part) == "--json" and index + 1 < len(command):
            return str(command[index + 1])
    output_dir = entry.get("rerun_output_dir")
    track = str(entry.get("track") or "")
    report_name = DEFAULT_REPORT_NAMES.get(track)
    if output_dir is None or report_name is None:
        return None
    return str(Path(str(output_dir)) / report_name)


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
        "frontier_stability_evidence_rerun_rollup": rollup_path,
        "frontier_stability_evidence_rerun_queue": queue_path,
    }
    for index, path in enumerate(observed_report_paths, start=1):
        artifacts[f"stability_rerun_report_{index}"] = Path(path)
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "rollup_frontier_stability_evidence_reruns",
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


def _validate_config(**values: float) -> None:
    for key, value in values.items():
        parsed = _optional_float(value)
        if parsed is None:
            raise ValueError(f"{key} must be finite.")
        if key != "min_verifier_delta_detection_mean" and not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{key} must be in [0, 1].")


def _min_check(name: str, value: Any, threshold: float) -> dict[str, Any]:
    parsed = _optional_float(value)
    passed = parsed is not None and parsed >= threshold
    return {
        "metric": name,
        "op": ">=",
        "value": parsed,
        "threshold": float(threshold),
        "passed": passed,
    }


def _max_check(name: str, value: Any, threshold: float) -> dict[str, Any]:
    parsed = _optional_float(value)
    passed = parsed is not None and parsed <= threshold
    return {
        "metric": name,
        "op": "<=",
        "value": parsed,
        "threshold": float(threshold),
        "passed": passed,
    }


def _failed_reasons(checks: Sequence[Mapping[str, Any]], *, prefix: str) -> tuple[str, ...]:
    reasons = []
    for check in checks:
        if check.get("passed") is True:
            continue
        value = check.get("value")
        if value is None:
            reasons.append(f"{prefix}.{check.get('metric')} is missing or non-finite")
        else:
            reasons.append(
                f"{prefix}.{check.get('metric')} {value} failed "
                f"{check.get('op')} {check.get('threshold')}"
            )
    return tuple(reasons)


def _metric_mean(payload: Mapping[str, Any], name: str) -> float | None:
    value = payload.get(name)
    if isinstance(value, Mapping):
        return _optional_float(value.get("mean"))
    return _optional_float(value)


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _positive_int(value: Any) -> int | None:
    parsed = _non_negative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _rank_float(value: Any, *, missing: float = 0.0) -> float:
    parsed = _optional_float(value)
    return missing if parsed is None else parsed


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
    candidates = (path,) if path.is_absolute() else (path, base / path, ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_path(path: Path, *, base: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    candidate = base / path
    return candidate if candidate.exists() else path


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
    payload = rollup_frontier_stability_evidence_reruns(
        queue_path=args.queue,
        report_json_path=args.json,
        report_paths=tuple(args.report or ()),
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        max_verified_false_alarm_mean=args.max_verified_false_alarm_mean,
        min_verified_detection_mean=args.min_verified_detection_mean,
        min_verifier_delta_detection_mean=args.min_verifier_delta_detection_mean,
        min_verifier_pass_seed_rate=args.min_verifier_pass_seed_rate,
        min_verifier_beats_internal_seed_rate=args.min_verifier_beats_internal_seed_rate,
        min_abstention_pass_seed_rate=args.min_abstention_pass_seed_rate,
        min_abstention_conditional_correctness_lower_bound_mean=(
            args.min_abstention_conditional_correctness_lower_bound_mean
        ),
        max_abstention_rate_mean=args.max_abstention_rate_mean,
        require_all_reports=bool(args.require_all_reports),
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "frontier_stability_evidence_rerun_rollup="
        f"{payload['status']} "
        f"candidates={summary['candidate_count']} "
        f"promotion_ready={payload['gate']['promotion_ready']} "
        f"missing_reports={summary['missing_report_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, help="frontier stability rerun queue JSON")
    parser.add_argument("--report", action="append", default=[], help="completed child stability report; repeatable")
    parser.add_argument("--json", required=True, help="rollup output JSON")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--max-verified-false-alarm-mean", type=float, default=DEFAULT_MAX_VERIFIED_FALSE_ALARM_MEAN)
    parser.add_argument("--min-verified-detection-mean", type=float, default=DEFAULT_MIN_VERIFIED_DETECTION_MEAN)
    parser.add_argument(
        "--min-verifier-delta-detection-mean",
        type=float,
        default=DEFAULT_MIN_VERIFIER_DELTA_DETECTION_MEAN,
    )
    parser.add_argument("--min-verifier-pass-seed-rate", type=float, default=DEFAULT_MIN_VERIFIER_PASS_SEED_RATE)
    parser.add_argument(
        "--min-verifier-beats-internal-seed-rate",
        type=float,
        default=DEFAULT_MIN_VERIFIER_BEATS_INTERNAL_SEED_RATE,
    )
    parser.add_argument("--min-abstention-pass-seed-rate", type=float, default=DEFAULT_MIN_ABSTENTION_PASS_SEED_RATE)
    parser.add_argument(
        "--min-abstention-conditional-correctness-lower-bound-mean",
        type=float,
        default=DEFAULT_MIN_ABSTENTION_CONDITIONAL_CORRECTNESS_LOWER_BOUND_MEAN,
    )
    parser.add_argument("--max-abstention-rate-mean", type=float, default=DEFAULT_MAX_ABSTENTION_RATE_MEAN)
    parser.add_argument("--require-all-reports", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
