"""Roll up frontier abstention rerun reports into release evidence.

The companion planner, ``plan_frontier_abstention_evidence_reruns.py``, emits a
profile x signal-family queue of ``eval_abstention_stability.py`` commands. This
workflow reads that queue plus any completed child reports and selects the best
candidate participation-gate configuration without executing model code.
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

WORKFLOW = "frontier_abstention_evidence_rerun_rollup"
QUEUE_WORKFLOW = "frontier_abstention_evidence_rerun_queue"
REPORT_WORKFLOW = "abstention_stability"

DEFAULT_MIN_CONDITIONAL_CORRECTNESS_LOWER_BOUND = 0.80
DEFAULT_MAX_ABSTENTION_RATE = 0.50
DEFAULT_MIN_PASS_SEED_RATE = 1.0


def rollup_frontier_abstention_evidence_reruns(
    *,
    queue_path: str | Path,
    report_json_path: str | Path,
    report_paths: Sequence[str | Path] = (),
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    min_conditional_correctness_lower_bound: float = DEFAULT_MIN_CONDITIONAL_CORRECTNESS_LOWER_BOUND,
    max_abstention_rate: float = DEFAULT_MAX_ABSTENTION_RATE,
    min_pass_seed_rate: float = DEFAULT_MIN_PASS_SEED_RATE,
    require_all_reports: bool = False,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Summarize abstention rerun reports and recommend a candidate config."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    _validate_thresholds(
        min_conditional_correctness_lower_bound=min_conditional_correctness_lower_bound,
        max_abstention_rate=max_abstention_rate,
        min_pass_seed_rate=min_pass_seed_rate,
    )

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
            min_conditional_correctness_lower_bound=min_conditional_correctness_lower_bound,
            max_abstention_rate=max_abstention_rate,
            min_pass_seed_rate=min_pass_seed_rate,
        )
        for entry in entries
    )
    summary = _summary(queue=queue, entries=entries, candidates=candidates)
    gate = _gate(
        candidates,
        require_all_reports=require_all_reports,
        summary=summary,
    )
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
            "min_conditional_correctness_lower_bound": min_conditional_correctness_lower_bound,
            "max_abstention_rate": max_abstention_rate,
            "min_pass_seed_rate": min_pass_seed_rate,
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
                "queue": str(queue_file),
                "artifact_manifest": str(manifest_path),
                "candidate_count": summary["candidate_count"],
                "passing_candidate_count": summary["passing_candidate_count"],
                "promotion_eligible_passing_candidate_count": (
                    summary["promotion_eligible_passing_candidate_count"]
                ),
                "candidate_gate_diagnostics": summary.get("candidate_gate_diagnostics"),
                "missing_report_count": summary["missing_report_count"],
                "best_profile": None if recommended is None else recommended.get("profile"),
                "best_signal_group": None if recommended is None else recommended.get("signal_group"),
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
    min_conditional_correctness_lower_bound: float,
    max_abstention_rate: float,
    min_pass_seed_rate: float,
) -> dict[str, Any]:
    expected_path = _expected_report_path(entry)
    observed = _load_expected_report(expected_path, queue_dir=queue_dir)
    report_source = "expected_path"
    if observed is None:
        observed = _match_explicit_report(entry, explicit_reports)
        report_source = "explicit_report" if observed is not None else "missing"

    base = _candidate_base(entry, expected_path=expected_path)
    if observed is None:
        reasons = ({"gate": "report", "reason": "Expected abstention rerun report is missing."},)
        return {
            **base,
            "candidate_status": "missing_report",
            "observed_report_path": None,
            "report_source": report_source,
            "metrics": {},
            "promotion_ready": False,
            "blocking_reasons": reasons,
        }
    report = _mapping(observed.get("report"))
    report_path = str(observed.get("path") or "")
    report_error = observed.get("error")
    if report_error:
        reasons = ({"gate": "report", "path": report_path, "reason": str(report_error)},)
        return {
            **base,
            "candidate_status": "invalid_report",
            "observed_report_path": report_path,
            "report_source": report_source,
            "metrics": {},
            "promotion_ready": False,
            "blocking_reasons": reasons,
        }

    run = _matching_run(report, str(entry.get("run") or ""))
    config_check = _config_check(entry, report)
    metrics = _candidate_metrics(report, run)
    reasons = _blocking_reasons(
        report=report,
        run=run,
        entry=entry,
        config_check=config_check,
        metrics=metrics,
        min_conditional_correctness_lower_bound=min_conditional_correctness_lower_bound,
        max_abstention_rate=max_abstention_rate,
        min_pass_seed_rate=min_pass_seed_rate,
    )
    promotion_ready = not reasons
    return {
        **base,
        "candidate_status": "promotion_ready" if promotion_ready else "blocked",
        "observed_report_path": report_path,
        "report_source": report_source,
        "report_workflow": report.get("workflow"),
        "report_status": report.get("status"),
        "config_check": config_check,
        "metrics": metrics,
        "promotion_ready": promotion_ready,
        "blocking_reasons": tuple(reasons),
    }


def _candidate_base(entry: Mapping[str, Any], *, expected_path: str | None) -> dict[str, Any]:
    profile_config = _mapping(entry.get("profile_config"))
    return {
        "run": str(entry.get("run") or ""),
        "profile": str(entry.get("profile") or ""),
        "signal_group": str(entry.get("signal_group") or ""),
        "signals": _string_tuple(entry.get("signals")),
        "command_status": str(entry.get("command_status") or ""),
        "promotion_eligible_profile": profile_config.get("promotion_eligible") is not False,
        "profile_config": dict(profile_config),
        "expected_report_path": expected_path,
        "source_report": entry.get("source_report"),
        "source_metrics": _mapping(entry.get("source_metrics")),
    }


def _candidate_metrics(report: Mapping[str, Any], run: Mapping[str, Any] | None) -> dict[str, Any]:
    if run is None:
        return {}
    stability = _mapping(run.get("stability"))
    feasibility = _mapping(run.get("supervised_feasibility_frontier"))
    feasible_best = _mapping(feasibility.get("best"))
    pass_count = _optional_int(stability.get("release_gate_pass_seed_count"))
    block_count = _optional_int(stability.get("release_gate_block_seed_count"))
    seed_count = _optional_int(stability.get("seed_count"))
    if seed_count is None and pass_count is not None and block_count is not None:
        seed_count = pass_count + block_count
    pass_seed_rate = None
    if seed_count and pass_count is not None:
        pass_seed_rate = pass_count / seed_count
    candidate_gate_diagnostics = _candidate_gate_diagnostics(stability)
    return {
        "seed_count": seed_count,
        "release_gate_pass_seed_count": pass_count,
        "release_gate_block_seed_count": block_count,
        "release_gate_pass_seed_rate": pass_seed_rate,
        "all_release_gates_passed": stability.get("all_release_gates_passed"),
        "conditional_correctness_lower_bound_mean": _metric_mean(
            stability.get("conditional_correctness_lower_bound")
        ),
        "empirical_abstention_rate_mean": _metric_mean(stability.get("empirical_abstention_rate")),
        "empirical_selective_accuracy_mean": _metric_mean(
            stability.get("empirical_selective_accuracy")
        ),
        "correct_retention_lower_bound_mean": _metric_mean(
            stability.get("correct_retention_lower_bound")
        ),
        "stable_recommended_score_name": stability.get("stable_recommended_score_name"),
        "recommended_score_name_counts": _mapping(stability.get("recommended_score_name_counts")),
        "candidate_gate_diagnostics": candidate_gate_diagnostics,
        "supervised_feasibility_target_passed": feasibility.get("target_passed"),
        "supervised_feasibility_score_name": feasible_best.get("score_name"),
        "supervised_feasibility_conditional_correctness_lower_bound": _optional_float(
            feasible_best.get("conditional_correctness_lower_bound")
        ),
        "supervised_feasibility_empirical_abstention_rate": _optional_float(
            feasible_best.get("empirical_abstention_rate")
        ),
        "supervised_feasibility_empirical_selective_accuracy": _optional_float(
            feasible_best.get("empirical_selective_accuracy")
        ),
    }


def _candidate_gate_diagnostics(stability: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(stability.get("candidate_gate_summary"))
    if not summary:
        return {}
    seed_count = _optional_int(stability.get("seed_count"))
    if seed_count is None:
        pass_count = _optional_int(stability.get("release_gate_pass_seed_count"))
        block_count = _optional_int(stability.get("release_gate_block_seed_count"))
        if pass_count is not None and block_count is not None:
            seed_count = pass_count + block_count
    seed_with_any_passing_count = _optional_int(
        summary.get("seed_with_any_passing_candidate_count")
    )
    seed_without_passing_count = _optional_int(
        summary.get("seed_without_passing_candidate_count")
    )
    missed_count = _optional_int(summary.get("recommended_missed_passing_candidate_count"))
    recommended_pass_count = _optional_int(summary.get("recommended_pass_seed_count"))
    recommended_block_count = _optional_int(summary.get("recommended_block_seed_count"))
    return {
        "seed_count": seed_count,
        "seed_with_any_passing_candidate_count": seed_with_any_passing_count,
        "seed_without_passing_candidate_count": seed_without_passing_count,
        "seed_with_any_passing_candidate_rate": _rate(
            seed_with_any_passing_count,
            seed_count,
        ),
        "all_seeds_have_passing_candidate": summary.get("all_seeds_have_passing_candidate"),
        "recommended_pass_seed_count": recommended_pass_count,
        "recommended_block_seed_count": recommended_block_count,
        "recommended_missed_passing_candidate_count": missed_count,
        "recommended_missed_passing_candidate_seed_rate": _rate(missed_count, seed_count),
        "recommended_blocking_reason_counts": _mapping(
            summary.get("recommended_blocking_reason_counts")
        ),
        "candidate_blocking_reason_counts": _mapping(
            summary.get("candidate_blocking_reason_counts")
        ),
        "best_passing_score_name_counts": _mapping(
            summary.get("best_passing_score_name_counts")
        ),
    }


def _blocking_reasons(
    *,
    report: Mapping[str, Any],
    run: Mapping[str, Any] | None,
    entry: Mapping[str, Any],
    config_check: Mapping[str, Any],
    metrics: Mapping[str, Any],
    min_conditional_correctness_lower_bound: float,
    max_abstention_rate: float,
    min_pass_seed_rate: float,
) -> tuple[dict[str, Any], ...]:
    reasons: list[dict[str, Any]] = []
    if report.get("workflow") != REPORT_WORKFLOW:
        reasons.append({
            "gate": "workflow",
            "reason": f"Unsupported report workflow {report.get('workflow')!r}.",
        })
    if report.get("status") not in {"complete", "promote"}:
        reasons.append({
            "gate": "report_status",
            "reason": f"Report status is {report.get('status')!r}.",
        })
    if run is None:
        reasons.append({
            "gate": "run",
            "reason": f"Report does not contain run {entry.get('run')!r}.",
        })
        return tuple(reasons)
    if not bool(config_check.get("matches")):
        reasons.append({
            "gate": "config",
            "reason": "Report config does not match queue entry.",
            "details": tuple(config_check.get("mismatches", ())),
        })
    if _mapping(entry.get("profile_config")).get("promotion_eligible") is False:
        reasons.append({
            "gate": "profile",
            "reason": "Queue profile is not marked promotion eligible.",
        })

    correctness = _optional_float(metrics.get("conditional_correctness_lower_bound_mean"))
    abstention = _optional_float(metrics.get("empirical_abstention_rate_mean"))
    pass_seed_rate = _optional_float(metrics.get("release_gate_pass_seed_rate"))
    if correctness is None:
        reasons.append({
            "gate": "conditional_correctness",
            "reason": "conditional_correctness_lower_bound.mean is missing or non-finite.",
        })
    elif correctness < min_conditional_correctness_lower_bound:
        reasons.append({
            "gate": "conditional_correctness",
            "reason": (
                f"conditional_correctness_lower_bound_mean {correctness} is below "
                f"{min_conditional_correctness_lower_bound}."
            ),
        })
    if abstention is None:
        reasons.append({
            "gate": "abstention_rate",
            "reason": "empirical_abstention_rate.mean is missing or non-finite.",
        })
    elif abstention > max_abstention_rate:
        reasons.append({
            "gate": "abstention_rate",
            "reason": f"empirical_abstention_rate_mean {abstention} exceeds {max_abstention_rate}.",
        })
    if pass_seed_rate is None:
        reasons.append({
            "gate": "seed_stability",
            "reason": "release_gate_pass_seed_rate is missing.",
        })
    elif pass_seed_rate < min_pass_seed_rate:
        reasons.append({
            "gate": "seed_stability",
            "reason": f"release_gate_pass_seed_rate {pass_seed_rate} is below {min_pass_seed_rate}.",
        })
    return tuple(reasons)


def _summary(
    *,
    queue: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    missing = tuple(candidate for candidate in candidates if candidate["candidate_status"] == "missing_report")
    invalid = tuple(candidate for candidate in candidates if candidate["candidate_status"] == "invalid_report")
    passing = tuple(candidate for candidate in candidates if bool(candidate.get("promotion_ready")))
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
        "missing_report_count": len(missing),
        "invalid_report_count": len(invalid),
        "blocked_candidate_count": sum(1 for candidate in candidates if candidate["candidate_status"] == "blocked"),
        "passing_candidate_count": len(passing),
        "promotion_eligible_passing_candidate_count": sum(
            1 for candidate in passing if bool(candidate.get("promotion_eligible_profile"))
        ),
        "runs": tuple(sorted({str(entry.get("run") or "") for entry in entries if entry.get("run")})),
        "profiles": tuple(sorted({str(entry.get("profile") or "") for entry in entries if entry.get("profile")})),
        "signal_groups": tuple(sorted(
            {str(entry.get("signal_group") or "") for entry in entries if entry.get("signal_group")}
        )),
        "candidate_gate_diagnostics": _candidate_gate_rollup_summary(candidates),
    }


def _candidate_gate_rollup_summary(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reports_with_diagnostics = 0
    reports_with_any_seed_passing_candidate = 0
    reports_with_all_seeds_passing_candidate = 0
    reports_with_recommended_miss = 0
    missed_seed_count = 0
    without_passing_seed_count = 0
    best_passing_counts: dict[str, int] = {}
    recommended_reason_counts: dict[str, int] = {}
    candidate_reason_counts: dict[str, int] = {}
    for candidate in candidates:
        diagnostics = _mapping(_mapping(candidate.get("metrics")).get("candidate_gate_diagnostics"))
        if not diagnostics:
            continue
        reports_with_diagnostics += 1
        if _optional_int(diagnostics.get("seed_with_any_passing_candidate_count")):
            reports_with_any_seed_passing_candidate += 1
        if diagnostics.get("all_seeds_have_passing_candidate") is True:
            reports_with_all_seeds_passing_candidate += 1
        missed_count = _optional_int(
            diagnostics.get("recommended_missed_passing_candidate_count")
        ) or 0
        if missed_count > 0:
            reports_with_recommended_miss += 1
            missed_seed_count += missed_count
        without_passing_seed_count += (
            _optional_int(diagnostics.get("seed_without_passing_candidate_count")) or 0
        )
        _merge_int_counts(
            best_passing_counts,
            diagnostics.get("best_passing_score_name_counts"),
        )
        _merge_int_counts(
            recommended_reason_counts,
            diagnostics.get("recommended_blocking_reason_counts"),
        )
        _merge_int_counts(
            candidate_reason_counts,
            diagnostics.get("candidate_blocking_reason_counts"),
        )
    return {
        "reports_with_candidate_gate_diagnostics_count": reports_with_diagnostics,
        "reports_without_candidate_gate_diagnostics_count": (
            len(candidates) - reports_with_diagnostics
        ),
        "reports_with_any_seed_passing_candidate_count": (
            reports_with_any_seed_passing_candidate
        ),
        "reports_with_all_seeds_passing_candidate_count": (
            reports_with_all_seeds_passing_candidate
        ),
        "reports_with_recommended_missed_passing_candidate_count": (
            reports_with_recommended_miss
        ),
        "recommended_missed_passing_candidate_seed_count": missed_seed_count,
        "seed_without_passing_candidate_count": without_passing_seed_count,
        "best_passing_score_name_counts": dict(sorted(best_passing_counts.items())),
        "recommended_blocking_reason_counts": dict(
            sorted(recommended_reason_counts.items())
        ),
        "candidate_blocking_reason_counts": dict(sorted(candidate_reason_counts.items())),
    }


def _merge_int_counts(target: dict[str, int], value: Any) -> None:
    if not isinstance(value, Mapping):
        return
    for name, count in value.items():
        parsed_count = _optional_int(count)
        if parsed_count is not None:
            target[str(name)] = target.get(str(name), 0) + parsed_count


def _gate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    require_all_reports: bool,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    blocking: list[dict[str, Any]] = []
    if not candidates:
        blocking.append({
            "gate": "queue",
            "reason": "Queue does not contain ready abstention rerun entries.",
        })
    if require_all_reports:
        for candidate in candidates:
            if candidate["candidate_status"] in {"missing_report", "invalid_report"}:
                blocking.append({
                    "gate": "report_coverage",
                    "run": candidate.get("run"),
                    "profile": candidate.get("profile"),
                    "signal_group": candidate.get("signal_group"),
                    "expected_report_path": candidate.get("expected_report_path"),
                    "reason": f"Candidate report status is {candidate['candidate_status']}.",
                })
    if not any(bool(candidate.get("promotion_ready")) for candidate in candidates):
        blocking.append({
            "gate": "promotion_candidate",
            "reason": "No abstention rerun candidate satisfies release thresholds.",
        })
    passed = not blocking
    promotion_ready = bool(
        passed
        and summary.get("promotion_eligible_passing_candidate_count", 0)
        and summary.get("candidate_count", 0)
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
    ranked = sorted(candidates, key=_candidate_rank, reverse=True)
    best = ranked[0]
    return {
        "run": best.get("run"),
        "profile": best.get("profile"),
        "signal_group": best.get("signal_group"),
        "signals": best.get("signals"),
        "candidate_status": best.get("candidate_status"),
        "promotion_ready": best.get("promotion_ready"),
        "promotion_eligible_profile": best.get("promotion_eligible_profile"),
        "expected_report_path": best.get("expected_report_path"),
        "observed_report_path": best.get("observed_report_path"),
        "metrics": best.get("metrics"),
        "blocking_reasons": best.get("blocking_reasons"),
    }


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = _mapping(candidate.get("metrics"))
    abstention = _optional_float(metrics.get("empirical_abstention_rate_mean"))
    return (
        bool(candidate.get("promotion_ready")),
        bool(candidate.get("promotion_eligible_profile")),
        _rank_float(metrics.get("conditional_correctness_lower_bound_mean")),
        _rank_float(metrics.get("release_gate_pass_seed_rate")),
        -_rank_float(abstention, missing=1.0),
        _rank_float(metrics.get("empirical_selective_accuracy_mean")),
        _rank_float(metrics.get("correct_retention_lower_bound_mean")),
        str(candidate.get("run") or ""),
        str(candidate.get("profile") or ""),
        str(candidate.get("signal_group") or ""),
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
        if entry.get("command_kind") == "abstention_stability_experiment"
        and entry.get("command_status") == "ready"
    )


def _load_explicit_reports(
    report_paths: Sequence[str | Path],
    *,
    queue_dir: Path,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _load_report_record(_resolve_path(Path(path), base=queue_dir))
        for path in report_paths
    )


def _load_expected_report(path: str | None, *, queue_dir: Path) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    resolved = _resolve_existing_path(candidate, base=queue_dir)
    if resolved is None:
        return None
    return _load_report_record(resolved)


def _load_report_record(path: Path) -> dict[str, Any]:
    try:
        return {"path": str(path), "report": _load_json_object(path), "error": None}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"path": str(path), "report": {}, "error": str(exc)}


def _match_explicit_report(
    entry: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    matches = []
    for report_record in reports:
        report = _mapping(report_record.get("report"))
        if report_record.get("error") or report.get("workflow") != REPORT_WORKFLOW:
            continue
        if _matching_run(report, str(entry.get("run") or "")) is None:
            continue
        check = _config_check(entry, report)
        if check.get("matches"):
            matches.append(report_record)
    return None if not matches else matches[0]


def _matching_run(report: Mapping[str, Any], run_name: str) -> Mapping[str, Any] | None:
    runs = tuple(_mapping_sequence(report.get("runs", ())))
    if run_name:
        for run in runs:
            if str(run.get("name") or "") == run_name:
                return run
    return runs[0] if len(runs) == 1 else None


def _config_check(entry: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    config = _mapping(report.get("config"))
    profile = _mapping(entry.get("profile_config"))
    mismatches: list[dict[str, Any]] = []
    entry_signals = set(_expected_config_signals(entry))
    report_signals = set(_string_tuple(config.get("signals")))
    if entry_signals and report_signals and entry_signals != report_signals:
        mismatches.append({
            "field": "signals",
            "expected": tuple(sorted(entry_signals)),
            "observed": tuple(sorted(report_signals)),
        })
    _check_float_match("alpha", profile.get("alpha"), config.get("alpha"), mismatches)
    expected_best_by = profile.get("best_by")
    observed_best_by = config.get("best_by")
    if expected_best_by and observed_best_by and str(expected_best_by) != str(observed_best_by):
        mismatches.append({
            "field": "best_by",
            "expected": str(expected_best_by),
            "observed": str(observed_best_by),
        })
    release_gate = _mapping(config.get("release_gate"))
    _check_float_match(
        "min_conditional_correctness_lower_bound",
        profile.get("min_conditional_correctness_lower_bound"),
        release_gate.get("min_conditional_correctness_lower_bound"),
        mismatches,
    )
    _check_float_match(
        "max_abstention_rate",
        profile.get("max_abstention_rate"),
        release_gate.get("max_abstention_rate"),
        mismatches,
    )
    return {"matches": not mismatches, "mismatches": tuple(mismatches)}


def _expected_config_signals(entry: Mapping[str, Any]) -> tuple[str, ...]:
    derived = _mapping(entry.get("derived_signal_config"))
    base_signals = _string_tuple(derived.get("base_signals"))
    return base_signals or _string_tuple(entry.get("signals"))


def _check_float_match(
    field: str,
    expected: Any,
    observed: Any,
    mismatches: list[dict[str, Any]],
) -> None:
    expected_float = _optional_float(expected)
    observed_float = _optional_float(observed)
    if expected_float is None or observed_float is None:
        return
    if not math.isclose(expected_float, observed_float, rel_tol=1e-9, abs_tol=1e-12):
        mismatches.append({"field": field, "expected": expected_float, "observed": observed_float})


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
        "frontier_abstention_evidence_rerun_rollup": rollup_path,
        "frontier_abstention_evidence_rerun_queue": queue_path,
    }
    for index, path in enumerate(observed_report_paths, start=1):
        artifacts[f"abstention_rerun_report_{index}"] = Path(path)
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "rollup_frontier_abstention_evidence_reruns",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "passed": _mapping(payload.get("gate")).get("passed"),
            "promotion_ready": _mapping(payload.get("gate")).get("promotion_ready"),
            "candidate_count": summary.get("candidate_count"),
            "passing_candidate_count": summary.get("passing_candidate_count"),
            "missing_report_count": summary.get("missing_report_count"),
            "candidate_gate_diagnostics": summary.get("candidate_gate_diagnostics"),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def _validate_thresholds(
    *,
    min_conditional_correctness_lower_bound: float,
    max_abstention_rate: float,
    min_pass_seed_rate: float,
) -> None:
    for name, value in (
        ("min_conditional_correctness_lower_bound", min_conditional_correctness_lower_bound),
        ("max_abstention_rate", max_abstention_rate),
        ("min_pass_seed_rate", min_pass_seed_rate),
    ):
        if _optional_float(value) is None:
            raise ValueError(f"{name} must be finite.")
    if not 0.0 <= min_pass_seed_rate <= 1.0:
        raise ValueError("min_pass_seed_rate must be between 0 and 1.")


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


def _metric_mean(value: Any) -> float | None:
    if isinstance(value, Mapping):
        return _optional_float(value.get("mean"))
    return _optional_float(value)


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


def _rate(count: int | None, total: int | None) -> float | None:
    if count is None or total is None or total <= 0:
        return None
    return count / total


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


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
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
    payload = rollup_frontier_abstention_evidence_reruns(
        queue_path=args.queue,
        report_json_path=args.json,
        report_paths=tuple(args.report or ()),
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        min_conditional_correctness_lower_bound=args.min_abstention_conditional_correctness_lower_bound,
        max_abstention_rate=args.max_abstention_rate,
        min_pass_seed_rate=args.min_pass_seed_rate,
        require_all_reports=bool(args.require_all_reports),
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "frontier_abstention_evidence_rerun_rollup="
        f"{payload['status']} "
        f"candidates={summary['candidate_count']} "
        f"passing={summary['passing_candidate_count']} "
        f"missing_reports={summary['missing_report_count']} "
        f"promotion_ready={payload['gate']['promotion_ready']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, help="frontier abstention rerun queue JSON")
    parser.add_argument("--report", action="append", default=[], help="completed abstention report; repeatable")
    parser.add_argument("--json", required=True, help="rollup output JSON")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument(
        "--min-abstention-conditional-correctness-lower-bound",
        type=float,
        default=DEFAULT_MIN_CONDITIONAL_CORRECTNESS_LOWER_BOUND,
    )
    parser.add_argument("--max-abstention-rate", type=float, default=DEFAULT_MAX_ABSTENTION_RATE)
    parser.add_argument("--min-pass-seed-rate", type=float, default=DEFAULT_MIN_PASS_SEED_RATE)
    parser.add_argument("--require-all-reports", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
