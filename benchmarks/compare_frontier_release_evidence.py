"""Combine frontier stability evidence into one fail-closed release verdict."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

DEFAULT_MAX_VERIFIED_FALSE_ALARM_MEAN = 0.02
DEFAULT_MIN_VERIFIED_DETECTION_MEAN = 0.20
DEFAULT_MIN_VERIFIER_DELTA_DETECTION_MEAN = 0.0
DEFAULT_MIN_VERIFIER_PASS_SEED_RATE = 1.0
DEFAULT_MIN_VERIFIER_BEATS_INTERNAL_SEED_RATE = 1.0
DEFAULT_MIN_ABSTENTION_PASS_SEED_RATE = 1.0
DEFAULT_MIN_ABSTENTION_CONDITIONAL_CORRECTNESS_LOWER_BOUND_MEAN = 0.8
DEFAULT_MAX_ABSTENTION_RATE_MEAN = 0.5
DEFAULT_MAX_DETECTABILITY_ENTRENCHED_FALSE_RATE = 0.25
FRONTIER_RERUN_ROLLUP_WORKFLOWS = {
    "frontier_stability_evidence_rerun_rollup": "stability",
    "frontier_abstention_evidence_rerun_rollup": "abstention",
    "frontier_detectability_evidence_rerun_rollup": "detectability",
    "frontier_multiple_testing_rerun_rollup": "multiple_testing",
}


def compare_frontier_release_evidence(
    *,
    verifier_stability_report_path: str | Path,
    abstention_stability_report_path: str | Path,
    detectability_taxonomy_report_paths: Sequence[str | Path] = (),
    frontier_workflow_report_paths: Sequence[str | Path] = (),
    citation_batch_rollup_report_paths: Sequence[str | Path] = (),
    frontier_rerun_rollup_report_paths: Sequence[str | Path] = (),
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
    max_detectability_entrenched_false_rate: float = DEFAULT_MAX_DETECTABILITY_ENTRENCHED_FALSE_RATE,
    require_input_manifests: bool = False,
    input_manifest_recursive: bool = True,
    manifest_fingerprint_workers: int = 1,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a release verdict from verifier and abstention stability reports."""
    context = ArtifactVerificationContext() if require_input_manifests else None
    verifier_path = Path(verifier_stability_report_path)
    abstention_path = Path(abstention_stability_report_path)
    detectability_paths = tuple(Path(path) for path in detectability_taxonomy_report_paths)
    frontier_workflow_paths = tuple(Path(path) for path in frontier_workflow_report_paths)
    verifier = _load_json_object(verifier_path)
    abstention = _load_json_object(abstention_path)
    detectability_reports = tuple((path, _load_json_object(path)) for path in detectability_paths)
    frontier_workflow_reports = tuple(
        (path, _load_json_object(path)) for path in frontier_workflow_paths
    )
    citation_batch_rollup_paths = tuple(Path(path) for path in citation_batch_rollup_report_paths)
    citation_batch_rollup_reports = tuple(
        (path, _load_json_object(path)) for path in citation_batch_rollup_paths
    )
    frontier_rerun_rollup_paths = tuple(Path(path) for path in frontier_rerun_rollup_report_paths)
    frontier_rerun_rollup_reports = tuple(
        (path, _load_json_object(path)) for path in frontier_rerun_rollup_paths
    )
    config = {
        "max_verified_false_alarm_mean": _unit_float(
            max_verified_false_alarm_mean,
            name="max_verified_false_alarm_mean",
        ),
        "min_verified_detection_mean": _unit_float(
            min_verified_detection_mean,
            name="min_verified_detection_mean",
        ),
        "min_verifier_delta_detection_mean": _finite_float(
            min_verifier_delta_detection_mean,
            name="min_verifier_delta_detection_mean",
        ),
        "min_verifier_pass_seed_rate": _unit_float(
            min_verifier_pass_seed_rate,
            name="min_verifier_pass_seed_rate",
        ),
        "min_verifier_beats_internal_seed_rate": _unit_float(
            min_verifier_beats_internal_seed_rate,
            name="min_verifier_beats_internal_seed_rate",
        ),
        "min_abstention_pass_seed_rate": _unit_float(
            min_abstention_pass_seed_rate,
            name="min_abstention_pass_seed_rate",
        ),
        "min_abstention_conditional_correctness_lower_bound_mean": _unit_float(
            min_abstention_conditional_correctness_lower_bound_mean,
            name="min_abstention_conditional_correctness_lower_bound_mean",
        ),
        "max_abstention_rate_mean": _unit_float(
            max_abstention_rate_mean,
            name="max_abstention_rate_mean",
        ),
        "max_detectability_entrenched_false_rate": _unit_float(
            max_detectability_entrenched_false_rate,
            name="max_detectability_entrenched_false_rate",
        ),
        "require_input_manifests": bool(require_input_manifests),
        "input_manifest_recursive": bool(input_manifest_recursive),
    }
    verifier_runs = _runs_by_name(verifier)
    abstention_runs = _runs_by_name(abstention)
    detectability_runs = _detectability_runs_by_name(detectability_reports)
    verifier_input = _input_summary(
        verifier,
        path=verifier_path,
        expected_workflow="verifier_stability",
        require_manifest=require_input_manifests,
        context=context,
        recursive=input_manifest_recursive,
        max_workers=manifest_fingerprint_workers,
    )
    abstention_input = _input_summary(
        abstention,
        path=abstention_path,
        expected_workflow="abstention_stability",
        require_manifest=require_input_manifests,
        context=context,
        recursive=input_manifest_recursive,
        max_workers=manifest_fingerprint_workers,
    )
    detectability_inputs = tuple(
        _input_summary(
            report,
            path=path,
            expected_workflow="detectability_taxonomy",
            require_manifest=require_input_manifests,
            context=context,
            recursive=input_manifest_recursive,
            max_workers=manifest_fingerprint_workers,
        )
        for path, report in detectability_reports
    )
    frontier_workflow_inputs = tuple(
        _input_summary(
            report,
            path=path,
            expected_workflow="truthfulqa_frontier_workflow",
            require_manifest=require_input_manifests,
            context=context,
            recursive=input_manifest_recursive,
            max_workers=manifest_fingerprint_workers,
        )
        for path, report in frontier_workflow_reports
    )
    citation_batch_rollup_inputs = tuple(
        _input_summary(
            report,
            path=path,
            expected_workflow="citation_search_batch_evidence_rollup",
            expected_statuses=("promote",),
            require_manifest=require_input_manifests,
            context=context,
            recursive=input_manifest_recursive,
            max_workers=manifest_fingerprint_workers,
        )
        for path, report in citation_batch_rollup_reports
    )
    frontier_rerun_rollup_inputs = tuple(
        _frontier_rerun_rollup_input_summary(
            path=path,
            report=report,
            require_manifest=require_input_manifests,
            context=context,
            recursive=input_manifest_recursive,
            max_workers=manifest_fingerprint_workers,
        )
        for path, report in frontier_rerun_rollup_reports
    )
    input_manifest_summary = _input_manifest_evidence_summary(
        (
            verifier_input,
            abstention_input,
            *detectability_inputs,
            *frontier_workflow_inputs,
            *citation_batch_rollup_inputs,
            *frontier_rerun_rollup_inputs,
        )
    )

    input_blocking_reasons = tuple(verifier_input["blocking_reasons"]) + tuple(
        abstention_input["blocking_reasons"]
    ) + tuple(
        reason
        for item in detectability_inputs
        for reason in item["blocking_reasons"]
    ) + tuple(
        reason
        for item in frontier_workflow_inputs
        for reason in item["blocking_reasons"]
    ) + tuple(
        reason
        for item in citation_batch_rollup_inputs
        for reason in item["blocking_reasons"]
    ) + tuple(
        reason
        for item in frontier_rerun_rollup_inputs
        for reason in item["blocking_reasons"]
    )
    run_decisions = []
    run_names = sorted(set(verifier_runs) | set(abstention_runs) | set(detectability_runs))
    if not run_names:
        input_blocking_reasons = input_blocking_reasons + ("no run evidence found",)
    for name in run_names:
        verifier_run = verifier_runs.get(name)
        abstention_run = abstention_runs.get(name)
        detectability_run = detectability_runs.get(name)
        run_decisions.append(
            _run_decision(
                name=name,
                verifier_run=verifier_run,
                abstention_run=abstention_run,
                detectability_run=detectability_run,
                detectability_required=bool(detectability_reports),
                config=config,
            )
        )

    base_verifier_track_status = _track_status(run_decisions, "verifier_decision")
    base_abstention_track_status = _track_status(run_decisions, "abstention_decision")
    base_detectability_track_status = (
        _track_status(run_decisions, "detectability_decision")
        if detectability_reports
        else "not_required"
    )
    multiple_testing_decisions = _frontier_workflow_multiple_testing_decisions(
        frontier_workflow_reports
    )
    base_multiple_testing_track_status = _multiple_testing_track_status(
        multiple_testing_decisions
    )
    citation_batch_decisions = _citation_batch_rollup_decisions(
        citation_batch_rollup_reports
    )
    citation_batch_track_status = _citation_batch_track_status(citation_batch_decisions)
    frontier_rerun_rollup_decisions = _frontier_rerun_rollup_decisions(
        frontier_rerun_rollup_reports
    )
    frontier_rerun_rollup_track_status = _frontier_rerun_rollup_track_status(
        frontier_rerun_rollup_decisions
    )
    frontier_rerun_rollup_promoted_tracks = _frontier_rerun_rollup_promoted_tracks(
        frontier_rerun_rollup_decisions
    )
    verifier_track_status = _effective_track_status(
        base_verifier_track_status,
        frontier_rerun_rollup_promoted_tracks,
        "verifier",
    )
    abstention_track_status = _effective_track_status(
        base_abstention_track_status,
        frontier_rerun_rollup_promoted_tracks,
        "abstention",
    )
    detectability_track_status = _effective_track_status(
        base_detectability_track_status,
        frontier_rerun_rollup_promoted_tracks,
        "detectability",
    )
    multiple_testing_track_status = _effective_track_status(
        base_multiple_testing_track_status,
        frontier_rerun_rollup_promoted_tracks,
        "multiple_testing",
    )
    blocking_reasons = list(input_blocking_reasons)
    for decision in run_decisions:
        _extend_direct_track_blocking(
            blocking_reasons,
            decision,
            key="verifier_decision",
            track="verifier",
            promoted_tracks=frontier_rerun_rollup_promoted_tracks,
        )
        _extend_direct_track_blocking(
            blocking_reasons,
            decision,
            key="abstention_decision",
            track="abstention",
            promoted_tracks=frontier_rerun_rollup_promoted_tracks,
        )
        _extend_direct_track_blocking(
            blocking_reasons,
            decision,
            key="detectability_decision",
            track="detectability",
            promoted_tracks=frontier_rerun_rollup_promoted_tracks,
        )
    for decision in multiple_testing_decisions:
        if "multiple_testing" not in frontier_rerun_rollup_promoted_tracks:
            blocking_reasons.extend(decision["blocking_reasons"])
    for decision in citation_batch_decisions:
        blocking_reasons.extend(decision["blocking_reasons"])
    for decision in frontier_rerun_rollup_decisions:
        blocking_reasons.extend(decision["blocking_reasons"])
    status = (
        "promote"
        if not blocking_reasons
        and verifier_track_status == "promote"
        and abstention_track_status == "promote"
        and detectability_track_status in {"promote", "not_required"}
        and multiple_testing_track_status in {"promote", "not_required"}
        and citation_batch_track_status in {"promote", "not_required"}
        and frontier_rerun_rollup_track_status in {"promote", "not_required"}
        else "blocked"
    )
    return {
        "schema_version": 1,
        "workflow": "frontier_release_evidence_comparison",
        "status": "complete",
        "config": config,
        "inputs": {
            "verifier_stability_report": verifier_input,
            "abstention_stability_report": abstention_input,
            "detectability_taxonomy_reports": detectability_inputs,
            "frontier_workflow_reports": frontier_workflow_inputs,
            "citation_batch_rollup_reports": citation_batch_rollup_inputs,
            "frontier_rerun_rollup_reports": frontier_rerun_rollup_inputs,
        },
        "evidence_summary": {
            "run_count": len(run_decisions),
            "run_names": run_names,
            "verifier_track_status": verifier_track_status,
            "abstention_track_status": abstention_track_status,
            "detectability_track_status": detectability_track_status,
            "multiple_testing_track_status": multiple_testing_track_status,
            "detectability_report_count": len(detectability_reports),
            "frontier_workflow_report_count": len(frontier_workflow_reports),
            "citation_batch_rollup_report_count": len(citation_batch_rollup_reports),
            "frontier_rerun_rollup_report_count": len(frontier_rerun_rollup_reports),
            **input_manifest_summary,
            **_multiple_testing_evidence_summary(multiple_testing_decisions),
            **_citation_batch_evidence_summary(citation_batch_decisions),
            **_frontier_rerun_rollup_evidence_summary(frontier_rerun_rollup_decisions),
            "base_verifier_track_status": base_verifier_track_status,
            "base_abstention_track_status": base_abstention_track_status,
            "base_detectability_track_status": base_detectability_track_status,
            "base_multiple_testing_track_status": base_multiple_testing_track_status,
            "frontier_rerun_rollup_promoted_tracks": tuple(sorted(
                frontier_rerun_rollup_promoted_tracks
            )),
            "verifier_signal": verifier.get("config", {}).get("signal")
            if isinstance(verifier.get("config"), Mapping)
            else None,
            "abstention_signals": list(abstention.get("config", {}).get("signals", ()))
            if isinstance(abstention.get("config"), Mapping)
            else [],
        },
        "run_decisions": run_decisions,
        "multiple_testing_decisions": multiple_testing_decisions,
        "citation_batch_decisions": citation_batch_decisions,
        "frontier_rerun_rollup_decisions": frontier_rerun_rollup_decisions,
        "decision": {
            "status": status,
            "verifier_track_status": verifier_track_status,
            "abstention_track_status": abstention_track_status,
            "detectability_track_status": detectability_track_status,
            "multiple_testing_track_status": multiple_testing_track_status,
            "citation_batch_track_status": citation_batch_track_status,
            "frontier_rerun_rollup_track_status": frontier_rerun_rollup_track_status,
            "base_verifier_track_status": base_verifier_track_status,
            "base_abstention_track_status": base_abstention_track_status,
            "base_detectability_track_status": base_detectability_track_status,
            "base_multiple_testing_track_status": base_multiple_testing_track_status,
            "frontier_rerun_rollup_promoted_tracks": tuple(sorted(
                frontier_rerun_rollup_promoted_tracks
            )),
            "blocking_reasons": tuple(blocking_reasons),
        },
        "notes": tuple(str(note) for note in notes),
    }


def _run_decision(
    *,
    name: str,
    verifier_run: Mapping[str, Any] | None,
    abstention_run: Mapping[str, Any] | None,
    detectability_run: Mapping[str, Any] | None,
    detectability_required: bool,
    config: Mapping[str, float],
) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    if verifier_run is None:
        verifier_decision = _missing_track_decision("verifier_stability", name)
    else:
        verifier_decision = _verifier_run_decision(name, verifier_run, config)
    if abstention_run is None:
        abstention_decision = _missing_track_decision("abstention_stability", name)
    else:
        abstention_decision = _abstention_run_decision(name, abstention_run, config)
    if not detectability_required:
        detectability_decision = _not_required_track_decision("detectability_taxonomy", name)
    elif detectability_run is None:
        detectability_decision = _missing_track_decision("detectability_taxonomy", name)
    else:
        detectability_decision = _detectability_run_decision(name, detectability_run, config)
    blocking_reasons.extend(verifier_decision["blocking_reasons"])
    blocking_reasons.extend(abstention_decision["blocking_reasons"])
    blocking_reasons.extend(detectability_decision["blocking_reasons"])
    status = (
        "promote"
        if verifier_decision["status"] == "promote"
        and abstention_decision["status"] == "promote"
        and detectability_decision["status"] in {"promote", "not_required"}
        else "blocked"
    )
    return {
        "name": name,
        "status": status,
        "verifier_decision": verifier_decision,
        "abstention_decision": abstention_decision,
        "detectability_decision": detectability_decision,
        "blocking_reasons": tuple(blocking_reasons),
    }


def _verifier_run_decision(
    name: str,
    run: Mapping[str, Any],
    config: Mapping[str, float],
) -> dict[str, Any]:
    stability = _mapping(run.get("stability"))
    seed_count = _positive_int(stability.get("seed_count"))
    metrics = {
        "verified_false_alarm_mean": _stat_mean(stability, "verified_false_alarm"),
        "verified_detection_mean": _stat_mean(stability, "verified_detection"),
        "delta_detection_mean": _stat_mean(stability, "delta_detection"),
        "verified_pass_seed_count": _non_negative_int(stability.get("verified_pass_seed_count")),
        "verified_beats_internal_detection_seed_count": _non_negative_int(
            stability.get("verified_beats_internal_detection_seed_count")
        ),
    }
    checks: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    if seed_count is None:
        blocking_reasons.append(f"verifier_stability.{name} missing valid seed_count")
    checks.extend(
        (
            _max_check(
                f"verifier_stability.{name}.verified_false_alarm_mean",
                metrics["verified_false_alarm_mean"],
                config["max_verified_false_alarm_mean"],
            ),
            _min_check(
                f"verifier_stability.{name}.verified_detection_mean",
                metrics["verified_detection_mean"],
                config["min_verified_detection_mean"],
            ),
            _min_check(
                f"verifier_stability.{name}.delta_detection_mean",
                metrics["delta_detection_mean"],
                config["min_verifier_delta_detection_mean"],
            ),
        )
    )
    if seed_count is not None:
        pass_rate = _ratio(metrics["verified_pass_seed_count"], seed_count)
        beats_rate = _ratio(metrics["verified_beats_internal_detection_seed_count"], seed_count)
        metrics["verified_pass_seed_rate"] = pass_rate
        metrics["verified_beats_internal_detection_seed_rate"] = beats_rate
        checks.extend(
            (
                _min_check(
                    f"verifier_stability.{name}.verified_pass_seed_rate",
                    pass_rate,
                    config["min_verifier_pass_seed_rate"],
                ),
                _min_check(
                    f"verifier_stability.{name}.verified_beats_internal_detection_seed_rate",
                    beats_rate,
                    config["min_verifier_beats_internal_seed_rate"],
                ),
            )
        )
    blocking_reasons.extend(_failed_reasons(checks))
    return _track_decision("verifier_stability", name, metrics, checks, blocking_reasons)


def _abstention_run_decision(
    name: str,
    run: Mapping[str, Any],
    config: Mapping[str, float],
) -> dict[str, Any]:
    stability = _mapping(run.get("stability"))
    feasibility = _mapping(run.get("supervised_feasibility_frontier"))
    feasible_best = _mapping(feasibility.get("best"))
    seed_count = _positive_int(stability.get("seed_count"))
    metrics = {
        "conditional_correctness_lower_bound_mean": _stat_mean(
            stability,
            "conditional_correctness_lower_bound",
        ),
        "empirical_abstention_rate_mean": _stat_mean(stability, "empirical_abstention_rate"),
        "release_gate_pass_seed_count": _non_negative_int(
            stability.get("release_gate_pass_seed_count")
        ),
        "release_gate_block_seed_count": _non_negative_int(
            stability.get("release_gate_block_seed_count")
        ),
        "stable_recommended_score_name": stability.get("stable_recommended_score_name"),
        "recommended_score_name_counts": _mapping(stability.get("recommended_score_name_counts")),
        "supervised_feasibility_target_passed": feasibility.get("target_passed"),
        "supervised_feasibility_score_name": feasible_best.get("score_name"),
        "supervised_feasibility_conditional_correctness_lower_bound": feasible_best.get(
            "conditional_correctness_lower_bound"
        ),
        "supervised_feasibility_empirical_selective_accuracy": feasible_best.get(
            "empirical_selective_accuracy"
        ),
        "supervised_feasibility_empirical_abstention_rate": feasible_best.get(
            "empirical_abstention_rate"
        ),
    }
    checks: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    if seed_count is None:
        blocking_reasons.append(f"abstention_stability.{name} missing valid seed_count")
    checks.extend(
        (
            _min_check(
                f"abstention_stability.{name}.conditional_correctness_lower_bound_mean",
                metrics["conditional_correctness_lower_bound_mean"],
                config["min_abstention_conditional_correctness_lower_bound_mean"],
            ),
            _max_check(
                f"abstention_stability.{name}.empirical_abstention_rate_mean",
                metrics["empirical_abstention_rate_mean"],
                config["max_abstention_rate_mean"],
            ),
        )
    )
    if seed_count is not None:
        pass_rate = _ratio(metrics["release_gate_pass_seed_count"], seed_count)
        metrics["release_gate_pass_seed_rate"] = pass_rate
        checks.append(
            _min_check(
                f"abstention_stability.{name}.release_gate_pass_seed_rate",
                pass_rate,
                config["min_abstention_pass_seed_rate"],
            )
        )
    blocking_reasons.extend(_failed_reasons(checks))
    return _track_decision("abstention_stability", name, metrics, checks, blocking_reasons)


def _detectability_run_decision(
    name: str,
    run: Mapping[str, Any],
    config: Mapping[str, float],
) -> dict[str, Any]:
    report = _mapping(run.get("report"))
    false_distribution = _mapping(report.get("false_distribution"))
    entrenched = _mapping(false_distribution.get("entrenched"))
    blind_spot = _mapping(report.get("blind_spot"))
    run_config = _mapping(run.get("config"))
    metrics = {
        "entrenched_false_rate": _finite_float_or_none(entrenched.get("rate")),
        "entrenched_false_count": _non_negative_int(entrenched.get("count")),
        "blind_spot_false_count": _non_negative_int(blind_spot.get("n_false")),
        "n_false": _non_negative_int(report.get("n_false")),
        "n_total": _non_negative_int(report.get("n_total")),
        "consistency_signal": run_config.get("consistency_signal"),
        "confidence_signal": run_config.get("confidence_signal"),
    }
    checks = [
        _max_check(
            f"detectability_taxonomy.{name}.entrenched_false_rate",
            metrics["entrenched_false_rate"],
            config["max_detectability_entrenched_false_rate"],
        )
    ]
    blocking_reasons = list(_failed_reasons(checks))
    return _track_decision("detectability_taxonomy", name, metrics, checks, blocking_reasons)


def _frontier_workflow_multiple_testing_decisions(
    reports: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _frontier_workflow_multiple_testing_decision(path=path, report=report)
        for path, report in reports
    )


def _frontier_workflow_multiple_testing_decision(
    *,
    path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    name = _frontier_workflow_report_name(path, report)
    gate = _mapping(report.get("multiple_testing_gate"))
    cell_count = _non_negative_int(gate.get("cell_count"))
    pass_count = _non_negative_int(gate.get("pass_count"))
    fail_count = _non_negative_int(gate.get("fail_count"))
    unknown_count = _non_negative_int(gate.get("unknown_count"))
    cell_summaries = _multiple_testing_cell_summaries(gate.get("cells", ()))
    failed_cells = tuple(cell for cell in cell_summaries if cell["status"] == "failed")
    unknown_cells = tuple(cell for cell in cell_summaries if cell["status"] == "unknown")
    passed_cells = tuple(cell for cell in cell_summaries if cell["status"] == "passed")
    missing_artifact_cells = tuple(
        cell["cell"]
        for cell in cell_summaries
        if not cell.get("report") or not cell.get("calibration")
    )
    metrics = {
        "enabled": gate.get("enabled"),
        "all_pass": gate.get("all_pass"),
        "signals": tuple(gate.get("signals", ()))
        if isinstance(gate.get("signals"), Sequence) and not isinstance(gate.get("signals"), (str, bytes))
        else (),
        "alpha": _finite_float_or_none(gate.get("alpha")),
        "method": gate.get("method"),
        "cell_count": cell_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "unknown_count": unknown_count,
        "failed_cells": failed_cells,
        "unknown_cells": unknown_cells,
        "blocked_cells": failed_cells + unknown_cells,
    }
    blocking_reasons: list[str] = []
    if not gate:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate missing"
        )
    if gate.get("enabled") is not True:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.enabled is not true"
        )
    if gate.get("all_pass") is not True:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.all_pass is not true"
        )
    if cell_count is None or cell_count < 1:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.cell_count is missing or zero"
        )
    elif len(cell_summaries) != cell_count:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.cells length "
            f"{len(cell_summaries)} does not match cell_count {cell_count}"
        )
    if pass_count is None:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.pass_count is missing"
        )
    elif cell_count is not None and pass_count != len(passed_cells):
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.pass_count "
            f"{pass_count} does not match passed cell list count {len(passed_cells)}"
        )
    if fail_count is None:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.fail_count is missing"
        )
    elif fail_count != len(failed_cells):
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.fail_count "
            f"{fail_count} does not match failed cell list count {len(failed_cells)}"
        )
    elif fail_count > 0:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.fail_count {fail_count} is non-zero"
        )
    if unknown_count is None:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.unknown_count is missing"
        )
    elif unknown_count != len(unknown_cells):
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.unknown_count "
            f"{unknown_count} does not match unknown cell list count {len(unknown_cells)}"
        )
    elif unknown_count > 0:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.unknown_count {unknown_count} is non-zero"
        )
    if (
        cell_count is not None
        and pass_count is not None
        and fail_count is not None
        and unknown_count is not None
        and pass_count + fail_count + unknown_count != cell_count
    ):
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate pass/fail/unknown "
            f"counts do not sum to cell_count {cell_count}"
        )
    if missing_artifact_cells:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.cells missing "
            f"report or calibration artifact: {', '.join(missing_artifact_cells)}"
        )
    if failed_cells:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.failed_cells: "
            f"{', '.join(cell['cell'] for cell in failed_cells)}"
        )
    if unknown_cells:
        blocking_reasons.append(
            f"truthfulqa_frontier_workflow.{name}.multiple_testing_gate.unknown_cells: "
            f"{', '.join(cell['cell'] for cell in unknown_cells)}"
        )
    return _track_decision(
        "truthfulqa_frontier_workflow.multiple_testing_gate",
        name,
        metrics,
        (),
        blocking_reasons,
    )


def _multiple_testing_cell_summaries(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    cells: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
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
            "false_alarm": _finite_float_or_none(item.get("false_alarm")),
            "detection": _finite_float_or_none(item.get("detection")),
            "report": item.get("report"),
            "calibration": item.get("calibration"),
        })
    return tuple(cells)


def _frontier_workflow_report_name(path: Path, report: Mapping[str, Any]) -> str:
    for key in ("name", "run_name"):
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = _mapping(report.get("metadata"))
    for key in ("name", "run_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return path.stem


def _multiple_testing_track_status(decisions: Sequence[Mapping[str, Any]]) -> str:
    if not decisions:
        return "not_required"
    statuses = {str(decision.get("status")) for decision in decisions}
    return "promote" if statuses == {"promote"} else "blocked"


def _multiple_testing_evidence_summary(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cell_count = 0
    pass_count = 0
    fail_count = 0
    unknown_count = 0
    failed_cells: list[dict[str, Any]] = []
    unknown_cells: list[dict[str, Any]] = []
    signals: set[str] = set()
    run_names = []
    for decision in decisions:
        run_name = decision.get("name")
        run_names.append(run_name)
        metrics = _mapping(decision.get("metrics"))
        cell_count += _non_negative_int(metrics.get("cell_count")) or 0
        pass_count += _non_negative_int(metrics.get("pass_count")) or 0
        fail_count += _non_negative_int(metrics.get("fail_count")) or 0
        unknown_count += _non_negative_int(metrics.get("unknown_count")) or 0
        failed_cells.extend(_annotated_multiple_testing_cells(metrics.get("failed_cells"), run_name=run_name))
        unknown_cells.extend(_annotated_multiple_testing_cells(metrics.get("unknown_cells"), run_name=run_name))
        for signal in metrics.get("signals", ()):
            if isinstance(signal, str):
                signals.add(signal)
    return {
        "multiple_testing_frontier_workflow_names": tuple(
            str(name) for name in run_names if name is not None
        ),
        "multiple_testing_signals": tuple(sorted(signals)),
        "multiple_testing_cell_count": cell_count,
        "multiple_testing_pass_count": pass_count,
        "multiple_testing_fail_count": fail_count,
        "multiple_testing_unknown_count": unknown_count,
        "multiple_testing_failed_cells": tuple(failed_cells),
        "multiple_testing_unknown_cells": tuple(unknown_cells),
        "multiple_testing_blocked_cells": tuple(failed_cells + unknown_cells),
    }


def _annotated_multiple_testing_cells(value: Any, *, run_name: Any) -> tuple[dict[str, Any], ...]:
    cells = []
    for item in _mapping_sequence(value):
        cells.append({
            "run": None if run_name is None else str(run_name),
            "cell": str(item.get("cell") or ""),
            "status": str(item.get("status") or "unknown"),
            "false_alarm": _finite_float_or_none(item.get("false_alarm")),
            "detection": _finite_float_or_none(item.get("detection")),
            "report": item.get("report"),
            "calibration": item.get("calibration"),
        })
    return tuple(cells)


def _citation_batch_rollup_decisions(
    reports: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _citation_batch_rollup_decision(path=path, report=report)
        for path, report in reports
    )


def _citation_batch_rollup_decision(
    *,
    path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    name = _citation_batch_rollup_report_name(path, report)
    summary = _mapping(report.get("summary"))
    gate = _mapping(report.get("gate"))
    paths = _mapping(report.get("paths"))
    metrics = {
        "workflow": report.get("workflow"),
        "status": report.get("status"),
        "gate_passed": gate.get("passed"),
        "promotion_ready": gate.get("promotion_ready"),
        "artifact_manifest": paths.get("artifact_manifest"),
        "report_count": _non_negative_int(summary.get("report_count")),
        "expected_batch_count": _non_negative_int(summary.get("expected_batch_count")),
        "observed_batch_count": _non_negative_int(summary.get("observed_batch_count")),
        "missing_expected_batch_count": _non_negative_int(
            summary.get("missing_expected_batch_count")
        ),
        "unexpected_batch_count": _non_negative_int(summary.get("unexpected_batch_count")),
        "duplicate_batch_count": _non_negative_int(summary.get("duplicate_batch_count")),
        "blocked_report_count": _non_negative_int(summary.get("blocked_report_count")),
        "unsupported_workflow_count": _non_negative_int(
            summary.get("unsupported_workflow_count")
        ),
        "child_manifest_failed_count": _non_negative_int(
            summary.get("child_manifest_failed_count")
        ),
        "child_manifest_missing_count": _non_negative_int(
            summary.get("child_manifest_missing_count")
        ),
        "adapter_request_count": _non_negative_int(summary.get("adapter_request_count")),
        "adapter_result_count": _non_negative_int(summary.get("adapter_result_count")),
        "source_document_count": _non_negative_int(summary.get("source_document_count")),
        "corpus_document_count": _non_negative_int(summary.get("corpus_document_count")),
        "expected_batch_ids": _string_tuple(summary.get("expected_batch_ids")),
        "observed_batch_ids": _string_tuple(summary.get("observed_batch_ids")),
        "missing_expected_batch_ids": _string_tuple(
            summary.get("missing_expected_batch_ids")
        ),
        "unexpected_batch_ids": _string_tuple(summary.get("unexpected_batch_ids")),
        "duplicate_batch_ids": _string_tuple(summary.get("duplicate_batch_ids")),
    }
    blocking_reasons: list[str] = []
    if report.get("workflow") != "citation_search_batch_evidence_rollup":
        blocking_reasons.append(
            f"citation_batch_rollup.{name}.workflow is "
            f"{report.get('workflow')!r}, expected 'citation_search_batch_evidence_rollup'"
        )
    if report.get("status") != "promote":
        blocking_reasons.append(
            f"citation_batch_rollup.{name}.status is {report.get('status')!r}, "
            "expected 'promote'"
        )
    if gate.get("passed") is not True:
        blocking_reasons.append(
            f"citation_batch_rollup.{name}.gate.passed is not true"
        )
    if gate.get("promotion_ready") is not True:
        blocking_reasons.append(
            f"citation_batch_rollup.{name}.gate.promotion_ready is not true"
        )
    if not paths.get("artifact_manifest"):
        blocking_reasons.append(
            f"citation_batch_rollup.{name}.paths.artifact_manifest is missing"
        )
    report_count = metrics["report_count"]
    if report_count is None or report_count < 1:
        blocking_reasons.append(
            f"citation_batch_rollup.{name}.summary.report_count is missing or zero"
        )
    for key in (
        "missing_expected_batch_count",
        "unexpected_batch_count",
        "duplicate_batch_count",
        "blocked_report_count",
        "unsupported_workflow_count",
        "child_manifest_failed_count",
    ):
        count = metrics[key]
        if count is None:
            blocking_reasons.append(
                f"citation_batch_rollup.{name}.summary.{key} is missing"
            )
        elif count > 0:
            blocking_reasons.append(
                f"citation_batch_rollup.{name}.summary.{key} {count} is non-zero"
            )
    for reason in _citation_batch_rollup_gate_reasons(gate.get("blocking_reasons")):
        blocking_reasons.append(f"citation_batch_rollup.{name}.{reason}")
    return _track_decision(
        "citation_search_batch_evidence_rollup",
        name,
        metrics,
        (),
        tuple(dict.fromkeys(blocking_reasons)),
    )


def _citation_batch_rollup_report_name(path: Path, report: Mapping[str, Any]) -> str:
    metadata = _mapping(report.get("metadata"))
    for payload in (report, metadata):
        for key in ("name", "run_name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return path.stem


def _citation_batch_rollup_gate_reasons(value: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    for item in value:
        if isinstance(item, Mapping):
            gate = item.get("gate")
            reason = item.get("reason")
            batch_id = item.get("batch_id")
            path = item.get("path")
            parts = []
            if gate:
                parts.append(f"gate={gate}")
            if batch_id:
                parts.append(f"batch_id={batch_id}")
            if path:
                parts.append(f"path={path}")
            if reason:
                parts.append(str(reason))
            if parts:
                reasons.append(" ".join(parts))
        elif item is not None:
            reasons.append(str(item))
    return tuple(reasons)


def _citation_batch_track_status(decisions: Sequence[Mapping[str, Any]]) -> str:
    if not decisions:
        return "not_required"
    statuses = {str(decision.get("status")) for decision in decisions}
    return "promote" if statuses == {"promote"} else "blocked"


def _citation_batch_evidence_summary(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rollup_names = []
    blocked_rollups = []
    expected_batch_ids: set[str] = set()
    observed_batch_ids: set[str] = set()
    missing_batch_rows: list[dict[str, str]] = []
    duplicate_batch_rows: list[dict[str, str]] = []
    unexpected_batch_rows: list[dict[str, str]] = []
    totals = {
        "citation_batch_child_report_count": 0,
        "citation_batch_expected_batch_count": 0,
        "citation_batch_observed_batch_count": 0,
        "citation_batch_missing_expected_batch_count": 0,
        "citation_batch_unexpected_batch_count": 0,
        "citation_batch_duplicate_batch_count": 0,
        "citation_batch_blocked_child_report_count": 0,
        "citation_batch_child_manifest_failed_count": 0,
        "citation_batch_adapter_request_count": 0,
        "citation_batch_adapter_result_count": 0,
        "citation_batch_source_document_count": 0,
        "citation_batch_corpus_document_count": 0,
    }
    for decision in decisions:
        name = str(decision.get("name") or "")
        rollup_names.append(name)
        if decision.get("status") != "promote":
            blocked_rollups.append(name)
        metrics = _mapping(decision.get("metrics"))
        for key, metric_key in (
            ("citation_batch_child_report_count", "report_count"),
            ("citation_batch_expected_batch_count", "expected_batch_count"),
            ("citation_batch_observed_batch_count", "observed_batch_count"),
            ("citation_batch_missing_expected_batch_count", "missing_expected_batch_count"),
            ("citation_batch_unexpected_batch_count", "unexpected_batch_count"),
            ("citation_batch_duplicate_batch_count", "duplicate_batch_count"),
            ("citation_batch_blocked_child_report_count", "blocked_report_count"),
            ("citation_batch_child_manifest_failed_count", "child_manifest_failed_count"),
            ("citation_batch_adapter_request_count", "adapter_request_count"),
            ("citation_batch_adapter_result_count", "adapter_result_count"),
            ("citation_batch_source_document_count", "source_document_count"),
            ("citation_batch_corpus_document_count", "corpus_document_count"),
        ):
            totals[key] += _non_negative_int(metrics.get(metric_key)) or 0
        expected_batch_ids.update(_string_tuple(metrics.get("expected_batch_ids")))
        observed_batch_ids.update(_string_tuple(metrics.get("observed_batch_ids")))
        for batch_id in _string_tuple(metrics.get("missing_expected_batch_ids")):
            missing_batch_rows.append({"rollup": name, "batch_id": batch_id})
        for batch_id in _string_tuple(metrics.get("duplicate_batch_ids")):
            duplicate_batch_rows.append({"rollup": name, "batch_id": batch_id})
        for batch_id in _string_tuple(metrics.get("unexpected_batch_ids")):
            unexpected_batch_rows.append({"rollup": name, "batch_id": batch_id})
    return {
        "citation_batch_rollup_names": tuple(name for name in rollup_names if name),
        "citation_batch_blocked_rollups": tuple(name for name in blocked_rollups if name),
        "citation_batch_rollup_count": len(decisions),
        "citation_batch_promotion_ready_count": sum(
            1 for decision in decisions if decision.get("status") == "promote"
        ),
        "citation_batch_expected_batch_ids": tuple(sorted(expected_batch_ids)),
        "citation_batch_observed_batch_ids": tuple(sorted(observed_batch_ids)),
        "citation_batch_missing_expected_batches": tuple(missing_batch_rows),
        "citation_batch_duplicate_batches": tuple(duplicate_batch_rows),
        "citation_batch_unexpected_batches": tuple(unexpected_batch_rows),
        **totals,
    }


def _frontier_rerun_rollup_decisions(
    reports: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _frontier_rerun_rollup_decision(path=path, report=report)
        for path, report in reports
    )


def _frontier_rerun_rollup_decision(
    *,
    path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    name = _frontier_rerun_rollup_report_name(path, report)
    workflow = report.get("workflow")
    track = FRONTIER_RERUN_ROLLUP_WORKFLOWS.get(str(workflow))
    gate = _mapping(report.get("gate"))
    summary = _mapping(report.get("summary"))
    paths = _mapping(report.get("paths"))
    candidate_count = _non_negative_int(summary.get("candidate_count"))
    if candidate_count is None:
        candidate_count = len(_mapping_sequence(report.get("candidates", ())))
    metrics = {
        "workflow": workflow,
        "track": track,
        "status": report.get("status"),
        "gate_passed": gate.get("passed"),
        "promotion_ready": gate.get("promotion_ready"),
        "audit_ready": gate.get("audit_ready"),
        "artifact_manifest": paths.get("artifact_manifest"),
        "candidate_count": candidate_count,
        "observed_report_count": _non_negative_int(summary.get("observed_report_count")),
        "missing_report_count": _non_negative_int(summary.get("missing_report_count")),
        "invalid_report_count": _non_negative_int(summary.get("invalid_report_count")),
        "blocked_candidate_count": _non_negative_int(summary.get("blocked_candidate_count")),
        "promotion_ready_count": _non_negative_int(summary.get("promotion_ready_count")),
        "passing_candidate_count": _non_negative_int(summary.get("passing_candidate_count")),
        "audit_ready_count": _non_negative_int(summary.get("audit_ready_count")),
        "tracks": _string_tuple(summary.get("tracks")),
        "track_statuses": _mapping(summary.get("track_statuses")),
        "runs": _string_tuple(summary.get("runs")),
        "cells": _string_tuple(summary.get("cells")),
    }
    blocking_reasons: list[str] = []
    if track is None:
        blocking_reasons.append(
            f"frontier_rerun_rollup.{name}.workflow {workflow!r} is unsupported"
        )
    if report.get("status") != "promote":
        blocking_reasons.append(
            f"frontier_rerun_rollup.{name}.status is {report.get('status')!r}, expected 'promote'"
        )
    if gate.get("passed") is not True:
        blocking_reasons.append(f"frontier_rerun_rollup.{name}.gate.passed is not true")
    if gate.get("promotion_ready") is not True:
        blocking_reasons.append(
            f"frontier_rerun_rollup.{name}.gate.promotion_ready is not true"
        )
    if not paths.get("artifact_manifest"):
        blocking_reasons.append(
            f"frontier_rerun_rollup.{name}.paths.artifact_manifest is missing"
        )
    if candidate_count is None or candidate_count < 1:
        blocking_reasons.append(
            f"frontier_rerun_rollup.{name}.summary.candidate_count is missing or zero"
        )
    for reason in _frontier_rerun_rollup_gate_reasons(gate.get("blocking_reasons")):
        blocking_reasons.append(f"frontier_rerun_rollup.{name}.{reason}")
    return _track_decision(
        "frontier_rerun_rollup",
        name,
        metrics,
        (),
        tuple(dict.fromkeys(blocking_reasons)),
    )


def _frontier_rerun_rollup_report_name(path: Path, report: Mapping[str, Any]) -> str:
    metadata = _mapping(report.get("metadata"))
    for payload in (report, metadata):
        for key in ("name", "run_name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return path.stem


def _frontier_rerun_rollup_gate_reasons(value: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    for item in _mapping_sequence(value):
        reason = item.get("reason")
        gate = item.get("gate")
        run = item.get("run")
        cell = item.get("cell")
        track = item.get("track")
        parts = []
        if gate:
            parts.append(f"gate={gate}")
        if track:
            parts.append(f"track={track}")
        if run:
            parts.append(f"run={run}")
        if cell:
            parts.append(f"cell={cell}")
        if reason:
            parts.append(str(reason))
        if parts:
            reasons.append(" ".join(parts))
    return tuple(reasons)


def _frontier_rerun_rollup_track_status(
    decisions: Sequence[Mapping[str, Any]],
) -> str:
    if not decisions:
        return "not_required"
    statuses = {str(decision.get("status")) for decision in decisions}
    return "promote" if statuses == {"promote"} else "blocked"


def _frontier_rerun_rollup_promoted_tracks(
    decisions: Sequence[Mapping[str, Any]],
) -> set[str]:
    promoted: set[str] = set()
    for decision in decisions:
        if decision.get("status") != "promote":
            continue
        metrics = _mapping(decision.get("metrics"))
        rollup_track = metrics.get("track")
        if rollup_track == "stability":
            track_statuses = _mapping(metrics.get("track_statuses"))
            for child_track, status in track_statuses.items():
                if status == "promote":
                    release_track = _release_track_from_rerun_track(child_track)
                    if release_track:
                        promoted.add(release_track)
            if not track_statuses:
                for child_track in _string_tuple(metrics.get("tracks")):
                    release_track = _release_track_from_rerun_track(child_track)
                    if release_track:
                        promoted.add(release_track)
        else:
            release_track = _release_track_from_rerun_track(rollup_track)
            if release_track:
                promoted.add(release_track)
    return promoted


def _release_track_from_rerun_track(value: Any) -> str | None:
    text = str(value or "")
    if text in {"verifier", "verifier_stability"}:
        return "verifier"
    if text in {"abstention", "abstention_stability"}:
        return "abstention"
    if text in {"detectability", "detectability_taxonomy"}:
        return "detectability"
    if text in {"multiple_testing", "truthfulqa_frontier_workflow"}:
        return "multiple_testing"
    return None


def _effective_track_status(
    base_status: str,
    promoted_tracks: set[str],
    track: str,
) -> str:
    return "promote" if track in promoted_tracks else base_status


def _extend_direct_track_blocking(
    blocking_reasons: list[str],
    decision: Mapping[str, Any],
    *,
    key: str,
    track: str,
    promoted_tracks: set[str],
) -> None:
    if track in promoted_tracks:
        return
    blocking_reasons.extend(_mapping(decision.get(key)).get("blocking_reasons", ()))


def _frontier_rerun_rollup_evidence_summary(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = []
    blocked = []
    workflows: set[str] = set()
    tracks: set[str] = set()
    totals = {
        "frontier_rerun_rollup_candidate_count": 0,
        "frontier_rerun_rollup_observed_report_count": 0,
        "frontier_rerun_rollup_missing_report_count": 0,
        "frontier_rerun_rollup_invalid_report_count": 0,
        "frontier_rerun_rollup_blocked_candidate_count": 0,
        "frontier_rerun_rollup_promotion_ready_count": 0,
    }
    for decision in decisions:
        name = str(decision.get("name") or "")
        names.append(name)
        if decision.get("status") != "promote":
            blocked.append(name)
        metrics = _mapping(decision.get("metrics"))
        workflow = metrics.get("workflow")
        track = metrics.get("track")
        if isinstance(workflow, str) and workflow:
            workflows.add(workflow)
        if isinstance(track, str) and track:
            tracks.add(track)
        for key, metric_key in (
            ("frontier_rerun_rollup_candidate_count", "candidate_count"),
            ("frontier_rerun_rollup_observed_report_count", "observed_report_count"),
            ("frontier_rerun_rollup_missing_report_count", "missing_report_count"),
            ("frontier_rerun_rollup_invalid_report_count", "invalid_report_count"),
            ("frontier_rerun_rollup_blocked_candidate_count", "blocked_candidate_count"),
            ("frontier_rerun_rollup_promotion_ready_count", "promotion_ready_count"),
        ):
            totals[key] += _non_negative_int(metrics.get(metric_key)) or 0
    return {
        "frontier_rerun_rollup_names": tuple(name for name in names if name),
        "frontier_rerun_rollup_blocked_names": tuple(name for name in blocked if name),
        "frontier_rerun_rollup_workflows": tuple(sorted(workflows)),
        "frontier_rerun_rollup_tracks": tuple(sorted(tracks)),
        **totals,
    }


def _track_decision(
    track: str,
    name: str,
    metrics: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    blocking_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "track": track,
        "name": name,
        "status": "promote" if not blocking_reasons else "blocked",
        "metrics": dict(metrics),
        "checks": tuple(dict(check) for check in checks),
        "blocking_reasons": tuple(blocking_reasons),
    }


def _missing_track_decision(track: str, name: str) -> dict[str, Any]:
    reason = f"{track}.{name} missing run evidence"
    return {
        "track": track,
        "name": name,
        "status": "blocked",
        "metrics": {},
        "checks": (),
        "blocking_reasons": (reason,),
    }


def _not_required_track_decision(track: str, name: str) -> dict[str, Any]:
    return {
        "track": track,
        "name": name,
        "status": "not_required",
        "metrics": {},
        "checks": (),
        "blocking_reasons": (),
    }


def _input_summary(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_workflow: str,
    expected_statuses: Sequence[str] = ("complete",),
    require_manifest: bool = False,
    context: ArtifactVerificationContext | None = None,
    recursive: bool = True,
    max_workers: int = 1,
) -> dict[str, Any]:
    workflow = payload.get("workflow")
    status = payload.get("status")
    paths = _mapping(payload.get("paths"))
    manifest_value = paths.get("artifact_manifest")
    manifest_verification = _input_manifest_verification_summary(
        path=path,
        manifest_value=manifest_value,
        require_manifest=require_manifest,
        context=context,
        recursive=recursive,
        max_workers=max_workers,
    )
    blocking_reasons = []
    if workflow != expected_workflow:
        blocking_reasons.append(f"{path} workflow {workflow!r} is not {expected_workflow!r}")
    if status not in expected_statuses:
        expected = ", ".join(repr(item) for item in expected_statuses)
        blocking_reasons.append(f"{path} status {status!r} is not one of {expected}")
    blocking_reasons.extend(manifest_verification["blocking_reasons"])
    return {
        "path": str(path),
        "workflow": workflow,
        "status": status,
        "artifact_manifest": manifest_value,
        "artifact_manifest_summary": _mapping(payload.get("artifact_manifest_summary")),
        "artifact_manifest_verification": manifest_verification,
        "blocking_reasons": tuple(blocking_reasons),
    }


def _frontier_rerun_rollup_input_summary(
    *,
    path: Path,
    report: Mapping[str, Any],
    require_manifest: bool = False,
    context: ArtifactVerificationContext | None = None,
    recursive: bool = True,
    max_workers: int = 1,
) -> dict[str, Any]:
    workflow = report.get("workflow")
    status = report.get("status")
    paths = _mapping(report.get("paths"))
    manifest_value = paths.get("artifact_manifest")
    manifest_verification = _input_manifest_verification_summary(
        path=path,
        manifest_value=manifest_value,
        require_manifest=require_manifest,
        context=context,
        recursive=recursive,
        max_workers=max_workers,
    )
    blocking_reasons = []
    if str(workflow) not in FRONTIER_RERUN_ROLLUP_WORKFLOWS:
        blocking_reasons.append(f"{path} workflow {workflow!r} is not a supported rerun rollup")
    if status not in {"promote", "blocked", "complete", "empty"}:
        blocking_reasons.append(
            f"{path} status {status!r} is not a supported rerun rollup status"
        )
    blocking_reasons.extend(manifest_verification["blocking_reasons"])
    return {
        "path": str(path),
        "workflow": workflow,
        "track": FRONTIER_RERUN_ROLLUP_WORKFLOWS.get(str(workflow)),
        "status": status,
        "artifact_manifest": manifest_value,
        "artifact_manifest_summary": _mapping(report.get("artifact_manifest_summary")),
        "artifact_manifest_verification": manifest_verification,
        "blocking_reasons": tuple(blocking_reasons),
    }


def _input_manifest_verification_summary(
    *,
    path: Path,
    manifest_value: Any,
    require_manifest: bool,
    context: ArtifactVerificationContext | None,
    recursive: bool,
    max_workers: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "required": bool(require_manifest),
        "path": None if manifest_value is None else str(manifest_value),
        "resolved_path": None,
        "passed": None,
        "checked": None,
        "failure_count": None,
        "nested_failure_count": None,
        "recursive": bool(recursive),
        "blocking_reasons": (),
    }
    if not require_manifest:
        return summary
    if not manifest_value:
        reason = f"{path} paths.artifact_manifest is required but missing"
        summary["blocking_reasons"] = (reason,)
        return summary
    manifest_path = _resolve_input_manifest_path(manifest_value, report_path=path)
    summary["resolved_path"] = str(manifest_path)
    verifier = context or ArtifactVerificationContext()
    try:
        verification = verifier.load_and_verify_artifact_manifest(
            manifest_path,
            recursive=recursive,
            max_workers=max_workers,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        reason = f"{path} artifact_manifest {manifest_path} could not be verified: {exc}"
        summary.update({
            "passed": False,
            "checked": 0,
            "failure_count": 1,
            "nested_failure_count": 0,
            "blocking_reasons": (reason,),
        })
        return summary
    payload = verification.to_dict()
    failure_count = len(tuple(payload.get("failures", ())))
    nested_failure_count = _nested_manifest_failure_count(payload.get("nested", ()))
    summary.update({
        "passed": bool(payload.get("passed")),
        "checked": payload.get("checked"),
        "failure_count": failure_count,
        "nested_failure_count": nested_failure_count,
    })
    if payload.get("passed") is not True:
        reason = (
            f"{path} artifact_manifest {manifest_path} verification failed "
            f"with {failure_count + nested_failure_count} failures"
        )
        summary["blocking_reasons"] = (reason,)
    return summary


def _resolve_input_manifest_path(value: Any, *, report_path: Path) -> Path:
    manifest_path = Path(str(value))
    if manifest_path.is_absolute() or manifest_path.exists():
        return manifest_path
    sibling_path = report_path.parent / manifest_path
    if sibling_path.exists():
        return sibling_path
    return manifest_path


def _nested_manifest_failure_count(value: Any) -> int:
    total = 0
    for item in _mapping_sequence(value):
        total += len(tuple(item.get("failures", ())))
        total += _nested_manifest_failure_count(item.get("nested", ()))
    return total


def _input_manifest_evidence_summary(
    inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    verifications = tuple(
        _mapping(input_summary.get("artifact_manifest_verification"))
        for input_summary in inputs
    )
    required = tuple(item for item in verifications if item.get("required") is True)
    return {
        "input_manifest_required": bool(required),
        "input_manifest_required_count": len(required),
        "input_manifest_verified_count": sum(1 for item in required if item.get("passed") is True),
        "input_manifest_failed_count": sum(1 for item in required if item.get("passed") is False),
        "input_manifest_missing_count": sum(
            1 for item in required
            if item.get("path") in {None, ""}
        ),
        "input_manifest_failure_count": sum(
            (_non_negative_int(item.get("failure_count")) or 0)
            + (_non_negative_int(item.get("nested_failure_count")) or 0)
            for item in required
        ),
    }


def _track_status(run_decisions: Sequence[Mapping[str, Any]], key: str) -> str:
    if not run_decisions:
        return "blocked"
    statuses = {str(_mapping(decision.get(key)).get("status")) for decision in run_decisions}
    return "promote" if statuses == {"promote"} else "blocked"


def _runs_by_name(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    runs = payload.get("runs", ())
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            continue
        name = str(run.get("name") or f"run_{index}")
        result[name] = run
    return result


def _detectability_runs_by_name(
    reports: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for path, report in reports:
        name = _detectability_report_name(path, report)
        if name in result:
            raise ValueError(f"duplicate detectability taxonomy run name: {name!r}")
        result[name] = report
    return result


def _detectability_report_name(path: Path, report: Mapping[str, Any]) -> str:
    metadata = _mapping(report.get("metadata"))
    for key in ("run_name", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source = _mapping(report.get("source"))
    summary = _mapping(source.get("score_dump_summary"))
    for key in ("name", "run_name", "model"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return path.stem


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must contain an object: {path}")
    return payload


def _stat_mean(payload: Mapping[str, Any], name: str) -> float | None:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        return None
    return _finite_float_or_none(value.get("mean"))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _positive_int(value: Any) -> int | None:
    numeric = _non_negative_int(value)
    if numeric is None or numeric < 1:
        return None
    return numeric


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return None
    return numeric


def _ratio(numerator: int | None, denominator: int) -> float | None:
    if numerator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _finite_float(value: Any, *, name: str) -> float:
    numeric = _finite_float_or_none(value)
    if numeric is None:
        raise ValueError(f"{name} must be finite.")
    return numeric


def _unit_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _min_check(name: str, value: float | None, threshold: float) -> dict[str, Any]:
    passed = value is not None and value >= threshold
    reason = None
    if not passed:
        if value is None:
            reason = f"{name} is missing or non-finite"
        else:
            reason = f"{name} {value:.6g} is below required minimum {threshold:.6g}"
    return {
        "name": name,
        "operator": ">=",
        "value": value,
        "threshold": threshold,
        "passed": passed,
        "reason": reason,
    }


def _max_check(name: str, value: float | None, threshold: float) -> dict[str, Any]:
    passed = value is not None and value <= threshold
    reason = None
    if not passed:
        if value is None:
            reason = f"{name} is missing or non-finite"
        else:
            reason = f"{name} {value:.6g} exceeds maximum {threshold:.6g}"
    return {
        "name": name,
        "operator": "<=",
        "value": value,
        "threshold": threshold,
        "passed": passed,
        "reason": reason,
    }


def _failed_reasons(checks: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        str(check["reason"])
        for check in checks
        if check.get("passed") is not True and check.get("reason")
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_artifact_manifest(
    *,
    context: ArtifactVerificationContext,
    report_path: Path,
    output_path: Path,
    payload: Mapping[str, Any],
    max_workers: int = 1,
) -> dict[str, Any]:
    inputs = _mapping(payload.get("inputs"))
    verifier_input = _mapping(inputs.get("verifier_stability_report"))
    abstention_input = _mapping(inputs.get("abstention_stability_report"))
    detectability_inputs = tuple(
        item for item in inputs.get("detectability_taxonomy_reports", ())
        if isinstance(item, Mapping)
    )
    frontier_workflow_inputs = tuple(
        item for item in inputs.get("frontier_workflow_reports", ())
        if isinstance(item, Mapping)
    )
    citation_batch_rollup_inputs = tuple(
        item for item in inputs.get("citation_batch_rollup_reports", ())
        if isinstance(item, Mapping)
    )
    frontier_rerun_rollup_inputs = tuple(
        item for item in inputs.get("frontier_rerun_rollup_reports", ())
        if isinstance(item, Mapping)
    )
    artifacts: dict[str, str | Path | None] = {
        "frontier_release_evidence_report": report_path,
        "verifier_stability_report": verifier_input.get("path"),
        "verifier_stability_manifest": verifier_input.get("artifact_manifest"),
        "abstention_stability_report": abstention_input.get("path"),
        "abstention_stability_manifest": abstention_input.get("artifact_manifest"),
    }
    for index, detectability_input in enumerate(detectability_inputs):
        artifacts[f"detectability_taxonomy_report_{index}"] = detectability_input.get("path")
        artifacts[f"detectability_taxonomy_manifest_{index}"] = detectability_input.get("artifact_manifest")
    for index, frontier_workflow_input in enumerate(frontier_workflow_inputs):
        artifacts[f"frontier_workflow_report_{index}"] = frontier_workflow_input.get("path")
        artifacts[f"frontier_workflow_manifest_{index}"] = frontier_workflow_input.get(
            "artifact_manifest"
        )
    for index, rollup_input in enumerate(citation_batch_rollup_inputs):
        artifacts[f"citation_batch_rollup_report_{index}"] = rollup_input.get("path")
        artifacts[f"citation_batch_rollup_manifest_{index}"] = rollup_input.get(
            "artifact_manifest"
        )
    for index, rollup_input in enumerate(frontier_rerun_rollup_inputs):
        artifacts[f"frontier_rerun_rollup_report_{index}"] = rollup_input.get("path")
        artifacts[f"frontier_rerun_rollup_manifest_{index}"] = rollup_input.get(
            "artifact_manifest"
        )
    manifest = context.build_artifact_manifest(
        artifacts,
        root=output_path.parent,
        metadata={
            "runner": "compare_frontier_release_evidence",
            "status": payload.get("decision", {}).get("status")
            if isinstance(payload.get("decision"), Mapping)
            else None,
            "verifier_track_status": payload.get("decision", {}).get("verifier_track_status")
            if isinstance(payload.get("decision"), Mapping)
            else None,
            "abstention_track_status": payload.get("decision", {}).get("abstention_track_status")
            if isinstance(payload.get("decision"), Mapping)
            else None,
            "detectability_track_status": payload.get("decision", {}).get("detectability_track_status")
            if isinstance(payload.get("decision"), Mapping)
            else None,
            "multiple_testing_track_status": (
                payload.get("decision", {}).get("multiple_testing_track_status")
                if isinstance(payload.get("decision"), Mapping)
                else None
            ),
            "citation_batch_track_status": (
                payload.get("decision", {}).get("citation_batch_track_status")
                if isinstance(payload.get("decision"), Mapping)
                else None
            ),
            "frontier_rerun_rollup_track_status": (
                payload.get("decision", {}).get("frontier_rerun_rollup_track_status")
                if isinstance(payload.get("decision"), Mapping)
                else None
            ),
            "frontier_rerun_rollup_promoted_tracks": (
                tuple(payload.get("decision", {}).get("frontier_rerun_rollup_promoted_tracks", ()))
                if isinstance(payload.get("decision"), Mapping)
                else ()
            ),
        },
        max_workers=max_workers,
    )
    _write_json(output_path, manifest)
    return manifest


def _verify_manifest(
    *,
    context: ArtifactVerificationContext,
    manifest_path: Path,
    output_path: Path,
    recursive: bool,
    max_workers: int = 1,
) -> dict[str, Any]:
    verification = context.load_and_verify_artifact_manifest(
        manifest_path,
        recursive=recursive,
        max_workers=max_workers,
    )
    payload = verification.to_dict()
    _write_json(output_path, payload)
    return payload


def _record_registry(
    *,
    registry_path: Path | None,
    name: str | None,
    version: str | None,
    report_path: Path,
    manifest_path: Path | None,
    verification_path: Path | None,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    verification: Mapping[str, Any] | None,
) -> None:
    if registry_path is None:
        return
    if not name or not version:
        raise ValueError("--registry requires --name and --version.")
    decision = _mapping(payload.get("decision"))
    evidence_summary = _mapping(payload.get("evidence_summary"))
    metadata = {
        "workflow": "compare_frontier_release_evidence",
        "status": decision.get("status"),
        "verifier_track_status": decision.get("verifier_track_status"),
        "abstention_track_status": decision.get("abstention_track_status"),
        "detectability_track_status": decision.get("detectability_track_status"),
        "multiple_testing_track_status": decision.get("multiple_testing_track_status"),
        "citation_batch_track_status": decision.get("citation_batch_track_status"),
        "frontier_rerun_rollup_track_status": decision.get("frontier_rerun_rollup_track_status"),
        "base_verifier_track_status": decision.get("base_verifier_track_status"),
        "base_abstention_track_status": decision.get("base_abstention_track_status"),
        "base_detectability_track_status": decision.get("base_detectability_track_status"),
        "base_multiple_testing_track_status": decision.get("base_multiple_testing_track_status"),
        "frontier_rerun_rollup_promoted_tracks": tuple(
            decision.get("frontier_rerun_rollup_promoted_tracks", ())
        ),
        "blocking_reasons": tuple(decision.get("blocking_reasons", ())),
        "run_names": tuple(evidence_summary.get("run_names", ())),
        "verifier_signal": evidence_summary.get("verifier_signal"),
        "abstention_signals": tuple(evidence_summary.get("abstention_signals", ())),
        "detectability_report_count": evidence_summary.get("detectability_report_count"),
        "frontier_workflow_report_count": evidence_summary.get("frontier_workflow_report_count"),
        "citation_batch_rollup_report_count": evidence_summary.get(
            "citation_batch_rollup_report_count"
        ),
        "citation_batch_rollup_names": tuple(
            evidence_summary.get("citation_batch_rollup_names", ())
        ),
        "citation_batch_rollup_count": evidence_summary.get("citation_batch_rollup_count"),
        "citation_batch_promotion_ready_count": evidence_summary.get(
            "citation_batch_promotion_ready_count"
        ),
        "input_manifest_required": evidence_summary.get("input_manifest_required"),
        "input_manifest_required_count": evidence_summary.get("input_manifest_required_count"),
        "input_manifest_verified_count": evidence_summary.get("input_manifest_verified_count"),
        "input_manifest_failed_count": evidence_summary.get("input_manifest_failed_count"),
        "input_manifest_missing_count": evidence_summary.get("input_manifest_missing_count"),
        "input_manifest_failure_count": evidence_summary.get("input_manifest_failure_count"),
        "frontier_rerun_rollup_report_count": evidence_summary.get(
            "frontier_rerun_rollup_report_count"
        ),
        "frontier_rerun_rollup_names": tuple(
            evidence_summary.get("frontier_rerun_rollup_names", ())
        ),
        "frontier_rerun_rollup_workflows": tuple(
            evidence_summary.get("frontier_rerun_rollup_workflows", ())
        ),
        "frontier_rerun_rollup_tracks": tuple(
            evidence_summary.get("frontier_rerun_rollup_tracks", ())
        ),
        "frontier_rerun_rollup_candidate_count": evidence_summary.get(
            "frontier_rerun_rollup_candidate_count"
        ),
        "frontier_rerun_rollup_missing_report_count": evidence_summary.get(
            "frontier_rerun_rollup_missing_report_count"
        ),
        "frontier_rerun_rollup_invalid_report_count": evidence_summary.get(
            "frontier_rerun_rollup_invalid_report_count"
        ),
        "frontier_rerun_rollup_blocked_candidate_count": evidence_summary.get(
            "frontier_rerun_rollup_blocked_candidate_count"
        ),
        "citation_batch_expected_batch_count": evidence_summary.get(
            "citation_batch_expected_batch_count"
        ),
        "citation_batch_observed_batch_count": evidence_summary.get(
            "citation_batch_observed_batch_count"
        ),
        "citation_batch_missing_expected_batch_count": evidence_summary.get(
            "citation_batch_missing_expected_batch_count"
        ),
        "citation_batch_duplicate_batch_count": evidence_summary.get(
            "citation_batch_duplicate_batch_count"
        ),
        "citation_batch_unexpected_batch_count": evidence_summary.get(
            "citation_batch_unexpected_batch_count"
        ),
        "citation_batch_blocked_child_report_count": evidence_summary.get(
            "citation_batch_blocked_child_report_count"
        ),
        "citation_batch_child_manifest_failed_count": evidence_summary.get(
            "citation_batch_child_manifest_failed_count"
        ),
        "citation_batch_expected_batch_ids": tuple(
            evidence_summary.get("citation_batch_expected_batch_ids", ())
        ),
        "citation_batch_observed_batch_ids": tuple(
            evidence_summary.get("citation_batch_observed_batch_ids", ())
        ),
        "citation_batch_missing_expected_batches": tuple(
            evidence_summary.get("citation_batch_missing_expected_batches", ())
        ),
        "citation_batch_duplicate_batches": tuple(
            evidence_summary.get("citation_batch_duplicate_batches", ())
        ),
        "citation_batch_unexpected_batches": tuple(
            evidence_summary.get("citation_batch_unexpected_batches", ())
        ),
        "multiple_testing_frontier_workflow_names": tuple(
            evidence_summary.get("multiple_testing_frontier_workflow_names", ())
        ),
        "multiple_testing_cell_count": evidence_summary.get("multiple_testing_cell_count"),
        "multiple_testing_pass_count": evidence_summary.get("multiple_testing_pass_count"),
        "multiple_testing_fail_count": evidence_summary.get("multiple_testing_fail_count"),
        "multiple_testing_unknown_count": evidence_summary.get("multiple_testing_unknown_count"),
        "multiple_testing_signals": tuple(evidence_summary.get("multiple_testing_signals", ())),
        "multiple_testing_failed_cells": tuple(
            evidence_summary.get("multiple_testing_failed_cells", ())
        ),
        "multiple_testing_unknown_cells": tuple(
            evidence_summary.get("multiple_testing_unknown_cells", ())
        ),
        "multiple_testing_blocked_cells": tuple(
            evidence_summary.get("multiple_testing_blocked_cells", ())
        ),
        "artifact_manifest": None if manifest_path is None else str(manifest_path),
        "manifest_verification_report": None if verification_path is None else str(verification_path),
        "manifest_verified": None if verification is None else bool(verification.get("passed")),
    }
    registry = ArtifactRegistry.load_json(registry_path)
    registry.record_report(
        name=name,
        path=report_path,
        version=version,
        metadata=metadata,
    )
    if manifest_path is not None and manifest is not None:
        registry.record_benchmark_manifest(
            name=name,
            path=manifest_path,
            version=version,
            metadata={
                **metadata,
                "manifest_summary": _mapping(manifest.get("summary")),
                "manifest_metadata": _mapping(manifest.get("metadata")),
                "checked": None if verification is None else verification.get("checked"),
                "failure_count": None
                if verification is None
                else len(tuple(verification.get("failures", ()))),
            },
        )
    if verification_path is not None and verification is not None:
        registry.record_manifest_verification(
            name=f"{name}-verification",
            path=verification_path,
            version=version,
            metadata={
                "manifest_name": name,
                "manifest_path": None if manifest_path is None else str(manifest_path),
                "passed": bool(verification.get("passed")),
                "recursive": bool(_mapping(payload.get("verification")).get("recursive", True)),
            },
        )
    registry.save_json()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.json)
    manifest_path = None if args.artifact_manifest is None else Path(args.artifact_manifest)
    verification_path = None if args.verification_report is None else Path(args.verification_report)
    context = ArtifactVerificationContext()
    payload = compare_frontier_release_evidence(
        verifier_stability_report_path=args.verifier_stability_report,
        abstention_stability_report_path=args.abstention_stability_report,
        detectability_taxonomy_report_paths=tuple(args.detectability_taxonomy_report or ()),
        frontier_workflow_report_paths=tuple(args.frontier_workflow_report or ()),
        citation_batch_rollup_report_paths=tuple(args.citation_batch_rollup_report or ()),
        frontier_rerun_rollup_report_paths=tuple(args.frontier_rerun_rollup_report or ()),
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
        max_detectability_entrenched_false_rate=args.max_detectability_entrenched_false_rate,
        require_input_manifests=args.require_input_manifests,
        input_manifest_recursive=not args.no_recursive,
        manifest_fingerprint_workers=args.manifest_fingerprint_workers,
        notes=args.note,
    )
    if manifest_path is not None:
        payload["paths"] = {
            "frontier_release_evidence_report": str(output_path),
            "artifact_manifest": str(manifest_path),
            "manifest_verification": None if verification_path is None else str(verification_path),
        }
    _write_json(output_path, payload)
    manifest = None
    verification = None
    if manifest_path is not None:
        manifest = _write_artifact_manifest(
            context=context,
            report_path=output_path,
            output_path=manifest_path,
            payload=payload,
            max_workers=args.manifest_fingerprint_workers,
        )
        payload["artifact_manifest_summary"] = manifest.get("summary", {})
        _write_json(output_path, payload)
        # Rebuild after embedding the manifest summary so the report fingerprint
        # in the manifest matches the final persisted report payload.
        manifest = _write_artifact_manifest(
            context=context,
            report_path=output_path,
            output_path=manifest_path,
            payload=payload,
            max_workers=args.manifest_fingerprint_workers,
        )
    if verification_path is not None:
        if manifest_path is None:
            raise ValueError("--verification-report requires --artifact-manifest.")
        verification = _verify_manifest(
            context=context,
            manifest_path=manifest_path,
            output_path=verification_path,
            recursive=not args.no_recursive,
            max_workers=args.manifest_fingerprint_workers,
        )
        payload["manifest_verification"] = {
            "path": str(verification_path),
            "passed": verification.get("passed"),
            "checked": verification.get("checked"),
            "failure_count": len(tuple(verification.get("failures", ()))),
            "recursive": not args.no_recursive,
        }
        _write_json(output_path, payload)
        # The report now contains verification metadata, so refresh the manifest
        # and verification report once more against the final report bytes.
        manifest = _write_artifact_manifest(
            context=context,
            report_path=output_path,
            output_path=manifest_path,
            payload=payload,
            max_workers=args.manifest_fingerprint_workers,
        )
        payload["artifact_manifest_summary"] = manifest.get("summary", {})
        _write_json(output_path, payload)
        manifest = _write_artifact_manifest(
            context=context,
            report_path=output_path,
            output_path=manifest_path,
            payload=payload,
            max_workers=args.manifest_fingerprint_workers,
        )
        verification = _verify_manifest(
            context=context,
            manifest_path=manifest_path,
            output_path=verification_path,
            recursive=not args.no_recursive,
            max_workers=args.manifest_fingerprint_workers,
        )
        payload["manifest_verification"] = {
            "path": str(verification_path),
            "passed": verification.get("passed"),
            "checked": verification.get("checked"),
            "failure_count": len(tuple(verification.get("failures", ()))),
            "recursive": not args.no_recursive,
        }
        _write_json(output_path, payload)
    _record_registry(
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        report_path=output_path,
        manifest_path=manifest_path,
        verification_path=verification_path,
        payload=payload,
        manifest=manifest,
        verification=verification,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare frontier stability evidence for release")
    parser.add_argument("--verifier-stability-report", required=True)
    parser.add_argument("--abstention-stability-report", required=True)
    parser.add_argument("--detectability-taxonomy-report", action="append", default=[])
    parser.add_argument("--frontier-workflow-report", action="append", default=[])
    parser.add_argument("--citation-batch-rollup-report", action="append", default=[])
    parser.add_argument("--frontier-rerun-rollup-report", action="append", default=[])
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--max-verified-false-alarm-mean", type=float,
                        default=DEFAULT_MAX_VERIFIED_FALSE_ALARM_MEAN)
    parser.add_argument("--min-verified-detection-mean", type=float,
                        default=DEFAULT_MIN_VERIFIED_DETECTION_MEAN)
    parser.add_argument("--min-verifier-delta-detection-mean", type=float,
                        default=DEFAULT_MIN_VERIFIER_DELTA_DETECTION_MEAN)
    parser.add_argument("--min-verifier-pass-seed-rate", type=float,
                        default=DEFAULT_MIN_VERIFIER_PASS_SEED_RATE)
    parser.add_argument("--min-verifier-beats-internal-seed-rate", type=float,
                        default=DEFAULT_MIN_VERIFIER_BEATS_INTERNAL_SEED_RATE)
    parser.add_argument("--min-abstention-pass-seed-rate", type=float,
                        default=DEFAULT_MIN_ABSTENTION_PASS_SEED_RATE)
    parser.add_argument("--min-abstention-conditional-correctness-lower-bound-mean", type=float,
                        default=DEFAULT_MIN_ABSTENTION_CONDITIONAL_CORRECTNESS_LOWER_BOUND_MEAN)
    parser.add_argument("--max-abstention-rate-mean", type=float,
                        default=DEFAULT_MAX_ABSTENTION_RATE_MEAN)
    parser.add_argument("--max-detectability-entrenched-false-rate", type=float,
                        default=DEFAULT_MAX_DETECTABILITY_ENTRENCHED_FALSE_RATE)
    parser.add_argument("--manifest-fingerprint-workers", type=int, default=1)
    parser.add_argument("--require-input-manifests", action="store_true",
                        help="fail closed unless every input report declares an artifact manifest that verifies")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--note", action="append", default=[])
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
