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


def compare_frontier_release_evidence(
    *,
    verifier_stability_report_path: str | Path,
    abstention_stability_report_path: str | Path,
    detectability_taxonomy_report_paths: Sequence[str | Path] = (),
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
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a release verdict from verifier and abstention stability reports."""
    verifier_path = Path(verifier_stability_report_path)
    abstention_path = Path(abstention_stability_report_path)
    detectability_paths = tuple(Path(path) for path in detectability_taxonomy_report_paths)
    verifier = _load_json_object(verifier_path)
    abstention = _load_json_object(abstention_path)
    detectability_reports = tuple((path, _load_json_object(path)) for path in detectability_paths)
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
    }
    verifier_runs = _runs_by_name(verifier)
    abstention_runs = _runs_by_name(abstention)
    detectability_runs = _detectability_runs_by_name(detectability_reports)
    verifier_input = _input_summary(
        verifier,
        path=verifier_path,
        expected_workflow="verifier_stability",
    )
    abstention_input = _input_summary(
        abstention,
        path=abstention_path,
        expected_workflow="abstention_stability",
    )
    detectability_inputs = tuple(
        _input_summary(
            report,
            path=path,
            expected_workflow="detectability_taxonomy",
        )
        for path, report in detectability_reports
    )

    input_blocking_reasons = tuple(verifier_input["blocking_reasons"]) + tuple(
        abstention_input["blocking_reasons"]
    ) + tuple(
        reason
        for item in detectability_inputs
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

    verifier_track_status = _track_status(run_decisions, "verifier_decision")
    abstention_track_status = _track_status(run_decisions, "abstention_decision")
    detectability_track_status = (
        _track_status(run_decisions, "detectability_decision")
        if detectability_reports
        else "not_required"
    )
    blocking_reasons = list(input_blocking_reasons)
    for decision in run_decisions:
        blocking_reasons.extend(decision["blocking_reasons"])
    status = (
        "promote"
        if not blocking_reasons
        and verifier_track_status == "promote"
        and abstention_track_status == "promote"
        and detectability_track_status in {"promote", "not_required"}
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
        },
        "evidence_summary": {
            "run_count": len(run_decisions),
            "run_names": run_names,
            "verifier_track_status": verifier_track_status,
            "abstention_track_status": abstention_track_status,
            "detectability_track_status": detectability_track_status,
            "detectability_report_count": len(detectability_reports),
            "verifier_signal": verifier.get("config", {}).get("signal")
            if isinstance(verifier.get("config"), Mapping)
            else None,
            "abstention_signals": list(abstention.get("config", {}).get("signals", ()))
            if isinstance(abstention.get("config"), Mapping)
            else [],
        },
        "run_decisions": run_decisions,
        "decision": {
            "status": status,
            "verifier_track_status": verifier_track_status,
            "abstention_track_status": abstention_track_status,
            "detectability_track_status": detectability_track_status,
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
) -> dict[str, Any]:
    workflow = payload.get("workflow")
    status = payload.get("status")
    paths = _mapping(payload.get("paths"))
    blocking_reasons = []
    if workflow != expected_workflow:
        blocking_reasons.append(f"{path} workflow {workflow!r} is not {expected_workflow!r}")
    if status != "complete":
        blocking_reasons.append(f"{path} status {status!r} is not 'complete'")
    return {
        "path": str(path),
        "workflow": workflow,
        "status": status,
        "artifact_manifest": paths.get("artifact_manifest"),
        "artifact_manifest_summary": _mapping(payload.get("artifact_manifest_summary")),
        "blocking_reasons": tuple(blocking_reasons),
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
        "blocking_reasons": tuple(decision.get("blocking_reasons", ())),
        "run_names": tuple(evidence_summary.get("run_names", ())),
        "verifier_signal": evidence_summary.get("verifier_signal"),
        "abstention_signals": tuple(evidence_summary.get("abstention_signals", ())),
        "detectability_report_count": evidence_summary.get("detectability_report_count"),
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
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--note", action="append", default=[])
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
