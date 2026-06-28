"""Compare forced-answer trajectory sweep evidence across runs.

This post-processing helper consumes reports produced by
``eval_trajectory_truthfulqa.py --layers`` and applies a fail-closed promotion
gate. The default gate is intentionally stricter than the first gpt2 smoke
artifact: trajectory signals need enough examples and more than one model
family before they should affect release decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

DEFAULT_MIN_REPORTS = 2
DEFAULT_MIN_MODEL_COUNT = 2
DEFAULT_MIN_N_EVALUATED = 100
DEFAULT_MIN_BEST_AUROC = 0.60
DEFAULT_MIN_CLASS_COUNT = 20
DEFAULT_MAX_SKIP_RATE = 0.20


def compare_trajectory_sweeps(
    reports: Sequence[tuple[str, Path]],
    *,
    min_reports: int = DEFAULT_MIN_REPORTS,
    min_model_count: int = DEFAULT_MIN_MODEL_COUNT,
    min_n_evaluated: int = DEFAULT_MIN_N_EVALUATED,
    min_best_auroc: float = DEFAULT_MIN_BEST_AUROC,
    min_class_count: int = DEFAULT_MIN_CLASS_COUNT,
    max_skip_rate: float = DEFAULT_MAX_SKIP_RATE,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a fail-closed evidence gate over trajectory sweep reports."""
    if not reports:
        raise ValueError("at least one trajectory report is required.")
    config = {
        "min_reports": _positive_int_value(min_reports, name="min_reports"),
        "min_model_count": _positive_int_value(min_model_count, name="min_model_count"),
        "min_n_evaluated": _positive_int_value(min_n_evaluated, name="min_n_evaluated"),
        "min_best_auroc": _unit_float(min_best_auroc, name="min_best_auroc"),
        "min_class_count": _non_negative_int_value(min_class_count, name="min_class_count"),
        "max_skip_rate": _unit_float(max_skip_rate, name="max_skip_rate"),
    }
    run_decisions = []
    for name, path in reports:
        payload = _load_json_object(path)
        run_decisions.append(_run_decision(name=name, path=path, report=payload, config=config))
    models = sorted({
        str(run["metrics"]["model"])
        for run in run_decisions
        if run["metrics"].get("model")
    })
    promoted_runs = [run for run in run_decisions if run["status"] == "promote"]
    blocking_reasons = []
    blocking_reasons.extend(_global_blocking_reasons(
        run_decisions,
        models=models,
        config=config,
    ))
    for run in run_decisions:
        blocking_reasons.extend(str(reason) for reason in run["blocking_reasons"])
    status = "promote" if not blocking_reasons else "blocked"
    best_run = _best_run(run_decisions)
    return {
        "schema_version": 1,
        "workflow": "trajectory_sweep_evidence_comparison",
        "status": "complete",
        "config": config,
        "evidence_summary": {
            "report_count": len(run_decisions),
            "promoted_report_count": len(promoted_runs),
            "model_count": len(models),
            "models": models,
            "mean_best_auroc": _mean(
                float(run["metrics"]["trajectory_score_best_auroc"])
                for run in run_decisions
                if run["metrics"].get("trajectory_score_best_auroc") is not None
            ),
            "min_best_auroc": _min_or_none(
                float(run["metrics"]["trajectory_score_best_auroc"])
                for run in run_decisions
                if run["metrics"].get("trajectory_score_best_auroc") is not None
            ),
            "recommended_run": None if best_run is None else {
                "name": best_run["name"],
                "path": best_run["path"],
                "model": best_run["metrics"].get("model"),
                "best_layer": best_run["metrics"].get("best_layer"),
                "trajectory_score_best_auroc": best_run["metrics"].get("trajectory_score_best_auroc"),
            },
        },
        "run_decisions": run_decisions,
        "decision": {
            "status": status,
            "trajectory_track_status": status,
            "blocking_reasons": tuple(blocking_reasons),
        },
        "notes": tuple(str(note) for note in notes),
    }


def _run_decision(
    *,
    name: str,
    path: Path,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = _trajectory_report_metrics(report)
    checks = (
        _eq_check(f"{name}.workflow", report.get("workflow"), {
            "truthfulqa_forced_answer_trajectory_layer_sweep",
            "truthfulqa_forced_answer_trajectory",
        }),
        _eq_check(f"{name}.status", report.get("status", "complete"), {"complete", None}),
        _min_check(f"{name}.n_evaluated", metrics.get("n_evaluated"), config["min_n_evaluated"]),
        _min_check(f"{name}.n_true", metrics.get("n_true"), config["min_class_count"]),
        _min_check(f"{name}.n_false", metrics.get("n_false"), config["min_class_count"]),
        _max_check(f"{name}.skip_rate", metrics.get("skip_rate"), config["max_skip_rate"]),
        _min_check(
            f"{name}.trajectory_score_best_auroc",
            metrics.get("trajectory_score_best_auroc"),
            config["min_best_auroc"],
        ),
    )
    blocking_reasons = _failed_reasons(checks)
    if not metrics.get("model"):
        blocking_reasons = blocking_reasons + (f"{name}.model is missing",)
    if metrics.get("trajectory_score_direction_for_false") not in {"higher", "lower"}:
        blocking_reasons = blocking_reasons + (
            f"{name}.trajectory_score_direction_for_false is missing or invalid",
        )
    return {
        "name": str(name),
        "path": str(path),
        "status": "promote" if not blocking_reasons else "blocked",
        "metrics": metrics,
        "checks": tuple(checks),
        "blocking_reasons": blocking_reasons,
    }


def _trajectory_report_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    metadata = _mapping(report.get("metadata"))
    config = _mapping(report.get("config"))
    layer_summaries = _layer_summaries(report)
    best_layer_summary = _best_layer_summary(layer_summaries)
    n_total = _finite_float_or_none(summary.get("n_total"))
    n_skipped = _finite_float_or_none(summary.get("n_skipped"))
    skip_rate = None
    if n_total is not None and n_total > 0 and n_skipped is not None:
        skip_rate = float(n_skipped) / float(n_total)
    return {
        "workflow": report.get("workflow"),
        "model": metadata.get("model"),
        "source_scores_path": _mapping(metadata.get("source_scores")).get("path"),
        "best_layer": summary.get("best_layer", config.get("layer")),
        "best_resolved_layer": summary.get("best_resolved_layer"),
        "layer_count": summary.get("layer_count", 1 if layer_summaries else 0),
        "n_total": _int_or_none(summary.get("n_total")),
        "n_evaluated": _int_or_none(summary.get("n_evaluated")),
        "n_skipped": _int_or_none(summary.get("n_skipped")),
        "skip_rate": skip_rate,
        "n_true": _int_or_none(best_layer_summary.get("n_true", summary.get("n_true"))),
        "n_false": _int_or_none(best_layer_summary.get("n_false", summary.get("n_false"))),
        "trajectory_score_best_auroc": _finite_float_or_none(
            summary.get("trajectory_score_best_auroc")
        ),
        "trajectory_score_direction_for_false": summary.get("trajectory_score_direction_for_false"),
        "spearman_convergence_false_label": _finite_float_or_none(
            summary.get("spearman_convergence_false_label")
        ),
        "nll_answer_higher_is_false_auroc": _finite_float_or_none(
            summary.get("nll_answer_higher_is_false_auroc")
        ),
        "layer_summaries": layer_summaries,
    }


def _layer_summaries(report: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = report.get("layer_summaries")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return tuple(dict(row) for row in rows if isinstance(row, Mapping))
    summary = _mapping(report.get("summary"))
    config = _mapping(report.get("config"))
    if not summary:
        return ()
    return ({
        "layer": config.get("layer"),
        "resolved_layer": None,
        "n_true": summary.get("n_true"),
        "n_false": summary.get("n_false"),
        "trajectory_score_best_auroc": summary.get("trajectory_score_best_auroc"),
        "trajectory_score_direction_for_false": summary.get("trajectory_score_direction_for_false"),
    },)


def _best_layer_summary(layer_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not layer_summaries:
        return {}
    return dict(max(
        layer_summaries,
        key=lambda row: _finite_float_or_none(row.get("trajectory_score_best_auroc")) or -math.inf,
    ))


def _global_blocking_reasons(
    run_decisions: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    reasons = []
    if len(run_decisions) < int(config["min_reports"]):
        reasons.append(
            f"trajectory report count {len(run_decisions)} is below required minimum {config['min_reports']}"
        )
    if len(models) < int(config["min_model_count"]):
        reasons.append(
            f"trajectory model count {len(models)} is below required minimum {config['min_model_count']}"
        )
    return tuple(reasons)


def _best_run(run_decisions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        run for run in run_decisions
        if run["metrics"].get("trajectory_score_best_auroc") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda run: float(run["metrics"]["trajectory_score_best_auroc"]))


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("report name cannot be empty.")
    return name, Path(raw_path)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_artifact_manifest(
    *,
    context: ArtifactVerificationContext,
    report_path: Path,
    input_reports: Sequence[tuple[str, Path]],
    output_path: Path,
    payload: Mapping[str, Any],
    max_workers: int,
) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {"trajectory_sweep_evidence_report": report_path}
    for index, (name, path) in enumerate(input_reports):
        artifacts[f"trajectory_sweep_report_{index}_{_artifact_key(name)}"] = path
    manifest = context.build_artifact_manifest(
        artifacts,
        root=output_path.parent,
        metadata={
            "runner": "compare_trajectory_sweeps",
            "status": _mapping(payload.get("decision")).get("status"),
            "trajectory_track_status": _mapping(payload.get("decision")).get("trajectory_track_status"),
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
    max_workers: int,
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
    summary = _mapping(payload.get("evidence_summary"))
    metadata = {
        "workflow": "compare_trajectory_sweeps",
        "status": decision.get("status"),
        "trajectory_track_status": decision.get("trajectory_track_status"),
        "blocking_reasons": tuple(decision.get("blocking_reasons", ())),
        "report_count": summary.get("report_count"),
        "model_count": summary.get("model_count"),
        "models": tuple(summary.get("models", ())),
        "artifact_manifest": None if manifest_path is None else str(manifest_path),
        "manifest_verification_report": None if verification_path is None else str(verification_path),
        "manifest_verified": None if verification is None else bool(verification.get("passed")),
    }
    registry = ArtifactRegistry.load_json(registry_path)
    registry.record_report(name=name, path=report_path, version=version, metadata=metadata)
    if manifest_path is not None and manifest is not None:
        registry.record_benchmark_manifest(
            name=name,
            path=manifest_path,
            version=version,
            metadata={
                **metadata,
                "manifest_summary": _mapping(manifest.get("summary")),
                "checked": None if verification is None else verification.get("checked"),
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
            },
        )
    registry.save_json()


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [_parse_named_path(value) for value in args.report]
    output_path = Path(args.json)
    manifest_path = None if args.artifact_manifest is None else Path(args.artifact_manifest)
    verification_path = None if args.verification_report is None else Path(args.verification_report)
    context = ArtifactVerificationContext()
    payload = compare_trajectory_sweeps(
        reports,
        min_reports=args.min_reports,
        min_model_count=args.min_model_count,
        min_n_evaluated=args.min_n_evaluated,
        min_best_auroc=args.min_best_auroc,
        min_class_count=args.min_class_count,
        max_skip_rate=args.max_skip_rate,
        notes=args.note,
    )
    if manifest_path is not None:
        payload["paths"] = {
            "trajectory_sweep_evidence_report": str(output_path),
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
            input_reports=reports,
            output_path=manifest_path,
            payload=payload,
            max_workers=args.manifest_fingerprint_workers,
        )
        payload["artifact_manifest_summary"] = manifest.get("summary", {})
        _write_json(output_path, payload)
        manifest = _write_artifact_manifest(
            context=context,
            report_path=output_path,
            input_reports=reports,
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
        manifest = _write_artifact_manifest(
            context=context,
            report_path=output_path,
            input_reports=reports,
            output_path=manifest_path,
            payload=payload,
            max_workers=args.manifest_fingerprint_workers,
        )
        payload["artifact_manifest_summary"] = manifest.get("summary", {})
        _write_json(output_path, payload)
        manifest = _write_artifact_manifest(
            context=context,
            report_path=output_path,
            input_reports=reports,
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
    if not args.quiet:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mean(values: Sequence[float]) -> float | None:
    materialized = [float(value) for value in values]
    if not materialized:
        return None
    return float(statistics.fmean(materialized))


def _min_or_none(values: Sequence[float]) -> float | None:
    materialized = [float(value) for value in values]
    if not materialized:
        return None
    return float(min(materialized))


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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int_value(value: Any, *, name: str) -> int:
    numeric = _int_or_none(value)
    if numeric is None or numeric < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return numeric


def _non_negative_int_value(value: Any, *, name: str) -> int:
    numeric = _int_or_none(value)
    if numeric is None or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return numeric


def _unit_float(value: Any, *, name: str) -> float:
    numeric = _finite_float_or_none(value)
    if numeric is None or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric


def _eq_check(name: str, value: Any, expected: set[Any]) -> dict[str, Any]:
    passed = value in expected
    return {
        "name": name,
        "operator": "in",
        "value": value,
        "threshold": tuple(sorted(str(item) for item in expected)),
        "passed": passed,
        "reason": None if passed else f"{name} is {value!r}, expected one of {sorted(expected, key=str)!r}",
    }


def _min_check(name: str, value: Any, threshold: float | int) -> dict[str, Any]:
    numeric = _finite_float_or_none(value)
    passed = numeric is not None and numeric >= float(threshold)
    return {
        "name": name,
        "operator": ">=",
        "value": numeric,
        "threshold": threshold,
        "passed": passed,
        "reason": None if passed else _threshold_reason(name, numeric, threshold, "below required minimum"),
    }


def _max_check(name: str, value: Any, threshold: float | int) -> dict[str, Any]:
    numeric = _finite_float_or_none(value)
    passed = numeric is not None and numeric <= float(threshold)
    return {
        "name": name,
        "operator": "<=",
        "value": numeric,
        "threshold": threshold,
        "passed": passed,
        "reason": None if passed else _threshold_reason(name, numeric, threshold, "exceeds maximum"),
    }


def _threshold_reason(name: str, value: float | None, threshold: float | int, label: str) -> str:
    if value is None:
        return f"{name} is missing or non-finite"
    return f"{name} {value:.6g} {label} {float(threshold):.6g}"


def _failed_reasons(checks: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        str(check["reason"])
        for check in checks
        if check.get("passed") is not True and check.get("reason")
    )


def _artifact_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_") or "report"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare trajectory sweep evidence for release readiness")
    parser.add_argument("--report", action="append", required=True,
                        help="trajectory sweep report path, optionally named as name=path; repeatable")
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--min-reports", type=int, default=DEFAULT_MIN_REPORTS)
    parser.add_argument("--min-model-count", type=int, default=DEFAULT_MIN_MODEL_COUNT)
    parser.add_argument("--min-n-evaluated", type=int, default=DEFAULT_MIN_N_EVALUATED)
    parser.add_argument("--min-best-auroc", type=float, default=DEFAULT_MIN_BEST_AUROC)
    parser.add_argument("--min-class-count", type=int, default=DEFAULT_MIN_CLASS_COUNT)
    parser.add_argument("--max-skip-rate", type=float, default=DEFAULT_MAX_SKIP_RATE)
    parser.add_argument("--manifest-fingerprint-workers", type=int, default=1)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
