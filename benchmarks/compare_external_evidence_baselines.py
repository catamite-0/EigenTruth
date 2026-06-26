"""Compare external-evidence candidates against stress and text redlines."""

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

from benchmarks.build_text_baseline_score_dump import DEFAULT_TEXT_BASELINE_SIGNALS  # noqa: E402
from benchmarks.compare_route_baselines import compare_route_baselines  # noqa: E402
from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

CANDIDATE_SUMMARY_FIELDS = (
    "best_geometry_fusion_at_alpha",
    "best_ensemble_at_alpha",
    "best_single_at_alpha",
)


def compare_external_evidence_baselines(
    *,
    route_registry_path: str | Path | None = None,
    route_baseline_keys: Sequence[str] = (),
    require_route_baseline: bool = False,
    recursive: bool = True,
    allow_unverified: bool = False,
    min_selected: int | None = None,
    min_decision_accuracy: float | None = None,
    max_false_supported_rate: float | None = None,
    min_false_refuted_rate: float | None = None,
    max_verified_false_alarm: float | None = None,
    min_verified_detection: float | None = None,
    require_non_oracle_evidence: bool = False,
    require_retrieval_provenance_filter: bool = False,
    required_retrieval_source_prefixes: Sequence[str] = (),
    required_retrieval_metadata: Mapping[str, Any] | None = None,
    min_retrieval_filter_score: float | None = None,
    require_retrieval_stress_control: bool = False,
    retrieval_stress_manifest: str | Path | None = None,
    min_stress_false_supported_rate: float | None = None,
    max_stress_false_refuted_rate: float | None = None,
    candidate_score_report_path: str | Path | None = None,
    text_baseline_report_path: str | Path | None = None,
    candidate_run: str | None = None,
    text_run: str | None = None,
    text_baseline_signals: Sequence[str] = DEFAULT_TEXT_BASELINE_SIGNALS,
    candidate_denied_signals: Sequence[str] = DEFAULT_TEXT_BASELINE_SIGNALS,
    require_text_redline: bool = False,
    min_text_detection_margin: float | None = 0.0,
    min_text_auroc_margin: float | None = None,
    min_candidate_detection: float | None = None,
    min_candidate_auroc: float | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a fail-closed comparison for external evidence promotion."""
    route_comparison = _route_comparison(
        route_registry_path=route_registry_path,
        route_baseline_keys=route_baseline_keys,
        require_route_baseline=require_route_baseline,
        recursive=recursive,
        allow_unverified=allow_unverified,
        min_selected=min_selected,
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
        require_non_oracle_evidence=require_non_oracle_evidence,
        require_retrieval_provenance_filter=require_retrieval_provenance_filter,
        required_retrieval_source_prefixes=required_retrieval_source_prefixes,
        required_retrieval_metadata=required_retrieval_metadata,
        min_retrieval_filter_score=min_retrieval_filter_score,
        require_retrieval_stress_control=require_retrieval_stress_control,
        retrieval_stress_manifest=retrieval_stress_manifest,
        min_stress_false_supported_rate=min_stress_false_supported_rate,
        max_stress_false_refuted_rate=max_stress_false_refuted_rate,
    )
    text_redline = compare_text_redline(
        candidate_score_report_path=candidate_score_report_path,
        text_baseline_report_path=text_baseline_report_path,
        candidate_run=candidate_run,
        text_run=text_run,
        text_baseline_signals=text_baseline_signals,
        candidate_denied_signals=candidate_denied_signals,
        require_text_redline=require_text_redline,
        min_text_detection_margin=min_text_detection_margin,
        min_text_auroc_margin=min_text_auroc_margin,
        min_candidate_detection=min_candidate_detection,
        min_candidate_auroc=min_candidate_auroc,
    )
    decision = _decision(route_comparison=route_comparison, text_redline=text_redline)
    return {
        "schema_version": 1,
        "workflow": "external_evidence_baseline_comparison",
        "config": {
            "route_registry_path": None if route_registry_path is None else str(route_registry_path),
            "route_baseline_keys": list(route_baseline_keys),
            "require_route_baseline": bool(require_route_baseline),
            "recursive": bool(recursive),
            "allow_unverified": bool(allow_unverified),
            "min_selected": min_selected,
            "min_decision_accuracy": min_decision_accuracy,
            "max_false_supported_rate": max_false_supported_rate,
            "min_false_refuted_rate": min_false_refuted_rate,
            "max_verified_false_alarm": max_verified_false_alarm,
            "min_verified_detection": min_verified_detection,
            "require_non_oracle_evidence": bool(require_non_oracle_evidence),
            "require_retrieval_provenance_filter": bool(require_retrieval_provenance_filter),
            "required_retrieval_source_prefixes": list(required_retrieval_source_prefixes),
            "required_retrieval_metadata": dict(required_retrieval_metadata or {}),
            "min_retrieval_filter_score": min_retrieval_filter_score,
            "require_retrieval_stress_control": bool(require_retrieval_stress_control),
            "retrieval_stress_manifest": None
            if retrieval_stress_manifest is None
            else str(retrieval_stress_manifest),
            "min_stress_false_supported_rate": min_stress_false_supported_rate,
            "max_stress_false_refuted_rate": max_stress_false_refuted_rate,
            "candidate_score_report_path": None
            if candidate_score_report_path is None
            else str(candidate_score_report_path),
            "text_baseline_report_path": None
            if text_baseline_report_path is None
            else str(text_baseline_report_path),
            "candidate_run": candidate_run,
            "text_run": text_run,
            "text_baseline_signals": list(text_baseline_signals),
            "candidate_denied_signals": list(candidate_denied_signals),
            "require_text_redline": bool(require_text_redline),
            "min_text_detection_margin": min_text_detection_margin,
            "min_text_auroc_margin": min_text_auroc_margin,
            "min_candidate_detection": min_candidate_detection,
            "min_candidate_auroc": min_candidate_auroc,
        },
        "summary": {
            "route_enabled": bool(route_comparison["enabled"]),
            "route_passed": bool(route_comparison["passed"]),
            "text_redline_enabled": bool(text_redline["enabled"]),
            "text_redline_passed": bool(text_redline["passed"]),
        },
        "decision": decision,
        "route_baseline_comparison": route_comparison,
        "text_redline_comparison": text_redline,
        "notes": list(notes),
    }


def compare_text_redline(
    *,
    candidate_score_report_path: str | Path | None,
    text_baseline_report_path: str | Path | None,
    candidate_run: str | None = None,
    text_run: str | None = None,
    text_baseline_signals: Sequence[str] = DEFAULT_TEXT_BASELINE_SIGNALS,
    candidate_denied_signals: Sequence[str] = DEFAULT_TEXT_BASELINE_SIGNALS,
    require_text_redline: bool = False,
    min_text_detection_margin: float | None = 0.0,
    min_text_auroc_margin: float | None = None,
    min_candidate_detection: float | None = None,
    min_candidate_auroc: float | None = None,
) -> dict[str, Any]:
    """Compare candidate signal/fusion quality against cheap text baselines."""
    enabled = bool(require_text_redline or candidate_score_report_path or text_baseline_report_path)
    result: dict[str, Any] = {
        "enabled": enabled,
        "passed": True,
        "blocking_reasons": [],
        "candidate_score_report_path": None
        if candidate_score_report_path is None
        else str(candidate_score_report_path),
        "text_baseline_report_path": None
        if text_baseline_report_path is None
        else str(text_baseline_report_path),
        "run_count": 0,
        "runs": [],
    }
    failures: list[str] = []
    if not enabled:
        return result
    if candidate_score_report_path is None:
        failures.append("candidate score report is required")
    if text_baseline_report_path is None:
        failures.append("text baseline report is required")
    if failures:
        result["passed"] = False
        result["blocking_reasons"] = failures
        return result

    candidate_report = _load_json_mapping(Path(candidate_score_report_path))
    text_report = _load_json_mapping(Path(text_baseline_report_path))
    candidate_runs = _select_runs(candidate_report, candidate_run, role="candidate")
    text_runs = _select_runs(text_report, text_run, role="text baseline")
    text_by_name = {str(run.get("name")): run for run in text_runs if run.get("name") is not None}
    alpha = _resolve_alpha(candidate_report, text_report)
    rows = []
    for candidate in candidate_runs:
        paired_text, pair_error = _paired_text_run(candidate, text_runs, text_by_name, explicit_text_run=text_run)
        if pair_error is not None:
            failures.append(pair_error)
            rows.append({
                "candidate_run": candidate.get("name"),
                "text_run": None,
                "passed": False,
                "blocking_reasons": (pair_error,),
            })
            continue
        row = _text_redline_row(
            candidate,
            paired_text,
            alpha=alpha,
            text_baseline_signals=text_baseline_signals,
            candidate_denied_signals=candidate_denied_signals,
            min_text_detection_margin=min_text_detection_margin,
            min_text_auroc_margin=min_text_auroc_margin,
            min_candidate_detection=min_candidate_detection,
            min_candidate_auroc=min_candidate_auroc,
        )
        rows.append(row)
        failures.extend(
            f"{row['candidate_run']}: {reason}"
            for reason in row["blocking_reasons"]
        )
    if not rows:
        failures.append("no candidate/text baseline runs were compared")
    result.update({
        "passed": not failures,
        "blocking_reasons": failures,
        "alpha": alpha,
        "run_count": len(rows),
        "runs": rows,
    })
    return result


def _route_comparison(
    *,
    route_registry_path: str | Path | None,
    route_baseline_keys: Sequence[str],
    require_route_baseline: bool,
    recursive: bool,
    allow_unverified: bool,
    min_selected: int | None,
    min_decision_accuracy: float | None,
    max_false_supported_rate: float | None,
    min_false_refuted_rate: float | None,
    max_verified_false_alarm: float | None,
    min_verified_detection: float | None,
    require_non_oracle_evidence: bool,
    require_retrieval_provenance_filter: bool,
    required_retrieval_source_prefixes: Sequence[str],
    required_retrieval_metadata: Mapping[str, Any] | None,
    min_retrieval_filter_score: float | None,
    require_retrieval_stress_control: bool,
    retrieval_stress_manifest: str | Path | None,
    min_stress_false_supported_rate: float | None,
    max_stress_false_refuted_rate: float | None,
) -> dict[str, Any]:
    enabled = bool(require_route_baseline or route_registry_path is not None)
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "blocking_reasons": [],
            "comparison": None,
        }
    if route_registry_path is None:
        reason = "route registry is required"
        return {
            "enabled": True,
            "passed": False,
            "blocking_reasons": [reason],
            "comparison": None,
        }
    comparison = compare_route_baselines(
        registry_path=route_registry_path,
        baseline_keys=route_baseline_keys,
        recursive=recursive,
        allow_unverified=allow_unverified,
        min_selected=min_selected,
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
        require_non_oracle_evidence=require_non_oracle_evidence,
        require_retrieval_provenance_filter=require_retrieval_provenance_filter,
        required_retrieval_source_prefixes=required_retrieval_source_prefixes,
        required_retrieval_metadata=required_retrieval_metadata,
        min_retrieval_filter_score=min_retrieval_filter_score,
        require_retrieval_stress_control=require_retrieval_stress_control,
        retrieval_stress_manifest=retrieval_stress_manifest,
        min_stress_false_supported_rate=min_stress_false_supported_rate,
        max_stress_false_refuted_rate=max_stress_false_refuted_rate,
    )
    decision = _mapping(comparison.get("decision"))
    passed = decision.get("status") == "promote"
    return {
        "enabled": True,
        "passed": passed,
        "blocking_reasons": [] if passed else list(decision.get("blocking_reasons", ())),
        "comparison": comparison,
    }


def _text_redline_row(
    candidate_run: Mapping[str, Any],
    text_run: Mapping[str, Any],
    *,
    alpha: float,
    text_baseline_signals: Sequence[str],
    candidate_denied_signals: Sequence[str],
    min_text_detection_margin: float | None,
    min_text_auroc_margin: float | None,
    min_candidate_detection: float | None,
    min_candidate_auroc: float | None,
) -> dict[str, Any]:
    candidate_best = _candidate_best(candidate_run, alpha=alpha, denied_signals=candidate_denied_signals)
    text_best = _text_best(text_run, alpha=alpha, text_baseline_signals=text_baseline_signals)
    failures: list[str] = []
    detection_margin = _metric_margin(candidate_best, text_best, "detection")
    auroc_margin = _metric_margin(candidate_best, text_best, "auroc")
    if candidate_best is None:
        failures.append("candidate report has no finite non-text candidate score at alpha")
    if text_best is None:
        failures.append("text baseline report has no finite text baseline score at alpha")
    if min_candidate_detection is not None:
        _check_min(
            failures,
            "candidate detection",
            None if candidate_best is None else candidate_best.get("detection"),
            min_candidate_detection,
        )
    if min_candidate_auroc is not None:
        _check_min(
            failures,
            "candidate AUROC",
            None if candidate_best is None else candidate_best.get("auroc"),
            min_candidate_auroc,
        )
    if min_text_detection_margin is not None:
        _check_min(failures, "candidate minus text detection", detection_margin, min_text_detection_margin)
    if min_text_auroc_margin is not None:
        _check_min(failures, "candidate minus text AUROC", auroc_margin, min_text_auroc_margin)
    return {
        "candidate_run": candidate_run.get("name"),
        "text_run": text_run.get("name"),
        "passed": not failures,
        "blocking_reasons": failures,
        "candidate_best": candidate_best,
        "text_best": text_best,
        "detection_margin": detection_margin,
        "auroc_margin": auroc_margin,
    }


def _candidate_best(
    run: Mapping[str, Any],
    *,
    alpha: float,
    denied_signals: Sequence[str],
) -> dict[str, Any] | None:
    denied = set(str(signal) for signal in denied_signals)
    candidates: list[dict[str, Any]] = []
    for field in CANDIDATE_SUMMARY_FIELDS:
        entry = _mapping(run.get(field))
        if not entry or entry.get("name") in denied:
            continue
        normalized = _summary_entry(entry, field=field)
        if normalized is not None:
            candidates.append(normalized)
    for group_name, group in (
        ("single_results", _mapping(run.get("single_results"))),
        ("ensemble_results", _mapping(run.get("ensemble_results"))),
        ("geometry_fusion_results", _mapping(run.get("geometry_fusion_results"))),
    ):
        for name, payload in group.items():
            if group_name == "single_results" and name in denied:
                continue
            normalized = _result_entry(str(name), _mapping(payload), alpha=alpha, field=group_name)
            if normalized is not None:
                candidates.append(normalized)
    if not candidates:
        return None
    return max(candidates, key=_quality_key)


def _text_best(
    run: Mapping[str, Any],
    *,
    alpha: float,
    text_baseline_signals: Sequence[str],
) -> dict[str, Any] | None:
    single_results = _mapping(run.get("single_results"))
    candidates = [
        _result_entry(signal, _mapping(single_results.get(signal)), alpha=alpha, field="single_results")
        for signal in text_baseline_signals
    ]
    finite = [candidate for candidate in candidates if candidate is not None]
    if not finite:
        return None
    return max(finite, key=_quality_key)


def _summary_entry(entry: Mapping[str, Any], *, field: str) -> dict[str, Any] | None:
    detection = _float_or_none(entry.get("detection"))
    auroc = _float_or_none(entry.get("auroc"))
    false_alarm = _float_or_none(entry.get("false_alarm"))
    if detection is None:
        return None
    return {
        "name": str(entry.get("name")),
        "source": field,
        "auroc": auroc,
        "false_alarm": false_alarm,
        "detection": detection,
    }


def _result_entry(
    name: str,
    payload: Mapping[str, Any],
    *,
    alpha: float,
    field: str,
) -> dict[str, Any] | None:
    alpha_payload = _alpha_payload(payload, alpha)
    if not alpha_payload:
        return None
    detection = _float_or_none(alpha_payload.get("detection"))
    false_alarm = _float_or_none(alpha_payload.get("false_alarm"))
    auroc = _float_or_none(payload.get("auroc"))
    if detection is None:
        return None
    return {
        "name": name,
        "source": field,
        "auroc": auroc,
        "false_alarm": false_alarm,
        "detection": detection,
    }


def _alpha_payload(payload: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    alphas = _mapping(payload.get("alphas"))
    for key in _alpha_keys(alpha):
        found = alphas.get(key)
        if isinstance(found, Mapping):
            return dict(found)
    return {}


def _alpha_keys(alpha: float) -> tuple[str, ...]:
    return tuple(dict.fromkeys((str(float(alpha)), str(alpha), f"{alpha:.2f}", f"{alpha:.3f}")))


def _quality_key(entry: Mapping[str, Any]) -> tuple[float, float, str]:
    return (
        _float_or_none(entry.get("detection")) or -math.inf,
        _float_or_none(entry.get("auroc")) or -math.inf,
        str(entry.get("name")),
    )


def _metric_margin(
    candidate_best: Mapping[str, Any] | None,
    text_best: Mapping[str, Any] | None,
    metric: str,
) -> float | None:
    if candidate_best is None or text_best is None:
        return None
    candidate = _float_or_none(candidate_best.get(metric))
    text = _float_or_none(text_best.get(metric))
    if candidate is None or text is None:
        return None
    return candidate - text


def _select_runs(report: Mapping[str, Any], run_name: str | None, *, role: str) -> tuple[dict[str, Any], ...]:
    runs = tuple(dict(run) for run in report.get("runs", ()) if isinstance(run, Mapping))
    if run_name is None:
        return runs
    selected = tuple(run for run in runs if str(run.get("name")) == run_name)
    if not selected:
        raise ValueError(f"{role} report has no run named {run_name!r}.")
    return selected


def _paired_text_run(
    candidate_run: Mapping[str, Any],
    text_runs: Sequence[Mapping[str, Any]],
    text_by_name: Mapping[str, Mapping[str, Any]],
    *,
    explicit_text_run: str | None,
) -> tuple[Mapping[str, Any], str | None]:
    if explicit_text_run is not None:
        return text_runs[0], None
    name = candidate_run.get("name")
    if name is not None and str(name) in text_by_name:
        return text_by_name[str(name)], None
    if len(text_runs) == 1:
        return text_runs[0], None
    return {}, f"text baseline run for candidate {name!r} is ambiguous or missing"


def _resolve_alpha(candidate_report: Mapping[str, Any], text_report: Mapping[str, Any]) -> float:
    candidate_alpha = _float_or_none(candidate_report.get("best_alpha"))
    text_alpha = _float_or_none(text_report.get("best_alpha"))
    if candidate_alpha is not None and text_alpha is not None and not math.isclose(candidate_alpha, text_alpha):
        raise ValueError(f"candidate best_alpha {candidate_alpha} differs from text best_alpha {text_alpha}.")
    alpha = candidate_alpha if candidate_alpha is not None else text_alpha
    if alpha is None:
        return 0.10
    if not (0.0 < alpha < 1.0):
        raise ValueError("best_alpha must be in (0, 1).")
    return alpha


def _decision(
    *,
    route_comparison: Mapping[str, Any],
    text_redline: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    if route_comparison.get("enabled") and not route_comparison.get("passed"):
        failures.extend(f"route_baseline: {reason}" for reason in route_comparison.get("blocking_reasons", ()))
    if text_redline.get("enabled") and not text_redline.get("passed"):
        failures.extend(f"text_redline: {reason}" for reason in text_redline.get("blocking_reasons", ()))
    if not failures:
        route_payload = _mapping(route_comparison.get("comparison"))
        route_decision = _mapping(route_payload.get("decision"))
        return {
            "status": "promote",
            "recommended_route_record": route_decision.get("recommended_record"),
            "recommended_route": route_decision.get("recommended_route"),
            "blocking_reasons": (),
        }
    return {
        "status": "blocked",
        "recommended_route_record": None,
        "recommended_route": None,
        "blocking_reasons": tuple(failures),
    }


def _load_json_mapping(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _check_min(
    failures: list[str],
    metric: str,
    value: float | int | None,
    limit: float | int | None,
) -> None:
    if limit is None:
        return
    if value is None or value < limit:
        failures.append(f"{metric} below {limit}")


def _parse_non_negative_float(value: str, *, flag: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{flag} must be a non-negative finite number.")
    return numeric


def _parse_non_negative_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{flag} must be a non-negative integer.")
    return numeric


def _parse_positive_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 1:
        raise ValueError(f"{flag} must be a positive integer.")
    return numeric


def _parse_csv(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tuple(parsed)


def _parse_key_values(values: Sequence[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not values:
        return parsed
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"metadata requirement {item!r} must use key=value format.")
            key, raw = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("metadata requirement key must be non-empty.")
            parsed[key] = raw.strip()
    return parsed


def _write_artifact_manifest(
    *,
    context: ArtifactVerificationContext,
    report_path: Path,
    output_path: Path,
    payload: Mapping[str, Any],
    max_workers: int = 1,
) -> dict[str, Any]:
    route_comparison = _mapping(payload.get("route_baseline_comparison"))
    text_redline = _mapping(payload.get("text_redline_comparison"))
    config = _mapping(payload.get("config"))
    artifacts: dict[str, str | Path | None] = {
        "external_evidence_baseline_comparison_report": report_path,
        "route_registry": config.get("route_registry_path"),
        "candidate_score_report": config.get("candidate_score_report_path"),
        "text_baseline_report": config.get("text_baseline_report_path"),
        "retrieval_stress_manifest": config.get("retrieval_stress_manifest"),
    }
    route_rows = route_comparison.get("rows")
    if isinstance(route_rows, Sequence) and not isinstance(route_rows, (str, bytes)):
        for idx, row in enumerate(route_rows, start=1):
            if not isinstance(row, Mapping):
                continue
            artifacts[f"route_manifest_{idx}"] = row.get("manifest_path")
    text_runs = text_redline.get("runs")
    decision = _mapping(payload.get("decision"))
    metadata = {
        "runner": "compare_external_evidence_baselines",
        "workflow": payload.get("workflow"),
        "decision_status": decision.get("status"),
        "recommended_route": decision.get("recommended_route"),
        "recommended_route_record": decision.get("recommended_route_record"),
        "route_passed": route_comparison.get("passed"),
        "text_redline_passed": text_redline.get("passed"),
        "text_redline_run_count": (
            len(text_runs)
            if isinstance(text_runs, Sequence) and not isinstance(text_runs, (str, bytes))
            else text_redline.get("run_count")
        ),
    }
    manifest = context.build_artifact_manifest(
        artifacts,
        root=output_path.parent,
        metadata=metadata,
        max_workers=max_workers,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    ).to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verification


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
    route_comparison = _mapping(payload.get("route_baseline_comparison"))
    text_redline = _mapping(payload.get("text_redline_comparison"))
    metadata = {
        "workflow": "compare_external_evidence_baselines",
        "status": decision.get("status"),
        "recommended_route": decision.get("recommended_route"),
        "recommended_route_record": decision.get("recommended_route_record"),
        "blocking_reasons": tuple(decision.get("blocking_reasons", ())),
        "route_passed": route_comparison.get("passed"),
        "text_redline_passed": text_redline.get("passed"),
        "text_redline_run_count": text_redline.get("run_count"),
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
            },
        )
    registry.save_json()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    if (
        args.json is None
        and (
            args.artifact_manifest is not None
            or args.verification_report is not None
            or args.registry is not None
        )
    ):
        raise ValueError("--artifact-manifest, --verification-report, and --registry require --json.")
    if args.verification_report is not None and args.artifact_manifest is None:
        raise ValueError("--verification-report requires --artifact-manifest.")
    output_path = None if args.json is None else Path(args.json)
    manifest_path = None if args.artifact_manifest is None else Path(args.artifact_manifest)
    verification_path = None if args.verification_report is None else Path(args.verification_report)
    context = ArtifactVerificationContext()
    text_baseline_signals = (
        _parse_csv(args.text_baseline_signal)
        or tuple(DEFAULT_TEXT_BASELINE_SIGNALS)
    )
    candidate_denied_signals = (
        _parse_csv(args.candidate_denied_signal)
        or tuple(DEFAULT_TEXT_BASELINE_SIGNALS)
    )
    payload = compare_external_evidence_baselines(
        route_registry_path=args.route_registry,
        route_baseline_keys=tuple(args.route_baseline_key or ()),
        require_route_baseline=bool(args.require_route_baseline),
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        min_selected=args.min_selected,
        min_decision_accuracy=args.min_decision_accuracy,
        max_false_supported_rate=args.max_false_supported_rate,
        min_false_refuted_rate=args.min_false_refuted_rate,
        max_verified_false_alarm=args.max_verified_false_alarm,
        min_verified_detection=args.min_verified_detection,
        require_non_oracle_evidence=bool(args.require_non_oracle_evidence),
        require_retrieval_provenance_filter=bool(args.require_retrieval_provenance_filter),
        required_retrieval_source_prefixes=_parse_csv(args.required_retrieval_source_prefix),
        required_retrieval_metadata=_parse_key_values(args.required_retrieval_metadata),
        min_retrieval_filter_score=args.min_retrieval_filter_score,
        require_retrieval_stress_control=bool(args.require_retrieval_stress_control),
        retrieval_stress_manifest=args.retrieval_stress_manifest,
        min_stress_false_supported_rate=args.min_stress_false_supported_rate,
        max_stress_false_refuted_rate=args.max_stress_false_refuted_rate,
        candidate_score_report_path=args.candidate_score_report,
        text_baseline_report_path=args.text_baseline_report,
        candidate_run=args.candidate_run,
        text_run=args.text_run,
        text_baseline_signals=text_baseline_signals,
        candidate_denied_signals=candidate_denied_signals,
        require_text_redline=bool(args.require_text_redline),
        min_text_detection_margin=args.min_text_detection_margin,
        min_text_auroc_margin=args.min_text_auroc_margin,
        min_candidate_detection=args.min_candidate_detection,
        min_candidate_auroc=args.min_candidate_auroc,
        notes=args.note,
    )
    if output_path is not None:
        if manifest_path is not None:
            payload["paths"] = {
                "external_evidence_baseline_comparison_report": str(output_path),
                "artifact_manifest": str(manifest_path),
                "manifest_verification": None
                if verification_path is None
                else str(verification_path),
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote external evidence baseline comparison to {output_path}")
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
            payload["artifact_manifest_summary"] = manifest.get("summary")
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest = _write_artifact_manifest(
                context=context,
                report_path=output_path,
                output_path=manifest_path,
                payload=payload,
                max_workers=args.manifest_fingerprint_workers,
            )
            if verification_path is not None:
                verification = _verify_manifest(
                    context=context,
                    manifest_path=manifest_path,
                    output_path=verification_path,
                    recursive=not args.no_recursive,
                    max_workers=args.manifest_fingerprint_workers,
                )
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
    decision = payload["decision"]
    print(
        "external_evidence_baseline_comparison="
        f"{decision['status']} route={decision.get('recommended_route')} "
        f"record={decision.get('recommended_route_record')}"
    )
    if args.fail_on_blocked and decision["status"] != "promote":
        raise SystemExit(1)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare external-evidence route candidates against answer-echo and text redlines"
    )
    parser.add_argument("--route-registry", default=None, help="local ArtifactRegistry JSON path")
    parser.add_argument("--route-baseline-key", action="append", default=[])
    parser.add_argument("--require-route-baseline", action="store_true")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument(
        "--artifact-manifest",
        default=None,
        help="optional artifact manifest path for the comparison report and inputs",
    )
    parser.add_argument(
        "--verification-report",
        default=None,
        help="optional recursive manifest verification report path",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="optional ArtifactRegistry JSON path to record the comparison report",
    )
    parser.add_argument("--name", default=None, help="registry artifact name")
    parser.add_argument("--version", default=None, help="registry artifact version")
    parser.add_argument(
        "--manifest-fingerprint-workers",
        type=lambda value: _parse_positive_int(
            value,
            flag="--manifest-fingerprint-workers",
        ),
        default=1,
    )
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--min-selected", type=lambda value: _parse_non_negative_int(
        value,
        flag="--min-selected",
    ), default=None)
    parser.add_argument("--min-decision-accuracy", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-decision-accuracy",
    ), default=None)
    parser.add_argument("--max-false-supported-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-false-supported-rate",
    ), default=None)
    parser.add_argument("--min-false-refuted-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-false-refuted-rate",
    ), default=None)
    parser.add_argument("--max-verified-false-alarm", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-verified-false-alarm",
    ), default=None)
    parser.add_argument("--min-verified-detection", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-verified-detection",
    ), default=None)
    parser.add_argument("--require-non-oracle-evidence", action="store_true")
    parser.add_argument("--require-retrieval-provenance-filter", action="store_true")
    parser.add_argument("--required-retrieval-source-prefix", action="append", default=None)
    parser.add_argument("--required-retrieval-metadata", action="append", default=None)
    parser.add_argument("--min-retrieval-filter-score", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-retrieval-filter-score",
    ), default=None)
    parser.add_argument("--require-retrieval-stress-control", action="store_true")
    parser.add_argument("--retrieval-stress-manifest", default=None)
    parser.add_argument("--min-stress-false-supported-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-stress-false-supported-rate",
    ), default=None)
    parser.add_argument("--max-stress-false-refuted-rate", type=lambda value: _parse_non_negative_float(
        value,
        flag="--max-stress-false-refuted-rate",
    ), default=None)
    parser.add_argument("--candidate-score-report", default=None)
    parser.add_argument("--text-baseline-report", default=None)
    parser.add_argument("--candidate-run", default=None)
    parser.add_argument("--text-run", default=None)
    parser.add_argument("--text-baseline-signal", action="append", default=None)
    parser.add_argument("--candidate-denied-signal", action="append", default=None)
    parser.add_argument("--require-text-redline", action="store_true")
    parser.add_argument("--min-text-detection-margin", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-text-detection-margin",
    ), default=0.0)
    parser.add_argument("--min-text-auroc-margin", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-text-auroc-margin",
    ), default=None)
    parser.add_argument("--min-candidate-detection", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-candidate-detection",
    ), default=None)
    parser.add_argument("--min-candidate-auroc", type=lambda value: _parse_non_negative_float(
        value,
        flag="--min-candidate-auroc",
    ), default=None)
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
