"""Build a deployment-oriented runtime recommendation from benchmark reports.

The cache-profile matrix and worker-count sweep reports are intentionally
experiment shaped. This helper turns their promotion decisions into one compact
machine-readable recommendation: layer, batch size, capture mode, token budget,
prefix-KV mode, worker count, and optional INSIDE sampling configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_runtime_recommendation(
    matrix_report: Mapping[str, Any],
    *,
    worker_sweep_report: Mapping[str, Any] | None = None,
    inside_sampling_report: Mapping[str, Any] | None = None,
    matrix_report_path: str | Path | None = None,
    worker_sweep_report_path: str | Path | None = None,
    inside_sampling_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return one runtime recommendation from matrix and optional worker evidence."""
    matrix_decision = _mapping(matrix_report.get("matrix_decision"))
    worker_decision = _mapping(
        None if worker_sweep_report is None else worker_sweep_report.get("worker_sweep_decision")
    )
    inside_sampling_decision = _inside_sampling_decision(
        inside_sampling_report,
        inside_sampling_report_path=inside_sampling_report_path,
    )
    matrix_status = str(matrix_decision.get("status") or "missing")
    worker_status = None if worker_sweep_report is None else str(worker_decision.get("status") or "missing")
    inside_sampling_status = (
        None
        if inside_sampling_report is None
        else str(inside_sampling_decision.get("status") or "missing")
    )
    status = _combined_status(matrix_status, worker_status, inside_sampling_status)
    recommended = None
    missing_promoted_runtime_cell = False
    if status == "promote":
        recommended = _recommendation(
            matrix_report,
            matrix_decision=matrix_decision,
            worker_sweep_report=worker_sweep_report,
            worker_decision=worker_decision,
            inside_sampling_decision=inside_sampling_decision,
            matrix_report_path=matrix_report_path,
        )
        if recommended is None:
            status = "no_candidate"
            missing_promoted_runtime_cell = True
    blocking_reasons = _blocking_reasons(
        matrix_decision=matrix_decision,
        worker_decision=worker_decision,
        worker_sweep_report=worker_sweep_report,
        inside_sampling_decision=inside_sampling_decision,
        inside_sampling_report=inside_sampling_report,
    )
    if missing_promoted_runtime_cell:
        blocking_reasons.append("matrix: promoted matrix did not include a recommended runtime cell")
    evidence = _evidence(
        matrix_report,
        matrix_decision=matrix_decision,
        worker_sweep_report=worker_sweep_report,
        worker_decision=worker_decision,
        inside_sampling_decision=inside_sampling_decision,
        inside_sampling_report=inside_sampling_report,
        matrix_report_path=matrix_report_path,
        worker_sweep_report_path=worker_sweep_report_path,
        inside_sampling_report_path=inside_sampling_report_path,
    )
    report = {
        "schema_version": 1,
        "workflow": "runtime_config_recommendation",
        "status": status,
        "recommendation": recommended,
        "evidence": evidence,
        "blocking_reasons": blocking_reasons,
    }
    if recommended is not None:
        report["benchmark_flags"] = _benchmark_flags(recommended, matrix_report)
    return report


def _recommendation(
    matrix_report: Mapping[str, Any],
    *,
    matrix_decision: Mapping[str, Any],
    worker_sweep_report: Mapping[str, Any] | None,
    worker_decision: Mapping[str, Any],
    inside_sampling_decision: Mapping[str, Any],
    matrix_report_path: str | Path | None,
) -> dict[str, Any] | None:
    matrix_recommended = _recommended_runtime_row(matrix_report, matrix_decision)
    if not matrix_recommended or any(
        matrix_recommended.get(key) is None
        for key in ("layer", "batch_size", "hidden_state_capture")
    ):
        return None
    worker_count = _recommended_worker_count(
        matrix_report,
        worker_sweep_report=worker_sweep_report,
        worker_decision=worker_decision,
    )
    quality = _quality_signal_summary(
        matrix_report,
        matrix_recommended,
        matrix_report_path=matrix_report_path,
    )
    totals = _runtime_totals(matrix_recommended)
    recommendation = {
        "cell_id": matrix_recommended.get("id") or matrix_decision.get("recommended_cell"),
        "layer": matrix_recommended.get("layer"),
        "batch_size": matrix_recommended.get("batch_size"),
        "hidden_state_capture": matrix_recommended.get("hidden_state_capture"),
        "max_batch_tokens": int(matrix_recommended.get("max_batch_tokens") or 0),
        "prefix_kv_cache": bool(matrix_recommended.get("prefix_kv_cache", False)),
        "max_workers": worker_count,
        "recommendation_metric": matrix_decision.get("recommendation_metric"),
        "uncached_total_seconds": totals["uncached_total_seconds"],
        "cached_total_seconds": totals["cached_total_seconds"],
        "cache_only_total_seconds": totals["cache_only_total_seconds"],
        "uncached_forced_answer_forward_seconds": totals["uncached_forced_answer_forward_seconds"],
        "truth_proj_auroc": matrix_recommended.get("truth_proj_auroc"),
        "quality_signals": quality["signals"],
        "best_quality_signal": quality["best"],
    }
    inside_sampling = _mapping(inside_sampling_decision.get("recommended"))
    if inside_sampling:
        recommendation["inside_sampling"] = inside_sampling
    return recommendation


def _runtime_totals(matrix_recommended: Mapping[str, Any]) -> dict[str, Any]:
    totals = _mapping(_mapping(matrix_recommended.get("summary")).get("totals"))
    uncached = _mapping(totals.get("uncached"))
    cached = _mapping(totals.get("cached"))
    cache_only = _mapping(totals.get("cache_only"))
    return {
        "uncached_total_seconds": _first_present(
            matrix_recommended.get("uncached_total_seconds"),
            uncached.get("total_seconds"),
        ),
        "cached_total_seconds": _first_present(
            matrix_recommended.get("cached_total_seconds"),
            cached.get("total_seconds"),
        ),
        "cache_only_total_seconds": _first_present(
            matrix_recommended.get("cache_only_total_seconds"),
            cache_only.get("total_seconds"),
        ),
        "uncached_forced_answer_forward_seconds": _first_present(
            matrix_recommended.get("uncached_forced_answer_forward_seconds"),
            uncached.get("forced_answer_forward_seconds"),
        ),
    }


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _quality_signal_summary(
    matrix_report: Mapping[str, Any],
    matrix_recommended: Mapping[str, Any],
    *,
    matrix_report_path: str | Path | None = None,
) -> dict[str, Any]:
    cell_id = matrix_recommended.get("id")
    cell = _find_by_id(matrix_report.get("cells"), cell_id) if cell_id is not None else {}
    signals, source = _quality_signals_from_cell(cell, matrix_report_path=matrix_report_path)
    if not signals:
        truth_proj = _float_or_none(matrix_recommended.get("truth_proj_auroc"))
        if truth_proj is not None:
            signals = {"truth_proj": truth_proj}
            source = "leaderboard"
    signals = {name: signals[name] for name in sorted(signals)}
    best = _best_quality_signal(signals)
    return {
        "signals": signals,
        "best": best,
        "source": source,
        "count": len(signals),
    }


def _quality_signals_from_cell(
    cell: Mapping[str, Any],
    *,
    matrix_report_path: str | Path | None,
) -> tuple[dict[str, float], str | None]:
    triplet = _mapping(cell.get("triplet"))
    results = _mapping(triplet.get("results"))
    for run_name in ("cache_only", "cached", "uncached"):
        result_path = results.get(run_name)
        if not result_path:
            continue
        signals, source = _quality_signals_from_result_path(
            Path(str(result_path)),
            base_dir=None if matrix_report_path is None else Path(str(matrix_report_path)).parent,
        )
        if signals:
            return signals, source
    summary_signals = _finite_float_mapping(_mapping(_mapping(cell.get("summary")).get("quality_signals")))
    if summary_signals:
        return summary_signals, "matrix_cell_summary"
    return {}, None


def _quality_signals_from_result_path(
    path: Path,
    *,
    base_dir: Path | None,
) -> tuple[dict[str, float], str | None]:
    candidates = [path]
    if base_dir is not None and not path.is_absolute():
        candidates.append(base_dir / path)
    for candidate in candidates:
        try:
            payload = _load_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        signals = _finite_float_mapping(_mapping(payload.get("auroc")))
        if signals:
            return signals, str(candidate)
    return {}, None


def _finite_float_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    signals = {}
    for key, value in values.items():
        numeric = _float_or_none(value)
        if numeric is not None:
            signals[str(key)] = numeric
    return signals


def _best_quality_signal(signals: Mapping[str, float]) -> dict[str, Any] | None:
    if not signals:
        return None
    name, auroc = sorted(signals.items(), key=lambda item: (-item[1], item[0]))[0]
    return {"name": name, "auroc": auroc}


def _recommended_runtime_row(
    matrix_report: Mapping[str, Any],
    matrix_decision: Mapping[str, Any],
) -> dict[str, Any]:
    recommended = _mapping(matrix_decision.get("recommended"))
    cell_id = recommended.get("id") or matrix_decision.get("recommended_cell")
    merged: dict[str, Any] = {}
    if cell_id is not None:
        merged.update(_find_by_id(matrix_report.get("cells"), cell_id))
        merged.update(_find_by_id(matrix_report.get("leaderboard"), cell_id))
    merged.update(recommended)
    if cell_id is not None and merged.get("id") is None:
        merged["id"] = cell_id
    return merged


def _find_by_id(values: Any, cell_id: Any) -> dict[str, Any]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        return {}
    for value in values:
        item = _mapping(value)
        if item.get("id") == cell_id:
            return item
    return {}


def _recommended_worker_count(
    matrix_report: Mapping[str, Any],
    *,
    worker_sweep_report: Mapping[str, Any] | None,
    worker_decision: Mapping[str, Any],
) -> int | None:
    worker_count = worker_decision.get("recommended_worker_count")
    if isinstance(worker_count, int) and not isinstance(worker_count, bool):
        return worker_count
    config = _mapping(matrix_report.get("config"))
    configured = config.get("max_workers")
    if isinstance(configured, int) and not isinstance(configured, bool):
        return configured
    if worker_sweep_report is not None:
        worker_reports = worker_sweep_report.get("worker_reports")
        if isinstance(worker_reports, Sequence) and not isinstance(worker_reports, str):
            for report in worker_reports:
                report_map = _mapping(report)
                value = report_map.get("worker_count")
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
    return None


def _inside_sampling_decision(
    inside_sampling_report: Mapping[str, Any] | None,
    *,
    inside_sampling_report_path: str | Path | None,
) -> dict[str, Any]:
    if inside_sampling_report is None:
        return {}
    gate = _mapping(inside_sampling_report.get("sample_efficiency_gate"))
    if gate.get("passed") is not True:
        return {
            "status": "blocked",
            "blocking_reasons": _inside_sampling_gate_failures(gate),
            "sample_efficiency_gate": gate,
        }
    recommendation = _mapping(inside_sampling_report.get("recommendation"))
    recommended_run = recommendation.get("recommended_run")
    if not recommended_run:
        return {
            "status": "no_candidate",
            "blocking_reasons": ["inside_sampling: promoted report did not include a recommended run"],
            "sample_efficiency_gate": gate,
        }
    row = _inside_sampling_report_row(inside_sampling_report, str(recommended_run))
    if not row:
        return {
            "status": "no_candidate",
            "recommended_run": str(recommended_run),
            "blocking_reasons": [
                f"inside_sampling: recommended run {recommended_run!r} was missing from report rows"
            ],
            "sample_efficiency_gate": gate,
        }
    result_payload, result_source = _inside_sampling_result_payload(
        row,
        inside_sampling_report_path=inside_sampling_report_path,
    )
    if not result_payload:
        return {
            "status": "no_candidate",
            "recommended_run": str(recommended_run),
            "blocking_reasons": [
                f"inside_sampling: recommended run {recommended_run!r} did not include a readable result payload"
            ],
            "sample_efficiency_gate": gate,
        }
    result_config = _mapping(result_payload.get("config"))
    inside_sampling = _mapping(result_payload.get("inside_sampling"))
    if not inside_sampling:
        return {
            "status": "no_candidate",
            "recommended_run": str(recommended_run),
            "blocking_reasons": [
                f"inside_sampling: recommended run {recommended_run!r} result did not include inside_sampling"
            ],
            "sample_efficiency_gate": gate,
        }
    return {
        "status": "promote",
        "recommended_run": str(recommended_run),
        "sample_efficiency_gate": gate,
        "recommended": _inside_sampling_recommendation(
            str(recommended_run),
            row=row,
            result_config=result_config,
            inside_sampling=inside_sampling,
            result_source=result_source,
        ),
    }


def _inside_sampling_gate_failures(gate: Mapping[str, Any]) -> list[str]:
    failures = gate.get("failures")
    if not isinstance(failures, Sequence) or isinstance(failures, str) or not failures:
        return ["inside_sampling: sample efficiency gate did not pass"]
    reasons = []
    for failure in failures:
        item = _mapping(failure)
        run = item.get("run", "unknown")
        metric = item.get("metric", "unknown_metric")
        value = item.get("value")
        max_allowed = item.get("max_allowed")
        reasons.append(
            f"inside_sampling: {run} failed {metric} gate "
            f"(value={value!r}, max_allowed={max_allowed!r})"
        )
    return reasons


def _inside_sampling_report_row(
    inside_sampling_report: Mapping[str, Any],
    recommended_run: str,
) -> dict[str, Any]:
    runs = _mapping(inside_sampling_report.get("runs"))
    row = _mapping(runs.get(recommended_run))
    if row:
        return row
    leaderboard = inside_sampling_report.get("leaderboard")
    if isinstance(leaderboard, Sequence) and not isinstance(leaderboard, str):
        for value in leaderboard:
            item = _mapping(value)
            if item.get("name") == recommended_run:
                return item
    return {}


def _inside_sampling_result_payload(
    row: Mapping[str, Any],
    *,
    inside_sampling_report_path: str | Path | None,
) -> tuple[dict[str, Any], str | None]:
    result_path = row.get("result_path")
    if not result_path:
        return {}, None
    path = Path(str(result_path))
    candidates = [path]
    if inside_sampling_report_path is not None and not path.is_absolute():
        candidates.append(Path(str(inside_sampling_report_path)).parent / path)
    for candidate in candidates:
        try:
            return _load_json(candidate), str(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return {}, None


def _inside_sampling_recommendation(
    recommended_run: str,
    *,
    row: Mapping[str, Any],
    result_config: Mapping[str, Any],
    inside_sampling: Mapping[str, Any],
    result_source: str | None,
) -> dict[str, Any]:
    return {
        "recommended_run": recommended_run,
        "result_path": result_source,
        "mode": inside_sampling.get("mode"),
        "adaptive": bool(
            _first_present(result_config.get("inside_adaptive_sampling"), inside_sampling.get("adaptive"))
        ),
        "selfcheck_early_stop": bool(
            _first_present(
                result_config.get("inside_selfcheck_early_stop"),
                inside_sampling.get("selfcheck_early_stop"),
            )
        ),
        "inside_samples": _int_or_none(
            _first_present(result_config.get("inside_samples"), inside_sampling.get("max_samples"))
        ),
        "inside_batch_size": _int_or_none(result_config.get("inside_batch_size")),
        "inside_max_new_tokens": _int_or_none(result_config.get("inside_max_new_tokens")),
        "inside_temperature": _float_or_none(result_config.get("inside_temperature")),
        "inside_top_p": _float_or_none(result_config.get("inside_top_p")),
        "inside_pooling": _first_present(result_config.get("inside_pooling"), "last"),
        "inside_embedding_threshold": _float_or_none(
            _first_present(
                result_config.get("inside_embedding_threshold"),
                inside_sampling.get("embedding_similarity_threshold"),
            )
        ),
        "inside_min_samples": _int_or_none(
            _first_present(result_config.get("inside_min_samples"), inside_sampling.get("min_samples"))
        ),
        "inside_sample_step": _int_or_none(
            _first_present(result_config.get("inside_sample_step"), inside_sampling.get("sample_step"))
        ),
        "inside_stability_delta": _float_or_none(
            _first_present(result_config.get("inside_stability_delta"), inside_sampling.get("stability_delta"))
        ),
        "inside_selfcheck_min_overlap": _float_or_none(
            _first_present(
                result_config.get("inside_selfcheck_min_overlap"),
                inside_sampling.get("selfcheck_min_overlap"),
            )
        ),
        "inside_selfcheck_support_threshold": _float_or_none(
            _first_present(
                result_config.get("inside_selfcheck_support_threshold"),
                inside_sampling.get("selfcheck_support_threshold"),
            )
        ),
        "inside_selfcheck_refute_threshold": _float_or_none(
            _first_present(
                result_config.get("inside_selfcheck_refute_threshold"),
                inside_sampling.get("selfcheck_refute_threshold"),
            )
        ),
        "inside_trigger_signal": result_config.get("inside_trigger_signal"),
        "inside_trigger_threshold": _float_or_none(result_config.get("inside_trigger_threshold")),
        "inside_trigger_top_fraction": _float_or_none(result_config.get("inside_trigger_top_fraction")),
        "total_generated_samples": _int_or_none(
            _first_present(row.get("total_generated_samples"), inside_sampling.get("total_generated_samples"))
        ),
        "sample_count_ratio_to_baseline": _float_or_none(row.get("sample_count_ratio_to_baseline")),
        "inside_generation_seconds": _float_or_none(row.get("inside_generation_seconds")),
        "inside_generation_seconds_ratio_to_baseline": _float_or_none(
            row.get("inside_generation_seconds_ratio_to_baseline")
        ),
        "stop_reason_counts": dict(row.get("stop_reason_counts", {}))
        if isinstance(row.get("stop_reason_counts", {}), Mapping)
        else {},
    }


def _benchmark_flags(
    recommendation: Mapping[str, Any],
    matrix_report: Mapping[str, Any],
) -> dict[str, list[str]]:
    layer = str(recommendation.get("layer"))
    batch_size = str(recommendation.get("batch_size"))
    capture = str(recommendation.get("hidden_state_capture"))
    max_batch_tokens = int(recommendation.get("max_batch_tokens") or 0)
    max_workers = recommendation.get("max_workers")
    prefix_kv_cache = bool(recommendation.get("prefix_kv_cache", False))
    length_bucketed = bool(_mapping(matrix_report.get("config")).get("length_bucketed_batches", False))

    eval_flags = ["--layer", layer, "--batch-size", batch_size, "--hidden-state-capture", capture]
    if max_batch_tokens > 0:
        eval_flags.extend(["--max-batch-tokens", str(max_batch_tokens)])
    if prefix_kv_cache:
        eval_flags.append("--prefix-kv-cache")
    if length_bucketed:
        eval_flags.append("--length-bucketed-batches")
    eval_flags.extend(_inside_sampling_eval_flags(_mapping(recommendation.get("inside_sampling"))))

    matrix_flags = ["--layers", layer, "--batch-sizes", batch_size, "--hidden-state-captures", capture]
    if max_batch_tokens > 0:
        matrix_flags.extend(["--max-batch-tokens", str(max_batch_tokens)])
    if prefix_kv_cache:
        matrix_flags.append("--prefix-kv-cache")
    if isinstance(max_workers, int) and max_workers >= 1:
        matrix_flags.extend(["--max-workers", str(max_workers)])

    readiness_flags = list(matrix_flags)
    flags = {
        "eval_truthfulqa": eval_flags,
        "run_cache_profile_matrix": matrix_flags,
        "run_adapter_readiness_workflow": readiness_flags,
    }
    inside_sampling = _mapping(recommendation.get("inside_sampling"))
    if inside_sampling:
        flags["run_inside_sampling_profile"] = _inside_sampling_profile_flags(inside_sampling)
    return flags


def _inside_sampling_eval_flags(inside_sampling: Mapping[str, Any]) -> list[str]:
    if not inside_sampling:
        return []
    flags = []
    _extend_flag(flags, "--inside-samples", inside_sampling.get("inside_samples"))
    _extend_flag(flags, "--inside-batch-size", inside_sampling.get("inside_batch_size"))
    _extend_flag(flags, "--inside-max-new-tokens", inside_sampling.get("inside_max_new_tokens"))
    _extend_flag(flags, "--inside-temperature", inside_sampling.get("inside_temperature"))
    _extend_flag(flags, "--inside-top-p", inside_sampling.get("inside_top_p"))
    _extend_flag(flags, "--inside-pooling", inside_sampling.get("inside_pooling"))
    _extend_flag(flags, "--inside-embedding-threshold", inside_sampling.get("inside_embedding_threshold"))
    if inside_sampling.get("adaptive"):
        flags.append("--inside-adaptive-sampling")
        _extend_flag(flags, "--inside-min-samples", inside_sampling.get("inside_min_samples"))
        _extend_flag(flags, "--inside-sample-step", inside_sampling.get("inside_sample_step"))
        _extend_flag(flags, "--inside-stability-delta", inside_sampling.get("inside_stability_delta"))
    if inside_sampling.get("selfcheck_early_stop"):
        flags.append("--inside-selfcheck-early-stop")
        _extend_flag(flags, "--inside-selfcheck-min-overlap", inside_sampling.get("inside_selfcheck_min_overlap"))
        _extend_flag(
            flags,
            "--inside-selfcheck-support-threshold",
            inside_sampling.get("inside_selfcheck_support_threshold"),
        )
        _extend_flag(
            flags,
            "--inside-selfcheck-refute-threshold",
            inside_sampling.get("inside_selfcheck_refute_threshold"),
        )
    _extend_flag(flags, "--inside-trigger-signal", inside_sampling.get("inside_trigger_signal"))
    _extend_flag(flags, "--inside-trigger-threshold", inside_sampling.get("inside_trigger_threshold"))
    _extend_flag(flags, "--inside-trigger-top-fraction", inside_sampling.get("inside_trigger_top_fraction"))
    return flags


def _inside_sampling_profile_flags(inside_sampling: Mapping[str, Any]) -> list[str]:
    flags = []
    _extend_flag(flags, "--inside-samples", inside_sampling.get("inside_samples"))
    _extend_flag(flags, "--inside-batch-size", inside_sampling.get("inside_batch_size"))
    _extend_flag(flags, "--inside-max-new-tokens", inside_sampling.get("inside_max_new_tokens"))
    _extend_flag(flags, "--inside-temperature", inside_sampling.get("inside_temperature"))
    _extend_flag(flags, "--inside-top-p", inside_sampling.get("inside_top_p"))
    _extend_flag(flags, "--inside-pooling", inside_sampling.get("inside_pooling"))
    _extend_flag(flags, "--inside-embedding-threshold", inside_sampling.get("inside_embedding_threshold"))
    _extend_flag(flags, "--inside-min-samples", inside_sampling.get("inside_min_samples"))
    _extend_flag(flags, "--inside-sample-step", inside_sampling.get("inside_sample_step"))
    _extend_flag(flags, "--inside-stability-delta", inside_sampling.get("inside_stability_delta"))
    _extend_flag(flags, "--inside-selfcheck-min-overlap", inside_sampling.get("inside_selfcheck_min_overlap"))
    _extend_flag(
        flags,
        "--inside-selfcheck-support-threshold",
        inside_sampling.get("inside_selfcheck_support_threshold"),
    )
    _extend_flag(
        flags,
        "--inside-selfcheck-refute-threshold",
        inside_sampling.get("inside_selfcheck_refute_threshold"),
    )
    _extend_flag(flags, "--inside-trigger-signal", inside_sampling.get("inside_trigger_signal"))
    _extend_flag(flags, "--inside-trigger-threshold", inside_sampling.get("inside_trigger_threshold"))
    _extend_flag(flags, "--inside-trigger-top-fraction", inside_sampling.get("inside_trigger_top_fraction"))
    if inside_sampling.get("recommended_run"):
        flags.extend(["--runs", str(inside_sampling["recommended_run"])])
    return flags


def _extend_flag(flags: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    flags.extend([flag, str(value)])


def _evidence(
    matrix_report: Mapping[str, Any],
    *,
    matrix_decision: Mapping[str, Any],
    worker_sweep_report: Mapping[str, Any] | None,
    worker_decision: Mapping[str, Any],
    inside_sampling_report: Mapping[str, Any] | None,
    inside_sampling_decision: Mapping[str, Any],
    matrix_report_path: str | Path | None,
    worker_sweep_report_path: str | Path | None,
    inside_sampling_report_path: str | Path | None,
) -> dict[str, Any]:
    matrix_recommended = _recommended_runtime_row(matrix_report, matrix_decision)
    worker_recommended = _mapping(worker_decision.get("recommended"))
    prefix_comparison = _prefix_comparison_for_recommendation(matrix_report, matrix_recommended)
    quality = _quality_signal_summary(
        matrix_report,
        matrix_recommended,
        matrix_report_path=matrix_report_path,
    )
    config = _mapping(matrix_report.get("config"))
    evidence = {
        "matrix_report": None if matrix_report_path is None else str(matrix_report_path),
        "matrix_status": matrix_decision.get("status"),
        "matrix_recommended_cell": matrix_decision.get("recommended_cell"),
        "matrix_recommendation_metric": matrix_decision.get("recommendation_metric"),
        "matrix_candidate_count": matrix_decision.get("candidate_count"),
        "matrix_checked_cell_count": matrix_decision.get("checked_cell_count"),
        "matrix_leaderboard_sort_metric": matrix_report.get("leaderboard_sort_metric"),
        "matrix_wall_clock_seconds": _mapping(matrix_report.get("execution")).get("wall_clock_seconds"),
        "configured_matrix_workers": config.get("max_workers"),
        "length_bucketed_batches": config.get("length_bucketed_batches"),
        "prefix_kv_comparison": prefix_comparison,
        "quality_signal_source": quality["source"],
        "quality_signal_count": quality["count"],
        "worker_sweep_report": None if worker_sweep_report_path is None else str(worker_sweep_report_path),
        "worker_sweep_status": None if worker_sweep_report is None else worker_decision.get("status"),
        "worker_recommended_worker_count": worker_decision.get("recommended_worker_count"),
        "worker_recommended_wall_clock_seconds": worker_recommended.get("wall_clock_seconds"),
        "worker_matrix_report_matches": _worker_matrix_report_matches(
            worker_recommended=worker_recommended,
            matrix_report_path=matrix_report_path,
        ),
        "inside_sampling_report": None
        if inside_sampling_report_path is None
        else str(inside_sampling_report_path),
        "inside_sampling_status": None
        if inside_sampling_report is None
        else inside_sampling_decision.get("status"),
        "inside_sampling_recommended_run": inside_sampling_decision.get("recommended_run"),
        "inside_sampling_baseline": None
        if inside_sampling_report is None
        else inside_sampling_report.get("baseline"),
        "inside_sampling_gate_passed": None
        if inside_sampling_report is None
        else _mapping(inside_sampling_decision.get("sample_efficiency_gate")).get("passed"),
    }
    return evidence


def _worker_matrix_report_matches(
    *,
    worker_recommended: Mapping[str, Any],
    matrix_report_path: str | Path | None,
) -> bool | None:
    worker_matrix_report = worker_recommended.get("matrix_report")
    if worker_matrix_report is None or matrix_report_path is None:
        return None
    try:
        return Path(str(worker_matrix_report)).resolve() == Path(str(matrix_report_path)).resolve()
    except OSError:
        return str(worker_matrix_report) == str(matrix_report_path)


def _prefix_comparison_for_recommendation(
    matrix_report: Mapping[str, Any],
    matrix_recommended: Mapping[str, Any],
) -> dict[str, Any] | None:
    comparisons = matrix_report.get("prefix_kv_comparisons")
    if not isinstance(comparisons, Sequence) or isinstance(comparisons, str):
        return None
    layer = matrix_recommended.get("layer")
    batch_size = matrix_recommended.get("batch_size")
    capture = matrix_recommended.get("hidden_state_capture")
    for comparison in comparisons:
        item = _mapping(comparison)
        if (
            item.get("layer") == layer
            and item.get("batch_size") == batch_size
            and item.get("hidden_state_capture") == capture
        ):
            return dict(item)
    return None


def _blocking_reasons(
    *,
    matrix_decision: Mapping[str, Any],
    worker_decision: Mapping[str, Any],
    worker_sweep_report: Mapping[str, Any] | None,
    inside_sampling_decision: Mapping[str, Any],
    inside_sampling_report: Mapping[str, Any] | None,
) -> list[str]:
    reasons = []
    for reason in matrix_decision.get("blocking_reasons") or ():
        reasons.append(f"matrix: {reason}")
    if worker_sweep_report is not None:
        for reason in worker_decision.get("blocking_reasons") or ():
            reasons.append(f"worker_sweep: {reason}")
    if inside_sampling_report is not None:
        for reason in inside_sampling_decision.get("blocking_reasons") or ():
            reasons.append(str(reason))
    return reasons


def _combined_status(*statuses: str | None) -> str:
    statuses = [status for status in statuses if status is not None]
    if any(status == "blocked" for status in statuses):
        return "blocked"
    if any(status == "dry_run" for status in statuses):
        return "needs_evidence"
    if any(status == "no_candidate" for status in statuses):
        return "no_candidate"
    if all(status == "promote" for status in statuses):
        return "promote"
    return "unknown"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        numeric = float(value)
    else:
        try:
            numeric = float(str(value))
        except (TypeError, ValueError):
            return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object.")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    matrix_report_path = Path(args.matrix_report)
    worker_sweep_report_path = Path(args.worker_sweep_report) if args.worker_sweep_report else None
    inside_sampling_report_path = Path(args.inside_sampling_report) if args.inside_sampling_report else None
    report = build_runtime_recommendation(
        _load_json(matrix_report_path),
        worker_sweep_report=None if worker_sweep_report_path is None else _load_json(worker_sweep_report_path),
        inside_sampling_report=(
            None if inside_sampling_report_path is None else _load_json(inside_sampling_report_path)
        ),
        matrix_report_path=matrix_report_path,
        worker_sweep_report_path=worker_sweep_report_path,
        inside_sampling_report_path=inside_sampling_report_path,
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output, end="")
    if args.fail_on_blocked and report["status"] != "promote":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a runtime configuration recommendation from benchmark reports"
    )
    parser.add_argument("--matrix-report", required=True,
                        help="cache-profile-matrix-report.json produced by run_cache_profile_matrix.py")
    parser.add_argument("--worker-sweep-report", default=None,
                        help="optional cache-worker-sweep-report.json produced by run_cache_worker_sweep.py")
    parser.add_argument("--inside-sampling-report", default=None,
                        help="optional inside-sampling-profile-comparison.json produced by "
                             "run_inside_sampling_profile.py")
    parser.add_argument("--output", default=None,
                        help="optional path to write the recommendation JSON")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the combined recommendation status is promote")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
