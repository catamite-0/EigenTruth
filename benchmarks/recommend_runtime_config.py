"""Build a deployment-oriented runtime recommendation from benchmark reports.

The cache-profile matrix and worker-count sweep reports are intentionally
experiment shaped. This helper turns their promotion decisions into one compact
machine-readable recommendation: layer, batch size, capture mode, token budget,
prefix-KV mode, worker count, and optional INSIDE sampling / trigger-budget
configuration.
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

INSIDE_TRIGGER_BUDGET_POLICIES = (
    "quality_balanced",
    "cost_first",
    "quality_first",
)
INSIDE_QUALITY_METRIC_PRIORITY = (
    "inside_semantic_entropy",
    "inside_embedding_entropy",
    "inside_eigenscore",
)
CACHE_TUNING_THRESHOLDS = {
    "low_shard_cache_hit_rate": 0.50,
    "high_cross_shard_read_rate": 0.25,
    "low_records_per_read": 2.0,
}


def build_runtime_recommendation(
    matrix_report: Mapping[str, Any],
    *,
    worker_sweep_report: Mapping[str, Any] | None = None,
    inside_sampling_report: Mapping[str, Any] | None = None,
    inside_trigger_budget_sweep_report: Mapping[str, Any] | None = None,
    inside_trigger_budget_policy: str = "quality_balanced",
    matrix_report_path: str | Path | None = None,
    worker_sweep_report_path: str | Path | None = None,
    inside_sampling_report_path: str | Path | None = None,
    inside_trigger_budget_sweep_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return one runtime recommendation from matrix and optional worker evidence."""
    inside_trigger_budget_policy = _normalize_inside_trigger_budget_policy(inside_trigger_budget_policy)
    matrix_decision = _mapping(matrix_report.get("matrix_decision"))
    worker_decision = _mapping(
        None if worker_sweep_report is None else worker_sweep_report.get("worker_sweep_decision")
    )
    inside_sampling_decision = _inside_sampling_decision(
        inside_sampling_report,
        inside_sampling_report_path=inside_sampling_report_path,
    )
    inside_trigger_budget_sweep_decision = _inside_trigger_budget_sweep_decision(
        inside_trigger_budget_sweep_report,
        policy=inside_trigger_budget_policy,
        inside_trigger_budget_sweep_report_path=inside_trigger_budget_sweep_report_path,
    )
    matrix_status = str(matrix_decision.get("status") or "missing")
    worker_status = None if worker_sweep_report is None else str(worker_decision.get("status") or "missing")
    inside_sampling_status = (
        None
        if inside_sampling_report is None
        else str(inside_sampling_decision.get("status") or "missing")
    )
    inside_trigger_budget_sweep_status = (
        None
        if inside_trigger_budget_sweep_report is None
        else str(inside_trigger_budget_sweep_decision.get("status") or "missing")
    )
    status = _combined_status(
        matrix_status,
        worker_status,
        inside_sampling_status,
        inside_trigger_budget_sweep_status,
    )
    recommended = None
    missing_promoted_runtime_cell = False
    if status == "promote":
        recommended = _recommendation(
            matrix_report,
            matrix_decision=matrix_decision,
            worker_sweep_report=worker_sweep_report,
            worker_decision=worker_decision,
            inside_sampling_decision=inside_sampling_decision,
            inside_trigger_budget_sweep_decision=inside_trigger_budget_sweep_decision,
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
        inside_trigger_budget_sweep_decision=inside_trigger_budget_sweep_decision,
        inside_trigger_budget_sweep_report=inside_trigger_budget_sweep_report,
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
        inside_trigger_budget_sweep_decision=inside_trigger_budget_sweep_decision,
        inside_trigger_budget_sweep_report=inside_trigger_budget_sweep_report,
        matrix_report_path=matrix_report_path,
        worker_sweep_report_path=worker_sweep_report_path,
        inside_sampling_report_path=inside_sampling_report_path,
        inside_trigger_budget_sweep_report_path=inside_trigger_budget_sweep_report_path,
        inside_trigger_budget_policy=inside_trigger_budget_policy,
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
    inside_trigger_budget_sweep_decision: Mapping[str, Any],
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
    matrix_config = _mapping(matrix_report.get("config"))
    eval_reps_shard_read_cache_size = _first_present(
        _int_or_none(matrix_recommended.get("eval_reps_shard_read_cache_size")),
        _int_or_none(matrix_config.get("eval_reps_shard_read_cache_size")),
    )
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
    if eval_reps_shard_read_cache_size is not None:
        recommendation["eval_reps_shard_read_cache_size"] = eval_reps_shard_read_cache_size
    cache_tuning = _cache_tuning_recommendation(matrix_recommended, matrix_report)
    if cache_tuning["status"] != "no_data":
        recommendation["cache_tuning"] = cache_tuning
    inside_sampling = _mapping(inside_sampling_decision.get("recommended"))
    trigger_sampling = _mapping(inside_trigger_budget_sweep_decision.get("inside_sampling"))
    if trigger_sampling:
        inside_sampling.update(trigger_sampling)
    if inside_sampling:
        recommendation["inside_sampling"] = inside_sampling
    trigger_budget = _mapping(inside_trigger_budget_sweep_decision.get("recommended"))
    if trigger_budget:
        recommendation["inside_trigger_budget_sweep"] = trigger_budget
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


def _cache_tuning_recommendation(
    matrix_recommended: Mapping[str, Any],
    matrix_report: Mapping[str, Any],
) -> dict[str, Any]:
    source_run, metrics = _recommended_cache_efficiency(matrix_recommended)
    if not metrics:
        return {
            "status": "no_data",
            "source_run": None,
            "metrics": {},
            "thresholds": dict(CACHE_TUNING_THRESHOLDS),
            "recommendations": [],
        }

    matrix_config = _mapping(matrix_report.get("config"))
    configured_read_cache_sizes = matrix_config.get("eval_reps_shard_read_cache_sizes")
    if isinstance(configured_read_cache_sizes, Sequence) and not isinstance(configured_read_cache_sizes, str):
        read_cache_sweep_sizes = tuple(
            value for value in (_int_or_none(item) for item in configured_read_cache_sizes) if value is not None
        )
    else:
        read_cache_sweep_sizes = ()
    read_cache_was_swept = len(set(read_cache_sweep_sizes)) > 1
    shard_hit_rate = _float_or_none(metrics.get("eval_reps_reader.shard_cache_hit_rate"))
    cross_shard_rate = _float_or_none(metrics.get("eval_reps_reader.cross_shard_read_rate"))
    records_per_read = _float_or_none(metrics.get("eval_reps_reader.records_per_read"))
    shard_count = _int_or_none(metrics.get("eval_reps_reader.shard_count"))
    shard_capacity = _int_or_none(metrics.get("eval_reps_reader.shard_cache_capacity"))
    read_cache_size = (
        _int_or_none(matrix_recommended.get("eval_reps_shard_read_cache_size"))
        or _int_or_none(matrix_config.get("eval_reps_shard_read_cache_size"))
        or shard_capacity
        or 2
    )
    shard_size = _int_or_none(matrix_config.get("eval_reps_cache_shard_size"))
    batch_size = _int_or_none(matrix_recommended.get("batch_size"))
    max_batch_tokens = _int_or_none(matrix_recommended.get("max_batch_tokens"))

    recommendations: list[dict[str, Any]] = []
    if (
        shard_hit_rate is not None
        and shard_hit_rate < CACHE_TUNING_THRESHOLDS["low_shard_cache_hit_rate"]
        and shard_count is not None
        and shard_count > read_cache_size
        and not read_cache_was_swept
    ):
        suggested = min(shard_count, max(read_cache_size + 1, read_cache_size * 2))
        recommendations.append({
            "action": "increase_eval_reps_shard_read_cache_size",
            "reason": "eval-reps shard cache hit rate is low while more shards exist than the read cache holds",
            "current": read_cache_size,
            "suggested": suggested,
            "suggested_flags": {
                "eval_truthfulqa": ["--eval-reps-shard-read-cache-size", str(suggested)],
                "run_cache_profile_matrix": ["--eval-reps-shard-read-cache-size", str(suggested)],
            },
        })

    if (
        cross_shard_rate is not None
        and cross_shard_rate > CACHE_TUNING_THRESHOLDS["high_cross_shard_read_rate"]
    ):
        suggested_shard_size = None if shard_size is None else max(shard_size + 1, shard_size * 2)
        recommendations.append({
            "action": "reduce_cross_shard_reads",
            "reason": "many eval-reps read requests span multiple shards",
            "current_eval_reps_cache_shard_size": shard_size,
            "suggested_eval_reps_cache_shard_size": suggested_shard_size,
            "suggested_flags": (
                {}
                if suggested_shard_size is None
                else {
                    "run_cache_profile_matrix": [
                        "--eval-reps-cache-shard-size",
                        str(suggested_shard_size),
                    ]
                }
            ),
        })

    if (
        records_per_read is not None
        and 0.0 < records_per_read < CACHE_TUNING_THRESHOLDS["low_records_per_read"]
    ):
        recommendations.append({
            "action": "increase_records_per_cache_read",
            "reason": "cache reader is serving very small ranges, which increases Python and shard IO overhead",
            "current_records_per_read": records_per_read,
            "current_batch_size": batch_size,
            "current_max_batch_tokens": max_batch_tokens,
            "suggested_next_step": (
                "increase --batch-size or --max-batch-tokens if memory allows, then rerun the cache profile matrix"
            ),
        })

    return {
        "status": "review" if recommendations else "ok",
        "source_run": source_run,
        "metrics": metrics,
        "thresholds": dict(CACHE_TUNING_THRESHOLDS),
        "read_cache_sweep": {
            "status": "swept" if read_cache_was_swept else "not_swept",
            "sizes": read_cache_sweep_sizes,
            "selected": read_cache_size,
        },
        "recommendations": recommendations,
    }


def _recommended_cache_efficiency(matrix_recommended: Mapping[str, Any]) -> tuple[str | None, dict[str, float]]:
    for run_name in ("cache_only", "cached"):
        top_level = _finite_float_mapping(_mapping(matrix_recommended.get(f"{run_name}_cache_efficiency")))
        if top_level:
            return run_name, top_level
        totals = _mapping(_mapping(matrix_recommended.get("summary")).get("totals"))
        run_total = _mapping(totals.get(run_name))
        nested = _finite_float_mapping(_mapping(run_total.get("cache_efficiency")))
        if nested:
            return run_name, nested
    return None, {}


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


def _inside_trigger_budget_sweep_decision(
    inside_trigger_budget_sweep_report: Mapping[str, Any] | None,
    *,
    policy: str,
    inside_trigger_budget_sweep_report_path: str | Path | None,
) -> dict[str, Any]:
    if inside_trigger_budget_sweep_report is None:
        return {}
    if inside_trigger_budget_sweep_report.get("dry_run") is True:
        return {
            "status": "dry_run",
            "blocking_reasons": [
                "inside_trigger_budget_sweep: report was dry-run only; run real profiles before promotion"
            ],
        }

    recommendation_source, recommendation = _inside_trigger_budget_recommendation_candidate(
        inside_trigger_budget_sweep_report,
        policy=policy,
    )
    if not recommendation:
        return {
            "status": "no_candidate",
            "blocking_reasons": [
                f"inside_trigger_budget_sweep: report did not include a {policy} budget recommendation"
            ],
            "selection_policy": policy,
        }
    budget_id = recommendation.get("budget_id")
    if not budget_id:
        return {
            "status": "no_candidate",
            "blocking_reasons": [
                "inside_trigger_budget_sweep: recommended budget did not include budget_id"
            ],
            "selection_policy": policy,
        }
    row = _inside_trigger_budget_row(
        inside_trigger_budget_sweep_report,
        budget_id=str(budget_id),
        recommended_run=recommendation.get("recommended_run"),
    )
    if not row:
        return {
            "status": "no_candidate",
            "recommended_budget_id": str(budget_id),
            "blocking_reasons": [
                f"inside_trigger_budget_sweep: recommended budget {budget_id!r} was missing from leaderboard"
            ],
            "selection_policy": policy,
        }

    budget_payload = _mapping(_mapping(inside_trigger_budget_sweep_report.get("budgets")).get(str(budget_id)))
    gate = _mapping(budget_payload.get("sample_efficiency_gate"))
    if gate and gate.get("passed") is not True:
        return {
            "status": "blocked",
            "recommended_budget_id": str(budget_id),
            "recommended_run": row.get("recommended_run") or recommendation.get("recommended_run"),
            "blocking_reasons": _inside_trigger_budget_gate_failures(gate, budget_id=str(budget_id)),
            "sample_efficiency_gate": gate,
            "selection_policy": policy,
        }

    config = _mapping(inside_trigger_budget_sweep_report.get("config"))
    recommended = _inside_trigger_budget_recommendation(
        row,
        recommendation=recommendation,
        recommendation_source=recommendation_source,
        selection_policy=policy,
        report=inside_trigger_budget_sweep_report,
        config=config,
        inside_trigger_budget_sweep_report_path=inside_trigger_budget_sweep_report_path,
    )
    return {
        "status": "promote",
        "recommended_budget_id": recommended.get("recommended_budget_id"),
        "recommended_run": recommended.get("recommended_run"),
        "recommendation_source": recommendation_source,
        "selection_policy": policy,
        "sample_efficiency_gate": gate,
        "recommended": recommended,
        "inside_sampling": _inside_trigger_budget_sampling(recommended, config=config, row=row),
    }


def _inside_trigger_budget_recommendation_candidate(
    report: Mapping[str, Any],
    *,
    policy: str,
) -> tuple[str | None, dict[str, Any]]:
    if policy == "cost_first":
        cost_first = _mapping(report.get("recommendation"))
        if cost_first:
            return "recommendation", cost_first
        quality_balanced = _mapping(report.get("quality_balanced_recommendation"))
        if quality_balanced:
            return "quality_balanced_recommendation", quality_balanced
        return None, {}
    if policy == "quality_first":
        quality_first = _quality_first_trigger_budget_recommendation(report)
        if quality_first:
            return "quality_first", quality_first
        quality_balanced = _mapping(report.get("quality_balanced_recommendation"))
        if quality_balanced:
            return "quality_balanced_recommendation", quality_balanced
        cost_first = _mapping(report.get("recommendation"))
        if cost_first:
            return "recommendation", cost_first
        return None, {}
    quality_balanced = _mapping(report.get("quality_balanced_recommendation"))
    if quality_balanced:
        return "quality_balanced_recommendation", quality_balanced
    cost_first = _mapping(report.get("recommendation"))
    if cost_first:
        return "recommendation", cost_first
    return None, {}


def _quality_first_trigger_budget_recommendation(report: Mapping[str, Any]) -> dict[str, Any]:
    leaderboard = report.get("leaderboard")
    if not isinstance(leaderboard, Sequence) or isinstance(leaderboard, str):
        return {}
    rows = [_mapping(value) for value in leaderboard]
    rows = [row for row in rows if row.get("budget_id")]
    if not rows:
        return {}
    metric = _inside_quality_metric(report, rows)
    if metric is None:
        return {}
    candidates = []
    for row in rows:
        quality = _float_or_none(_mapping(row.get("inside_auroc")).get(metric))
        if quality is None:
            continue
        candidates.append((row, quality, _inside_trigger_budget_cost(row)))
    if not candidates:
        return {}
    row, quality, cost = sorted(
        candidates,
        key=lambda item: (
            -item[1],
            math.inf if item[2] is None else item[2],
            str(item[0].get("budget_id")),
            str(item[0].get("recommended_run") or ""),
        ),
    )[0]
    payload = {
        "budget_id": row.get("budget_id"),
        "recommended_run": row.get("recommended_run"),
        "reason": "highest_inside_quality_metric_then_lowest_cost",
        "quality_metric": metric,
        "quality_value": quality,
        "best_quality_value": quality,
    }
    cost_metric = _inside_trigger_budget_cost_metric(row)
    if cost_metric is not None and cost is not None:
        payload["cost_metric"] = cost_metric
        payload["cost_value"] = cost
    return payload


def _inside_quality_metric(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str | None:
    recommended_metric = _mapping(report.get("quality_balanced_recommendation")).get("quality_metric")
    if recommended_metric:
        metric = str(recommended_metric)
        if any(_float_or_none(_mapping(row.get("inside_auroc")).get(metric)) is not None for row in rows):
            return metric
    available = {
        str(metric)
        for row in rows
        for metric, value in _mapping(row.get("inside_auroc")).items()
        if _float_or_none(value) is not None
    }
    for metric in INSIDE_QUALITY_METRIC_PRIORITY:
        if metric in available:
            return metric
    if available:
        return sorted(available)[0]
    return None


def _inside_trigger_budget_cost_metric(row: Mapping[str, Any]) -> str | None:
    for metric in (
        "inside_generation_seconds_ratio_to_reference",
        "inside_generation_seconds_ratio_to_budget_fixed",
        "inside_generation_seconds",
        "sample_count_ratio_to_reference",
        "total_generated_samples",
    ):
        if _float_or_none(row.get(metric)) is not None:
            return metric
    return None


def _inside_trigger_budget_cost(row: Mapping[str, Any]) -> float | None:
    metric = _inside_trigger_budget_cost_metric(row)
    if metric is None:
        return None
    return _float_or_none(row.get(metric))


def _inside_trigger_budget_row(
    report: Mapping[str, Any],
    *,
    budget_id: str,
    recommended_run: Any,
) -> dict[str, Any]:
    leaderboard = report.get("leaderboard")
    if not isinstance(leaderboard, Sequence) or isinstance(leaderboard, str):
        return {}
    fallback = {}
    for value in leaderboard:
        row = _mapping(value)
        if row.get("budget_id") != budget_id:
            continue
        if not fallback:
            fallback = row
        if recommended_run is None or row.get("recommended_run") == recommended_run:
            return row
    return fallback


def _inside_trigger_budget_recommendation(
    row: Mapping[str, Any],
    *,
    recommendation: Mapping[str, Any],
    recommendation_source: str | None,
    selection_policy: str,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    inside_trigger_budget_sweep_report_path: str | Path | None,
) -> dict[str, Any]:
    budget_kind = str(row.get("budget_kind") or "")
    budget_value = _float_or_none(row.get("budget_value"))
    trigger_signal = _first_present(config.get("trigger_signal"), row.get("trigger_signal"))
    derived_from_max_budget = (
        report.get("derived_from_max_budget") is True
        or config.get("derive_from_max_budget") is True
        or row.get("derived") is True
    )
    payload = {
        "report_path": None
        if inside_trigger_budget_sweep_report_path is None
        else str(inside_trigger_budget_sweep_report_path),
        "recommendation_source": recommendation_source,
        "selection_policy": selection_policy,
        "recommended_budget_id": row.get("budget_id"),
        "recommended_run": _first_present(row.get("recommended_run"), recommendation.get("recommended_run")),
        "reason": recommendation.get("reason"),
        "trigger_signal": trigger_signal,
        "budget_kind": budget_kind,
        "budget_value": budget_value,
        "budgets": _trigger_budget_specs(config, fallback_row=row),
        "derive_from_max_budget": derived_from_max_budget,
        "derived_source_budget_id": report.get("derived_source_budget_id"),
        "derived_source_score_dump": report.get("derived_source_score_dump"),
        "source_score_dump": row.get("source_score_dump"),
        "inside_generation_seconds_source": row.get("inside_generation_seconds_source"),
        "model": config.get("model"),
        "dtype": config.get("dtype"),
        "layer": _int_or_none(config.get("layer")),
        "limit": _int_or_none(config.get("limit")),
        "manifold_questions": _int_or_none(config.get("manifold_questions")),
        "batch_size": _int_or_none(config.get("batch_size")),
        "max_batch_tokens": _int_or_none(config.get("max_batch_tokens")),
        "max_length": _int_or_none(config.get("max_length")),
        "hidden_state_capture": config.get("hidden_state_capture"),
        "progress_every": _int_or_none(config.get("progress_every")),
        "offline": config.get("offline"),
        "length_bucketed_batches": config.get("length_bucketed_batches"),
        "inside_samples": _int_or_none(config.get("inside_samples")),
        "inside_batch_size": _int_or_none(config.get("inside_batch_size")),
        "inside_max_new_tokens": _int_or_none(config.get("inside_max_new_tokens")),
        "inside_temperature": _float_or_none(config.get("inside_temperature")),
        "inside_top_p": _float_or_none(config.get("inside_top_p")),
        "inside_pooling": config.get("inside_pooling"),
        "inside_embedding_threshold": _float_or_none(config.get("inside_embedding_threshold")),
        "inside_min_samples": _int_or_none(config.get("inside_min_samples")),
        "inside_sample_step": _int_or_none(config.get("inside_sample_step")),
        "inside_stability_delta": _float_or_none(config.get("inside_stability_delta")),
        "inside_selfcheck_min_overlap": _float_or_none(config.get("inside_selfcheck_min_overlap")),
        "inside_selfcheck_support_threshold": _float_or_none(
            config.get("inside_selfcheck_support_threshold")
        ),
        "inside_selfcheck_refute_threshold": _float_or_none(config.get("inside_selfcheck_refute_threshold")),
        "run_names": _string_sequence(config.get("run_names")),
        "reference_report": config.get("reference_report"),
        "shared_cache_dir": config.get("shared_cache_dir"),
        "eval_reps_cache_shard_size": _int_or_none(config.get("eval_reps_cache_shard_size")),
        "refresh_shared_caches": config.get("refresh_shared_caches"),
        "total_generated_samples": _int_or_none(row.get("total_generated_samples")),
        "sampled": _int_or_none(row.get("sampled")),
        "skipped_by_trigger": _int_or_none(row.get("skipped_by_trigger")),
        "mean_samples_per_record": _float_or_none(row.get("mean_samples_per_record")),
        "mean_samples_per_sampled_record": _float_or_none(row.get("mean_samples_per_sampled_record")),
        "inside_generation_seconds": _float_or_none(row.get("inside_generation_seconds")),
        "sample_count_ratio_to_budget_fixed": _float_or_none(
            row.get("sample_count_ratio_to_budget_fixed")
        ),
        "inside_generation_seconds_ratio_to_budget_fixed": _float_or_none(
            row.get("inside_generation_seconds_ratio_to_budget_fixed")
        ),
        "sample_count_ratio_to_reference": _float_or_none(row.get("sample_count_ratio_to_reference")),
        "inside_generation_seconds_ratio_to_reference": _float_or_none(
            row.get("inside_generation_seconds_ratio_to_reference")
        ),
        "inside_auroc": _finite_float_mapping(_mapping(row.get("inside_auroc"))),
        "quality_metric": recommendation.get("quality_metric"),
        "quality_value": _float_or_none(recommendation.get("quality_value")),
        "best_quality_value": _float_or_none(recommendation.get("best_quality_value")),
        "quality_tolerance": _float_or_none(recommendation.get("quality_tolerance")),
        "cost_metric": recommendation.get("cost_metric"),
        "cost_value": _float_or_none(recommendation.get("cost_value")),
        "stop_reason_counts": dict(row.get("stop_reason_counts", {}))
        if isinstance(row.get("stop_reason_counts", {}), Mapping)
        else {},
    }
    return {key: value for key, value in payload.items() if value is not None}


def _inside_trigger_budget_sampling(
    recommended: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    run_name = str(recommended.get("recommended_run") or "")
    budget_kind = str(recommended.get("budget_kind") or "")
    budget_value = _float_or_none(recommended.get("budget_value"))
    sampling = {
        "recommended_run": run_name or None,
        "mode": "triggered",
        "adaptive": run_name in {"adaptive", "adaptive_selfcheck"},
        "selfcheck_early_stop": run_name == "adaptive_selfcheck",
        "inside_samples": _int_or_none(recommended.get("inside_samples")),
        "inside_batch_size": _int_or_none(recommended.get("inside_batch_size")),
        "inside_max_new_tokens": _int_or_none(recommended.get("inside_max_new_tokens")),
        "inside_temperature": _float_or_none(recommended.get("inside_temperature")),
        "inside_top_p": _float_or_none(recommended.get("inside_top_p")),
        "inside_pooling": _first_present(recommended.get("inside_pooling"), "last"),
        "inside_embedding_threshold": _float_or_none(recommended.get("inside_embedding_threshold")),
        "inside_min_samples": _int_or_none(recommended.get("inside_min_samples")),
        "inside_sample_step": _int_or_none(recommended.get("inside_sample_step")),
        "inside_stability_delta": _float_or_none(recommended.get("inside_stability_delta")),
        "inside_selfcheck_min_overlap": _float_or_none(recommended.get("inside_selfcheck_min_overlap")),
        "inside_selfcheck_support_threshold": _float_or_none(
            recommended.get("inside_selfcheck_support_threshold")
        ),
        "inside_selfcheck_refute_threshold": _float_or_none(
            recommended.get("inside_selfcheck_refute_threshold")
        ),
        "inside_trigger_signal": _first_present(
            recommended.get("trigger_signal"),
            config.get("trigger_signal"),
        ),
        "inside_trigger_threshold": budget_value if budget_kind == "threshold" else None,
        "inside_trigger_top_fraction": budget_value if budget_kind == "top_fraction" else None,
        "inside_trigger_budget_id": recommended.get("recommended_budget_id"),
        "inside_trigger_budget_source": "inside_trigger_budget_sweep",
        "inside_trigger_budget_policy": recommended.get("selection_policy"),
        "derive_from_max_budget": recommended.get("derive_from_max_budget"),
        "derived_source_budget_id": recommended.get("derived_source_budget_id"),
        "total_generated_samples": _int_or_none(recommended.get("total_generated_samples")),
        "sample_count_ratio_to_baseline": _float_or_none(
            recommended.get("sample_count_ratio_to_budget_fixed")
        ),
        "inside_generation_seconds": _float_or_none(recommended.get("inside_generation_seconds")),
        "inside_generation_seconds_ratio_to_baseline": _float_or_none(
            recommended.get("inside_generation_seconds_ratio_to_budget_fixed")
        ),
        "sample_count_ratio_to_reference": _float_or_none(recommended.get("sample_count_ratio_to_reference")),
        "inside_generation_seconds_ratio_to_reference": _float_or_none(
            recommended.get("inside_generation_seconds_ratio_to_reference")
        ),
        "inside_generation_seconds_source": recommended.get("inside_generation_seconds_source"),
        "stop_reason_counts": dict(row.get("stop_reason_counts", {}))
        if isinstance(row.get("stop_reason_counts", {}), Mapping)
        else {},
    }
    return {key: value for key, value in sampling.items() if value is not None}


def _inside_trigger_budget_gate_failures(gate: Mapping[str, Any], *, budget_id: str) -> list[str]:
    failures = gate.get("failures")
    if not isinstance(failures, Sequence) or isinstance(failures, str) or not failures:
        return [f"inside_trigger_budget_sweep: selected budget {budget_id} failed sample efficiency gate"]
    reasons = []
    for failure in failures:
        item = _mapping(failure)
        run = item.get("run", "unknown")
        metric = item.get("metric", "unknown_metric")
        value = item.get("value")
        max_allowed = item.get("max_allowed")
        reasons.append(
            f"inside_trigger_budget_sweep: {budget_id}/{run} failed {metric} gate "
            f"(value={value!r}, max_allowed={max_allowed!r})"
        )
    return reasons


def _trigger_budget_specs(
    config: Mapping[str, Any],
    *,
    fallback_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_budgets = config.get("budgets")
    budgets = []
    if isinstance(raw_budgets, Sequence) and not isinstance(raw_budgets, str):
        for raw_budget in raw_budgets:
            item = _mapping(raw_budget)
            kind = item.get("kind")
            value = _float_or_none(item.get("value"))
            if kind and value is not None:
                budgets.append({
                    "kind": str(kind),
                    "value": value,
                    "id": item.get("id") or _trigger_budget_id(str(kind), value),
                })
    if budgets:
        return budgets
    kind = fallback_row.get("budget_kind")
    value = _float_or_none(fallback_row.get("budget_value"))
    if kind and value is not None:
        return [{"kind": str(kind), "value": value, "id": fallback_row.get("budget_id")}]
    return []


def _trigger_budget_id(kind: str, value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    prefix = "top" if kind == "top_fraction" else "threshold"
    return f"{prefix}_{text}"


def _string_sequence(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item) for item in value]


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
    eval_reps_shard_read_cache_size = _int_or_none(recommendation.get("eval_reps_shard_read_cache_size"))
    length_bucketed = bool(_mapping(matrix_report.get("config")).get("length_bucketed_batches", False))

    eval_flags = ["--layer", layer, "--batch-size", batch_size, "--hidden-state-capture", capture]
    if max_batch_tokens > 0:
        eval_flags.extend(["--max-batch-tokens", str(max_batch_tokens)])
    if prefix_kv_cache:
        eval_flags.append("--prefix-kv-cache")
    if eval_reps_shard_read_cache_size is not None and eval_reps_shard_read_cache_size != 2:
        eval_flags.extend(["--eval-reps-shard-read-cache-size", str(eval_reps_shard_read_cache_size)])
    if length_bucketed:
        eval_flags.append("--length-bucketed-batches")
    eval_flags.extend(_inside_sampling_eval_flags(_mapping(recommendation.get("inside_sampling"))))

    matrix_flags = ["--layers", layer, "--batch-sizes", batch_size, "--hidden-state-captures", capture]
    if max_batch_tokens > 0:
        matrix_flags.extend(["--max-batch-tokens", str(max_batch_tokens)])
    if prefix_kv_cache:
        matrix_flags.append("--prefix-kv-cache")
    if eval_reps_shard_read_cache_size is not None and eval_reps_shard_read_cache_size != 2:
        matrix_flags.extend(["--eval-reps-shard-read-cache-size", str(eval_reps_shard_read_cache_size)])
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
    trigger_budget = _mapping(recommendation.get("inside_trigger_budget_sweep"))
    if trigger_budget:
        flags["run_inside_trigger_budget_sweep"] = _inside_trigger_budget_sweep_flags(trigger_budget)
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


def _inside_trigger_budget_sweep_flags(trigger_budget: Mapping[str, Any]) -> list[str]:
    flags = []
    _extend_flag(flags, "--trigger-signal", trigger_budget.get("trigger_signal"))
    budgets = _trigger_budget_specs(trigger_budget, fallback_row={})
    top_fractions = [
        str(budget["value"])
        for budget in budgets
        if budget.get("kind") == "top_fraction"
    ]
    thresholds = [
        str(budget["value"])
        for budget in budgets
        if budget.get("kind") == "threshold"
    ]
    if not top_fractions and trigger_budget.get("budget_kind") == "top_fraction":
        value = _float_or_none(trigger_budget.get("budget_value"))
        if value is not None:
            top_fractions = [str(value)]
    if not thresholds and trigger_budget.get("budget_kind") == "threshold":
        value = _float_or_none(trigger_budget.get("budget_value"))
        if value is not None:
            thresholds = [str(value)]
    if top_fractions:
        flags.extend(["--top-fractions", ",".join(top_fractions)])
    if thresholds:
        flags.extend(["--thresholds", ",".join(thresholds)])
    _extend_flag(flags, "--reference-report", trigger_budget.get("reference_report"))
    _extend_flag(flags, "--model", trigger_budget.get("model"))
    _extend_flag(flags, "--dtype", trigger_budget.get("dtype"))
    _extend_flag(flags, "--layer", trigger_budget.get("layer"))
    _extend_flag(flags, "--limit", trigger_budget.get("limit"))
    _extend_flag(flags, "--manifold-questions", trigger_budget.get("manifold_questions"))
    _extend_flag(flags, "--batch-size", trigger_budget.get("batch_size"))
    _extend_flag(flags, "--max-batch-tokens", trigger_budget.get("max_batch_tokens"))
    _extend_flag(flags, "--max-length", trigger_budget.get("max_length"))
    _extend_flag(flags, "--hidden-state-capture", trigger_budget.get("hidden_state_capture"))
    _extend_flag(flags, "--progress-every", trigger_budget.get("progress_every"))
    if trigger_budget.get("offline") is False:
        flags.append("--real-truthfulqa")
    if trigger_budget.get("length_bucketed_batches") is False:
        flags.append("--no-length-bucketed-batches")
    _extend_flag(flags, "--inside-samples", trigger_budget.get("inside_samples"))
    _extend_flag(flags, "--inside-batch-size", trigger_budget.get("inside_batch_size"))
    _extend_flag(flags, "--inside-max-new-tokens", trigger_budget.get("inside_max_new_tokens"))
    _extend_flag(flags, "--inside-temperature", trigger_budget.get("inside_temperature"))
    _extend_flag(flags, "--inside-top-p", trigger_budget.get("inside_top_p"))
    _extend_flag(flags, "--inside-pooling", trigger_budget.get("inside_pooling"))
    _extend_flag(flags, "--inside-embedding-threshold", trigger_budget.get("inside_embedding_threshold"))
    _extend_flag(flags, "--inside-min-samples", trigger_budget.get("inside_min_samples"))
    _extend_flag(flags, "--inside-sample-step", trigger_budget.get("inside_sample_step"))
    _extend_flag(flags, "--inside-stability-delta", trigger_budget.get("inside_stability_delta"))
    _extend_flag(flags, "--inside-selfcheck-min-overlap", trigger_budget.get("inside_selfcheck_min_overlap"))
    _extend_flag(
        flags,
        "--inside-selfcheck-support-threshold",
        trigger_budget.get("inside_selfcheck_support_threshold"),
    )
    _extend_flag(
        flags,
        "--inside-selfcheck-refute-threshold",
        trigger_budget.get("inside_selfcheck_refute_threshold"),
    )
    run_names = _string_sequence(trigger_budget.get("run_names"))
    if not run_names and trigger_budget.get("recommended_run"):
        run_names = [str(trigger_budget["recommended_run"])]
    if run_names:
        flags.extend(["--runs", ",".join(run_names)])
    _extend_flag(flags, "--shared-cache-dir", trigger_budget.get("shared_cache_dir"))
    _extend_flag(flags, "--eval-reps-cache-shard-size", trigger_budget.get("eval_reps_cache_shard_size"))
    if trigger_budget.get("refresh_shared_caches") is True:
        flags.append("--refresh-shared-caches")
    if trigger_budget.get("derive_from_max_budget") is True:
        flags.append("--derive-from-max-budget")
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
    inside_trigger_budget_sweep_report: Mapping[str, Any] | None,
    inside_trigger_budget_sweep_decision: Mapping[str, Any],
    matrix_report_path: str | Path | None,
    worker_sweep_report_path: str | Path | None,
    inside_sampling_report_path: str | Path | None,
    inside_trigger_budget_sweep_report_path: str | Path | None,
    inside_trigger_budget_policy: str,
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
    trigger_budget = _mapping(inside_trigger_budget_sweep_decision.get("recommended"))
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
            matrix_recommended=matrix_recommended,
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
        "inside_trigger_budget_sweep_report": None
        if inside_trigger_budget_sweep_report_path is None
        else str(inside_trigger_budget_sweep_report_path),
        "inside_trigger_budget_policy": None
        if inside_trigger_budget_sweep_report is None
        else inside_trigger_budget_policy,
        "inside_trigger_budget_sweep_status": None
        if inside_trigger_budget_sweep_report is None
        else inside_trigger_budget_sweep_decision.get("status"),
        "inside_trigger_budget_recommended_budget_id": inside_trigger_budget_sweep_decision.get(
            "recommended_budget_id"
        ),
        "inside_trigger_budget_recommended_run": inside_trigger_budget_sweep_decision.get(
            "recommended_run"
        ),
        "inside_trigger_budget_recommendation_source": inside_trigger_budget_sweep_decision.get(
            "recommendation_source"
        ),
        "inside_trigger_budget_derive_from_max_budget": trigger_budget.get("derive_from_max_budget"),
        "inside_trigger_budget_derived_source_budget_id": trigger_budget.get("derived_source_budget_id"),
        "inside_trigger_budget_gate_passed": None
        if inside_trigger_budget_sweep_report is None
        else _mapping(inside_trigger_budget_sweep_decision.get("sample_efficiency_gate")).get("passed"),
    }
    return evidence


def _worker_matrix_report_matches(
    *,
    worker_recommended: Mapping[str, Any],
    matrix_recommended: Mapping[str, Any],
    matrix_report_path: str | Path | None,
) -> bool | None:
    worker_matrix_report = worker_recommended.get("matrix_report")
    if worker_matrix_report is not None and matrix_report_path is not None:
        try:
            if Path(str(worker_matrix_report)).resolve() == Path(str(matrix_report_path)).resolve():
                return True
        except OSError:
            if str(worker_matrix_report) == str(matrix_report_path):
                return True
    semantic_match = _worker_matrix_recommended_cell_matches(
        worker_recommended=worker_recommended,
        matrix_recommended=matrix_recommended,
    )
    if semantic_match is not None:
        return semantic_match
    if worker_matrix_report is None or matrix_report_path is None:
        return None
    return False


def _worker_matrix_recommended_cell_matches(
    *,
    worker_recommended: Mapping[str, Any],
    matrix_recommended: Mapping[str, Any],
) -> bool | None:
    matched_any = False
    worker_cell = worker_recommended.get("recommended_cell")
    matrix_cell = matrix_recommended.get("id")
    if worker_cell is not None and matrix_cell is not None:
        matched_any = True
        if str(worker_cell) != str(matrix_cell):
            return False
    for worker_key, matrix_key, rel_tol, abs_tol in (
        ("recommended_truth_proj_auroc", "truth_proj_auroc", 1e-9, 1e-12),
        ("recommended_cache_only_total_seconds", "cache_only_total_seconds", 0.05, 1e-3),
    ):
        worker_value = _float_or_none(worker_recommended.get(worker_key))
        matrix_value = _float_or_none(matrix_recommended.get(matrix_key))
        if worker_value is None or matrix_value is None:
            continue
        matched_any = True
        if not math.isclose(worker_value, matrix_value, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    if matched_any:
        return True
    return None


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
    inside_trigger_budget_sweep_decision: Mapping[str, Any],
    inside_trigger_budget_sweep_report: Mapping[str, Any] | None,
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
    if inside_trigger_budget_sweep_report is not None:
        for reason in inside_trigger_budget_sweep_decision.get("blocking_reasons") or ():
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


def _normalize_inside_trigger_budget_policy(policy: str) -> str:
    normalized = str(policy or "").strip().lower().replace("-", "_")
    if normalized not in INSIDE_TRIGGER_BUDGET_POLICIES:
        choices = ", ".join(INSIDE_TRIGGER_BUDGET_POLICIES)
        raise ValueError(f"inside_trigger_budget_policy must be one of: {choices}")
    return normalized


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
    inside_trigger_budget_sweep_report_arg = getattr(args, "inside_trigger_budget_sweep_report", None)
    inside_trigger_budget_sweep_report_path = (
        Path(inside_trigger_budget_sweep_report_arg) if inside_trigger_budget_sweep_report_arg else None
    )
    report = build_runtime_recommendation(
        _load_json(matrix_report_path),
        worker_sweep_report=None if worker_sweep_report_path is None else _load_json(worker_sweep_report_path),
        inside_sampling_report=(
            None if inside_sampling_report_path is None else _load_json(inside_sampling_report_path)
        ),
        inside_trigger_budget_sweep_report=(
            None
            if inside_trigger_budget_sweep_report_path is None
            else _load_json(inside_trigger_budget_sweep_report_path)
        ),
        inside_trigger_budget_policy=args.inside_trigger_budget_policy,
        matrix_report_path=matrix_report_path,
        worker_sweep_report_path=worker_sweep_report_path,
        inside_sampling_report_path=inside_sampling_report_path,
        inside_trigger_budget_sweep_report_path=inside_trigger_budget_sweep_report_path,
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
    parser.add_argument("--inside-trigger-budget-sweep-report", default=None,
                        help="optional inside-trigger-budget-sweep.json produced by "
                             "run_inside_trigger_budget_sweep.py")
    parser.add_argument("--inside-trigger-budget-policy", default="quality_balanced",
                        choices=INSIDE_TRIGGER_BUDGET_POLICIES,
                        help="budget selection policy when --inside-trigger-budget-sweep-report is provided; "
                             "quality_balanced preserves the previous default, cost_first minimizes sampled "
                             "INSIDE work, and quality_first chooses the highest inside quality metric")
    parser.add_argument("--output", default=None,
                        help="optional path to write the recommendation JSON")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero unless the combined recommendation status is promote")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
