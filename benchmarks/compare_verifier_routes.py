"""Compare verifier-ensemble route metrics across reports.

This is a post-processing helper for ``eval_verifier_ensemble.py`` JSON output.
It does not load models or rerun verification. It extracts per-route verifier
quality and per-alpha control impact so routes can be ranked before wiring a
real production adapter behind them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("report name cannot be empty.")
    return name, Path(path)


def _alpha_payload(run: Mapping[str, Any], alpha: float) -> Mapping[str, Any]:
    alphas = run.get("alphas", {})
    if not isinstance(alphas, Mapping):
        return {}
    direct = alphas.get(str(float(alpha)))
    if isinstance(direct, Mapping):
        return direct
    for key, payload in alphas.items():
        try:
            matches = abs(float(key) - float(alpha)) < 1e-12
        except (TypeError, ValueError):
            matches = False
        if matches and isinstance(payload, Mapping):
            return payload
    return {}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return statistics.fmean(present)


def _weighted_mean(values: Sequence[tuple[float | None, int | float]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        if value is None or weight <= 0:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return _safe_div(numerator, denominator)


def _label_counts(route_quality: Mapping[str, Any]) -> dict[str, int]:
    matrix = route_quality.get("label_status_matrix", {})
    true = matrix.get("true", {}) if isinstance(matrix, Mapping) else {}
    false = matrix.get("false", {}) if isinstance(matrix, Mapping) else {}
    return {
        "true_supported": int(true.get("supported", 0)),
        "true_refuted": int(true.get("refuted", 0)),
        "true_insufficient_evidence": int(true.get("insufficient_evidence", 0)),
        "false_supported": int(false.get("supported", 0)),
        "false_refuted": int(false.get("refuted", 0)),
        "false_insufficient_evidence": int(false.get("insufficient_evidence", 0)),
    }


def _impact_for_route(run: Mapping[str, Any], route: str, alpha: float) -> Mapping[str, Any]:
    alpha_payload = _alpha_payload(run, alpha)
    impacts = alpha_payload.get("route_control_impact", {})
    if not isinstance(impacts, Mapping):
        return {}
    route_impact = impacts.get(route, {})
    return route_impact if isinstance(route_impact, Mapping) else {}


def _route_row(
    *,
    report_name: str,
    source: Path,
    run: Mapping[str, Any],
    route: str,
    route_quality: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    impact = _impact_for_route(run, route, alpha)
    internal = impact.get("internal", {}) if isinstance(impact.get("internal", {}), Mapping) else {}
    verified = impact.get("verified", {}) if isinstance(impact.get("verified", {}), Mapping) else {}
    delta = impact.get("delta", {}) if isinstance(impact.get("delta", {}), Mapping) else {}
    counts = _label_counts(route_quality)
    return {
        "report": report_name,
        "run": str(run.get("name", report_name)),
        "source": str(source),
        "route": route,
        "alpha": float(alpha),
        "selected": int(route_quality.get("selected", 0)),
        "selection_rate": _as_float(route_quality.get("selection_rate")),
        "n_true": int(route_quality.get("n_true", 0)),
        "n_false": int(route_quality.get("n_false", 0)),
        **counts,
        "true_supported_rate": _as_float(route_quality.get("true_supported_rate")),
        "true_refuted_rate": _as_float(route_quality.get("true_refuted_rate")),
        "false_refuted_rate": _as_float(route_quality.get("false_refuted_rate")),
        "false_supported_rate": _as_float(route_quality.get("false_supported_rate")),
        "insufficient_evidence_rate": _as_float(route_quality.get("insufficient_evidence_rate")),
        "decision_accuracy": _as_float(route_quality.get("decision_accuracy")),
        "decision_error_rate": _as_float(route_quality.get("decision_error_rate")),
        "duration_observations": int(route_quality.get("duration_observations", 0)),
        "total_duration_seconds": _as_float(route_quality.get("total_duration_seconds")),
        "mean_duration_seconds": _as_float(route_quality.get("mean_duration_seconds")),
        "p95_duration_seconds": _as_float(route_quality.get("p95_duration_seconds")),
        "p99_duration_seconds": _as_float(route_quality.get("p99_duration_seconds")),
        "max_duration_seconds": _as_float(route_quality.get("max_duration_seconds")),
        "selected_route_duration_observations": int(
            route_quality.get("selected_route_duration_observations", 0)
        ),
        "total_selected_route_duration_seconds": _as_float(
            route_quality.get("total_selected_route_duration_seconds")
        ),
        "mean_selected_route_duration_seconds": _as_float(
            route_quality.get("mean_selected_route_duration_seconds")
        ),
        "p95_selected_route_duration_seconds": _as_float(
            route_quality.get("p95_selected_route_duration_seconds")
        ),
        "p99_selected_route_duration_seconds": _as_float(
            route_quality.get("p99_selected_route_duration_seconds")
        ),
        "attempted_route_count_observations": int(route_quality.get("attempted_route_count_observations", 0)),
        "total_attempted_route_count": _as_float(route_quality.get("total_attempted_route_count")),
        "mean_attempted_route_count": _as_float(route_quality.get("mean_attempted_route_count")),
        "used_retrieval_count": int(route_quality.get("used_retrieval_count", 0)),
        "retrieval_use_rate": _as_float(route_quality.get("retrieval_use_rate")),
        "retrieval_hit_count": int(route_quality.get("retrieval_hit_count", 0)),
        "mean_retrieval_hits": _as_float(route_quality.get("mean_retrieval_hits")),
        "internal_false_alarm": _as_float(internal.get("false_alarm")),
        "verified_false_alarm": _as_float(verified.get("false_alarm")),
        "delta_false_alarm": _as_float(delta.get("false_alarm")),
        "internal_detection": _as_float(internal.get("detection")),
        "verified_detection": _as_float(verified.get("detection")),
        "delta_detection": _as_float(delta.get("detection")),
        "suppressed_false_alarm_rate": _as_float(delta.get("suppressed_false_alarm_rate")),
        "rescued_detection_rate": _as_float(delta.get("rescued_detection_rate")),
    }


def _cache_record(
    *,
    report_name: str,
    source: Path,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    stats = run.get("cache_stats", {})
    total = stats.get("total", {}) if isinstance(stats, Mapping) else {}
    total = total if isinstance(total, Mapping) else {}
    hits = _as_int_or_none(total.get("hits"))
    misses = _as_int_or_none(total.get("misses"))
    requests = _as_int_or_none(total.get("requests"))
    if requests is None and hits is not None and misses is not None:
        requests = hits + misses
    return {
        "report": report_name,
        "run": str(run.get("name", report_name)),
        "source": str(source),
        "has_cache_stats": isinstance(stats, Mapping) and bool(stats),
        "size": _as_int_or_none(total.get("size")),
        "hits": hits,
        "misses": misses,
        "requests": requests,
        "hit_rate": _as_float(total.get("hit_rate")) if total.get("hit_rate") is not None else None,
    }


def _extract_route_rows_and_cache_records(
    reports: Sequence[tuple[str, Path]],
    *,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    cache_records = []
    for report_name, path in reports:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        runs = report.get("runs", ())
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
            raise ValueError(f"report {path} has invalid runs field.")
        for run in runs:
            if not isinstance(run, Mapping):
                continue
            cache_records.append(_cache_record(report_name=report_name, source=path, run=run))
            route_quality = run.get("route_quality", {})
            if not isinstance(route_quality, Mapping):
                continue
            for route, payload in route_quality.items():
                if not isinstance(payload, Mapping):
                    continue
                rows.append(_route_row(
                    report_name=report_name,
                    source=path,
                    run=run,
                    route=str(route),
                    route_quality=payload,
                    alpha=alpha,
                ))
    return rows, cache_records


def _extract_route_rows(
    reports: Sequence[tuple[str, Path]],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    rows, _ = _extract_route_rows_and_cache_records(reports, alpha=alpha)
    return rows


def _none_low(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def _none_high(value: float | None) -> float:
    return 1.0 if value is None else float(value)


def _leaderboard_rows(rows: Sequence[Mapping[str, Any]], *, min_selected: int) -> list[dict[str, Any]]:
    eligible = [dict(row) for row in rows if int(row.get("selected", 0)) >= min_selected]
    eligible.sort(
        key=lambda row: (
            _none_low(row.get("decision_accuracy")),
            _none_low(row.get("false_refuted_rate")),
            -_none_high(row.get("false_supported_rate")),
            _none_low(row.get("verified_detection")),
            -_none_high(row.get("verified_false_alarm")),
            -_none_high(row.get("mean_duration_seconds")),
            int(row.get("selected", 0)),
        ),
        reverse=True,
    )
    return eligible


def _sum_finite_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    values = [_finite_float(row.get(metric)) for row in rows]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(sum(present))


def _max_finite_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float | None:
    values = [_finite_float(row.get(metric)) for row in rows]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _mean_from_total(total: float | None, observations: int) -> float | None:
    if observations == 0 or total is None:
        return None
    return _safe_div(total, observations)


def _aggregate_route(route: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = sum(int(row.get("selected", 0)) for row in rows)
    n_true = sum(int(row.get("n_true", 0)) for row in rows)
    n_false = sum(int(row.get("n_false", 0)) for row in rows)
    true_supported = sum(int(row.get("true_supported", 0)) for row in rows)
    true_refuted = sum(int(row.get("true_refuted", 0)) for row in rows)
    false_supported = sum(int(row.get("false_supported", 0)) for row in rows)
    false_refuted = sum(int(row.get("false_refuted", 0)) for row in rows)
    decided = true_supported + true_refuted + false_supported + false_refuted
    correct = true_supported + false_refuted
    wrong = true_refuted + false_supported
    duration_observations = sum(int(row.get("duration_observations", 0)) for row in rows)
    selected_route_duration_observations = sum(
        int(row.get("selected_route_duration_observations", 0)) for row in rows
    )
    attempted_route_count_observations = sum(
        int(row.get("attempted_route_count_observations", 0)) for row in rows
    )
    total_duration = _sum_finite_metric(rows, "total_duration_seconds")
    total_selected_route_duration = _sum_finite_metric(rows, "total_selected_route_duration_seconds")
    total_attempted_route_count = _sum_finite_metric(rows, "total_attempted_route_count")
    used_retrieval_count = sum(int(row.get("used_retrieval_count", 0)) for row in rows)
    retrieval_hit_count = sum(int(row.get("retrieval_hit_count", 0)) for row in rows)
    best = _leaderboard_rows(rows, min_selected=0)[0] if rows else None
    return {
        "route": route,
        "n_entries": len(rows),
        "selected": selected,
        "n_true": n_true,
        "n_false": n_false,
        "mean_selection_rate": _mean([row.get("selection_rate") for row in rows]),
        "true_supported_rate": _safe_div(true_supported, n_true),
        "true_refuted_rate": _safe_div(true_refuted, n_true),
        "false_refuted_rate": _safe_div(false_refuted, n_false),
        "false_supported_rate": _safe_div(false_supported, n_false),
        "decision_accuracy": _safe_div(correct, decided),
        "decision_error_rate": _safe_div(wrong, decided),
        "duration_observations": duration_observations,
        "total_duration_seconds": total_duration,
        "mean_duration_seconds": _mean_from_total(total_duration, duration_observations),
        "p95_duration_seconds": _max_finite_metric(rows, "p95_duration_seconds"),
        "p99_duration_seconds": _max_finite_metric(rows, "p99_duration_seconds"),
        "max_duration_seconds": _max_finite_metric(rows, "max_duration_seconds"),
        "selected_route_duration_observations": selected_route_duration_observations,
        "total_selected_route_duration_seconds": total_selected_route_duration,
        "mean_selected_route_duration_seconds": _mean_from_total(
            total_selected_route_duration,
            selected_route_duration_observations,
        ),
        "p95_selected_route_duration_seconds": _max_finite_metric(
            rows,
            "p95_selected_route_duration_seconds",
        ),
        "p99_selected_route_duration_seconds": _max_finite_metric(
            rows,
            "p99_selected_route_duration_seconds",
        ),
        "attempted_route_count_observations": attempted_route_count_observations,
        "total_attempted_route_count": total_attempted_route_count,
        "mean_attempted_route_count": _mean_from_total(
            total_attempted_route_count,
            attempted_route_count_observations,
        ),
        "used_retrieval_count": used_retrieval_count,
        "retrieval_use_rate": _safe_div(used_retrieval_count, selected),
        "retrieval_hit_count": retrieval_hit_count,
        "mean_retrieval_hits": _safe_div(retrieval_hit_count, selected),
        "verified_false_alarm": _weighted_mean(
            [(row.get("verified_false_alarm"), row.get("n_true", 0)) for row in rows]
        ),
        "verified_detection": _weighted_mean(
            [(row.get("verified_detection"), row.get("n_false", 0)) for row in rows]
        ),
        "delta_false_alarm": _weighted_mean(
            [(row.get("delta_false_alarm"), row.get("n_true", 0)) for row in rows]
        ),
        "delta_detection": _weighted_mean(
            [(row.get("delta_detection"), row.get("n_false", 0)) for row in rows]
        ),
        "suppressed_false_alarm_rate": _weighted_mean(
            [(row.get("suppressed_false_alarm_rate"), row.get("n_true", 0)) for row in rows]
        ),
        "rescued_detection_rate": _weighted_mean(
            [(row.get("rescued_detection_rate"), row.get("n_false", 0)) for row in rows]
        ),
        "best_entry": None if best is None else {
            "report": best["report"],
            "run": best["run"],
            "selected": best["selected"],
            "decision_accuracy": best["decision_accuracy"],
            "false_refuted_rate": best["false_refuted_rate"],
            "false_supported_rate": best["false_supported_rate"],
        },
    }


def _aggregate_by_route(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["route"]), []).append(row)
    return {
        route: _aggregate_route(route, grouped[route])
        for route in sorted(grouped)
    }


def _aggregate_cache_summary(cache_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    present = [row for row in cache_records if row.get("has_cache_stats")]
    hits = sum(int(row.get("hits") or 0) for row in present)
    misses = sum(int(row.get("misses") or 0) for row in present)
    size = sum(int(row.get("size") or 0) for row in present)
    requests = hits + misses
    return {
        "n_runs": len(cache_records),
        "n_runs_with_cache_stats": len(present),
        "total": {
            "size": size,
            "hits": hits,
            "misses": misses,
            "requests": requests,
            "hit_rate": None if requests == 0 else hits / requests,
        },
        "runs": [dict(row) for row in cache_records],
    }


PARETO_MAXIMIZE_METRICS = (
    "decision_accuracy",
    "false_refuted_rate",
    "verified_detection",
    "rescued_detection_rate",
    "selected",
)
PARETO_MINIMIZE_METRICS = (
    "false_supported_rate",
    "verified_false_alarm",
    "mean_duration_seconds",
    "p95_duration_seconds",
    "p99_duration_seconds",
    "mean_attempted_route_count",
    "retrieval_use_rate",
)


def _metric_value(payload: Mapping[str, Any], metric: str) -> float | None:
    if metric == "selected":
        return float(int(payload.get("selected", 0)))
    return _finite_float(payload.get(metric))


def _all_pareto_metrics() -> tuple[str, ...]:
    return (*PARETO_MAXIMIZE_METRICS, *PARETO_MINIMIZE_METRICS)


def _missing_pareto_metrics(payload: Mapping[str, Any]) -> list[str]:
    return [
        metric
        for metric in _all_pareto_metrics()
        if _metric_value(payload, metric) is None
    ]


def _better_or_equal(
    challenger: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    metric: str,
    maximize: bool,
) -> bool | None:
    challenger_value = _metric_value(challenger, metric)
    incumbent_value = _metric_value(incumbent, metric)
    if challenger_value is None or incumbent_value is None:
        return None
    if maximize:
        return challenger_value >= incumbent_value
    return challenger_value <= incumbent_value


def _strictly_better(
    challenger: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    metric: str,
    maximize: bool,
) -> bool | None:
    challenger_value = _metric_value(challenger, metric)
    incumbent_value = _metric_value(incumbent, metric)
    if challenger_value is None or incumbent_value is None:
        return None
    if maximize:
        return challenger_value > incumbent_value
    return challenger_value < incumbent_value


def _dominates(challenger: Mapping[str, Any], incumbent: Mapping[str, Any]) -> bool:
    """Return whether challenger is at least as good on shared metrics and better on one."""
    comparisons = []
    improvements = []
    for metric in PARETO_MAXIMIZE_METRICS:
        better_or_equal = _better_or_equal(challenger, incumbent, metric=metric, maximize=True)
        strictly_better = _strictly_better(challenger, incumbent, metric=metric, maximize=True)
        if better_or_equal is not None:
            comparisons.append(better_or_equal)
        if strictly_better is not None:
            improvements.append(strictly_better)
    for metric in PARETO_MINIMIZE_METRICS:
        better_or_equal = _better_or_equal(challenger, incumbent, metric=metric, maximize=False)
        strictly_better = _strictly_better(challenger, incumbent, metric=metric, maximize=False)
        if better_or_equal is not None:
            comparisons.append(better_or_equal)
        if strictly_better is not None:
            improvements.append(strictly_better)
    return bool(comparisons) and all(comparisons) and any(improvements)


def _bounded_metric(value: Any, *, default: float = 0.0) -> float:
    observed = _finite_float(value)
    if observed is None:
        return default
    return max(0.0, min(1.0, observed))


def _efficiency_score(value: Any, *, default: float = 0.5) -> float:
    observed = _finite_float(value)
    if observed is None or observed < 0.0:
        return default
    return 1.0 / (1.0 + observed)


def _promotion_score(payload: Mapping[str, Any]) -> float:
    """Return a deterministic quality/cost score for ordering Pareto candidates."""
    components = (
        2.0 * _bounded_metric(payload.get("decision_accuracy")),
        1.5 * _bounded_metric(payload.get("false_refuted_rate")),
        1.5 * (1.0 - _bounded_metric(payload.get("false_supported_rate"), default=1.0)),
        1.0 * _bounded_metric(payload.get("verified_detection")),
        1.0 * (1.0 - _bounded_metric(payload.get("verified_false_alarm"), default=1.0)),
        0.75 * _efficiency_score(payload.get("mean_duration_seconds")),
        0.50 * _efficiency_score(payload.get("p95_duration_seconds")),
        0.50 * _efficiency_score(payload.get("p99_duration_seconds")),
        0.50 * _efficiency_score(payload.get("mean_attempted_route_count")),
        0.50 * (1.0 - _bounded_metric(payload.get("retrieval_use_rate"), default=0.5)),
    )
    return sum(components) / 9.75


def _pareto_entry(route: str, payload: Mapping[str, Any], *, dominated_by: str | None = None) -> dict[str, Any]:
    return {
        "route": route,
        "selected": int(payload.get("selected", 0)),
        "promotion_score": _promotion_score(payload),
        "dominated_by": dominated_by,
        "missing_metrics": _missing_pareto_metrics(payload),
        "metrics": {
            metric: _metric_value(payload, metric)
            for metric in _all_pareto_metrics()
        },
    }


def _entry_metric_or(item: Mapping[str, Any], metric: str, default: float) -> float:
    metrics = item.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return default
    value = _finite_float(metrics.get(metric))
    return default if value is None else value


def build_route_pareto_frontier(
    by_route: Mapping[str, Any],
    *,
    min_selected: int = 1,
) -> dict[str, Any]:
    """Build a route quality/cost Pareto frontier over aggregate route metrics."""
    if min_selected < 0:
        raise ValueError("pareto min_selected must be >= 0.")
    eligible = {
        str(route): payload
        for route, payload in by_route.items()
        if isinstance(payload, Mapping) and int(payload.get("selected", 0)) >= min_selected
    }
    frontier = []
    dominated = []
    for route, payload in sorted(eligible.items()):
        dominator = None
        for other_route, other_payload in sorted(eligible.items()):
            if other_route == route:
                continue
            if _dominates(other_payload, payload):
                dominator = other_route
                break
        if dominator is None:
            frontier.append(_pareto_entry(route, payload))
        else:
            dominated.append(_pareto_entry(route, payload, dominated_by=dominator))

    frontier.sort(
        key=lambda item: (
            item["promotion_score"],
            _entry_metric_or(item, "decision_accuracy", -1.0),
            _entry_metric_or(item, "false_refuted_rate", -1.0),
            -_entry_metric_or(item, "mean_duration_seconds", math.inf),
            item["selected"],
        ),
        reverse=True,
    )
    dominated.sort(key=lambda item: (item["dominated_by"] or "", item["route"]))
    return {
        "config": {
            "min_selected": int(min_selected),
            "maximize": list(PARETO_MAXIMIZE_METRICS),
            "minimize": list(PARETO_MINIMIZE_METRICS),
            "score": "weighted_quality_cost_v1",
        },
        "n_eligible": len(eligible),
        "recommended": None if not frontier else frontier[0],
        "frontier": frontier,
        "dominated": dominated,
    }


def _thresholds_enabled(
    *,
    min_decision_accuracy: float | None,
    max_false_supported_rate: float | None,
    min_false_refuted_rate: float | None,
    max_verified_false_alarm: float | None,
    min_verified_detection: float | None,
    max_mean_duration_seconds: float | None,
    max_p95_duration_seconds: float | None,
    max_p99_duration_seconds: float | None,
    max_max_duration_seconds: float | None,
    max_mean_attempted_route_count: float | None,
    max_retrieval_use_rate: float | None,
    min_cache_hit_rate: float | None,
) -> bool:
    return any(
        value is not None
        for value in (
            min_decision_accuracy,
            max_false_supported_rate,
            min_false_refuted_rate,
            max_verified_false_alarm,
            min_verified_detection,
            max_mean_duration_seconds,
            max_p95_duration_seconds,
            max_p99_duration_seconds,
            max_max_duration_seconds,
            max_mean_attempted_route_count,
            max_retrieval_use_rate,
            min_cache_hit_rate,
        )
    )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _validated_limit(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    limit = float(value)
    if not math.isfinite(limit):
        raise ValueError(f"{name} must be finite.")
    return limit


def _check_min_metric(
    *,
    route: str,
    metric: str,
    value: Any,
    limit: float,
) -> dict[str, Any] | None:
    observed = _finite_float(value)
    if observed is None or observed < float(limit):
        return {
            "route": route,
            "metric": metric,
            "limit_type": "min",
            "limit": float(limit),
            "value": observed,
            "raw_value": None if value is None else repr(value),
        }
    return None


def _check_max_metric(
    *,
    route: str,
    metric: str,
    value: Any,
    limit: float,
) -> dict[str, Any] | None:
    observed = _finite_float(value)
    if observed is None or observed > float(limit):
        return {
            "route": route,
            "metric": metric,
            "limit_type": "max",
            "limit": float(limit),
            "value": observed,
            "raw_value": None if value is None else repr(value),
        }
    return None


def build_route_quality_gate(
    by_route: Mapping[str, Any],
    *,
    cache_summary: Mapping[str, Any] | None = None,
    routes: Sequence[str] = (),
    min_selected: int = 1,
    min_decision_accuracy: float | None = None,
    max_false_supported_rate: float | None = None,
    min_false_refuted_rate: float | None = None,
    max_verified_false_alarm: float | None = None,
    min_verified_detection: float | None = None,
    max_mean_duration_seconds: float | None = None,
    max_p95_duration_seconds: float | None = None,
    max_p99_duration_seconds: float | None = None,
    max_max_duration_seconds: float | None = None,
    max_mean_attempted_route_count: float | None = None,
    max_retrieval_use_rate: float | None = None,
    min_cache_hit_rate: float | None = None,
) -> dict[str, Any] | None:
    """Build a fail-closed route quality gate over aggregate route metrics."""
    if min_selected < 0:
        raise ValueError("gate min_selected must be >= 0.")
    min_decision_accuracy = _validated_limit("min_decision_accuracy", min_decision_accuracy)
    max_false_supported_rate = _validated_limit("max_false_supported_rate", max_false_supported_rate)
    min_false_refuted_rate = _validated_limit("min_false_refuted_rate", min_false_refuted_rate)
    max_verified_false_alarm = _validated_limit("max_verified_false_alarm", max_verified_false_alarm)
    min_verified_detection = _validated_limit("min_verified_detection", min_verified_detection)
    max_mean_duration_seconds = _validated_limit("max_mean_duration_seconds", max_mean_duration_seconds)
    max_p95_duration_seconds = _validated_limit("max_p95_duration_seconds", max_p95_duration_seconds)
    max_p99_duration_seconds = _validated_limit("max_p99_duration_seconds", max_p99_duration_seconds)
    max_max_duration_seconds = _validated_limit("max_max_duration_seconds", max_max_duration_seconds)
    max_mean_attempted_route_count = _validated_limit(
        "max_mean_attempted_route_count",
        max_mean_attempted_route_count,
    )
    max_retrieval_use_rate = _validated_limit("max_retrieval_use_rate", max_retrieval_use_rate)
    min_cache_hit_rate = _validated_limit("min_cache_hit_rate", min_cache_hit_rate)
    enabled = bool(routes) or _thresholds_enabled(
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
        max_mean_duration_seconds=max_mean_duration_seconds,
        max_p95_duration_seconds=max_p95_duration_seconds,
        max_p99_duration_seconds=max_p99_duration_seconds,
        max_max_duration_seconds=max_max_duration_seconds,
        max_mean_attempted_route_count=max_mean_attempted_route_count,
        max_retrieval_use_rate=max_retrieval_use_rate,
        min_cache_hit_rate=min_cache_hit_rate,
    )
    if not enabled:
        return None

    route_names = tuple(str(route) for route in routes) or tuple(
        route
        for route, payload in sorted(by_route.items())
        if isinstance(payload, Mapping) and int(payload.get("selected", 0)) >= min_selected
    )
    failures = []
    checked_routes = []
    if not route_names:
        failures.append({
            "route": None,
            "metric": "eligible_routes",
            "limit_type": "min",
            "limit": 1,
            "value": 0,
            "reason": "no aggregate routes met gate min_selected",
        })
    for route in route_names:
        payload = by_route.get(route)
        if not isinstance(payload, Mapping):
            failures.append({
                "route": route,
                "metric": "route",
                "limit_type": "present",
                "limit": True,
                "value": None,
                "reason": "route missing from aggregate report",
            })
            continue
        selected = int(payload.get("selected", 0))
        if selected < min_selected:
            failures.append({
                "route": route,
                "metric": "selected",
                "limit_type": "min",
                "limit": int(min_selected),
                "value": selected,
            })
            continue
        checked_routes.append(route)
        metric_checks = (
            ("decision_accuracy", min_decision_accuracy, _check_min_metric),
            ("false_supported_rate", max_false_supported_rate, _check_max_metric),
            ("false_refuted_rate", min_false_refuted_rate, _check_min_metric),
            ("verified_false_alarm", max_verified_false_alarm, _check_max_metric),
            ("verified_detection", min_verified_detection, _check_min_metric),
            ("mean_duration_seconds", max_mean_duration_seconds, _check_max_metric),
            ("p95_duration_seconds", max_p95_duration_seconds, _check_max_metric),
            ("p99_duration_seconds", max_p99_duration_seconds, _check_max_metric),
            ("max_duration_seconds", max_max_duration_seconds, _check_max_metric),
            ("mean_attempted_route_count", max_mean_attempted_route_count, _check_max_metric),
            ("retrieval_use_rate", max_retrieval_use_rate, _check_max_metric),
        )
        for metric, limit, checker in metric_checks:
            if limit is None:
                continue
            failure = checker(route=route, metric=metric, value=payload.get(metric), limit=float(limit))
            if failure is not None:
                failures.append(failure)

    if min_cache_hit_rate is not None:
        total_cache = {}
        if isinstance(cache_summary, Mapping):
            raw_total = cache_summary.get("total", {})
            if isinstance(raw_total, Mapping):
                total_cache = raw_total
        failure = _check_min_metric(
            route=None,
            metric="cache_hit_rate",
            value=total_cache.get("hit_rate"),
            limit=float(min_cache_hit_rate),
        )
        if failure is not None:
            failure["reason"] = "aggregate report cache hit rate is missing or below threshold"
            failures.append(failure)

    return {
        "enabled": True,
        "passed": not failures,
        "checked_routes": checked_routes,
        "config": {
            "routes": list(route_names),
            "min_selected": int(min_selected),
            "min_decision_accuracy": min_decision_accuracy,
            "max_false_supported_rate": max_false_supported_rate,
            "min_false_refuted_rate": min_false_refuted_rate,
            "max_verified_false_alarm": max_verified_false_alarm,
            "min_verified_detection": min_verified_detection,
            "max_mean_duration_seconds": max_mean_duration_seconds,
            "max_p95_duration_seconds": max_p95_duration_seconds,
            "max_p99_duration_seconds": max_p99_duration_seconds,
            "max_max_duration_seconds": max_max_duration_seconds,
            "max_mean_attempted_route_count": max_mean_attempted_route_count,
            "max_retrieval_use_rate": max_retrieval_use_rate,
            "min_cache_hit_rate": min_cache_hit_rate,
        },
        "failures": failures,
    }


def _gate_failures_for_route(gate: Mapping[str, Any], route: str) -> list[dict[str, Any]]:
    failures = gate.get("failures", ())
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes, bytearray)):
        return []
    return [
        dict(failure)
        for failure in failures
        if isinstance(failure, Mapping) and failure.get("route") == route
    ]


def _global_gate_failures(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = gate.get("failures", ())
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes, bytearray)):
        return []
    return [
        dict(failure)
        for failure in failures
        if isinstance(failure, Mapping) and failure.get("route") is None
    ]


def build_adapter_promotion_decision(
    pareto_frontier: Mapping[str, Any],
    quality_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a route-specific adapter promotion decision from frontier and gate output."""
    recommended = pareto_frontier.get("recommended")
    if not isinstance(recommended, Mapping):
        return {
            "schema_version": 1,
            "status": "no_candidate",
            "recommended_route": None,
            "frontier_recommended_route": None,
            "reason": "no route met the Pareto frontier eligibility floor",
            "gate_status": "not_configured" if quality_gate is None else "present",
            "blocking_failures": [],
            "warnings": [],
        }

    recommended_route = str(recommended.get("route"))
    base = {
        "schema_version": 1,
        "recommended_route": recommended_route,
        "frontier_recommended_route": recommended_route,
        "frontier_rank": 1,
        "promotion_score": recommended.get("promotion_score"),
        "missing_metrics": list(recommended.get("missing_metrics", ())),
        "blocking_failures": [],
        "warnings": [],
    }
    if quality_gate is None:
        return {
            **base,
            "status": "needs_gate",
            "reason": "configure fail-closed quality/cost gate thresholds before promoting an adapter",
            "gate_status": "not_configured",
            "gate_checked_route": False,
            "route_gate_passed": None,
        }

    checked_routes = {
        str(route)
        for route in quality_gate.get("checked_routes", ())
        if route is not None
    }
    gate_failures = _gate_failures_for_route(quality_gate, recommended_route)
    global_failures = _global_gate_failures(quality_gate)
    other_failures = [
        dict(failure)
        for failure in quality_gate.get("failures", ())
        if (
            isinstance(failure, Mapping)
            and failure.get("route") not in {recommended_route, None}
        )
    ]

    if recommended_route not in checked_routes:
        return {
            **base,
            "status": "needs_gate_for_recommended",
            "reason": "recommended Pareto route was not covered by the quality/cost gate",
            "gate_status": "failed" if quality_gate.get("passed") is False else "passed",
            "gate_checked_route": False,
            "route_gate_passed": None,
            "blocking_failures": global_failures,
            "warnings": [
                {
                    "type": "unchecked_recommended_route",
                    "route": recommended_route,
                    "checked_routes": sorted(checked_routes),
                },
            ],
        }

    blocking_failures = [*global_failures, *gate_failures]
    if blocking_failures:
        return {
            **base,
            "status": "blocked_by_gate",
            "reason": "recommended Pareto route failed one or more quality/cost gate checks",
            "gate_status": "failed",
            "gate_checked_route": True,
            "route_gate_passed": False,
            "blocking_failures": blocking_failures,
            "warnings": [{"type": "other_route_gate_failures", "failures": other_failures}] if other_failures else [],
        }

    return {
        **base,
        "status": "promote",
        "reason": "recommended Pareto route is covered by the gate and has no route-specific failures",
        "gate_status": "failed" if quality_gate.get("passed") is False else "passed",
        "gate_checked_route": True,
        "route_gate_passed": True,
        "warnings": [{"type": "other_route_gate_failures", "failures": other_failures}] if other_failures else [],
    }


def build_route_comparison_report(
    reports: Sequence[tuple[str, Path]],
    *,
    alpha: float = 0.10,
    min_selected: int = 1,
    notes: Sequence[str] = (),
    gate_routes: Sequence[str] = (),
    gate_min_selected: int | None = None,
    min_decision_accuracy: float | None = None,
    max_false_supported_rate: float | None = None,
    min_false_refuted_rate: float | None = None,
    max_verified_false_alarm: float | None = None,
    min_verified_detection: float | None = None,
    max_mean_duration_seconds: float | None = None,
    max_p95_duration_seconds: float | None = None,
    max_p99_duration_seconds: float | None = None,
    max_max_duration_seconds: float | None = None,
    max_mean_attempted_route_count: float | None = None,
    max_retrieval_use_rate: float | None = None,
    min_cache_hit_rate: float | None = None,
) -> dict[str, Any]:
    """Build a route comparison report from verifier-ensemble JSON files."""
    if not reports:
        raise ValueError("at least one verifier ensemble report is required.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if min_selected < 0:
        raise ValueError("min_selected must be >= 0.")

    rows, cache_records = _extract_route_rows_and_cache_records(reports, alpha=float(alpha))
    leaderboard = _leaderboard_rows(rows, min_selected=min_selected)
    by_route = _aggregate_by_route(rows)
    cache_summary = _aggregate_cache_summary(cache_records)
    pareto_frontier = build_route_pareto_frontier(by_route, min_selected=min_selected)
    gate = build_route_quality_gate(
        by_route,
        cache_summary=cache_summary,
        routes=gate_routes,
        min_selected=min_selected if gate_min_selected is None else gate_min_selected,
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
        max_mean_duration_seconds=max_mean_duration_seconds,
        max_p95_duration_seconds=max_p95_duration_seconds,
        max_p99_duration_seconds=max_p99_duration_seconds,
        max_max_duration_seconds=max_max_duration_seconds,
        max_mean_attempted_route_count=max_mean_attempted_route_count,
        max_retrieval_use_rate=max_retrieval_use_rate,
        min_cache_hit_rate=min_cache_hit_rate,
    )
    promotion_decision = build_adapter_promotion_decision(pareto_frontier, gate)
    payload = {
        "schema_version": 1,
        "alpha": float(alpha),
        "min_selected": int(min_selected),
        "n_reports": len(reports),
        "n_route_entries": len(rows),
        "n_leaderboard_entries": len(leaderboard),
        "leaderboard": leaderboard,
        "by_route": by_route,
        "cache_summary": cache_summary,
        "pareto_frontier": pareto_frontier,
        "promotion_decision": promotion_decision,
        "rows": rows,
        "notes": list(notes),
    }
    if gate is not None:
        payload["quality_gate"] = gate
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [_parse_named_path(value) for value in args.report]
    payload = build_route_comparison_report(
        reports,
        alpha=args.alpha,
        min_selected=args.min_selected,
        notes=args.note,
        gate_routes=args.gate_route,
        gate_min_selected=args.gate_min_selected,
        min_decision_accuracy=args.min_decision_accuracy,
        max_false_supported_rate=args.max_false_supported_rate,
        min_false_refuted_rate=args.min_false_refuted_rate,
        max_verified_false_alarm=args.max_verified_false_alarm,
        min_verified_detection=args.min_verified_detection,
        max_mean_duration_seconds=args.max_mean_duration_seconds,
        max_p95_duration_seconds=args.max_p95_duration_seconds,
        max_p99_duration_seconds=args.max_p99_duration_seconds,
        max_max_duration_seconds=args.max_max_duration_seconds,
        max_mean_attempted_route_count=args.max_mean_attempted_route_count,
        max_retrieval_use_rate=args.max_retrieval_use_rate,
        min_cache_hit_rate=args.min_cache_hit_rate,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote verifier route comparison to {output_path}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare verifier route metrics across reports")
    parser.add_argument("--report", action="append", required=True,
                        help="verifier ensemble report path, optionally named as name=path; repeatable")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="alpha key to use for route_control_impact")
    parser.add_argument("--min-selected", type=int, default=1,
                        help="minimum selected records required for leaderboard rows")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the output report; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--gate-route", action="append", default=[],
                        help="aggregate route to gate; repeatable. Defaults to all routes passing --min-selected")
    parser.add_argument("--gate-min-selected", type=int, default=None,
                        help="minimum selected records for gated routes; defaults to --min-selected")
    parser.add_argument("--min-decision-accuracy", type=float, default=None,
                        help="fail gate when decision_accuracy is below this value")
    parser.add_argument("--max-false-supported-rate", type=float, default=None,
                        help="fail gate when false_supported_rate exceeds this value")
    parser.add_argument("--min-false-refuted-rate", type=float, default=None,
                        help="fail gate when false_refuted_rate is below this value")
    parser.add_argument("--max-verified-false-alarm", type=float, default=None,
                        help="fail gate when verified_false_alarm exceeds this value")
    parser.add_argument("--min-verified-detection", type=float, default=None,
                        help="fail gate when verified_detection is below this value")
    parser.add_argument("--max-mean-duration-seconds", type=float, default=None,
                        help="fail gate when aggregate mean_duration_seconds exceeds this value")
    parser.add_argument("--max-p95-duration-seconds", type=float, default=None,
                        help="fail gate when aggregate p95_duration_seconds exceeds this value")
    parser.add_argument("--max-p99-duration-seconds", type=float, default=None,
                        help="fail gate when aggregate p99_duration_seconds exceeds this value")
    parser.add_argument("--max-max-duration-seconds", type=float, default=None,
                        help="fail gate when aggregate max_duration_seconds exceeds this value")
    parser.add_argument("--max-mean-attempted-route-count", type=float, default=None,
                        help="fail gate when mean_attempted_route_count exceeds this value")
    parser.add_argument("--max-retrieval-use-rate", type=float, default=None,
                        help="fail gate when retrieval_use_rate exceeds this value")
    parser.add_argument("--min-cache-hit-rate", type=float, default=None,
                        help="fail gate when aggregate report cache hit rate is below this value")
    parser.add_argument("--fail-on-gate", action="store_true",
                        help="exit non-zero when the route quality gate fails")
    parser.add_argument("--fail-on-promotion", action="store_true",
                        help="exit non-zero unless promotion_decision.status is promote")
    args = parser.parse_args(argv)
    payload = run(args)
    for item in payload["leaderboard"][:10]:
        print(
            f"{item['report']}/{item['run']}:{item['route']} "
            f"selected={item['selected']} "
            f"decision_accuracy={item['decision_accuracy']} "
            f"false_refuted={item['false_refuted_rate']} "
            f"false_supported={item['false_supported_rate']}"
        )
    gate = payload.get("quality_gate")
    if gate is not None:
        print(f"quality_gate={'passed' if gate['passed'] else 'failed'}")
        if args.fail_on_gate and not gate["passed"]:
            raise SystemExit(1)
    decision = payload.get("promotion_decision")
    if isinstance(decision, Mapping):
        print(
            f"promotion_decision={decision['status']} "
            f"route={decision.get('recommended_route')}"
        )
        if args.fail_on_promotion and decision["status"] != "promote":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
