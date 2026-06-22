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
        "internal_false_alarm": _as_float(internal.get("false_alarm")),
        "verified_false_alarm": _as_float(verified.get("false_alarm")),
        "delta_false_alarm": _as_float(delta.get("false_alarm")),
        "internal_detection": _as_float(internal.get("detection")),
        "verified_detection": _as_float(verified.get("detection")),
        "delta_detection": _as_float(delta.get("detection")),
        "suppressed_false_alarm_rate": _as_float(delta.get("suppressed_false_alarm_rate")),
        "rescued_detection_rate": _as_float(delta.get("rescued_detection_rate")),
    }


def _extract_route_rows(
    reports: Sequence[tuple[str, Path]],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    rows = []
    for report_name, path in reports:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        runs = report.get("runs", ())
        if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
            raise ValueError(f"report {path} has invalid runs field.")
        for run in runs:
            if not isinstance(run, Mapping):
                continue
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
            int(row.get("selected", 0)),
        ),
        reverse=True,
    )
    return eligible


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


def _thresholds_enabled(
    *,
    min_decision_accuracy: float | None,
    max_false_supported_rate: float | None,
    min_false_refuted_rate: float | None,
    max_verified_false_alarm: float | None,
    min_verified_detection: float | None,
) -> bool:
    return any(
        value is not None
        for value in (
            min_decision_accuracy,
            max_false_supported_rate,
            min_false_refuted_rate,
            max_verified_false_alarm,
            min_verified_detection,
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
    routes: Sequence[str] = (),
    min_selected: int = 1,
    min_decision_accuracy: float | None = None,
    max_false_supported_rate: float | None = None,
    min_false_refuted_rate: float | None = None,
    max_verified_false_alarm: float | None = None,
    min_verified_detection: float | None = None,
) -> dict[str, Any] | None:
    """Build a fail-closed route quality gate over aggregate route metrics."""
    if min_selected < 0:
        raise ValueError("gate min_selected must be >= 0.")
    min_decision_accuracy = _validated_limit("min_decision_accuracy", min_decision_accuracy)
    max_false_supported_rate = _validated_limit("max_false_supported_rate", max_false_supported_rate)
    min_false_refuted_rate = _validated_limit("min_false_refuted_rate", min_false_refuted_rate)
    max_verified_false_alarm = _validated_limit("max_verified_false_alarm", max_verified_false_alarm)
    min_verified_detection = _validated_limit("min_verified_detection", min_verified_detection)
    enabled = bool(routes) or _thresholds_enabled(
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
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
        )
        for metric, limit, checker in metric_checks:
            if limit is None:
                continue
            failure = checker(route=route, metric=metric, value=payload.get(metric), limit=float(limit))
            if failure is not None:
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
        },
        "failures": failures,
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
) -> dict[str, Any]:
    """Build a route comparison report from verifier-ensemble JSON files."""
    if not reports:
        raise ValueError("at least one verifier ensemble report is required.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if min_selected < 0:
        raise ValueError("min_selected must be >= 0.")

    rows = _extract_route_rows(reports, alpha=float(alpha))
    leaderboard = _leaderboard_rows(rows, min_selected=min_selected)
    by_route = _aggregate_by_route(rows)
    payload = {
        "schema_version": 1,
        "alpha": float(alpha),
        "min_selected": int(min_selected),
        "n_reports": len(reports),
        "n_route_entries": len(rows),
        "n_leaderboard_entries": len(leaderboard),
        "leaderboard": leaderboard,
        "by_route": by_route,
        "rows": rows,
        "notes": list(notes),
    }
    gate = build_route_quality_gate(
        by_route,
        routes=gate_routes,
        min_selected=min_selected if gate_min_selected is None else gate_min_selected,
        min_decision_accuracy=min_decision_accuracy,
        max_false_supported_rate=max_false_supported_rate,
        min_false_refuted_rate=min_false_refuted_rate,
        max_verified_false_alarm=max_verified_false_alarm,
        min_verified_detection=min_verified_detection,
    )
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
    parser.add_argument("--fail-on-gate", action="store_true",
                        help="exit non-zero when the route quality gate fails")
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


if __name__ == "__main__":
    main()
