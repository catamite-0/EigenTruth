"""Compare EigenTruth benchmark profile payloads across runs.

This is a post-processing helper for ``eval_truthfulqa.py --profile`` and
``--profile-json`` outputs. It does not load models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("profile name cannot be empty.")
    return name, Path(path)


def _parse_named_float(value: str, *, flag: str) -> tuple[str, float]:
    if "=" not in value:
        raise ValueError(f"{flag} must be formatted as name=value.")
    name, raw_value = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"{flag} name cannot be empty.")
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{flag} value for {name!r} must be numeric.") from exc
    if threshold < 0:
        raise ValueError(f"{flag} value for {name!r} must be non-negative.")
    return name, threshold


def _load_profile(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    profile = payload.get("profile", payload)
    if not isinstance(profile, Mapping):
        raise ValueError(f"profile payload in {path} must be a JSON object.")
    if "total_seconds" not in profile or "phases" not in profile:
        raise ValueError(f"profile payload in {path} must include total_seconds and phases.")
    phases = profile["phases"]
    if not isinstance(phases, Mapping):
        raise ValueError(f"profile phases in {path} must be a JSON object.")
    total_seconds = float(profile["total_seconds"])
    phase_seconds = {str(name): float(seconds) for name, seconds in phases.items()}
    if total_seconds < 0 or any(seconds < 0 for seconds in phase_seconds.values()):
        raise ValueError(f"profile payload in {path} contains negative timing values.")
    summary = profile.get("summary")
    if not isinstance(summary, Mapping):
        summary = _minimal_profile_summary(phase_seconds, total_seconds)
    return {
        "total_seconds": total_seconds,
        "phases": phase_seconds,
        "summary": dict(summary),
    }


def _minimal_profile_summary(phases: Mapping[str, float], total_seconds: float) -> dict[str, Any]:
    top_phases = [
        {"name": name, "seconds": seconds, "share": _safe_div(seconds, total_seconds)}
        for name, seconds in sorted(phases.items(), key=lambda item: item[1], reverse=True)[:5]
    ]
    return {
        "bottleneck": top_phases[0]["name"] if top_phases else None,
        "top_phases": top_phases,
        "groups": {},
        "throughput": {},
    }


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _delta(current: float, baseline: float) -> dict[str, float | None]:
    ratio = _safe_div(current, baseline)
    return {
        "seconds": current,
        "baseline_seconds": baseline,
        "delta_seconds": current - baseline,
        "ratio_to_baseline": ratio,
        "speedup_vs_baseline": None if ratio in {None, 0.0} else 1.0 / ratio,
    }


def _phase_deltas(current: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, dict[str, float | None]]:
    names = sorted(set(current) | set(baseline))
    return {
        name: _delta(float(current.get(name, 0.0)), float(baseline.get(name, 0.0)))
        for name in names
    }


def _group_seconds(summary: Mapping[str, Any]) -> dict[str, float]:
    groups = summary.get("groups", {})
    if not isinstance(groups, Mapping):
        return {}
    result = {}
    for name, payload in groups.items():
        if isinstance(payload, Mapping):
            result[str(name)] = float(payload.get("seconds", 0.0))
    return result


def _throughput_values(summary: Mapping[str, Any]) -> dict[str, float]:
    values = summary.get("throughput", {})
    if not isinstance(values, Mapping):
        return {}
    return {str(name): float(value) for name, value in values.items()}


def _throughput_deltas(
    current: Mapping[str, float],
    baseline: Mapping[str, float],
) -> dict[str, dict[str, float | None]]:
    names = sorted(set(current) | set(baseline))
    payload = {}
    for name in names:
        current_value = float(current.get(name, 0.0))
        baseline_value = float(baseline.get(name, 0.0))
        payload[name] = {
            "value": current_value,
            "baseline_value": baseline_value,
            "delta": current_value - baseline_value,
            "ratio_to_baseline": _safe_div(current_value, baseline_value),
        }
    return payload


def _check_max_ratio(
    *,
    ratio: float | None,
    current: float,
    baseline: float,
    maximum: float,
) -> bool:
    if ratio is not None:
        return ratio <= maximum
    return current <= baseline


def _check_min_ratio(*, ratio: float | None, minimum: float, current: float, baseline: float) -> bool:
    if ratio is not None:
        return ratio >= minimum
    return current >= baseline


def _build_regression_gate(
    runs: Sequence[dict[str, Any]],
    *,
    baseline: str,
    max_total_ratio: float | None,
    max_run_total_ratios: Mapping[str, float],
    max_phase_ratios: Mapping[str, float],
    min_throughput_ratios: Mapping[str, float],
) -> dict[str, Any] | None:
    if (
        max_total_ratio is None
        and not max_run_total_ratios
        and not max_phase_ratios
        and not min_throughput_ratios
    ):
        return None
    failures = []
    checked_runs = [run for run in runs if run["name"] != baseline]

    for run in checked_runs:
        name = run["name"]
        run_total_limit = max_run_total_ratios.get(name, max_total_ratio)
        if run_total_limit is not None:
            total = run["total_delta"]
            if not _check_max_ratio(
                ratio=total["ratio_to_baseline"],
                current=total["seconds"],
                baseline=total["baseline_seconds"],
                maximum=run_total_limit,
            ):
                failures.append({
                    "run": name,
                    "metric": "total_seconds",
                    "limit_type": "max_ratio_to_baseline",
                    "limit": run_total_limit,
                    "value": total["ratio_to_baseline"],
                    "seconds": total["seconds"],
                    "baseline_seconds": total["baseline_seconds"],
                })

        for phase_name, limit in max_phase_ratios.items():
            phase = run["phase_deltas"].get(phase_name)
            if phase is None:
                failures.append({
                    "run": name,
                    "metric": f"phase:{phase_name}",
                    "limit_type": "max_ratio_to_baseline",
                    "limit": limit,
                    "value": None,
                    "reason": "phase missing from comparison",
                })
                continue
            if not _check_max_ratio(
                ratio=phase["ratio_to_baseline"],
                current=phase["seconds"],
                baseline=phase["baseline_seconds"],
                maximum=limit,
            ):
                failures.append({
                    "run": name,
                    "metric": f"phase:{phase_name}",
                    "limit_type": "max_ratio_to_baseline",
                    "limit": limit,
                    "value": phase["ratio_to_baseline"],
                    "seconds": phase["seconds"],
                    "baseline_seconds": phase["baseline_seconds"],
                })

        for metric_name, limit in min_throughput_ratios.items():
            throughput = run["throughput_deltas"].get(metric_name)
            if throughput is None:
                failures.append({
                    "run": name,
                    "metric": f"throughput:{metric_name}",
                    "limit_type": "min_ratio_to_baseline",
                    "limit": limit,
                    "value": None,
                    "reason": "throughput metric missing from comparison",
                })
                continue
            if not _check_min_ratio(
                ratio=throughput["ratio_to_baseline"],
                minimum=limit,
                current=throughput["value"],
                baseline=throughput["baseline_value"],
            ):
                failures.append({
                    "run": name,
                    "metric": f"throughput:{metric_name}",
                    "limit_type": "min_ratio_to_baseline",
                    "limit": limit,
                    "value": throughput["ratio_to_baseline"],
                    "throughput": throughput["value"],
                    "baseline_throughput": throughput["baseline_value"],
                })

    return {
        "enabled": True,
        "passed": not failures,
        "checked_runs": [run["name"] for run in checked_runs],
        "config": {
            "max_total_ratio": max_total_ratio,
            "max_run_total_ratios": dict(max_run_total_ratios),
            "max_phase_ratios": dict(max_phase_ratios),
            "min_throughput_ratios": dict(min_throughput_ratios),
        },
        "failures": failures,
    }


def _validate_non_negative_thresholds(
    *,
    max_total_ratio: float | None,
    max_run_total_ratios: Mapping[str, float],
    max_phase_ratios: Mapping[str, float],
    min_throughput_ratios: Mapping[str, float],
) -> None:
    if max_total_ratio is not None and max_total_ratio < 0:
        raise ValueError("max_total_ratio must be non-negative.")
    for name, value in max_run_total_ratios.items():
        if value < 0:
            raise ValueError(f"max_run_total_ratios[{name!r}] must be non-negative.")
    for name, value in max_phase_ratios.items():
        if value < 0:
            raise ValueError(f"max_phase_ratios[{name!r}] must be non-negative.")
    for name, value in min_throughput_ratios.items():
        if value < 0:
            raise ValueError(f"min_throughput_ratios[{name!r}] must be non-negative.")


def build_profile_comparison(
    profiles: Sequence[tuple[str, Path]],
    *,
    baseline: str | None = None,
    notes: Sequence[str] = (),
    max_total_ratio: float | None = None,
    max_run_total_ratios: Mapping[str, float] | None = None,
    max_phase_ratios: Mapping[str, float] | None = None,
    min_throughput_ratios: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if not profiles:
        raise ValueError("at least one profile is required.")
    max_run_total_ratios = dict(max_run_total_ratios or {})
    max_phase_ratios = dict(max_phase_ratios or {})
    min_throughput_ratios = dict(min_throughput_ratios or {})
    _validate_non_negative_thresholds(
        max_total_ratio=max_total_ratio,
        max_run_total_ratios=max_run_total_ratios,
        max_phase_ratios=max_phase_ratios,
        min_throughput_ratios=min_throughput_ratios,
    )
    loaded = []
    seen = set()
    for name, path in profiles:
        if name in seen:
            raise ValueError(f"profile name {name!r} is duplicated.")
        seen.add(name)
        profile = _load_profile(path)
        loaded.append({
            "name": name,
            "source": str(path),
            **profile,
        })

    baseline_name = baseline or loaded[0]["name"]
    baseline_profile = next((item for item in loaded if item["name"] == baseline_name), None)
    if baseline_profile is None:
        raise ValueError(f"baseline profile {baseline_name!r} was not provided.")

    baseline_total = float(baseline_profile["total_seconds"])
    baseline_phases = baseline_profile["phases"]
    baseline_groups = _group_seconds(baseline_profile["summary"])
    baseline_throughput = _throughput_values(baseline_profile["summary"])

    runs = []
    for item in loaded:
        total_seconds = float(item["total_seconds"])
        summary = item["summary"]
        runs.append({
            "name": item["name"],
            "source": item["source"],
            "total_seconds": total_seconds,
            "bottleneck": summary.get("bottleneck"),
            "total_delta": _delta(total_seconds, baseline_total),
            "phase_deltas": _phase_deltas(item["phases"], baseline_phases),
            "group_deltas": _phase_deltas(_group_seconds(summary), baseline_groups),
            "throughput_deltas": _throughput_deltas(_throughput_values(summary), baseline_throughput),
            "top_phases": summary.get("top_phases", []),
        })

    fastest = min(runs, key=lambda run: float(run["total_seconds"]))
    slowest = max(runs, key=lambda run: float(run["total_seconds"]))
    payload = {
        "baseline": baseline_name,
        "n_profiles": len(runs),
        "fastest": {
            "name": fastest["name"],
            "total_seconds": fastest["total_seconds"],
            "speedup_vs_baseline": fastest["total_delta"]["speedup_vs_baseline"],
        },
        "slowest": {
            "name": slowest["name"],
            "total_seconds": slowest["total_seconds"],
            "speedup_vs_baseline": slowest["total_delta"]["speedup_vs_baseline"],
        },
        "runs": runs,
        "notes": list(notes),
    }
    gate = _build_regression_gate(
        runs,
        baseline=baseline_name,
        max_total_ratio=max_total_ratio,
        max_run_total_ratios=max_run_total_ratios,
        max_phase_ratios=max_phase_ratios,
        min_throughput_ratios=min_throughput_ratios,
    )
    if gate is not None:
        payload["regression_gate"] = gate
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    profiles = [_parse_named_path(value) for value in args.profile]
    max_run_total_ratios = dict(_parse_named_float(value, flag="--max-run-total-ratio")
                                for value in args.max_run_total_ratio)
    max_phase_ratios = dict(_parse_named_float(value, flag="--max-phase-ratio")
                            for value in args.max_phase_ratio)
    min_throughput_ratios = dict(_parse_named_float(value, flag="--min-throughput-ratio")
                                 for value in args.min_throughput_ratio)
    payload = build_profile_comparison(
        profiles,
        baseline=args.baseline,
        notes=args.note,
        max_total_ratio=args.max_total_ratio,
        max_run_total_ratios=max_run_total_ratios,
        max_phase_ratios=max_phase_ratios,
        min_throughput_ratios=min_throughput_ratios,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote profile comparison to {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EigenTruth benchmark profile payloads")
    parser.add_argument("--profile", action="append", required=True,
                        help="profile JSON path, optionally named as name=path; repeatable")
    parser.add_argument("--baseline", default=None,
                        help="profile name to use as the baseline; defaults to the first --profile")
    parser.add_argument("--note", action="append", default=[],
                        help="optional note to include in the output report; repeatable")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--max-total-ratio", type=float, default=None,
                        help="fail when any non-baseline run exceeds this total-time ratio")
    parser.add_argument("--max-run-total-ratio", action="append", default=[],
                        help="fail when one named run exceeds this total-time ratio, formatted as run=ratio; "
                             "overrides --max-total-ratio for that run")
    parser.add_argument("--max-phase-ratio", action="append", default=[],
                        help="fail when a phase exceeds this ratio, formatted as phase=ratio; repeatable")
    parser.add_argument("--min-throughput-ratio", action="append", default=[],
                        help="fail when throughput drops below this ratio, formatted as metric=ratio; repeatable")
    args = parser.parse_args()
    if args.max_total_ratio is not None and args.max_total_ratio < 0:
        raise ValueError("--max-total-ratio must be non-negative.")
    payload = run(args)
    for item in payload["runs"]:
        delta = item["total_delta"]
        speedup = delta["speedup_vs_baseline"]
        print(
            f"{item['name']}: total={item['total_seconds']:.3f}s "
            f"delta={delta['delta_seconds']:+.3f}s "
            f"speedup={speedup if speedup is not None else 'n/a'} "
            f"bottleneck={item['bottleneck']}"
        )
    gate = payload.get("regression_gate")
    if gate is not None:
        status = "passed" if gate["passed"] else "failed"
        print(f"regression_gate={status}")
        if not gate["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
