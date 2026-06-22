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


def build_profile_comparison(
    profiles: Sequence[tuple[str, Path]],
    *,
    baseline: str | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    if not profiles:
        raise ValueError("at least one profile is required.")
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
    return {
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    profiles = [_parse_named_path(value) for value in args.profile]
    payload = build_profile_comparison(profiles, baseline=args.baseline, notes=args.note)
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
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
