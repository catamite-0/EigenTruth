"""Evaluate multi-seed stability for conformal abstention promotion gates.

This is a model-free post-hoc benchmark. It consumes existing score dumps,
replays conformal abstention candidate selection across seeded calibration
splits, and records whether the recommended participation gate remains stable
and passes release thresholds.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.calibration import DEFAULT_SCORE_DIRECTIONS  # noqa: E402
from eigentruth.eval.conformal import (  # noqa: E402
    ABSTENTION_COMPARISON_METRICS,
    ConformalAbstentionComparisonCandidate,
    ConformalAbstentionComparisonReport,
    ConformalAbstentionReport,
    conformal_abstention_release_gate,
    directional_conformal_threshold,
    evaluate_conformal_abstention,
)
from eigentruth.eval.score_dump import (  # noqa: E402
    load_score_dump_columns,
    score_dump_cache_summary,
    score_dump_file_metadata,
)
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("scores name cannot be empty.")
    return name, Path(path)


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...]:
    if value is None:
        raise ValueError(f"{name} must contain at least one value.")
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    return parts


def _parse_int_csv(value: str | None, *, name: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in _parse_csv(value, name=name))
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate integers.")
    return values


def _float_stats(values: Sequence[float | None]) -> dict[str, Any]:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return {"count": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "count": len(finite_values),
        "mean": statistics.fmean(finite_values),
        "stdev": statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0,
        "min": min(finite_values),
        "max": max(finite_values),
    }


def _validate_unique_score_dump_names(score_dumps: Sequence[tuple[str, Path]]) -> None:
    counts = Counter(name for name, _ in score_dumps)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"score dump names must be unique; duplicate name(s): {duplicates}.")


def _summary_int(summary: Mapping[str, Any], key: str) -> int | None:
    value = summary.get(key)
    if value is None:
        return None
    return int(value)


def _validate_score_dump_metadata(
    name: str,
    path: Path,
    metadata: Mapping[str, Any],
) -> None:
    if not metadata.get("exists"):
        raise ValueError(f"score dump {name!r} does not exist: {path}.")
    if metadata.get("kind") != "file":
        raise ValueError(f"score dump {name!r} must be a file: {path}.")
    records = metadata.get("records")
    if isinstance(records, Mapping) and not records.get("exists"):
        raise ValueError(f"score dump {name!r} records sidecar does not exist: {records.get('path')}.")
    summary = metadata.get("summary")
    if not isinstance(summary, Mapping):
        return
    n_true = _summary_int(summary, "n_true")
    n_false = _summary_int(summary, "n_false")
    if n_true is not None and n_true < 2:
        raise ValueError(
            f"score dump {name!r} must contain at least 2 true labels for abstention stability; "
            f"got {n_true}."
        )
    if n_false is not None and n_false < 1:
        raise ValueError(
            f"score dump {name!r} must contain at least 1 false label for abstention stability; "
            f"got {n_false}."
        )


def _direction_for(signal: str, override: str | None) -> str:
    return override or DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")


def _split_group(indices: Sequence[int], rng: random.Random) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return (), tuple(shuffled)
    split = max(1, len(shuffled) // 2)
    if split >= len(shuffled):
        split = len(shuffled) - 1
    return tuple(shuffled[:split]), tuple(shuffled[split:])


def _split_indices(labels: Sequence[int], *, seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    true_indices = tuple(index for index, label in enumerate(labels) if int(label) == 0)
    false_indices = tuple(index for index, label in enumerate(labels) if int(label) == 1)
    if len(true_indices) < 2:
        raise ValueError("abstention stability requires at least two true labels.")
    if not false_indices:
        raise ValueError("abstention stability requires at least one false label.")
    rng = random.Random(int(seed))
    true_calib, true_eval = _split_group(true_indices, rng)
    false_calib, false_eval = _split_group(false_indices, rng)
    calibration = tuple(sorted((*true_calib, *false_calib)))
    evaluation = tuple(sorted((*true_eval, *false_eval)))
    if not true_calib:
        raise ValueError("seeded split produced no true calibration records.")
    if not evaluation:
        raise ValueError("seeded split produced no evaluation records.")
    return calibration, evaluation


def _subset(values: Sequence[float], indices: Sequence[int]) -> tuple[float, ...]:
    return tuple(float(values[index]) for index in indices)


def _correctness(labels: Sequence[int], indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(1 if int(labels[index]) == 0 else 0 for index in indices)


def _candidate_metric(report: ConformalAbstentionReport, metric: str) -> float | None:
    value = getattr(report, metric)
    if value is None:
        return None
    return float(value)


def _sortable_value(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def _rank_abstention_reports(
    reports: Sequence[ConformalAbstentionReport],
    *,
    alpha: float,
    best_by: str,
) -> ConformalAbstentionComparisonReport:
    if best_by not in ABSTENTION_COMPARISON_METRICS:
        raise ValueError(f"best_by must be one of {ABSTENTION_COMPARISON_METRICS}.")

    def sort_key(report: ConformalAbstentionReport) -> tuple[float, float, float, float, float, str]:
        score_name = "" if report.score_name is None else report.score_name
        return (
            -_sortable_value(_candidate_metric(report, best_by)),
            -_sortable_value(_candidate_metric(report, "conditional_correctness_lower_bound")),
            -_sortable_value(_candidate_metric(report, "empirical_selective_accuracy")),
            -_sortable_value(_candidate_metric(report, "empirical_participation_rate")),
            -_sortable_value(_candidate_metric(report, "correct_retention_lower_bound")),
            score_name,
        )

    ranked_reports = sorted(reports, key=sort_key)
    return ConformalAbstentionComparisonReport(
        alpha=alpha,
        best_by=best_by,
        candidates=tuple(
            ConformalAbstentionComparisonCandidate(
                rank=rank,
                score_name=str(report.score_name),
                direction=report.direction,
                selection_metric=best_by,
                selection_value=_candidate_metric(report, best_by),
                report=report,
            )
            for rank, report in enumerate(ranked_reports, start=1)
        ),
    )


def _seed_abstention_comparison(
    labels: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    *,
    signals: Sequence[str],
    directions: Mapping[str, str],
    seed: int,
    alpha: float,
    best_by: str,
) -> ConformalAbstentionComparisonReport:
    calibration_indices, evaluation_indices = _split_indices(labels, seed=seed)
    reports: list[ConformalAbstentionReport] = []
    for signal in signals:
        signal_scores = scores[signal]
        direction = directions[signal]
        correct_calibration_indices = tuple(
            index for index in calibration_indices if int(labels[index]) == 0
        )
        threshold = directional_conformal_threshold(
            _subset(signal_scores, correct_calibration_indices),
            alpha,
            direction,
        )
        reports.append(
            evaluate_conformal_abstention(
                _subset(signal_scores, evaluation_indices),
                _correctness(labels, evaluation_indices),
                threshold=threshold,
                alpha=alpha,
                direction=direction,
                score_name=signal,
            )
        )
    return _rank_abstention_reports(reports, alpha=alpha, best_by=best_by)


def _candidate_summary(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    report = candidate.get("report")
    if not isinstance(report, Mapping):
        report = {}
    return {
        "rank": candidate.get("rank"),
        "score_name": candidate.get("score_name"),
        "direction": candidate.get("direction"),
        "selection_metric": candidate.get("selection_metric"),
        "selection_value": candidate.get("selection_value"),
        "threshold": report.get("threshold"),
        "empirical_participation_rate": report.get("empirical_participation_rate"),
        "empirical_abstention_rate": report.get("empirical_abstention_rate"),
        "empirical_selective_accuracy": report.get("empirical_selective_accuracy"),
        "correct_retention_lower_bound": report.get("correct_retention_lower_bound"),
        "conditional_correctness_lower_bound": report.get(
            "conditional_correctness_lower_bound"
        ),
    }


def _report_summary(report: ConformalAbstentionReport) -> dict[str, Any]:
    return {
        "score_name": report.score_name,
        "direction": report.direction,
        "threshold": report.threshold,
        "empirical_participation_rate": report.empirical_participation_rate,
        "empirical_abstention_rate": report.empirical_abstention_rate,
        "empirical_selective_accuracy": report.empirical_selective_accuracy,
        "correct_retention_lower_bound": report.correct_retention_lower_bound,
        "conditional_correctness_lower_bound": report.conditional_correctness_lower_bound,
        "retained_count": report.retained_count,
        "correct_retained_count": report.correct_retained_count,
        "abstained_count": report.abstained_count,
    }


def _best_supervised_threshold_report(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    signal: str,
    direction: str,
    alpha: float,
    max_abstention_rate: float,
) -> ConformalAbstentionReport | None:
    thresholds = sorted({float(score) for score in scores})
    if not thresholds:
        return None
    best: ConformalAbstentionReport | None = None
    correctness = _correctness(labels, tuple(range(len(labels))))
    for threshold in thresholds:
        report = evaluate_conformal_abstention(
            scores,
            correctness,
            threshold=threshold,
            alpha=alpha,
            direction=direction,
            score_name=signal,
        )
        if report.empirical_abstention_rate > max_abstention_rate:
            continue
        if best is None or _supervised_feasibility_sort_key(report) > _supervised_feasibility_sort_key(best):
            best = report
    return best


def _supervised_feasibility_sort_key(report: ConformalAbstentionReport) -> tuple[float, float, float, float, str]:
    return (
        report.conditional_correctness_lower_bound,
        -1.0 if report.empirical_selective_accuracy is None else report.empirical_selective_accuracy,
        report.correct_retention_lower_bound,
        -report.empirical_abstention_rate,
        "" if report.score_name is None else report.score_name,
    )


def _supervised_feasibility_frontier(
    labels: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    *,
    signals: Sequence[str],
    directions: Mapping[str, str],
    alpha: float,
    min_conditional_correctness_lower_bound: float,
    max_abstention_rate: float,
) -> dict[str, Any]:
    signal_reports = []
    for signal in signals:
        report = _best_supervised_threshold_report(
            labels,
            scores[signal],
            signal=signal,
            direction=directions[signal],
            alpha=alpha,
            max_abstention_rate=max_abstention_rate,
        )
        if report is None:
            signal_reports.append({
                "score_name": signal,
                "direction": directions[signal],
                "target_passed": False,
                "blocking_reasons": ("no threshold satisfied the abstention budget",),
            })
            continue
        summary = _report_summary(report)
        passed = (
            report.conditional_correctness_lower_bound
            >= min_conditional_correctness_lower_bound
            and report.empirical_abstention_rate <= max_abstention_rate
        )
        summary.update({
            "target_passed": passed,
            "blocking_reasons": (
                []
                if passed
                else [
                    "conditional_correctness_lower_bound "
                    f"{report.conditional_correctness_lower_bound:.6g} "
                    "is below required minimum "
                    f"{min_conditional_correctness_lower_bound:.6g}"
                ]
            ),
        })
        signal_reports.append(summary)

    ranked = sorted(
        signal_reports,
        key=lambda item: (
            -_sortable_value(item.get("conditional_correctness_lower_bound")),
            -_sortable_value(item.get("empirical_selective_accuracy")),
            _sortable_value(item.get("empirical_abstention_rate")),
            str(item.get("score_name", "")),
        ),
    )
    best = None if not ranked else ranked[0]
    return {
        "scope": "supervised_full_run_threshold_sweep",
        "uses_labels": True,
        "promotion_eligible": False,
        "note": (
            "Diagnostic upper bound only: this sweep uses held-out labels and "
            "must not be used as a runtime calibration artifact."
        ),
        "target": {
            "min_conditional_correctness_lower_bound": float(
                min_conditional_correctness_lower_bound
            ),
            "max_abstention_rate": float(max_abstention_rate),
        },
        "target_passed": bool(best and best.get("target_passed") is True),
        "best": best,
        "signals": ranked,
    }


def _metric_values(
    seed_entries: Sequence[Mapping[str, Any]],
    section: str,
    metric: str,
) -> tuple[float | None, ...]:
    values = []
    for entry in seed_entries:
        payload = entry.get(section, {})
        if isinstance(payload, Mapping):
            value = payload.get(metric)
            values.append(None if value is None else float(value))
    return tuple(values)


def _recommended_counts(seed_entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for entry in seed_entries:
        recommended = entry.get("recommended")
        if not isinstance(recommended, Mapping):
            counter["<none>"] += 1
        else:
            counter[str(recommended.get("score_name", "<missing>"))] += 1
    return dict(sorted(counter.items()))


def _summarize_seed_entries(seed_entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recommended_counts = _recommended_counts(seed_entries)
    stable_score = None
    if len(recommended_counts) == 1:
        score_name, count = next(iter(recommended_counts.items()))
        if count == len(seed_entries):
            stable_score = score_name
    release_pass_count = sum(
        1
        for entry in seed_entries
        if isinstance(entry.get("release_gate"), Mapping)
        and entry["release_gate"].get("passed") is True
    )
    return {
        "seed_count": len(seed_entries),
        "recommended_score_name_counts": recommended_counts,
        "stable_recommended_score_name": stable_score,
        "release_gate_pass_seed_count": release_pass_count,
        "release_gate_block_seed_count": len(seed_entries) - release_pass_count,
        "all_release_gates_passed": release_pass_count == len(seed_entries),
        "conditional_correctness_lower_bound": _float_stats(
            _metric_values(seed_entries, "recommended", "conditional_correctness_lower_bound")
        ),
        "empirical_abstention_rate": _float_stats(
            _metric_values(seed_entries, "recommended", "empirical_abstention_rate")
        ),
        "empirical_participation_rate": _float_stats(
            _metric_values(seed_entries, "recommended", "empirical_participation_rate")
        ),
        "empirical_selective_accuracy": _float_stats(
            _metric_values(seed_entries, "recommended", "empirical_selective_accuracy")
        ),
        "correct_retention_lower_bound": _float_stats(
            _metric_values(seed_entries, "recommended", "correct_retention_lower_bound")
        ),
    }


def build_abstention_stability_report(
    score_dumps: Sequence[tuple[str, Path]],
    *,
    signals: Sequence[str],
    seeds: Sequence[int],
    alpha: float = 0.10,
    best_by: str = "conditional_correctness_lower_bound",
    direction_override: str | None = None,
    min_conditional_correctness_lower_bound: float = 0.80,
    max_abstention_rate: float = 0.50,
) -> dict[str, Any]:
    """Build a compact abstention-gate stability report from existing score dumps."""
    if not score_dumps:
        raise ValueError("at least one score dump is required.")
    if not signals:
        raise ValueError("at least one signal is required.")
    if not seeds:
        raise ValueError("at least one seed is required.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if best_by not in ABSTENTION_COMPARISON_METRICS:
        raise ValueError(f"best_by must be one of {ABSTENTION_COMPARISON_METRICS}.")
    if direction_override is not None and direction_override not in {"higher", "lower"}:
        raise ValueError("direction_override must be 'higher', 'lower', or None.")
    _validate_unique_score_dump_names(score_dumps)

    score_dump_cache: dict[str, Any] = {}
    source_metadata_by_name = {}
    column_view_by_name = {}
    for name, path in score_dumps:
        metadata = score_dump_file_metadata(path, cache=score_dump_cache)
        _validate_score_dump_metadata(name, path, metadata)
        source_metadata_by_name[name] = metadata
        column_view_by_name[name] = load_score_dump_columns(
            path,
            signals,
            cache=score_dump_cache,
        )

    directions = {signal: _direction_for(signal, direction_override) for signal in signals}
    seed_reports = []
    seed_run_map: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in score_dumps}
    for seed in seeds:
        compact_runs = []
        for name, _ in score_dumps:
            columns = column_view_by_name[name]
            comparison = _seed_abstention_comparison(
                columns.labels,
                columns.scores,
                signals=signals,
                directions=directions,
                seed=int(seed),
                alpha=float(alpha),
                best_by=best_by,
            )
            comparison_payload = comparison.to_dict()
            gate = conformal_abstention_release_gate(
                comparison,
                min_conditional_correctness_lower_bound=(
                    min_conditional_correctness_lower_bound
                ),
                max_abstention_rate=max_abstention_rate,
            ).to_dict()
            entry = {
                "seed": int(seed),
                "recommended": _candidate_summary(comparison_payload.get("recommended")),
                "release_gate": gate,
                "candidates": [
                    _candidate_summary(candidate)
                    for candidate in comparison_payload.get("candidates", ())
                ],
            }
            seed_run_map[name].append(entry)
            compact_runs.append({"name": name, **entry})
        seed_reports.append({"seed": int(seed), "runs": compact_runs})

    runs = []
    for name, path in score_dumps:
        seed_entries = seed_run_map.get(name, [])
        columns = column_view_by_name[name]
        runs.append({
            "name": name,
            "scores_path": str(path),
            "score_dump": source_metadata_by_name[name],
            "seed_runs": seed_entries,
            "stability": _summarize_seed_entries(seed_entries),
            "supervised_feasibility_frontier": _supervised_feasibility_frontier(
                columns.labels,
                columns.scores,
                signals=signals,
                directions=directions,
                alpha=float(alpha),
                min_conditional_correctness_lower_bound=(
                    min_conditional_correctness_lower_bound
                ),
                max_abstention_rate=max_abstention_rate,
            ),
        })

    return {
        "schema_version": 1,
        "workflow": "abstention_stability",
        "status": "complete",
        "config": {
            "signals": list(signals),
            "directions": dict(directions),
            "alpha": float(alpha),
            "best_by": best_by,
            "seeds": [int(seed) for seed in seeds],
            "release_gate": {
                "min_conditional_correctness_lower_bound": float(
                    min_conditional_correctness_lower_bound
                ),
                "max_abstention_rate": float(max_abstention_rate),
            },
        },
        "score_dump_cache": score_dump_cache_summary(score_dump_cache),
        "seed_reports": seed_reports,
        "runs": runs,
    }


def _artifact_paths(
    *,
    output_path: Path,
    score_dumps: Sequence[tuple[str, Path]],
    payload: Mapping[str, Any] | None = None,
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {"abstention_stability_report": output_path}
    for name, path in score_dumps:
        artifacts[f"input_scores.{name}"] = path
    if payload is not None:
        for run in payload.get("runs", ()):
            if not isinstance(run, Mapping):
                continue
            name = str(run.get("name", "unknown"))
            score_dump = run.get("score_dump")
            if not isinstance(score_dump, Mapping):
                continue
            records = score_dump.get("records")
            if isinstance(records, Mapping) and records.get("path") is not None:
                artifacts[f"input_score_records.{name}"] = Path(str(records["path"]))
    return artifacts


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    score_dumps: Sequence[tuple[str, Path]],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(output_path=output_path, score_dumps=score_dumps, payload=payload),
        root=manifest_path.parent,
        metadata={
            "runner": "eval_abstention_stability",
            "status": payload.get("status"),
            "seeds": tuple(payload.get("config", {}).get("seeds", ())),
            "alpha": payload.get("config", {}).get("alpha"),
            "best_by": payload.get("config", {}).get("best_by"),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _registry_run_summaries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for run in payload.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        stability = run.get("stability")
        if not isinstance(stability, Mapping):
            stability = {}
        feasibility = run.get("supervised_feasibility_frontier")
        if not isinstance(feasibility, Mapping):
            feasibility = {}
        feasible_best = feasibility.get("best")
        if not isinstance(feasible_best, Mapping):
            feasible_best = {}
        correctness = stability.get("conditional_correctness_lower_bound")
        abstention = stability.get("empirical_abstention_rate")
        summaries.append({
            "name": run.get("name"),
            "stable_recommended_score_name": stability.get("stable_recommended_score_name"),
            "recommended_score_name_counts": stability.get("recommended_score_name_counts"),
            "release_gate_pass_seed_count": stability.get("release_gate_pass_seed_count"),
            "release_gate_block_seed_count": stability.get("release_gate_block_seed_count"),
            "all_release_gates_passed": stability.get("all_release_gates_passed"),
            "conditional_correctness_lower_bound_mean": (
                correctness.get("mean") if isinstance(correctness, Mapping) else None
            ),
            "empirical_abstention_rate_mean": (
                abstention.get("mean") if isinstance(abstention, Mapping) else None
            ),
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
        })
    return summaries


def _record_registry(
    *,
    registry_path: Path | None,
    name: str | None,
    version: str | None,
    output_path: Path,
    manifest_path: Path | None,
    payload: Mapping[str, Any],
) -> None:
    if registry_path is None:
        return
    if not name or not version:
        raise ValueError("--registry requires --name and --version.")
    registry = ArtifactRegistry.load_json(registry_path)
    registry.record_report(
        name=name,
        path=output_path,
        version=version,
        metadata={
            "workflow": "eval_abstention_stability",
            "status": payload.get("status"),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "signals": tuple(payload.get("config", {}).get("signals", ())),
            "alpha": payload.get("config", {}).get("alpha"),
            "best_by": payload.get("config", {}).get("best_by"),
            "seeds": tuple(payload.get("config", {}).get("seeds", ())),
            "release_gate": payload.get("config", {}).get("release_gate"),
            "runs": tuple(run.get("name") for run in payload.get("runs", ())),
            "run_summaries": _registry_run_summaries(payload),
        },
    )
    registry.save_json()


def run(args: argparse.Namespace) -> dict[str, Any]:
    score_dumps = [_parse_named_path(value) for value in args.scores]
    signals = _parse_csv(args.signals, name="signals")
    seeds = _parse_int_csv(args.seeds, name="seeds")
    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_abstention_stability_report(
        score_dumps,
        signals=signals,
        seeds=seeds,
        alpha=float(args.alpha),
        best_by=str(args.best_by),
        direction_override=args.direction,
        min_conditional_correctness_lower_bound=float(
            args.min_abstention_conditional_correctness_lower_bound
        ),
        max_abstention_rate=float(args.max_abstention_rate),
    )
    payload["paths"] = {"abstention_stability_report": str(output_path)}

    manifest_path = None if args.artifact_manifest is None else Path(args.artifact_manifest)
    if manifest_path is not None:
        payload["paths"]["artifact_manifest"] = str(manifest_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if manifest_path is not None:
        initial_manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            score_dumps=score_dumps,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = initial_manifest["summary"]
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            score_dumps=score_dumps,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = manifest["summary"]

    _record_registry(
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        output_path=output_path,
        manifest_path=manifest_path,
        payload=payload,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate conformal abstention gate stability across seeds"
    )
    parser.add_argument("--scores", action="append", required=True, help="name=score_dump path; repeatable")
    parser.add_argument("--signals", required=True, help="comma-separated abstention candidate signals")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--best-by", choices=ABSTENTION_COMPARISON_METRICS,
                        default="conditional_correctness_lower_bound")
    parser.add_argument("--direction", choices=("higher", "lower"), default=None,
                        help="optional override applied to every abstention signal")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--min-abstention-conditional-correctness-lower-bound",
                        type=float, default=0.80)
    parser.add_argument("--max-abstention-rate", type=float, default=0.50)
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    payload = run(parser.parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
