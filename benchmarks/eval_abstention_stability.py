"""Evaluate multi-seed stability for conformal abstention promotion gates.

This is a model-free post-hoc benchmark. It consumes existing score dumps,
replays conformal abstention candidate selection across seeded calibration
splits, and records whether the recommended participation gate remains stable
and passes release thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
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
from eigentruth.eval.score_fusion import (  # noqa: E402
    GEOMETRY_UNCERTAINTY_FUSION_METHODS,
    RANK_SCORE_FUSION_METHODS,
    combine_geometry_uncertainty_scores,
    combine_rank_anomaly_scores,
    directional_rank_anomaly_scores,
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


def _parse_optional_csv(value: str | None, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return _parse_csv(value, name=name)


def _parse_int_csv(value: str | None, *, name: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in _parse_csv(value, name=name))
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate integers.")
    return values


def _parse_float_csv(value: str | None, *, name: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in _parse_csv(value, name=name))
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values.")
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
    prefer_release_gate_passing: bool = False,
    min_conditional_correctness_lower_bound: float | None = None,
    max_abstention_rate: float | None = None,
) -> ConformalAbstentionComparisonReport:
    if best_by not in ABSTENTION_COMPARISON_METRICS:
        raise ValueError(f"best_by must be one of {ABSTENTION_COMPARISON_METRICS}.")
    if prefer_release_gate_passing and (
        min_conditional_correctness_lower_bound is None or max_abstention_rate is None
    ):
        raise ValueError("release gate thresholds are required when gate-aware ranking is enabled.")

    def release_gate_rank(report: ConformalAbstentionReport) -> int:
        if not prefer_release_gate_passing:
            return 0
        assert min_conditional_correctness_lower_bound is not None
        assert max_abstention_rate is not None
        if (
            report.conditional_correctness_lower_bound
            >= min_conditional_correctness_lower_bound
            and report.empirical_abstention_rate <= max_abstention_rate
        ):
            return 0
        return 1

    def sort_key(report: ConformalAbstentionReport) -> tuple[int, float, float, float, float, float, str]:
        score_name = "" if report.score_name is None else report.score_name
        return (
            release_gate_rank(report),
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


def _validate_unique_values(values: Sequence[str], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values.")


def _validate_fusion_config(
    *,
    rank_fusion_signals: Sequence[str],
    rank_fusion_methods: Sequence[str],
    geometry_signals: Sequence[str],
    uncertainty_signals: Sequence[str],
    geometry_method: str,
    uncertainty_method: str,
    geometry_fusion_methods: Sequence[str],
) -> None:
    _validate_unique_values(rank_fusion_signals, name="rank_fusion_signals")
    _validate_unique_values(geometry_signals, name="geometry_signals")
    _validate_unique_values(uncertainty_signals, name="uncertainty_signals")
    if rank_fusion_signals and not rank_fusion_methods:
        raise ValueError("rank_fusion_methods must be set when rank_fusion_signals are set.")
    if rank_fusion_methods and not rank_fusion_signals:
        raise ValueError("rank_fusion_signals must be set when rank_fusion_methods are set.")
    for method in rank_fusion_methods:
        if method not in RANK_SCORE_FUSION_METHODS:
            raise ValueError(f"rank_fusion_methods must be one of {RANK_SCORE_FUSION_METHODS}.")
    if bool(geometry_signals) != bool(uncertainty_signals):
        raise ValueError("geometry_signals and uncertainty_signals must be set together.")
    if geometry_fusion_methods and (not geometry_signals or not uncertainty_signals):
        raise ValueError(
            "geometry_signals and uncertainty_signals are required when geometry_fusion_methods are set."
        )
    if (geometry_signals or uncertainty_signals) and not geometry_fusion_methods:
        raise ValueError("geometry_fusion_methods must be set for geometry/uncertainty fusion.")
    if geometry_method not in RANK_SCORE_FUSION_METHODS:
        raise ValueError(f"geometry_method must be one of {RANK_SCORE_FUSION_METHODS}.")
    if uncertainty_method not in RANK_SCORE_FUSION_METHODS:
        raise ValueError(f"uncertainty_method must be one of {RANK_SCORE_FUSION_METHODS}.")
    for method in geometry_fusion_methods:
        if method not in GEOMETRY_UNCERTAINTY_FUSION_METHODS:
            raise ValueError(
                f"geometry_fusion_methods must be one of {GEOMETRY_UNCERTAINTY_FUSION_METHODS}."
            )


def _resolve_budget_target_rates(
    *,
    enforce_abstention_budget: bool,
    abstention_budget_target_rate: float | None,
    abstention_budget_target_rates: Sequence[float],
    max_abstention_rate: float,
) -> tuple[float | None, ...]:
    if abstention_budget_target_rate is not None and abstention_budget_target_rates:
        raise ValueError(
            "use either abstention_budget_target_rate or abstention_budget_target_rates, not both."
        )
    if abstention_budget_target_rates and not enforce_abstention_budget:
        raise ValueError("abstention_budget_target_rates requires enforce_abstention_budget.")
    for rate in abstention_budget_target_rates:
        if not (0.0 <= float(rate) <= 1.0):
            raise ValueError("abstention_budget_target_rates values must be in [0, 1].")
    if not enforce_abstention_budget:
        return (None,)
    if abstention_budget_target_rates:
        return tuple(float(rate) for rate in abstention_budget_target_rates)
    return (
        float(max_abstention_rate)
        if abstention_budget_target_rate is None
        else float(abstention_budget_target_rate),
    )


def _abstention_budget_threshold(
    scores: Sequence[float],
    *,
    max_abstention_rate: float,
    direction: str,
) -> float:
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    values = sorted(float(score) for score in scores)
    if not values:
        raise ValueError("budget reference scores must be non-empty.")
    budget = float(max_abstention_rate)
    if not (0.0 <= budget <= 1.0):
        raise ValueError("max_abstention_rate must be in [0, 1].")
    n = len(values)
    if direction == "higher":
        retained_count = math.ceil((1.0 - budget) * n)
        index = min(max(retained_count - 1, 0), n - 1)
        return values[index]
    allowed_abstentions = math.floor(budget * n)
    index = min(max(allowed_abstentions, 0), n - 1)
    return values[index]


def _abstention_threshold(
    scores: Sequence[float],
    *,
    conformal_calibration_indices: Sequence[int],
    budget_reference_indices: Sequence[int],
    alpha: float,
    direction: str,
    budget_max_abstention_rate: float | None,
) -> float:
    conformal_threshold = directional_conformal_threshold(
        _subset(scores, conformal_calibration_indices),
        alpha,
        direction,
    )
    if budget_max_abstention_rate is None:
        return conformal_threshold
    budget_threshold = _abstention_budget_threshold(
        _subset(scores, budget_reference_indices),
        max_abstention_rate=budget_max_abstention_rate,
        direction=direction,
    )
    if direction == "higher":
        return max(conformal_threshold, budget_threshold)
    return min(conformal_threshold, budget_threshold)


def _budget_score_name(
    score_name: str,
    budget_max_abstention_rate: float | None,
    *,
    include_budget_suffix: bool,
) -> str:
    if not include_budget_suffix:
        return score_name
    label = "conformal" if budget_max_abstention_rate is None else f"{budget_max_abstention_rate:g}"
    return f"{score_name}@budget={label}"


def _fusion_name(prefix: str, method: str, signals: Sequence[str]) -> str:
    return f"{prefix}:{method}[{'+'.join(signals)}]"


def _geometry_fusion_name(
    *,
    geometry_method: str,
    uncertainty_method: str,
    fusion_method: str,
    geometry_signals: Sequence[str],
    uncertainty_signals: Sequence[str],
) -> str:
    geometry = "+".join(geometry_signals)
    uncertainty = "+".join(uncertainty_signals)
    return (
        "geometry_uncertainty_fusion:"
        f"{fusion_method}[geometry={geometry_method}:{geometry};"
        f"uncertainty={uncertainty_method}:{uncertainty}]"
    )


def _rank_anomaly_scores_for_signals(
    scores: Mapping[str, Sequence[float]],
    *,
    signals: Sequence[str],
    directions: Mapping[str, str],
    calibration_indices: Sequence[int],
) -> tuple[Any, ...]:
    return tuple(
        directional_rank_anomaly_scores(
            _subset(scores[signal], calibration_indices),
            scores[signal],
            direction=directions[signal],
        )
        for signal in signals
    )


def _rank_fusion_candidate_scores(
    scores: Mapping[str, Sequence[float]],
    *,
    rank_fusion_signals: Sequence[str],
    rank_fusion_methods: Sequence[str],
    directions: Mapping[str, str],
    calibration_indices: Sequence[int],
) -> tuple[tuple[str, Any], ...]:
    if not rank_fusion_signals or not rank_fusion_methods:
        return ()
    rank_scores = _rank_anomaly_scores_for_signals(
        scores,
        signals=rank_fusion_signals,
        directions=directions,
        calibration_indices=calibration_indices,
    )
    return tuple(
        (
            _fusion_name("rank_fusion", method, rank_fusion_signals),
            combine_rank_anomaly_scores(rank_scores, method),
        )
        for method in rank_fusion_methods
    )


def _geometry_uncertainty_candidate_scores(
    scores: Mapping[str, Sequence[float]],
    *,
    geometry_signals: Sequence[str],
    uncertainty_signals: Sequence[str],
    geometry_method: str,
    uncertainty_method: str,
    geometry_fusion_methods: Sequence[str],
    directions: Mapping[str, str],
    calibration_indices: Sequence[int],
) -> tuple[tuple[str, Any], ...]:
    if not geometry_signals or not uncertainty_signals or not geometry_fusion_methods:
        return ()
    geometry_rank_scores = _rank_anomaly_scores_for_signals(
        scores,
        signals=geometry_signals,
        directions=directions,
        calibration_indices=calibration_indices,
    )
    uncertainty_rank_scores = _rank_anomaly_scores_for_signals(
        scores,
        signals=uncertainty_signals,
        directions=directions,
        calibration_indices=calibration_indices,
    )
    geometry_scores = combine_rank_anomaly_scores(geometry_rank_scores, geometry_method)
    uncertainty_scores = combine_rank_anomaly_scores(uncertainty_rank_scores, uncertainty_method)
    return tuple(
        (
            _geometry_fusion_name(
                geometry_method=geometry_method,
                uncertainty_method=uncertainty_method,
                fusion_method=fusion_method,
                geometry_signals=geometry_signals,
                uncertainty_signals=uncertainty_signals,
            ),
            combine_geometry_uncertainty_scores(
                geometry_scores,
                uncertainty_scores,
                method=fusion_method,
            ),
        )
        for fusion_method in geometry_fusion_methods
    )


def _evaluate_fused_candidate(
    fused_scores: Sequence[float],
    *,
    labels: Sequence[int],
    conformal_calibration_indices: Sequence[int],
    budget_reference_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    alpha: float,
    score_name: str,
    budget_max_abstention_rate: float | None,
) -> ConformalAbstentionReport:
    threshold = _abstention_threshold(
        fused_scores,
        conformal_calibration_indices=conformal_calibration_indices,
        budget_reference_indices=budget_reference_indices,
        alpha=alpha,
        direction="higher",
        budget_max_abstention_rate=budget_max_abstention_rate,
    )
    return evaluate_conformal_abstention(
        _subset(fused_scores, evaluation_indices),
        _correctness(labels, evaluation_indices),
        threshold=threshold,
        alpha=alpha,
        direction="higher",
        score_name=score_name,
    )


def _seed_abstention_comparison(
    labels: Sequence[int],
    scores: Mapping[str, Sequence[float]],
    *,
    signals: Sequence[str],
    directions: Mapping[str, str],
    rank_fusion_signals: Sequence[str] = (),
    rank_fusion_methods: Sequence[str] = (),
    geometry_signals: Sequence[str] = (),
    uncertainty_signals: Sequence[str] = (),
    geometry_method: str = "mean_rank",
    uncertainty_method: str = "mean_rank",
    geometry_fusion_methods: Sequence[str] = (),
    budget_max_abstention_rates: Sequence[float | None] = (None,),
    include_budget_suffix: bool = False,
    prefer_release_gate_passing: bool = False,
    min_conditional_correctness_lower_bound: float | None = None,
    max_abstention_rate: float | None = None,
    seed: int,
    alpha: float,
    best_by: str,
) -> ConformalAbstentionComparisonReport:
    calibration_indices, evaluation_indices = _split_indices(labels, seed=seed)
    correct_calibration_indices = tuple(
        index for index in calibration_indices if int(labels[index]) == 0
    )
    reports: list[ConformalAbstentionReport] = []
    for budget_max_abstention_rate in budget_max_abstention_rates:
        for signal in signals:
            signal_scores = scores[signal]
            direction = directions[signal]
            threshold = _abstention_threshold(
                signal_scores,
                conformal_calibration_indices=correct_calibration_indices,
                budget_reference_indices=calibration_indices,
                alpha=alpha,
                direction=direction,
                budget_max_abstention_rate=budget_max_abstention_rate,
            )
            reports.append(
                evaluate_conformal_abstention(
                    _subset(signal_scores, evaluation_indices),
                    _correctness(labels, evaluation_indices),
                    threshold=threshold,
                    alpha=alpha,
                    direction=direction,
                    score_name=_budget_score_name(
                        signal,
                        budget_max_abstention_rate,
                        include_budget_suffix=include_budget_suffix,
                    ),
                )
            )
    fused_candidates = (
        *_rank_fusion_candidate_scores(
            scores,
            rank_fusion_signals=rank_fusion_signals,
            rank_fusion_methods=rank_fusion_methods,
            directions=directions,
            calibration_indices=correct_calibration_indices,
        ),
        *_geometry_uncertainty_candidate_scores(
            scores,
            geometry_signals=geometry_signals,
            uncertainty_signals=uncertainty_signals,
            geometry_method=geometry_method,
            uncertainty_method=uncertainty_method,
            geometry_fusion_methods=geometry_fusion_methods,
            directions=directions,
            calibration_indices=correct_calibration_indices,
        ),
    )
    for budget_max_abstention_rate in budget_max_abstention_rates:
        for score_name, fused_scores in fused_candidates:
            reports.append(
                _evaluate_fused_candidate(
                    fused_scores,
                    labels=labels,
                    conformal_calibration_indices=correct_calibration_indices,
                    budget_reference_indices=calibration_indices,
                    evaluation_indices=evaluation_indices,
                    alpha=alpha,
                    score_name=_budget_score_name(
                        score_name,
                        budget_max_abstention_rate,
                        include_budget_suffix=include_budget_suffix,
                    ),
                    budget_max_abstention_rate=budget_max_abstention_rate,
                )
            )
    return _rank_abstention_reports(
        reports,
        alpha=alpha,
        best_by=best_by,
        prefer_release_gate_passing=prefer_release_gate_passing,
        min_conditional_correctness_lower_bound=min_conditional_correctness_lower_bound,
        max_abstention_rate=max_abstention_rate,
    )


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


def _candidate_float(candidate: Mapping[str, Any], metric: str) -> float | None:
    value = candidate.get(metric)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _candidate_blocking_reason_codes(
    candidate: Mapping[str, Any],
    *,
    min_conditional_correctness_lower_bound: float,
    max_abstention_rate: float,
) -> tuple[str, ...]:
    reasons = []
    conditional_correctness = _candidate_float(
        candidate,
        "conditional_correctness_lower_bound",
    )
    abstention_rate = _candidate_float(candidate, "empirical_abstention_rate")
    if (
        conditional_correctness is None
        or conditional_correctness < min_conditional_correctness_lower_bound
    ):
        reasons.append("conditional_correctness_lower_bound")
    if abstention_rate is None or abstention_rate > max_abstention_rate:
        reasons.append("empirical_abstention_rate")
    return tuple(reasons)


def _candidate_release_gate_passed(
    candidate: Mapping[str, Any],
    *,
    min_conditional_correctness_lower_bound: float,
    max_abstention_rate: float,
) -> bool:
    return not _candidate_blocking_reason_codes(
        candidate,
        min_conditional_correctness_lower_bound=min_conditional_correctness_lower_bound,
        max_abstention_rate=max_abstention_rate,
    )


def _candidate_gate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, int, str]:
    rank = candidate.get("rank")
    try:
        parsed_rank = int(rank)
    except (TypeError, ValueError):
        parsed_rank = 1_000_000
    return (
        -_sortable_value(_candidate_float(candidate, "conditional_correctness_lower_bound")),
        -_sortable_value(_candidate_float(candidate, "empirical_selective_accuracy")),
        -_sortable_value(_candidate_float(candidate, "correct_retention_lower_bound")),
        _sortable_value(_candidate_float(candidate, "empirical_abstention_rate")),
        parsed_rank,
        str(candidate.get("score_name", "")),
    )


def _candidate_gate_summary(
    candidates: Sequence[Mapping[str, Any] | None],
    recommended: Mapping[str, Any] | None,
    release_gate: Mapping[str, Any],
    *,
    min_conditional_correctness_lower_bound: float,
    max_abstention_rate: float,
) -> dict[str, Any]:
    candidate_list = tuple(candidate for candidate in candidates if isinstance(candidate, Mapping))
    passing_candidates = tuple(
        candidate
        for candidate in candidate_list
        if _candidate_release_gate_passed(
            candidate,
            min_conditional_correctness_lower_bound=min_conditional_correctness_lower_bound,
            max_abstention_rate=max_abstention_rate,
        )
    )
    blocking_reason_counts: Counter[str] = Counter()
    for candidate in candidate_list:
        blocking_reason_counts.update(
            _candidate_blocking_reason_codes(
                candidate,
                min_conditional_correctness_lower_bound=(
                    min_conditional_correctness_lower_bound
                ),
                max_abstention_rate=max_abstention_rate,
            )
        )
    recommended_passed = bool(release_gate.get("passed") is True)
    recommended_reason_codes = (
        ()
        if recommended is None
        else _candidate_blocking_reason_codes(
            recommended,
            min_conditional_correctness_lower_bound=(
                min_conditional_correctness_lower_bound
            ),
            max_abstention_rate=max_abstention_rate,
        )
    )
    best_passing_candidate = (
        None if not passing_candidates else sorted(passing_candidates, key=_candidate_gate_sort_key)[0]
    )
    return {
        "candidate_count": len(candidate_list),
        "passing_candidate_count": len(passing_candidates),
        "blocked_candidate_count": len(candidate_list) - len(passing_candidates),
        "any_candidate_passed": bool(passing_candidates),
        "recommended_passed": recommended_passed,
        "recommended_missed_passing_candidate": (
            not recommended_passed and bool(passing_candidates)
        ),
        "recommended_blocking_reasons": list(release_gate.get("blocking_reasons", ())),
        "recommended_blocking_reason_codes": list(recommended_reason_codes),
        "candidate_blocking_reason_counts": dict(sorted(blocking_reason_counts.items())),
        "best_passing_candidate": best_passing_candidate,
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
    rank_fusion_signals: Sequence[str] = (),
    rank_fusion_methods: Sequence[str] = (),
    geometry_signals: Sequence[str] = (),
    uncertainty_signals: Sequence[str] = (),
    geometry_method: str = "mean_rank",
    uncertainty_method: str = "mean_rank",
    geometry_fusion_methods: Sequence[str] = (),
    alpha: float,
    min_conditional_correctness_lower_bound: float,
    max_abstention_rate: float,
) -> dict[str, Any]:
    signal_reports = []
    signal_candidates: list[tuple[str, Sequence[float], str]] = [
        (signal, scores[signal], directions[signal]) for signal in signals
    ]
    true_indices = tuple(index for index, label in enumerate(labels) if int(label) == 0)
    for score_name, fused_scores in (
        *_rank_fusion_candidate_scores(
            scores,
            rank_fusion_signals=rank_fusion_signals,
            rank_fusion_methods=rank_fusion_methods,
            directions=directions,
            calibration_indices=true_indices,
        ),
        *_geometry_uncertainty_candidate_scores(
            scores,
            geometry_signals=geometry_signals,
            uncertainty_signals=uncertainty_signals,
            geometry_method=geometry_method,
            uncertainty_method=uncertainty_method,
            geometry_fusion_methods=geometry_fusion_methods,
            directions=directions,
            calibration_indices=true_indices,
        ),
    ):
        signal_candidates.append((score_name, fused_scores, "higher"))

    for signal, signal_scores, direction in signal_candidates:
        report = _best_supervised_threshold_report(
            labels,
            signal_scores,
            signal=signal,
            direction=direction,
            alpha=alpha,
            max_abstention_rate=max_abstention_rate,
        )
        if report is None:
            signal_reports.append({
                "score_name": signal,
                "direction": direction,
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


def _candidate_gate_stability_summary(
    seed_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seed_with_any_passing_candidate_count = 0
    recommended_pass_seed_count = 0
    recommended_missed_passing_candidate_count = 0
    recommended_reason_counts: Counter[str] = Counter()
    candidate_reason_counts: Counter[str] = Counter()
    best_passing_counts: Counter[str] = Counter()
    for entry in seed_entries:
        summary = entry.get("candidate_gate_summary")
        if not isinstance(summary, Mapping):
            continue
        if summary.get("any_candidate_passed") is True:
            seed_with_any_passing_candidate_count += 1
        if summary.get("recommended_passed") is True:
            recommended_pass_seed_count += 1
        if summary.get("recommended_missed_passing_candidate") is True:
            recommended_missed_passing_candidate_count += 1
        recommended_reason_counts.update(
            str(reason) for reason in summary.get("recommended_blocking_reason_codes", ())
        )
        raw_candidate_reasons = summary.get("candidate_blocking_reason_counts", {})
        if isinstance(raw_candidate_reasons, Mapping):
            for reason, count in raw_candidate_reasons.items():
                candidate_reason_counts[str(reason)] += int(count)
        best_passing_candidate = summary.get("best_passing_candidate")
        if isinstance(best_passing_candidate, Mapping):
            best_passing_counts[str(best_passing_candidate.get("score_name", "<missing>"))] += 1
    seed_count = len(seed_entries)
    return {
        "seed_with_any_passing_candidate_count": seed_with_any_passing_candidate_count,
        "seed_without_passing_candidate_count": (
            seed_count - seed_with_any_passing_candidate_count
        ),
        "all_seeds_have_passing_candidate": (
            seed_with_any_passing_candidate_count == seed_count
        ),
        "recommended_pass_seed_count": recommended_pass_seed_count,
        "recommended_block_seed_count": seed_count - recommended_pass_seed_count,
        "recommended_missed_passing_candidate_count": (
            recommended_missed_passing_candidate_count
        ),
        "recommended_blocking_reason_counts": dict(
            sorted(recommended_reason_counts.items())
        ),
        "candidate_blocking_reason_counts": dict(sorted(candidate_reason_counts.items())),
        "best_passing_score_name_counts": dict(sorted(best_passing_counts.items())),
    }


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
        "candidate_gate_summary": _candidate_gate_stability_summary(seed_entries),
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
    rank_fusion_signals: Sequence[str] = (),
    rank_fusion_methods: Sequence[str] = (),
    geometry_signals: Sequence[str] = (),
    uncertainty_signals: Sequence[str] = (),
    geometry_method: str = "mean_rank",
    uncertainty_method: str = "mean_rank",
    geometry_fusion_methods: Sequence[str] = (),
    enforce_abstention_budget: bool = False,
    abstention_budget_target_rate: float | None = None,
    abstention_budget_target_rates: Sequence[float] = (),
    prefer_release_gate_passing: bool = False,
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
    if not (0.0 <= float(max_abstention_rate) <= 1.0):
        raise ValueError("max_abstention_rate must be in [0, 1].")
    if abstention_budget_target_rate is not None and not (
        0.0 <= float(abstention_budget_target_rate) <= 1.0
    ):
        raise ValueError("abstention_budget_target_rate must be in [0, 1].")
    if best_by not in ABSTENTION_COMPARISON_METRICS:
        raise ValueError(f"best_by must be one of {ABSTENTION_COMPARISON_METRICS}.")
    if direction_override is not None and direction_override not in {"higher", "lower"}:
        raise ValueError("direction_override must be 'higher', 'lower', or None.")
    _validate_fusion_config(
        rank_fusion_signals=rank_fusion_signals,
        rank_fusion_methods=rank_fusion_methods,
        geometry_signals=geometry_signals,
        uncertainty_signals=uncertainty_signals,
        geometry_method=geometry_method,
        uncertainty_method=uncertainty_method,
        geometry_fusion_methods=geometry_fusion_methods,
    )
    _validate_unique_score_dump_names(score_dumps)

    required_signals = tuple(dict.fromkeys((
        *signals,
        *rank_fusion_signals,
        *geometry_signals,
        *uncertainty_signals,
    )))
    score_dump_cache: dict[str, Any] = {}
    source_metadata_by_name = {}
    column_view_by_name = {}
    for name, path in score_dumps:
        metadata = score_dump_file_metadata(path, cache=score_dump_cache)
        _validate_score_dump_metadata(name, path, metadata)
        source_metadata_by_name[name] = metadata
        column_view_by_name[name] = load_score_dump_columns(
            path,
            required_signals,
            cache=score_dump_cache,
        )

    directions = {
        signal: _direction_for(signal, direction_override) for signal in required_signals
    }
    seed_reports = []
    seed_run_map: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in score_dumps}
    budget_max_abstention_rates = _resolve_budget_target_rates(
        enforce_abstention_budget=enforce_abstention_budget,
        abstention_budget_target_rate=abstention_budget_target_rate,
        abstention_budget_target_rates=abstention_budget_target_rates,
        max_abstention_rate=float(max_abstention_rate),
    )
    include_budget_suffix = len(budget_max_abstention_rates) > 1
    for seed in seeds:
        compact_runs = []
        for name, _ in score_dumps:
            columns = column_view_by_name[name]
            comparison = _seed_abstention_comparison(
                columns.labels,
                columns.scores,
                signals=signals,
                directions=directions,
                rank_fusion_signals=rank_fusion_signals,
                rank_fusion_methods=rank_fusion_methods,
                geometry_signals=geometry_signals,
                uncertainty_signals=uncertainty_signals,
                geometry_method=geometry_method,
                uncertainty_method=uncertainty_method,
                geometry_fusion_methods=geometry_fusion_methods,
                budget_max_abstention_rates=budget_max_abstention_rates,
                include_budget_suffix=include_budget_suffix,
                prefer_release_gate_passing=prefer_release_gate_passing,
                min_conditional_correctness_lower_bound=(
                    min_conditional_correctness_lower_bound
                ),
                max_abstention_rate=max_abstention_rate,
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
            entry["candidate_gate_summary"] = _candidate_gate_summary(
                entry["candidates"],
                entry["recommended"],
                gate,
                min_conditional_correctness_lower_bound=(
                    min_conditional_correctness_lower_bound
                ),
                max_abstention_rate=max_abstention_rate,
            )
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
                rank_fusion_signals=rank_fusion_signals,
                rank_fusion_methods=rank_fusion_methods,
                geometry_signals=geometry_signals,
                uncertainty_signals=uncertainty_signals,
                geometry_method=geometry_method,
                uncertainty_method=uncertainty_method,
                geometry_fusion_methods=geometry_fusion_methods,
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
            "fusion": {
                "rank": {
                    "signals": list(rank_fusion_signals),
                    "methods": list(rank_fusion_methods),
                },
                "geometry_uncertainty": {
                    "geometry_signals": list(geometry_signals),
                    "uncertainty_signals": list(uncertainty_signals),
                    "geometry_method": geometry_method,
                    "uncertainty_method": uncertainty_method,
                    "fusion_methods": list(geometry_fusion_methods),
                },
            },
            "threshold_policy": {
                "mode": (
                    "conformal_with_abstention_budget"
                    if enforce_abstention_budget
                    else "conformal"
                ),
                "enforce_abstention_budget": bool(enforce_abstention_budget),
                "abstention_budget_target_rate": (
                    budget_max_abstention_rates[0]
                    if len(budget_max_abstention_rates) == 1
                    else None
                ),
                "abstention_budget_target_rates": [
                    None if rate is None else float(rate)
                    for rate in budget_max_abstention_rates
                ],
                "prefer_release_gate_passing": bool(prefer_release_gate_passing),
                "budget_reference": (
                    "seed_calibration_split_unlabeled_scores"
                    if enforce_abstention_budget
                    else None
                ),
            },
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
            "fusion": payload.get("config", {}).get("fusion"),
            "threshold_policy": payload.get("config", {}).get("threshold_policy"),
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
            "candidate_gate_summary": stability.get("candidate_gate_summary"),
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
            "fusion": payload.get("config", {}).get("fusion"),
            "threshold_policy": payload.get("config", {}).get("threshold_policy"),
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
    rank_fusion_signals = _parse_optional_csv(
        args.rank_fusion_signals,
        name="rank_fusion_signals",
    )
    rank_fusion_methods = _parse_optional_csv(
        args.rank_fusion_methods,
        name="rank_fusion_methods",
    )
    geometry_signals = _parse_optional_csv(args.geometry_signals, name="geometry_signals")
    uncertainty_signals = _parse_optional_csv(
        args.uncertainty_signals,
        name="uncertainty_signals",
    )
    geometry_fusion_methods = _parse_optional_csv(
        args.geometry_fusion_methods,
        name="geometry_fusion_methods",
    )
    abstention_budget_target_rates = (
        ()
        if args.abstention_budget_target_rates is None
        else _parse_float_csv(
            args.abstention_budget_target_rates,
            name="abstention_budget_target_rates",
        )
    )
    seeds = _parse_int_csv(args.seeds, name="seeds")
    output_path = Path(args.json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_abstention_stability_report(
        score_dumps,
        signals=signals,
        seeds=seeds,
        rank_fusion_signals=rank_fusion_signals,
        rank_fusion_methods=rank_fusion_methods,
        geometry_signals=geometry_signals,
        uncertainty_signals=uncertainty_signals,
        geometry_method=str(args.geometry_method),
        uncertainty_method=str(args.uncertainty_method),
        geometry_fusion_methods=geometry_fusion_methods,
        enforce_abstention_budget=bool(args.enforce_abstention_budget),
        abstention_budget_target_rate=(
            None
            if args.abstention_budget_target_rate is None
            else float(args.abstention_budget_target_rate)
        ),
        abstention_budget_target_rates=abstention_budget_target_rates,
        prefer_release_gate_passing=bool(args.prefer_release_gate_passing),
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
    parser.add_argument(
        "--rank-fusion-signals",
        default=None,
        help="optional comma-separated signals to rank-calibrate and fuse as abstention candidates",
    )
    parser.add_argument(
        "--rank-fusion-methods",
        default=None,
        help=f"comma-separated rank fusion methods; choices: {','.join(RANK_SCORE_FUSION_METHODS)}",
    )
    parser.add_argument(
        "--geometry-signals",
        default=None,
        help="optional comma-separated representation/geometry signals for geometry-uncertainty fusion",
    )
    parser.add_argument(
        "--uncertainty-signals",
        default=None,
        help="optional comma-separated verifier/uncertainty signals for geometry-uncertainty fusion",
    )
    parser.add_argument(
        "--geometry-method",
        choices=RANK_SCORE_FUSION_METHODS,
        default="mean_rank",
    )
    parser.add_argument(
        "--uncertainty-method",
        choices=RANK_SCORE_FUSION_METHODS,
        default="mean_rank",
    )
    parser.add_argument(
        "--geometry-fusion-methods",
        default=None,
        help=(
            "comma-separated geometry/uncertainty fusion methods; choices: "
            f"{','.join(GEOMETRY_UNCERTAINTY_FUSION_METHODS)}"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--enforce-abstention-budget",
        action="store_true",
        help=(
            "raise/lower candidate thresholds with the seeded calibration split score "
            "distribution so total abstention stays within --max-abstention-rate"
        ),
    )
    parser.add_argument(
        "--abstention-budget-target-rate",
        type=float,
        default=None,
        help=(
            "optional calibration-split abstention target used by "
            "--enforce-abstention-budget; defaults to --max-abstention-rate"
        ),
    )
    parser.add_argument(
        "--abstention-budget-target-rates",
        default=None,
        help=(
            "optional comma-separated calibration-split abstention targets; "
            "each target becomes a distinct candidate suffix when budget enforcement is enabled"
        ),
    )
    parser.add_argument(
        "--prefer-release-gate-passing",
        action="store_true",
        help=(
            "rank candidates that satisfy the configured abstention release gate ahead of "
            "higher-scoring but gate-failing candidates"
        ),
    )
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
