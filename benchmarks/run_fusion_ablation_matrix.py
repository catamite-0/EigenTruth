"""Evaluate score-fusion ablation candidates over aligned score dumps."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.calibration import DEFAULT_SCORE_DIRECTIONS  # noqa: E402
from eigentruth.eval import (  # noqa: E402
    combine_rank_anomaly_scores,
    directional_conformal_threshold,
    directional_rank_anomaly_scores,
    directional_trigger_rate,
    load_score_dump_columns,
    native_anomaly_scores,
    roc_auc,
    score_dump_cache_summary,
    score_dump_file_metadata,
)

DEFAULT_ALPHAS = (0.05, 0.10, 0.20)
DEFAULT_METHODS = ("max_rank", "mean_rank")
TOLERANCE = 0.03


def build_fusion_ablation_matrix(
    score_dumps: Sequence[tuple[str, Path]],
    *,
    candidates: Sequence[tuple[str, Sequence[str]]],
    methods: Sequence[str] = DEFAULT_METHODS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    repeats: int = 20,
    seed: int = 0,
    best_alpha: float = 0.10,
    score_dump_cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a repeated-split ablation matrix over named signal candidates."""
    if not score_dumps:
        raise ValueError("at least one score dump is required.")
    if not candidates:
        raise ValueError("at least one candidate is required.")
    if int(repeats) < 1:
        raise ValueError("repeats must be >= 1.")
    parsed_alphas = tuple(float(alpha) for alpha in alphas)
    if any(not (0.0 < alpha < 1.0) for alpha in parsed_alphas):
        raise ValueError("alphas must be in (0, 1).")
    if float(best_alpha) not in set(parsed_alphas):
        raise ValueError("best_alpha must be one of alphas.")
    parsed_methods = tuple(str(method) for method in methods)
    if any(method not in {"max_rank", "mean_rank", "noisy_or_rank"} for method in parsed_methods):
        raise ValueError("methods must contain rank fusion methods.")
    parsed_candidates = tuple(_candidate_tuple(name, signals) for name, signals in candidates)
    load_signals = tuple(dict.fromkeys(signal for _, signals in parsed_candidates for signal in signals))
    cache = {} if score_dump_cache is None else score_dump_cache
    runs = []
    for name, path in score_dumps:
        dump = load_score_dump_columns(path, load_signals, cache=cache)
        labels = torch.as_tensor(dump.labels, dtype=torch.int64)
        _validate_labels_for_split(labels)
        scores = {
            signal: torch.as_tensor(dump.scores[signal], dtype=torch.float64)
            for signal in load_signals
        }
        directions = _directions_for_run(dump.config, load_signals)
        candidate_results = {}
        for candidate_name, signals in parsed_candidates:
            candidate_results.update(_evaluate_candidate(
                candidate_name=candidate_name,
                signals=signals,
                selected_scores=scores,
                labels=labels,
                directions=directions,
                methods=parsed_methods,
                alphas=parsed_alphas,
                repeats=int(repeats),
                seed=int(seed),
            ))
        best = _best_at_alpha(candidate_results, float(best_alpha))
        runs.append({
            "name": str(name),
            "scores_path": str(path),
            "score_dump": {
                **score_dump_file_metadata(path, cache=cache),
                "summary": dict(dump.summary),
                "source_format": dump.source_format,
            },
            "n_total": int(labels.numel()),
            "n_true": int((labels == 0).sum().item()),
            "n_false": int((labels == 1).sum().item()),
            "loaded_signals": list(load_signals),
            "directions": directions,
            "candidate_results": candidate_results,
            "best_at_alpha": best,
        })
    return {
        "schema_version": 1,
        "workflow": "fusion_ablation_matrix",
        "status": "complete",
        "candidates": [
            {"name": name, "signals": list(signals)}
            for name, signals in parsed_candidates
        ],
        "methods": list(parsed_methods),
        "alphas": list(parsed_alphas),
        "repeats": int(repeats),
        "seed": int(seed),
        "best_alpha": float(best_alpha),
        "score_dump_cache": score_dump_cache_summary(cache),
        "runs": runs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entry point with a testable args namespace."""
    score_dumps = tuple(_parse_named_path(value) for value in args.scores)
    candidates = tuple(_parse_candidate(value) for value in args.candidate)
    methods = _parse_csv(args.methods, name="methods") or DEFAULT_METHODS
    alphas = tuple(float(value) for value in (_parse_csv(args.alphas, name="alphas") or ()))
    payload = build_fusion_ablation_matrix(
        score_dumps,
        candidates=candidates,
        methods=methods,
        alphas=alphas or DEFAULT_ALPHAS,
        repeats=int(args.repeats),
        seed=int(args.seed),
        best_alpha=float(args.best_alpha),
    )
    if args.json is not None:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        for run_payload in payload["runs"]:
            best = run_payload["best_at_alpha"]
            print(
                f"{run_payload['name']}: "
                f"best@{payload['best_alpha']}={best['name']} "
                f"auroc={best['auroc']:.3f} "
                f"det={best['detection']:.3f} "
                f"fa={best['false_alarm']:.3f}"
            )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a score-fusion ablation matrix")
    parser.add_argument("--scores", action="append", required=True,
                        help="score dump path, optionally named as name=path; repeatable")
    parser.add_argument("--candidate", action="append", required=True,
                        help="candidate as name=signal1,signal2; repeatable")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in DEFAULT_ALPHAS))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-alpha", type=float, default=0.10)
    parser.add_argument("--json", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    run(args)


def _evaluate_candidate(
    *,
    candidate_name: str,
    signals: Sequence[str],
    selected_scores: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    directions: Mapping[str, str],
    methods: Sequence[str],
    alphas: Sequence[float],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if len(signals) == 1:
        signal = signals[0]
        return {
            candidate_name: _score_single_signal(
                signal=signal,
                scores=selected_scores[signal],
                labels=labels,
                direction=directions[signal],
                alphas=alphas,
                repeats=repeats,
                seed=seed,
            )
        }
    return {
        f"{candidate_name}:{method}": _score_rank_fusion(
            candidate_name=candidate_name,
            signals=signals,
            selected_scores=selected_scores,
            labels=labels,
            directions=directions,
            method=method,
            alphas=alphas,
            repeats=repeats,
            seed=seed,
        )
        for method in methods
    }


def _score_single_signal(
    *,
    signal: str,
    scores: torch.Tensor,
    labels: torch.Tensor,
    direction: str,
    alphas: Sequence[float],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    true_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()
    false_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
    false_alarm_by_alpha = {alpha: [] for alpha in alphas}
    detection_by_alpha = {alpha: [] for alpha in alphas}
    for repeat in range(repeats):
        calib_idx, test_true_idx = _split_true_indices(true_idx, seed=seed + repeat)
        for alpha in alphas:
            threshold = directional_conformal_threshold(scores[calib_idx], alpha, direction)
            false_alarm_by_alpha[alpha].append(directional_trigger_rate(scores[test_true_idx], threshold, direction))
            detection_by_alpha[alpha].append(directional_trigger_rate(scores[false_idx], threshold, direction))
    return {
        "name": signal,
        "signals": [signal],
        "method": "native",
        "direction": direction,
        "auroc": roc_auc(native_anomaly_scores(scores, direction), labels),
        "alphas": _alpha_payloads(false_alarm_by_alpha, detection_by_alpha),
    }


def _score_rank_fusion(
    *,
    candidate_name: str,
    signals: Sequence[str],
    selected_scores: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    directions: Mapping[str, str],
    method: str,
    alphas: Sequence[float],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    true_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()
    false_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
    false_alarm_by_alpha = {alpha: [] for alpha in alphas}
    detection_by_alpha = {alpha: [] for alpha in alphas}
    aurocs = []
    for repeat in range(repeats):
        calib_idx, test_true_idx = _split_true_indices(true_idx, seed=seed + repeat)
        rank_scores = [
            directional_rank_anomaly_scores(
                selected_scores[signal][calib_idx],
                selected_scores[signal],
                direction=directions[signal],
            )
            for signal in signals
        ]
        fused = combine_rank_anomaly_scores(rank_scores, method)
        aurocs.append(roc_auc(fused, labels))
        for alpha in alphas:
            threshold = directional_conformal_threshold(fused[calib_idx], alpha, "higher")
            false_alarm_by_alpha[alpha].append(directional_trigger_rate(fused[test_true_idx], threshold, "higher"))
            detection_by_alpha[alpha].append(directional_trigger_rate(fused[false_idx], threshold, "higher"))
    return {
        "name": candidate_name,
        "signals": list(signals),
        "method": method,
        "direction": "higher",
        "auroc": _mean(aurocs),
        "alphas": _alpha_payloads(false_alarm_by_alpha, detection_by_alpha),
    }


def _split_true_indices(true_idx: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    perm = true_idx[torch.randperm(int(true_idx.numel()), generator=generator)]
    half = max(1, int(true_idx.numel()) // 2)
    calib_idx = perm[:half]
    test_true_idx = perm[half:]
    if int(test_true_idx.numel()) == 0:
        raise ValueError("at least two true records are required for split conformal evaluation.")
    return calib_idx, test_true_idx


def _alpha_payloads(
    false_alarm_by_alpha: Mapping[float, Sequence[float]],
    detection_by_alpha: Mapping[float, Sequence[float]],
) -> dict[str, Any]:
    return {
        str(alpha): _rate_payload(false_alarm_by_alpha[alpha], detection_by_alpha[alpha], alpha)
        for alpha in false_alarm_by_alpha
    }


def _rate_payload(false_alarms: Sequence[float], detections: Sequence[float], alpha: float) -> dict[str, Any]:
    false_alarm = _mean(false_alarms)
    detection = _mean(detections)
    return {
        "alpha": float(alpha),
        "false_alarm": false_alarm,
        "coverage": 1.0 - false_alarm,
        "detection": detection,
        "pass": abs(false_alarm - float(alpha)) <= TOLERANCE,
        "repeats": len(false_alarms),
    }


def _best_at_alpha(results: Mapping[str, Mapping[str, Any]], alpha: float) -> dict[str, Any]:
    key = str(float(alpha))
    candidates = []
    for name, result in results.items():
        alpha_payload = result.get("alphas", {}).get(key)
        if alpha_payload is None:
            continue
        candidates.append((name, result, alpha_payload))
    if not candidates:
        raise ValueError(f"no candidate has alpha {alpha}.")
    name, result, alpha_payload = max(
        candidates,
        key=lambda item: (
            float(item[2]["detection"]),
            float(item[1]["auroc"]),
            -float(item[2]["false_alarm"]),
        ),
    )
    return {
        "name": name,
        "signals": list(result["signals"]),
        "method": result["method"],
        "auroc": float(result["auroc"]),
        "false_alarm": float(alpha_payload["false_alarm"]),
        "detection": float(alpha_payload["detection"]),
        "coverage": float(alpha_payload["coverage"]),
    }


def _directions_for_run(config: Mapping[str, Any], signals: Sequence[str]) -> dict[str, str]:
    directions = {signal: DEFAULT_SCORE_DIRECTIONS.get(signal, "higher") for signal in signals}
    trajectory_config = config.get("trajectory_signal_score_dump")
    if isinstance(trajectory_config, Mapping):
        for signal, direction in dict(trajectory_config.get("directions", {})).items():
            if str(signal) in directions:
                if str(direction) not in {"higher", "lower"}:
                    raise ValueError(f"direction for signal {signal!r} must be 'higher' or 'lower'.")
                directions[str(signal)] = str(direction)
    return directions


def _candidate_tuple(name: str, signals: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    candidate_name = str(name).strip()
    signal_tuple = tuple(str(signal).strip() for signal in signals if str(signal).strip())
    if not candidate_name:
        raise ValueError("candidate name must be non-empty.")
    if not signal_tuple:
        raise ValueError(f"candidate {candidate_name!r} must contain at least one signal.")
    if len(set(signal_tuple)) != len(signal_tuple):
        raise ValueError(f"candidate {candidate_name!r} must contain unique signals.")
    return candidate_name, signal_tuple


def _parse_candidate(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        signals = _parse_csv(value, name="candidate")
        assert signals is not None
        return "+".join(signals), signals
    name, raw_signals = value.split("=", 1)
    signals = _parse_csv(raw_signals, name="candidate")
    assert signals is not None
    return _candidate_tuple(name, signals)


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("scores name cannot be empty.")
    return name, Path(path)


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"{name} must contain unique values.")
    return parts


def _validate_labels_for_split(labels: torch.Tensor) -> None:
    if int((labels == 0).sum().item()) < 2:
        raise ValueError("at least two true records are required.")
    if int((labels == 1).sum().item()) < 1:
        raise ValueError("at least one false record is required.")


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(float(value) for value in values) if values else float("nan")


if __name__ == "__main__":
    main()
