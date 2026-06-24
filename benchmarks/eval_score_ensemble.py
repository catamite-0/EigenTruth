"""Evaluate calibrated ensembles of EigenTruth diagnostic scores.

The script consumes ``eval_truthfulqa.py --dump-scores`` artifacts and compares
single diagnostic scores against simple rank-normalized ensembles. It does not
load models.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from eigentruth.calibration import DEFAULT_SCORE_DIRECTIONS
from eigentruth.eval.conformal import directional_conformal_threshold, directional_trigger_rate
from eigentruth.eval.metrics import roc_auc
from eigentruth.eval.score_dump import load_score_dump_columns, score_dump_file_metadata

ALPHAS = (0.05, 0.10, 0.20)
METHODS = ("max_rank", "mean_rank")
TOLERANCE = 0.03


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
    return parts


def _directional_rank_anomaly_scores(
    calibration_scores: torch.Tensor,
    scores: torch.Tensor,
    *,
    direction: str,
) -> torch.Tensor:
    """Map native scores to calibration-set anomaly ranks in [0, 1]."""
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    calibration_scores = calibration_scores.to(torch.float64).flatten()
    scores = scores.to(torch.float64).flatten()
    if calibration_scores.numel() == 0:
        raise ValueError("calibration scores must be non-empty.")
    sorted_calib, _ = torch.sort(calibration_scores)
    n = float(calibration_scores.numel())
    if direction == "higher":
        counts = torch.searchsorted(sorted_calib, scores, right=True).to(torch.float64)
    else:
        counts = (calibration_scores.numel() - torch.searchsorted(
            sorted_calib,
            scores,
            right=False,
        )).to(torch.float64)
    return counts / n


def _combine_rank_scores(rank_scores: Sequence[torch.Tensor], method: str) -> torch.Tensor:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}.")
    if not rank_scores:
        raise ValueError("at least one rank score is required.")
    stacked = torch.stack([score.to(torch.float64).flatten() for score in rank_scores], dim=0)
    if method == "max_rank":
        return stacked.max(dim=0).values
    return stacked.mean(dim=0)


def _native_anomaly_scores(scores: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "higher":
        return scores
    if direction == "lower":
        return -scores
    raise ValueError("direction must be 'higher' or 'lower'.")


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(float(value) for value in values) if values else float("nan")


def _rate_payload(false_alarms: Sequence[float], detections: Sequence[float], alpha: float) -> dict[str, Any]:
    false_alarm = _mean(false_alarms)
    detection = _mean(detections)
    return {
        "alpha": alpha,
        "false_alarm": false_alarm,
        "coverage": 1.0 - false_alarm,
        "detection": detection,
        "pass": abs(false_alarm - alpha) <= TOLERANCE,
        "repeats": len(false_alarms),
    }


def _load_scores(path: Path, *, signals: Sequence[str]) -> dict[str, Any]:
    dump = load_score_dump_columns(path, signals)
    labels = torch.tensor(dump.labels, dtype=torch.int64)
    scores = {
        name: torch.tensor(values, dtype=torch.float64)
        for name, values in dump.scores.items()
    }
    return {
        "config": dict(dump.config),
        "labels": labels,
        "scores": scores,
        "score_dump_summary": dict(dump.summary),
        "score_dump_source_format": dump.source_format,
    }


def _score_signal(
    *,
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
        generator = torch.Generator().manual_seed(seed + repeat)
        perm = true_idx[torch.randperm(true_idx.numel(), generator=generator)]
        half = true_idx.numel() // 2
        calib_idx = perm[:half]
        test_true_idx = perm[half:]
        for alpha in alphas:
            threshold = directional_conformal_threshold(scores[calib_idx], alpha, direction)
            false_alarm_by_alpha[alpha].append(
                directional_trigger_rate(scores[test_true_idx], threshold, direction)
            )
            detection_by_alpha[alpha].append(
                directional_trigger_rate(scores[false_idx], threshold, direction)
            )
    return {
        "direction": direction,
        "auroc": roc_auc(_native_anomaly_scores(scores, direction), labels),
        "alphas": {
            str(alpha): _rate_payload(false_alarm_by_alpha[alpha], detection_by_alpha[alpha], alpha)
            for alpha in alphas
        },
    }


def _score_ensemble(
    *,
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
        generator = torch.Generator().manual_seed(seed + repeat)
        perm = true_idx[torch.randperm(true_idx.numel(), generator=generator)]
        half = true_idx.numel() // 2
        calib_idx = perm[:half]
        test_true_idx = perm[half:]
        rank_scores = [
            _directional_rank_anomaly_scores(
                scores[calib_idx],
                scores,
                direction=directions[name],
            )
            for name, scores in selected_scores.items()
        ]
        ensemble_scores = _combine_rank_scores(rank_scores, method)
        aurocs.append(roc_auc(ensemble_scores, labels))
        for alpha in alphas:
            threshold = directional_conformal_threshold(ensemble_scores[calib_idx], alpha, "higher")
            false_alarm_by_alpha[alpha].append(
                directional_trigger_rate(ensemble_scores[test_true_idx], threshold, "higher")
            )
            detection_by_alpha[alpha].append(
                directional_trigger_rate(ensemble_scores[false_idx], threshold, "higher")
            )
    return {
        "method": method,
        "direction": "higher",
        "auroc": _mean(aurocs),
        "alphas": {
            str(alpha): _rate_payload(false_alarm_by_alpha[alpha], detection_by_alpha[alpha], alpha)
            for alpha in alphas
        },
    }


def _best_at_alpha(results: Mapping[str, Mapping[str, Any]], alpha: float) -> dict[str, Any] | None:
    key = str(alpha)
    available = [
        (name, payload) for name, payload in results.items()
        if key in payload.get("alphas", {})
    ]
    if not available:
        return None
    name, payload = max(
        available,
        key=lambda item: float(item[1]["alphas"][key]["detection"]),
    )
    alpha_payload = payload["alphas"][key]
    return {
        "name": name,
        "auroc": payload.get("auroc"),
        "false_alarm": alpha_payload["false_alarm"],
        "detection": alpha_payload["detection"],
    }


def build_ensemble_report(
    score_dumps: Sequence[tuple[str, Path]],
    *,
    signals: Sequence[str],
    methods: Sequence[str] = METHODS,
    alphas: Sequence[float] = ALPHAS,
    repeats: int = 20,
    seed: int = 0,
    best_alpha: float = 0.10,
) -> dict[str, Any]:
    if not score_dumps:
        raise ValueError("at least one score dump is required.")
    if not signals:
        raise ValueError("at least one signal is required.")
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    if any(not (0.0 < float(alpha) < 1.0) for alpha in alphas):
        raise ValueError("alphas must be in (0, 1).")

    runs = []
    score_dump_metadata_cache = {}
    for name, path in score_dumps:
        dump = _load_scores(path, signals=signals)
        labels = dump["labels"]
        missing = [signal for signal in signals if signal not in dump["scores"]]
        if missing:
            raise ValueError(f"{path} is missing requested score(s): {missing}.")
        selected_scores = {signal: dump["scores"][signal] for signal in signals}
        directions = {
            signal: DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")
            for signal in signals
        }
        single_results = {
            signal: _score_signal(
                scores=scores,
                labels=labels,
                direction=directions[signal],
                alphas=alphas,
                repeats=repeats,
                seed=seed,
            )
            for signal, scores in selected_scores.items()
        }
        ensemble_results = {
            method: _score_ensemble(
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
        runs.append({
            "name": name,
            "scores_path": str(path),
            "score_dump": {
                **score_dump_file_metadata(path, cache=score_dump_metadata_cache),
                "summary": dump["score_dump_summary"],
                "source_format": dump["score_dump_source_format"],
            },
            "config": dump["config"],
            "signals": list(signals),
            "directions": directions,
            "n_total": int(labels.numel()),
            "n_true": int((labels == 0).sum().item()),
            "n_false": int((labels == 1).sum().item()),
            "single_results": single_results,
            "ensemble_results": ensemble_results,
            "best_single_at_alpha": _best_at_alpha(single_results, best_alpha),
            "best_ensemble_at_alpha": _best_at_alpha(ensemble_results, best_alpha),
        })

    return {
        "schema_version": 1,
        "signals": list(signals),
        "methods": list(methods),
        "alphas": [float(alpha) for alpha in alphas],
        "repeats": int(repeats),
        "seed": int(seed),
        "best_alpha": float(best_alpha),
        "runs": runs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    score_dumps = [_parse_named_path(value) for value in args.scores]
    signals = _parse_csv(args.signals, name="signals")
    if signals is None:
        raise ValueError("--signals is required.")
    methods = _parse_csv(args.methods, name="methods") or METHODS
    alphas = tuple(float(value) for value in (_parse_csv(args.alphas, name="alphas") or ()))
    payload = build_ensemble_report(
        score_dumps,
        signals=signals,
        methods=methods,
        alphas=alphas or ALPHAS,
        repeats=args.repeats,
        seed=args.seed,
        best_alpha=args.best_alpha,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote score ensemble report to {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibrated ensembles over score dumps")
    parser.add_argument("--scores", action="append", required=True,
                        help="score dump path, optionally named as name=path; repeatable")
    parser.add_argument("--signals", required=True,
                        help="comma-list of score names to combine")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="comma-list of ensemble methods: max_rank,mean_rank")
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in ALPHAS),
                        help="comma-list of conformal alpha values")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-alpha", type=float, default=0.10,
                        help="alpha used for best single/ensemble summary")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    args = parser.parse_args()
    payload = run(args)
    key = str(float(args.best_alpha))
    for run_payload in payload["runs"]:
        best_single = run_payload["best_single_at_alpha"]
        best_ensemble = run_payload["best_ensemble_at_alpha"]
        print(
            f"{run_payload['name']}: "
            f"best_single@{key}={None if best_single is None else best_single['name']} "
            f"det={None if best_single is None else best_single['detection']}  "
            f"best_ensemble@{key}={None if best_ensemble is None else best_ensemble['name']} "
            f"det={None if best_ensemble is None else best_ensemble['detection']}"
        )


if __name__ == "__main__":
    main()
