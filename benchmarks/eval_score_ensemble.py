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
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from eigentruth.calibration import (
    DEFAULT_SCORE_DIRECTIONS,
    GeometryScoreFusionArtifact,
    GeometryScoreFusionCalibrator,
    RankScoreFusionArtifact,
    RankScoreFusionCalibrator,
)
from eigentruth.eval.conformal import directional_conformal_threshold, directional_trigger_rate
from eigentruth.eval.metrics import roc_auc
from eigentruth.eval.score_dump import (
    load_score_dump_columns,
    score_dump_cache_summary,
    score_dump_file_metadata,
)
from eigentruth.eval.score_fusion import (
    combine_geometry_uncertainty_scores,
    combine_rank_anomaly_scores,
    directional_rank_anomaly_scores,
    native_anomaly_scores,
)

ALPHAS = (0.05, 0.10, 0.20)
METHODS = ("max_rank", "mean_rank")
GEOMETRY_FUSION_METHODS = ("interaction", "product")
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


def _dedupe_signals(*groups: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        signal
        for group in groups
        if group is not None
        for signal in group
    ))


def _resolve_geometry_groups(
    geometry_signals: Sequence[str] | None,
    uncertainty_signals: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    geometry = tuple(geometry_signals or ())
    uncertainty = tuple(uncertainty_signals or ())
    if bool(geometry) != bool(uncertainty):
        raise ValueError("geometry_signals and uncertainty_signals must be provided together.")
    if len(set(geometry)) != len(geometry):
        raise ValueError("geometry_signals must contain unique values.")
    if len(set(uncertainty)) != len(uncertainty):
        raise ValueError("uncertainty_signals must contain unique values.")
    if set(geometry) & set(uncertainty):
        raise ValueError("geometry_signals and uncertainty_signals must not overlap.")
    return geometry, uncertainty


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


def _load_scores(
    path: Path,
    *,
    signals: Sequence[str],
    cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    dump = load_score_dump_columns(path, signals, cache=cache)
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
        "auroc": roc_auc(native_anomaly_scores(scores, direction), labels),
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
            directional_rank_anomaly_scores(
                scores[calib_idx],
                scores,
                direction=directions[name],
            )
            for name, scores in selected_scores.items()
        ]
        ensemble_scores = combine_rank_anomaly_scores(rank_scores, method)
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


def _score_geometry_fusion(
    *,
    selected_scores: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    directions: Mapping[str, str],
    geometry_signals: Sequence[str],
    uncertainty_signals: Sequence[str],
    geometry_method: str,
    uncertainty_method: str,
    fusion_method: str,
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
        geometry_rank_scores = [
            directional_rank_anomaly_scores(
                selected_scores[name][calib_idx],
                selected_scores[name],
                direction=directions[name],
            )
            for name in geometry_signals
        ]
        uncertainty_rank_scores = [
            directional_rank_anomaly_scores(
                selected_scores[name][calib_idx],
                selected_scores[name],
                direction=directions[name],
            )
            for name in uncertainty_signals
        ]
        geometry_scores = combine_rank_anomaly_scores(geometry_rank_scores, geometry_method)
        uncertainty_scores = combine_rank_anomaly_scores(uncertainty_rank_scores, uncertainty_method)
        fusion_scores = combine_geometry_uncertainty_scores(
            geometry_scores,
            uncertainty_scores,
            method=fusion_method,
        )
        aurocs.append(roc_auc(fusion_scores, labels))
        for alpha in alphas:
            threshold = directional_conformal_threshold(fusion_scores[calib_idx], alpha, "higher")
            false_alarm_by_alpha[alpha].append(
                directional_trigger_rate(fusion_scores[test_true_idx], threshold, "higher")
            )
            detection_by_alpha[alpha].append(
                directional_trigger_rate(fusion_scores[false_idx], threshold, "higher")
            )
    return {
        "geometry_method": geometry_method,
        "uncertainty_method": uncertainty_method,
        "fusion_method": fusion_method,
        "fusion_style": _geometry_fusion_style(fusion_method),
        "direction": "higher",
        "geometry_signals": list(geometry_signals),
        "uncertainty_signals": list(uncertainty_signals),
        "auroc": _mean(aurocs),
        "alphas": {
            str(alpha): _rate_payload(false_alarm_by_alpha[alpha], detection_by_alpha[alpha], alpha)
            for alpha in alphas
        },
    }


def _geometry_fusion_style(fusion_method: str) -> str:
    if fusion_method == "product":
        return "global_local_uncertainty"
    if fusion_method == "interaction":
        return "geometry_uncertainty_interaction"
    return "geometry_uncertainty_fusion"


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
    geometry_signals: Sequence[str] | None = None,
    uncertainty_signals: Sequence[str] | None = None,
    geometry_method: str = "mean_rank",
    uncertainty_method: str = "mean_rank",
    geometry_fusion_methods: Sequence[str] = GEOMETRY_FUSION_METHODS,
    alphas: Sequence[float] = ALPHAS,
    repeats: int = 20,
    seed: int = 0,
    best_alpha: float = 0.10,
    score_dump_cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not score_dumps:
        raise ValueError("at least one score dump is required.")
    if not signals:
        raise ValueError("at least one signal is required.")
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    if any(not (0.0 < float(alpha) < 1.0) for alpha in alphas):
        raise ValueError("alphas must be in (0, 1).")
    geometry_signals, uncertainty_signals = _resolve_geometry_groups(geometry_signals, uncertainty_signals)
    load_signals = _dedupe_signals(signals, geometry_signals, uncertainty_signals)

    runs = []
    score_dump_metadata_cache = {} if score_dump_cache is None else score_dump_cache
    for name, path in score_dumps:
        dump = _load_scores(path, signals=load_signals, cache=score_dump_metadata_cache)
        labels = dump["labels"]
        missing = [signal for signal in load_signals if signal not in dump["scores"]]
        if missing:
            raise ValueError(f"{path} is missing requested score(s): {missing}.")
        selected_scores = {signal: dump["scores"][signal] for signal in load_signals}
        directions = {
            signal: DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")
            for signal in load_signals
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
                selected_scores={signal: selected_scores[signal] for signal in signals},
                labels=labels,
                directions=directions,
                method=method,
                alphas=alphas,
                repeats=repeats,
                seed=seed,
            )
            for method in methods
        }
        geometry_fusion_results = {}
        if geometry_signals and uncertainty_signals:
            geometry_fusion_results = {
                method: _score_geometry_fusion(
                    selected_scores=selected_scores,
                    labels=labels,
                    directions=directions,
                    geometry_signals=geometry_signals,
                    uncertainty_signals=uncertainty_signals,
                    geometry_method=geometry_method,
                    uncertainty_method=uncertainty_method,
                    fusion_method=method,
                    alphas=alphas,
                    repeats=repeats,
                    seed=seed,
                )
                for method in geometry_fusion_methods
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
            "loaded_signals": list(load_signals),
            "directions": directions,
            "geometry_signals": list(geometry_signals),
            "uncertainty_signals": list(uncertainty_signals),
            "n_total": int(labels.numel()),
            "n_true": int((labels == 0).sum().item()),
            "n_false": int((labels == 1).sum().item()),
            "single_results": single_results,
            "ensemble_results": ensemble_results,
            "geometry_fusion_results": geometry_fusion_results,
            "best_single_at_alpha": _best_at_alpha(single_results, best_alpha),
            "best_ensemble_at_alpha": _best_at_alpha(ensemble_results, best_alpha),
            "best_geometry_fusion_at_alpha": _best_at_alpha(geometry_fusion_results, best_alpha),
        })

    return {
        "schema_version": 1,
        "signals": list(signals),
        "methods": list(methods),
        "geometry_signals": list(geometry_signals),
        "uncertainty_signals": list(uncertainty_signals),
        "geometry_method": geometry_method,
        "uncertainty_method": uncertainty_method,
        "geometry_fusion_methods": list(geometry_fusion_methods),
        "alphas": [float(alpha) for alpha in alphas],
        "repeats": int(repeats),
        "seed": int(seed),
        "best_alpha": float(best_alpha),
        "score_dump_cache": score_dump_cache_summary(score_dump_metadata_cache),
        "runs": runs,
    }


def build_fusion_artifact_from_score_dump(
    score_dump: tuple[str, Path],
    *,
    signals: Sequence[str],
    method: str,
    alpha: float,
    cache: MutableMapping[str, Any] | None = None,
) -> RankScoreFusionArtifact:
    """Fit a deployable fusion artifact from all normal records in one score dump."""
    name, path = score_dump
    dump = _load_scores(path, signals=signals, cache=cache)
    missing = [signal for signal in signals if signal not in dump["scores"]]
    if missing:
        raise ValueError(f"{path} is missing requested score(s): {missing}.")
    config = dump["config"]
    directions = {
        signal: DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")
        for signal in signals
    }
    target_layer = config.get("layer")
    calibrator = RankScoreFusionCalibrator(alpha=alpha, method=method)
    return calibrator.calibrate(
        labels=dump["labels"].tolist(),
        scores={signal: dump["scores"][signal].tolist() for signal in signals},
        directions=directions,
        model_id=None if config.get("model") is None else str(config.get("model")),
        target_layer=None if target_layer is None else int(target_layer),
        model_revision=None if config.get("model_revision") is None else str(config.get("model_revision")),
        score_dump_metadata={
            "run_name": name,
            **score_dump_file_metadata(path, cache=cache),
            "summary": dump["score_dump_summary"],
            "source_format": dump["score_dump_source_format"],
        },
    )


def build_geometry_fusion_artifact_from_score_dump(
    score_dump: tuple[str, Path],
    *,
    geometry_signals: Sequence[str],
    uncertainty_signals: Sequence[str],
    geometry_method: str,
    uncertainty_method: str,
    fusion_method: str,
    alpha: float,
    cache: MutableMapping[str, Any] | None = None,
) -> GeometryScoreFusionArtifact:
    """Fit a deployable geometry-by-uncertainty fusion artifact from one score dump."""
    name, path = score_dump
    load_signals = _dedupe_signals(geometry_signals, uncertainty_signals)
    dump = _load_scores(path, signals=load_signals, cache=cache)
    missing = [signal for signal in load_signals if signal not in dump["scores"]]
    if missing:
        raise ValueError(f"{path} is missing requested score(s): {missing}.")
    config = dump["config"]
    directions = {
        signal: DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")
        for signal in load_signals
    }
    target_layer = config.get("layer")
    calibrator = GeometryScoreFusionCalibrator(
        alpha=alpha,
        geometry_method=geometry_method,
        uncertainty_method=uncertainty_method,
        fusion_method=fusion_method,
    )
    return calibrator.calibrate(
        labels=dump["labels"].tolist(),
        scores={signal: dump["scores"][signal].tolist() for signal in load_signals},
        geometry_signals=geometry_signals,
        uncertainty_signals=uncertainty_signals,
        directions=directions,
        model_id=None if config.get("model") is None else str(config.get("model")),
        target_layer=None if target_layer is None else int(target_layer),
        model_revision=None if config.get("model_revision") is None else str(config.get("model_revision")),
        score_dump_metadata={
            "run_name": name,
            **score_dump_file_metadata(path, cache=cache),
            "summary": dump["score_dump_summary"],
            "source_format": dump["score_dump_source_format"],
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    score_dumps = [_parse_named_path(value) for value in args.scores]
    signals = _parse_csv(args.signals, name="signals")
    if signals is None:
        raise ValueError("--signals is required.")
    methods = _parse_csv(args.methods, name="methods") or METHODS
    geometry_signals = _parse_csv(getattr(args, "geometry_signals", None), name="geometry_signals")
    uncertainty_signals = _parse_csv(getattr(args, "uncertainty_signals", None), name="uncertainty_signals")
    geometry_method = str(getattr(args, "geometry_method", "mean_rank"))
    uncertainty_method = str(getattr(args, "uncertainty_method", "mean_rank"))
    geometry_fusion_methods = (
        _parse_csv(getattr(args, "geometry_fusion_methods", None), name="geometry_fusion_methods")
        or GEOMETRY_FUSION_METHODS
    )
    alphas = tuple(float(value) for value in (_parse_csv(args.alphas, name="alphas") or ()))
    payload = build_ensemble_report(
        score_dumps,
        signals=signals,
        methods=methods,
        geometry_signals=geometry_signals,
        uncertainty_signals=uncertainty_signals,
        geometry_method=geometry_method,
        uncertainty_method=uncertainty_method,
        geometry_fusion_methods=geometry_fusion_methods,
        alphas=alphas or ALPHAS,
        repeats=args.repeats,
        seed=args.seed,
        best_alpha=args.best_alpha,
    )
    if args.save_best_fusion_artifact:
        if len(score_dumps) != 1:
            raise ValueError("--save-best-fusion-artifact requires exactly one --scores input.")
        run_payload = payload["runs"][0]
        best_ensemble = run_payload["best_ensemble_at_alpha"]
        if best_ensemble is None:
            raise ValueError("no best ensemble is available at --best-alpha.")
        artifact_path = Path(args.save_best_fusion_artifact)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = build_fusion_artifact_from_score_dump(
            score_dumps[0],
            signals=signals,
            method=str(best_ensemble["name"]),
            alpha=args.best_alpha,
            cache={},
        )
        artifact.save_json(artifact_path)
        run_payload["best_fusion_artifact"] = {
            "path": str(artifact_path),
            "method": artifact.method,
            "threshold": artifact.threshold,
            "conformal_alpha": artifact.conformal_alpha,
            "signals": list(artifact.signal_names()),
            "calibration_size": artifact.calibration_size(),
        }
        print(f"Wrote score fusion artifact to {artifact_path}")
    if getattr(args, "save_best_geometry_fusion_artifact", None):
        if len(score_dumps) != 1:
            raise ValueError("--save-best-geometry-fusion-artifact requires exactly one --scores input.")
        if geometry_signals is None or uncertainty_signals is None:
            raise ValueError(
                "--save-best-geometry-fusion-artifact requires --geometry-signals and --uncertainty-signals."
            )
        run_payload = payload["runs"][0]
        best_geometry = run_payload["best_geometry_fusion_at_alpha"]
        if best_geometry is None:
            raise ValueError("no best geometry fusion is available at --best-alpha.")
        artifact_path = Path(args.save_best_geometry_fusion_artifact)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = build_geometry_fusion_artifact_from_score_dump(
            score_dumps[0],
            geometry_signals=geometry_signals,
            uncertainty_signals=uncertainty_signals,
            geometry_method=geometry_method,
            uncertainty_method=uncertainty_method,
            fusion_method=str(best_geometry["name"]),
            alpha=args.best_alpha,
            cache={},
        )
        artifact.save_json(artifact_path)
        run_payload["best_geometry_fusion_artifact"] = {
            "path": str(artifact_path),
            "fusion_method": artifact.fusion_method,
            "fusion_style": _geometry_fusion_style(artifact.fusion_method),
            "geometry_method": artifact.geometry_method,
            "uncertainty_method": artifact.uncertainty_method,
            "threshold": artifact.threshold,
            "conformal_alpha": artifact.conformal_alpha,
            "geometry_signals": [signal.name for signal in artifact.geometry_signals],
            "uncertainty_signals": [signal.name for signal in artifact.uncertainty_signals],
            "calibration_size": artifact.calibration_size(),
        }
        print(f"Wrote geometry fusion artifact to {artifact_path}")
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
    parser.add_argument("--geometry-signals", default=None,
                        help="optional comma-list of representation-geometry scores for geometry fusion")
    parser.add_argument("--uncertainty-signals", default=None,
                        help="optional comma-list of confidence or uncertainty scores for geometry fusion")
    parser.add_argument("--geometry-method", default="mean_rank",
                        help="rank fusion method for geometry signals")
    parser.add_argument("--uncertainty-method", default="mean_rank",
                        help="rank fusion method for uncertainty signals")
    parser.add_argument("--geometry-fusion-methods", default=",".join(GEOMETRY_FUSION_METHODS),
                        help="comma-list of geometry/uncertainty fusion methods")
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in ALPHAS),
                        help="comma-list of conformal alpha values")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-alpha", type=float, default=0.10,
                        help="alpha used for best single/ensemble summary")
    parser.add_argument("--save-best-fusion-artifact", default=None,
                        help="optional path to save a deployable artifact for the best ensemble")
    parser.add_argument("--save-best-geometry-fusion-artifact", default=None,
                        help="optional path to save a deployable artifact for the best geometry fusion")
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    args = parser.parse_args()
    payload = run(args)
    key = str(float(args.best_alpha))
    for run_payload in payload["runs"]:
        best_single = run_payload["best_single_at_alpha"]
        best_ensemble = run_payload["best_ensemble_at_alpha"]
        best_geometry = run_payload["best_geometry_fusion_at_alpha"]
        print(
            f"{run_payload['name']}: "
            f"best_single@{key}={None if best_single is None else best_single['name']} "
            f"det={None if best_single is None else best_single['detection']}  "
            f"best_ensemble@{key}={None if best_ensemble is None else best_ensemble['name']} "
            f"det={None if best_ensemble is None else best_ensemble['detection']}  "
            f"best_geometry_fusion@{key}={None if best_geometry is None else best_geometry['name']} "
            f"det={None if best_geometry is None else best_geometry['detection']}"
        )


if __name__ == "__main__":
    main()
