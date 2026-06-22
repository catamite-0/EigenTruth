"""Layer and score sweep calibration utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from eigentruth import __version__
from eigentruth.calibration.artifacts import CalibrationArtifact, CalibrationScore, SteeringPolicyConfig
from eigentruth.eval.conformal import directional_conformal_threshold, directional_trigger_rate
from eigentruth.eval.metrics import roc_auc

ArrayLike = torch.Tensor | Sequence[float]

DEFAULT_SCORE_DIRECTIONS: dict[str, str] = {
    "maha_last": "higher",
    "maha": "higher",
    "truth_proj": "higher",
    "subspace_resid": "higher",
    "disp_euclid": "higher",
    "disp_hse": "higher",
    "eigenscore": "higher",
    "nll_answer": "higher",
}


@dataclass(frozen=True)
class SweepScoreResult:
    """Calibration and ranking metrics for one layer/score pair."""

    layer: int
    score_name: str
    direction: str
    threshold: float
    conformal_alpha: float
    auroc: float
    false_alarm: float
    detection: float
    n_true: int
    n_false: int

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        if not (0.0 < self.conformal_alpha < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")

    def score_config(self) -> CalibrationScore:
        """Return this sweep result as a calibration score config."""
        return CalibrationScore(
            name=self.score_name,
            threshold=self.threshold,
            conformal_alpha=self.conformal_alpha,
            direction=self.direction,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "layer": self.layer,
            "score_name": self.score_name,
            "direction": self.direction,
            "threshold": self.threshold,
            "conformal_alpha": self.conformal_alpha,
            "auroc": self.auroc,
            "false_alarm": self.false_alarm,
            "detection": self.detection,
            "n_true": self.n_true,
            "n_false": self.n_false,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SweepScoreResult":
        """Build a sweep score result from JSON-like data."""
        return cls(
            layer=int(data["layer"]),
            score_name=str(data["score_name"]),
            direction=str(data.get("direction", "higher")),
            threshold=float(data["threshold"]),
            conformal_alpha=float(data["conformal_alpha"]),
            auroc=float(data["auroc"]),
            false_alarm=float(data["false_alarm"]),
            detection=float(data["detection"]),
            n_true=int(data["n_true"]),
            n_false=int(data["n_false"]),
        )


@dataclass(frozen=True)
class LayerScoreSweepResult:
    """Sweep results for one model layer."""

    layer: int
    scores: tuple[SweepScoreResult, ...]

    def best_score(self, *, best_by: str = "auroc") -> SweepScoreResult:
        """Return the best score result for this layer."""
        if not self.scores:
            raise ValueError("layer sweep result contains no scores.")
        return _best_result(self.scores, best_by=best_by)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {"layer": self.layer, "scores": [score.to_dict() for score in self.scores]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerScoreSweepResult":
        """Build a layer sweep result from JSON-like data."""
        return cls(
            layer=int(data["layer"]),
            scores=tuple(SweepScoreResult.from_dict(score) for score in data.get("scores", ())),
        )


@dataclass(frozen=True)
class LayerScoreSweepReport:
    """Versioned report for layer/score calibration sweeps."""

    model_id: str
    conformal_alpha: float
    layers: tuple[LayerScoreSweepResult, ...]
    best_by: str = "auroc"
    eigentruth_version: str = __version__
    model_revision: Optional[str] = None
    scores_path: Optional[str] = None
    created_at: Optional[str] = None
    commit_sha: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not (0.0 < self.conformal_alpha < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")
        if self.best_by not in {"auroc", "detection"}:
            raise ValueError("best_by must be 'auroc' or 'detection'.")

    def score_results(self) -> tuple[SweepScoreResult, ...]:
        """Return all layer/score results in report order."""
        return tuple(score for layer in self.layers for score in layer.scores)

    def best_score(self) -> SweepScoreResult:
        """Return the best layer/score result according to ``best_by``."""
        results = self.score_results()
        if not results:
            raise ValueError("sweep report contains no score results.")
        return _best_result(results, best_by=self.best_by)

    def best_artifact(
        self,
        *,
        steering_policy: Optional[SteeringPolicyConfig] = None,
        warmup_dataset_metadata: Optional[Mapping[str, Any]] = None,
        calibration_dataset_metadata: Optional[Mapping[str, Any]] = None,
    ) -> CalibrationArtifact:
        """Build a single-score calibration artifact from the best sweep result."""
        best = self.best_score()
        metadata = {
            "source": "LayerScoreSweepReport",
            "best_by": self.best_by,
            "scores_path": self.scores_path,
            "n_true": best.n_true,
            "n_false": best.n_false,
        }
        if calibration_dataset_metadata:
            metadata.update(calibration_dataset_metadata)
        return CalibrationArtifact(
            model_id=self.model_id,
            model_revision=self.model_revision,
            target_layer=best.layer,
            scores=(best.score_config(),),
            eigentruth_version=self.eigentruth_version,
            steering_policy=steering_policy or SteeringPolicyConfig(),
            warmup_dataset_metadata=warmup_dataset_metadata or {},
            calibration_dataset_metadata=metadata,
            created_at=self.created_at,
            commit_sha=self.commit_sha,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        best = self.best_score() if self.score_results() else None
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "conformal_alpha": self.conformal_alpha,
            "best_by": self.best_by,
            "best": None if best is None else best.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
            "eigentruth_version": self.eigentruth_version,
            "scores_path": self.scores_path,
            "created_at": self.created_at,
            "commit_sha": self.commit_sha,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerScoreSweepReport":
        """Build a sweep report from JSON-like data."""
        return cls(
            model_id=str(data["model_id"]),
            model_revision=None if data.get("model_revision") is None else str(data["model_revision"]),
            conformal_alpha=float(data["conformal_alpha"]),
            layers=tuple(LayerScoreSweepResult.from_dict(layer) for layer in data.get("layers", ())),
            best_by=str(data.get("best_by", "auroc")),
            eigentruth_version=str(data.get("eigentruth_version", __version__)),
            scores_path=None if data.get("scores_path") is None else str(data["scores_path"]),
            created_at=None if data.get("created_at") is None else str(data["created_at"]),
            commit_sha=None if data.get("commit_sha") is None else str(data["commit_sha"]),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", 1)),
        )

    def save_json(self, path: str | Path) -> None:
        """Save the sweep report as UTF-8 JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "LayerScoreSweepReport":
        """Load a sweep report from UTF-8 JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class LayerScoreSweepCalibrator:
    """Build calibration reports across layers and diagnostic scores."""

    alpha: float = 0.1
    best_by: str = "auroc"

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if self.best_by not in {"auroc", "detection"}:
            raise ValueError("best_by must be 'auroc' or 'detection'.")

    def calibrate_from_file(
        self,
        path: str | Path,
        *,
        signals: Optional[Sequence[str]] = None,
        directions: Optional[Mapping[str, str]] = None,
        model_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LayerScoreSweepReport:
        """Load a score dump and build a layer/score sweep report."""
        dump_path = Path(path)
        dump = json.loads(dump_path.read_text(encoding="utf-8"))
        return self.calibrate_from_dump(
            dump,
            signals=signals,
            directions=directions,
            model_id=model_id,
            model_revision=model_revision,
            scores_path=str(dump_path),
            created_at=created_at,
            commit_sha=commit_sha,
            eigentruth_version=eigentruth_version,
            metadata=metadata,
        )

    def calibrate_from_dump(
        self,
        dump: Mapping[str, Any],
        *,
        signals: Optional[Sequence[str]] = None,
        directions: Optional[Mapping[str, str]] = None,
        model_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        scores_path: Optional[str] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LayerScoreSweepReport:
        """Build a layer/score sweep report from an ``eval_truthfulqa`` score dump."""
        labels = torch.as_tensor(dump["labels"], dtype=torch.int64)
        config = dict(dump.get("config", {}))
        layer_scores = _collect_layer_scores(dump)
        selected = set(signals) if signals is not None else _all_score_names(layer_scores)
        results = []
        for layer in sorted(layer_scores):
            score_results = []
            for score_name in sorted(layer_scores[layer]):
                if score_name not in selected:
                    continue
                direction = _score_direction(score_name, directions)
                score_results.append(
                    _calibrate_score(
                        layer=layer,
                        score_name=score_name,
                        scores=layer_scores[layer][score_name],
                        labels=labels,
                        alpha=self.alpha,
                        direction=direction,
                    )
                )
            if score_results:
                results.append(LayerScoreSweepResult(layer=layer, scores=tuple(score_results)))

        if not results:
            raise ValueError("no matching layer/score results were found in the score dump.")

        return LayerScoreSweepReport(
            model_id=model_id or str(config.get("model", "unknown")),
            model_revision=model_revision,
            conformal_alpha=self.alpha,
            layers=tuple(results),
            best_by=self.best_by,
            eigentruth_version=eigentruth_version,
            scores_path=scores_path,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            commit_sha=commit_sha,
            metadata=metadata or {"source": "eval_truthfulqa.py", "config": config},
        )


def _collect_layer_scores(dump: Mapping[str, Any]) -> dict[int, dict[str, Sequence[float]]]:
    config = dict(dump.get("config", {}))
    primary_layer = int(config.get("layer", 0))
    layer_scores: dict[int, dict[str, Sequence[float]]] = {primary_layer: dict(dump.get("scores", {}))}
    for layer_key, scores in dict(dump.get("sweep_scores", {})).items():
        layer = int(layer_key)
        layer_scores.setdefault(layer, {}).update(dict(scores))
    return layer_scores


def _all_score_names(layer_scores: Mapping[int, Mapping[str, Sequence[float]]]) -> set[str]:
    return {score_name for scores in layer_scores.values() for score_name in scores}


def _score_direction(score_name: str, directions: Optional[Mapping[str, str]]) -> str:
    direction = (directions or {}).get(score_name, DEFAULT_SCORE_DIRECTIONS.get(score_name, "higher"))
    if direction not in {"higher", "lower"}:
        raise ValueError("directions values must be 'higher' or 'lower'.")
    return direction


def _calibrate_score(
    *,
    layer: int,
    score_name: str,
    scores: ArrayLike,
    labels: torch.Tensor,
    alpha: float,
    direction: str,
) -> SweepScoreResult:
    scores_t = torch.as_tensor(scores, dtype=torch.float64).flatten()
    if scores_t.numel() != labels.numel():
        raise ValueError(f"score '{score_name}' has {scores_t.numel()} values but labels has {labels.numel()}.")
    true_scores = scores_t[labels == 0]
    false_scores = scores_t[labels == 1]
    if true_scores.numel() == 0 or false_scores.numel() == 0:
        raise ValueError("sweep calibration requires at least one true and one false labeled score.")

    threshold = directional_conformal_threshold(true_scores, alpha, direction)
    anomaly_scores = _anomaly_scores(scores_t, direction)
    false_alarm = directional_trigger_rate(true_scores, threshold, direction)
    detection = directional_trigger_rate(false_scores, threshold, direction)
    return SweepScoreResult(
        layer=layer,
        score_name=score_name,
        direction=direction,
        threshold=threshold,
        conformal_alpha=alpha,
        auroc=roc_auc(anomaly_scores, labels),
        false_alarm=false_alarm,
        detection=detection,
        n_true=int(true_scores.numel()),
        n_false=int(false_scores.numel()),
    )



def _anomaly_scores(scores: torch.Tensor, direction: str) -> torch.Tensor:
    return scores if direction == "higher" else -scores



def _best_result(results: Sequence[SweepScoreResult], *, best_by: str) -> SweepScoreResult:
    if best_by == "auroc":
        return max(results, key=lambda result: _rankable(result.auroc))
    if best_by == "detection":
        return max(results, key=lambda result: (_rankable(result.detection), _rankable(result.auroc)))
    raise ValueError("best_by must be 'auroc' or 'detection'.")


def _rankable(value: float) -> float:
    return float("-inf") if math.isnan(value) else value
