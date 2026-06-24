"""Serializable artifacts for calibrated multi-signal score fusion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from eigentruth.eval.conformal import directional_conformal_threshold
from eigentruth.eval.metrics import roc_auc
from eigentruth.eval.score_fusion import (
    RANK_SCORE_FUSION_METHODS,
    combine_rank_anomaly_scores,
    directional_rank_anomaly_scores,
)


def _finite_float_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values.")
    return result


@dataclass(frozen=True)
class ScoreFusionSignal:
    """One score input and its empirical calibration distribution."""

    name: str
    direction: str = "higher"
    calibration_scores: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty.")
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        object.__setattr__(
            self,
            "calibration_scores",
            _finite_float_tuple(self.calibration_scores, name="calibration_scores"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "direction": self.direction,
            "calibration_scores": list(self.calibration_scores),
            "calibration_size": len(self.calibration_scores),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScoreFusionSignal":
        """Build a signal from JSON-like data."""
        return cls(
            name=str(data["name"]),
            direction=str(data.get("direction", "higher")),
            calibration_scores=tuple(float(value) for value in data["calibration_scores"]),
        )


@dataclass(frozen=True)
class RankScoreFusionArtifact:
    """Deployable rank-calibrated fusion artifact for diagnostic scores."""

    signals: tuple[ScoreFusionSignal, ...]
    method: str = "max_rank"
    threshold: float | None = None
    conformal_alpha: float | None = None
    model_id: str | None = None
    target_layer: int | None = None
    model_revision: str | None = None
    eigentruth_version: str | None = None
    score_dump_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    commit_sha: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.signals:
            raise ValueError("signals must be non-empty.")
        if self.method not in RANK_SCORE_FUSION_METHODS:
            raise ValueError(f"method must be one of {RANK_SCORE_FUSION_METHODS}.")
        names = [signal.name for signal in self.signals]
        if len(set(names)) != len(names):
            raise ValueError("signals must have unique names.")
        if self.threshold is not None and not math.isfinite(float(self.threshold)):
            if not math.isinf(float(self.threshold)):
                raise ValueError("threshold must be finite or infinite.")
        if self.conformal_alpha is not None and not (0.0 < self.conformal_alpha < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")

    def signal_names(self) -> tuple[str, ...]:
        """Return signal names in artifact order."""
        return tuple(signal.name for signal in self.signals)

    def calibration_size(self) -> int:
        """Return the shared calibration size, requiring aligned signal calibration sets."""
        sizes = {len(signal.calibration_scores) for signal in self.signals}
        if len(sizes) != 1:
            raise ValueError("all fusion signals must have the same calibration size.")
        return sizes.pop()

    def score(self, scores: Mapping[str, Any]) -> torch.Tensor:
        """Return fused anomaly scores in [0, 1] for one or more records."""
        rank_scores = []
        expected_length: int | None = None
        for signal in self.signals:
            if signal.name not in scores:
                raise KeyError(signal.name)
            values = torch.as_tensor(scores[signal.name], dtype=torch.float64).flatten()
            if expected_length is None:
                expected_length = int(values.numel())
            elif values.numel() != expected_length:
                raise ValueError("all score inputs must have the same length.")
            rank_scores.append(
                directional_rank_anomaly_scores(
                    signal.calibration_scores,
                    values,
                    direction=signal.direction,
                )
            )
        return combine_rank_anomaly_scores(rank_scores, self.method)

    def calibration_fusion_scores(self) -> torch.Tensor:
        """Return fused scores for the artifact's own aligned calibration records."""
        self.calibration_size()
        return self.score({signal.name: signal.calibration_scores for signal in self.signals})

    def flags(self, scores: Mapping[str, Any]) -> torch.Tensor:
        """Return threshold-trigger flags for fused scores."""
        if self.threshold is None:
            raise ValueError("threshold is required to compute flags.")
        fused = self.score(scores)
        if math.isinf(float(self.threshold)):
            return torch.zeros_like(fused, dtype=torch.bool)
        return fused > float(self.threshold)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "threshold": self.threshold,
            "conformal_alpha": self.conformal_alpha,
            "model_id": self.model_id,
            "target_layer": self.target_layer,
            "model_revision": self.model_revision,
            "eigentruth_version": self.eigentruth_version,
            "signals": [signal.to_dict() for signal in self.signals],
            "score_dump_metadata": dict(self.score_dump_metadata),
            "created_at": self.created_at,
            "commit_sha": self.commit_sha,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RankScoreFusionArtifact":
        """Build an artifact from JSON-like data."""
        return cls(
            signals=tuple(ScoreFusionSignal.from_dict(signal) for signal in data["signals"]),
            method=str(data.get("method", "max_rank")),
            threshold=None if data.get("threshold") is None else float(data["threshold"]),
            conformal_alpha=None if data.get("conformal_alpha") is None else float(data["conformal_alpha"]),
            model_id=None if data.get("model_id") is None else str(data["model_id"]),
            target_layer=None if data.get("target_layer") is None else int(data["target_layer"]),
            model_revision=None if data.get("model_revision") is None else str(data["model_revision"]),
            eigentruth_version=None if data.get("eigentruth_version") is None else str(data["eigentruth_version"]),
            score_dump_metadata=dict(data.get("score_dump_metadata", {})),
            created_at=None if data.get("created_at") is None else str(data["created_at"]),
            commit_sha=None if data.get("commit_sha") is None else str(data["commit_sha"]),
            schema_version=int(data.get("schema_version", 1)),
        )

    def save_json(self, path: str | Path) -> None:
        """Save artifact metadata as UTF-8 JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "RankScoreFusionArtifact":
        """Load artifact metadata from UTF-8 JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class RankScoreFusionCalibrator:
    """Fit rank-calibrated score fusion artifacts from labeled score columns."""

    def __init__(self, *, alpha: float = 0.1, method: str = "max_rank") -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if method not in RANK_SCORE_FUSION_METHODS:
            raise ValueError(f"method must be one of {RANK_SCORE_FUSION_METHODS}.")
        self.alpha = float(alpha)
        self.method = method

    def calibrate(
        self,
        *,
        labels: Sequence[int],
        scores: Mapping[str, Sequence[float]],
        directions: Mapping[str, str] | None = None,
        model_id: str | None = None,
        target_layer: int | None = None,
        model_revision: str | None = None,
        eigentruth_version: str | None = None,
        score_dump_metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        commit_sha: str | None = None,
    ) -> RankScoreFusionArtifact:
        """Fit a deployment artifact using label 0 records as calibration normals."""
        if not scores:
            raise ValueError("scores must contain at least one signal.")
        labels_t = torch.as_tensor(labels, dtype=torch.int64).flatten()
        if labels_t.numel() == 0:
            raise ValueError("labels must be non-empty.")
        if not torch.logical_or(labels_t == 0, labels_t == 1).all():
            raise ValueError("labels must be binary values in {0, 1}.")
        normal_mask = labels_t == 0
        if int(normal_mask.sum().item()) == 0:
            raise ValueError("at least one label 0 calibration record is required.")

        resolved_directions = {} if directions is None else dict(directions)
        signals: list[ScoreFusionSignal] = []
        for name, values in scores.items():
            values_t = torch.as_tensor(values, dtype=torch.float64).flatten()
            if values_t.numel() != labels_t.numel():
                raise ValueError(f"score {name!r} length does not match labels.")
            if not torch.isfinite(values_t).all():
                raise ValueError(f"score {name!r} contains non-finite values.")
            direction = str(resolved_directions.get(name, "higher"))
            signals.append(
                ScoreFusionSignal(
                    name=str(name),
                    direction=direction,
                    calibration_scores=tuple(float(value) for value in values_t[normal_mask].tolist()),
                )
            )
        artifact = RankScoreFusionArtifact(
            signals=tuple(signals),
            method=self.method,
            conformal_alpha=self.alpha,
            model_id=model_id,
            target_layer=target_layer,
            model_revision=model_revision,
            eigentruth_version=eigentruth_version,
            score_dump_metadata={} if score_dump_metadata is None else dict(score_dump_metadata),
            created_at=created_at,
            commit_sha=commit_sha,
        )
        threshold = directional_conformal_threshold(artifact.calibration_fusion_scores(), self.alpha, "higher")
        return RankScoreFusionArtifact(
            signals=artifact.signals,
            method=artifact.method,
            threshold=threshold,
            conformal_alpha=artifact.conformal_alpha,
            model_id=artifact.model_id,
            target_layer=artifact.target_layer,
            model_revision=artifact.model_revision,
            eigentruth_version=artifact.eigentruth_version,
            score_dump_metadata=artifact.score_dump_metadata,
            created_at=artifact.created_at,
            commit_sha=artifact.commit_sha,
            schema_version=artifact.schema_version,
        )

    def evaluate(
        self,
        *,
        labels: Sequence[int],
        scores: Mapping[str, Sequence[float]],
        directions: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Fit and evaluate a fusion artifact on the provided labeled score columns."""
        artifact = self.calibrate(labels=labels, scores=scores, directions=directions)
        labels_t = torch.as_tensor(labels, dtype=torch.int64).flatten()
        fused = artifact.score(scores)
        flags = artifact.flags(scores)
        normal = labels_t == 0
        anomalous = labels_t == 1
        false_alarm = (
            0.0
            if int(normal.sum().item()) == 0
            else float(flags[normal].double().mean().item())
        )
        detection = (
            0.0
            if int(anomalous.sum().item()) == 0
            else float(flags[anomalous].double().mean().item())
        )
        return {
            "method": self.method,
            "threshold": artifact.threshold,
            "conformal_alpha": self.alpha,
            "auroc": roc_auc(fused, labels_t),
            "false_alarm": false_alarm,
            "detection": detection,
            "artifact": artifact.to_dict(),
        }
