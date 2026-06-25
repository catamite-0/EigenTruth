"""Serializable calibration metadata for EigenTruth diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CalibrationScore:
    """Threshold and conformal metadata for one diagnostic score.

    Args:
        name: Stable score name, such as ``maha`` or ``truth_proj``.
        threshold: Alarm threshold in the score's native units.
        conformal_alpha: Optional false-alarm budget used to choose the threshold.
        direction: Whether higher or lower values are more anomalous.
    """

    name: str
    threshold: float
    conformal_alpha: Optional[float] = None
    direction: str = "higher"

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        if isinstance(self.threshold, bool):
            raise ValueError("threshold must be numeric and must not be NaN.")
        threshold = float(self.threshold)
        if math.isnan(threshold):
            raise ValueError("threshold must be numeric and must not be NaN.")
        object.__setattr__(self, "threshold", threshold)
        if self.conformal_alpha is not None and not (0.0 < self.conformal_alpha < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "threshold": self.threshold,
            "conformal_alpha": self.conformal_alpha,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationScore":
        """Build a score config from JSON-like data."""
        return cls(
            name=str(data["name"]),
            threshold=float(data["threshold"]),
            conformal_alpha=(None if data.get("conformal_alpha") is None else float(data["conformal_alpha"])),
            direction=str(data.get("direction", "higher")),
        )


@dataclass(frozen=True)
class SteeringPolicyConfig:
    """Configuration for optional activation steering decisions."""

    mode: str = "disabled"
    steering_lambda: float = 0.0
    max_steering_lambda: Optional[float] = None

    def __post_init__(self) -> None:
        if self.steering_lambda < 0.0:
            raise ValueError("steering_lambda must be non-negative.")
        if self.max_steering_lambda is not None and self.max_steering_lambda < self.steering_lambda:
            raise ValueError("max_steering_lambda must be >= steering_lambda.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "mode": self.mode,
            "steering_lambda": self.steering_lambda,
            "max_steering_lambda": self.max_steering_lambda,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SteeringPolicyConfig":
        """Build a steering config from JSON-like data."""
        return cls(
            mode=str(data.get("mode", "disabled")),
            steering_lambda=float(data.get("steering_lambda", 0.0)),
            max_steering_lambda=(
                None if data.get("max_steering_lambda") is None else float(data["max_steering_lambda"])
            ),
        )


@dataclass(frozen=True)
class CalibrationArtifact:
    """Versioned calibration settings for a model/layer/domain.

    Artifacts are intentionally plain dataclasses so they can be saved as JSON,
    YAML, or embedded into benchmark output without pulling in storage libraries.
    """

    model_id: str
    target_layer: int
    scores: tuple[CalibrationScore, ...]
    eigentruth_version: str
    model_revision: Optional[str] = None
    steering_policy: SteeringPolicyConfig = field(default_factory=SteeringPolicyConfig)
    warmup_dataset_metadata: Mapping[str, Any] = field(default_factory=dict)
    calibration_dataset_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    commit_sha: Optional[str] = None
    schema_version: int = 1

    def score_names(self) -> tuple[str, ...]:
        """Return score names in artifact order."""
        return tuple(score.name for score in self.scores)

    def get_score(self, name: str) -> CalibrationScore:
        """Return one score config by name."""
        for score in self.scores:
            if score.name == name:
                return score
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "target_layer": self.target_layer,
            "scores": [score.to_dict() for score in self.scores],
            "eigentruth_version": self.eigentruth_version,
            "steering_policy": self.steering_policy.to_dict(),
            "warmup_dataset_metadata": dict(self.warmup_dataset_metadata),
            "calibration_dataset_metadata": dict(self.calibration_dataset_metadata),
            "created_at": self.created_at,
            "commit_sha": self.commit_sha,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationArtifact":
        """Build an artifact from JSON-like data."""
        return cls(
            model_id=str(data["model_id"]),
            model_revision=None if data.get("model_revision") is None else str(data["model_revision"]),
            target_layer=int(data["target_layer"]),
            scores=tuple(CalibrationScore.from_dict(score) for score in data["scores"]),
            eigentruth_version=str(data["eigentruth_version"]),
            steering_policy=SteeringPolicyConfig.from_dict(data.get("steering_policy", {})),
            warmup_dataset_metadata=dict(data.get("warmup_dataset_metadata", {})),
            calibration_dataset_metadata=dict(data.get("calibration_dataset_metadata", {})),
            created_at=None if data.get("created_at") is None else str(data["created_at"]),
            commit_sha=None if data.get("commit_sha") is None else str(data["commit_sha"]),
            schema_version=int(data.get("schema_version", 1)),
        )

    def save_json(self, path: str | Path) -> None:
        """Save artifact metadata as UTF-8 JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "CalibrationArtifact":
        """Load artifact metadata from UTF-8 JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
