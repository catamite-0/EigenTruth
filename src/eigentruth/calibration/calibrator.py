"""Calibrators that turn diagnostic scores into reusable artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import torch

from eigentruth import __version__
from eigentruth.calibration.artifacts import CalibrationArtifact, CalibrationScore, SteeringPolicyConfig
from eigentruth.eval.conformal import (
    AdaptiveScoreTransform,
    conformal_threshold,
    directional_conformal_threshold,
)

ArrayLike = torch.Tensor | Sequence[float]


@dataclass(frozen=True)
class ConformalCalibrator:
    """Build split-conformal calibration artifacts from held-out normal scores.

    The underlying conformal helper assumes higher scores are more anomalous. For
    score families where lower is more anomalous, scores are negated for
    calibration and the resulting threshold is transformed back to native units.
    """

    alpha: float = 0.1

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")

    def calibrate(
        self,
        *,
        model_id: str,
        target_layer: int,
        calibration_scores: Mapping[str, ArrayLike],
        directions: Optional[Mapping[str, str]] = None,
        model_revision: Optional[str] = None,
        steering_policy: Optional[SteeringPolicyConfig] = None,
        warmup_dataset_metadata: Optional[Mapping[str, Any]] = None,
        calibration_dataset_metadata: Optional[Mapping[str, Any]] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
    ) -> CalibrationArtifact:
        """Create a calibration artifact for one model/layer setting.

        Args:
            model_id: Model identifier or local model name.
            target_layer: Layer index used by the diagnostic score.
            calibration_scores: Mapping of score name to held-out normal scores.
            directions: Optional score directions, ``higher`` or ``lower`` anomalous.
            model_revision: Optional model revision or checkpoint hash.
            steering_policy: Optional steering policy metadata.
            warmup_dataset_metadata: Metadata for the manifold warmup data.
            calibration_dataset_metadata: Metadata for calibration scores.
            created_at: Optional timestamp. Defaults to current UTC time.
            commit_sha: Optional repository commit SHA.
            eigentruth_version: EigenTruth version string to store.
        """
        if len(calibration_scores) == 0:
            raise ValueError("calibration_scores must contain at least one score family.")

        score_configs = []
        for name, scores in calibration_scores.items():
            direction = (directions or {}).get(name, "higher")
            if direction not in {"higher", "lower"}:
                raise ValueError("directions values must be 'higher' or 'lower'.")
            threshold = directional_conformal_threshold(scores, self.alpha, direction)
            score_configs.append(
                CalibrationScore(
                    name=name,
                    threshold=threshold,
                    conformal_alpha=self.alpha,
                    direction=direction,
                )
            )

        return CalibrationArtifact(
            model_id=model_id,
            model_revision=model_revision,
            target_layer=target_layer,
            scores=tuple(score_configs),
            eigentruth_version=eigentruth_version,
            steering_policy=steering_policy or SteeringPolicyConfig(),
            warmup_dataset_metadata=warmup_dataset_metadata or {},
            calibration_dataset_metadata=calibration_dataset_metadata or {},
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            commit_sha=commit_sha,
        )


@dataclass(frozen=True)
class AdaptiveConformalCalibrator:
    """Build calibration artifacts from feature-adjusted nonconformity scores."""

    alpha: float = 0.1
    transform: AdaptiveScoreTransform = field(default_factory=AdaptiveScoreTransform)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")

    def calibrate(
        self,
        *,
        model_id: str,
        target_layer: int,
        score_name: str,
        calibration_scores: ArrayLike,
        feature_values: Mapping[str, ArrayLike] | None = None,
        output_score_name: str | None = None,
        model_revision: Optional[str] = None,
        steering_policy: Optional[SteeringPolicyConfig] = None,
        warmup_dataset_metadata: Optional[Mapping[str, Any]] = None,
        calibration_dataset_metadata: Optional[Mapping[str, Any]] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
    ) -> CalibrationArtifact:
        """Create an artifact for an adaptive conformal diagnostic score."""
        adjusted_scores = self.transform.transform(calibration_scores, feature_values)
        threshold = conformal_threshold(adjusted_scores, self.alpha)
        resolved_score_name = output_score_name or f"{score_name}_adaptive"
        metadata = dict(calibration_dataset_metadata or {})
        metadata["adaptive_conformal"] = {
            "base_score_name": str(score_name),
            "output_score_name": resolved_score_name,
            "transform": self.transform.to_dict(),
            "n_calibration": int(adjusted_scores.numel()),
        }

        return CalibrationArtifact(
            model_id=model_id,
            model_revision=model_revision,
            target_layer=target_layer,
            scores=(
                CalibrationScore(
                    name=resolved_score_name,
                    threshold=threshold,
                    conformal_alpha=self.alpha,
                    direction="higher",
                ),
            ),
            eigentruth_version=eigentruth_version,
            steering_policy=steering_policy or SteeringPolicyConfig(),
            warmup_dataset_metadata=warmup_dataset_metadata or {},
            calibration_dataset_metadata=metadata,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            commit_sha=commit_sha,
        )
