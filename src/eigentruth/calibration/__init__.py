"""Calibration artifacts and parameter-sweep interfaces.

This module defines lightweight data structures for reproducible EigenTruth
calibration. Implementations should stay CPU-testable and avoid model-loading or
network dependencies.
"""

from __future__ import annotations

from eigentruth.calibration.artifacts import CalibrationArtifact, CalibrationScore, SteeringPolicyConfig
from eigentruth.calibration.calibrator import AdaptiveConformalCalibrator, ConformalCalibrator
from eigentruth.calibration.fusion import (
    GeometryScoreFusionArtifact,
    GeometryScoreFusionCalibrator,
    RankScoreFusionArtifact,
    RankScoreFusionCalibrator,
    ScoreFusionSignal,
)
from eigentruth.calibration.sweeps import (
    DEFAULT_SCORE_DIRECTIONS,
    LayerScoreSweepCalibrator,
    LayerScoreSweepReport,
    LayerScoreSweepResult,
    SweepScoreResult,
)

__all__ = [
    "AdaptiveConformalCalibrator",
    "CalibrationArtifact",
    "CalibrationScore",
    "ConformalCalibrator",
    "DEFAULT_SCORE_DIRECTIONS",
    "GeometryScoreFusionArtifact",
    "GeometryScoreFusionCalibrator",
    "LayerScoreSweepCalibrator",
    "LayerScoreSweepReport",
    "LayerScoreSweepResult",
    "RankScoreFusionArtifact",
    "RankScoreFusionCalibrator",
    "ScoreFusionSignal",
    "SteeringPolicyConfig",
    "SweepScoreResult",
]
