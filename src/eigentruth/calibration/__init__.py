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
from eigentruth.calibration.multiple_testing import (
    MultipleTestingConformalArtifact,
    MultipleTestingConformalCalibrator,
    MultipleTestingConformalSignal,
)
from eigentruth.calibration.sequential import (
    SequentialConformalArtifact,
    SequentialConformalCalibrator,
)
from eigentruth.calibration.sweeps import (
    DEFAULT_SCORE_DIRECTIONS,
    LayerScoreSweepCalibrator,
    LayerScoreSweepReport,
    LayerScoreSweepResult,
    SweepScoreResult,
)
from eigentruth.calibration.trajectory_fusion import (
    DEFAULT_NLL_SIGNAL_NAME,
    DEFAULT_TRAJECTORY_SIGNAL_NAME,
    TrajectoryFusionDataset,
    calibrate_trajectory_fusion_from_report,
    trajectory_fusion_dataset_from_report,
)

__all__ = [
    "AdaptiveConformalCalibrator",
    "CalibrationArtifact",
    "CalibrationScore",
    "ConformalCalibrator",
    "DEFAULT_SCORE_DIRECTIONS",
    "DEFAULT_NLL_SIGNAL_NAME",
    "DEFAULT_TRAJECTORY_SIGNAL_NAME",
    "GeometryScoreFusionArtifact",
    "GeometryScoreFusionCalibrator",
    "LayerScoreSweepCalibrator",
    "LayerScoreSweepReport",
    "LayerScoreSweepResult",
    "MultipleTestingConformalArtifact",
    "MultipleTestingConformalCalibrator",
    "MultipleTestingConformalSignal",
    "RankScoreFusionArtifact",
    "RankScoreFusionCalibrator",
    "ScoreFusionSignal",
    "SequentialConformalArtifact",
    "SequentialConformalCalibrator",
    "SteeringPolicyConfig",
    "SweepScoreResult",
    "TrajectoryFusionDataset",
    "calibrate_trajectory_fusion_from_report",
    "trajectory_fusion_dataset_from_report",
]
