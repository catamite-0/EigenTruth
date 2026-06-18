"""Calibration artifact and conformal calibrator tests."""

import json

import pytest

from eigentruth.calibration import (
    CalibrationArtifact,
    CalibrationScore,
    ConformalCalibrator,
    SteeringPolicyConfig,
)


def test_artifact_json_roundtrip(tmp_path):
    artifact = CalibrationArtifact(
        model_id="gpt2",
        model_revision="main",
        target_layer=-8,
        scores=(
            CalibrationScore("maha", threshold=4.0, conformal_alpha=0.2),
            CalibrationScore("loss_margin", threshold=2.0, conformal_alpha=0.2, direction="lower"),
        ),
        eigentruth_version="0.1.0",
        steering_policy=SteeringPolicyConfig(mode="adaptive", steering_lambda=0.1, max_steering_lambda=0.3),
        warmup_dataset_metadata={"name": "warmup"},
        calibration_dataset_metadata={"name": "calib", "n": 4},
        created_at="2026-06-16T00:00:00+00:00",
        commit_sha="abc123",
    )

    path = tmp_path / "calibration.json"
    artifact.save_json(path)
    loaded = CalibrationArtifact.load_json(path)

    assert loaded == artifact
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == 1
    assert raw["scores"][1]["direction"] == "lower"


def test_conformal_calibrator_builds_higher_is_anomalous_artifact():
    calibrator = ConformalCalibrator(alpha=0.4)
    artifact = calibrator.calibrate(
        model_id="tiny-model",
        target_layer=-4,
        calibration_scores={"maha": [1.0, 2.0, 3.0, 4.0]},
        model_revision="rev1",
        created_at="2026-06-16T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    score = artifact.get_score("maha")
    assert score.threshold == 3.0
    assert score.conformal_alpha == 0.4
    assert score.direction == "higher"
    assert artifact.model_revision == "rev1"


def test_conformal_calibrator_supports_lower_is_anomalous_scores():
    calibrator = ConformalCalibrator(alpha=0.4)
    artifact = calibrator.calibrate(
        model_id="tiny-model",
        target_layer=-4,
        calibration_scores={"support_score": [1.0, 2.0, 3.0, 4.0]},
        directions={"support_score": "lower"},
        created_at="2026-06-16T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    score = artifact.get_score("support_score")
    assert score.threshold == 2.0
    assert score.direction == "lower"


def test_conformal_calibrator_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="alpha"):
        ConformalCalibrator(alpha=1.0)

    calibrator = ConformalCalibrator(alpha=0.1)
    with pytest.raises(ValueError, match="at least one"):
        calibrator.calibrate(model_id="m", target_layer=0, calibration_scores={})
    with pytest.raises(ValueError, match="directions"):
        calibrator.calibrate(
            model_id="m",
            target_layer=0,
            calibration_scores={"score": [1.0, 2.0]},
            directions={"score": "sideways"},
        )
