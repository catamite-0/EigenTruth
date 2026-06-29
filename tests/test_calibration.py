"""Calibration artifact and conformal calibrator tests."""

import json
import math
from unittest.mock import patch

import pytest

from eigentruth.calibration import (
    AdaptiveConformalCalibrator,
    CalibrationArtifact,
    CalibrationScore,
    ConformalCalibrator,
    GeometryScoreFusionArtifact,
    GeometryScoreFusionCalibrator,
    RankScoreFusionArtifact,
    RankScoreFusionCalibrator,
    SteeringPolicyConfig,
)
from eigentruth.eval import (
    AdaptiveScoreTransform,
    combine_geometry_uncertainty_scores,
    geometry_calibrated_anomaly_scores,
    global_local_uncertainty_scores,
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


def test_calibration_artifacts_write_strict_json_for_infinite_threshold(tmp_path):
    artifact = CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(CalibrationScore("maha", threshold=math.inf, conformal_alpha=0.05),),
        eigentruth_version="0.1.0",
    )

    path = tmp_path / "calibration.json"
    artifact.save_json(path)
    raw = path.read_text(encoding="utf-8")

    assert "Infinity" not in raw
    assert json.loads(raw)["scores"][0]["threshold"] == "inf"
    assert CalibrationArtifact.load_json(path).get_score("maha").threshold == math.inf


def test_calibration_score_from_dict_rejects_bool_threshold():
    with pytest.raises(ValueError, match="threshold"):
        CalibrationScore.from_dict({"name": "maha", "threshold": True})


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


def test_adaptive_conformal_calibrator_builds_feature_adjusted_artifact():
    transform = AdaptiveScoreTransform(
        feature_weights={"semantic_entropy": 2.0},
        intercept=0.5,
        direction="higher",
    )
    calibrator = AdaptiveConformalCalibrator(alpha=0.4, transform=transform)

    artifact = calibrator.calibrate(
        model_id="tiny-model",
        target_layer=-4,
        score_name="maha",
        calibration_scores=[1.0, 2.0, 3.0, 4.0],
        feature_values={"semantic_entropy": [0.0, 0.0, 1.0, 1.0]},
        created_at="2026-06-25T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    score = artifact.get_score("maha_adaptive")
    adaptive_metadata = artifact.calibration_dataset_metadata["adaptive_conformal"]

    assert score.threshold == pytest.approx(5.5)
    assert score.direction == "higher"
    assert score.conformal_alpha == pytest.approx(0.4)
    assert adaptive_metadata["base_score_name"] == "maha"
    assert adaptive_metadata["transform"] == transform.to_dict()
    assert adaptive_metadata["n_calibration"] == 4


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

    with pytest.raises(ValueError, match="alpha"):
        AdaptiveConformalCalibrator(alpha=0.0)


def test_rank_score_fusion_artifact_roundtrip_and_directional_flags(tmp_path):
    labels = [0, 0, 0, 0, 1]
    scores = {
        "truth_proj": [0.0, 1.0, 2.0, 3.0, 10.0],
        "support_score": [10.0, 9.0, 8.0, 7.0, 0.0],
    }
    calibrator = RankScoreFusionCalibrator(alpha=0.4, method="max_rank")
    artifact = calibrator.calibrate(
        labels=labels,
        scores=scores,
        directions={"truth_proj": "higher", "support_score": "lower"},
        model_id="synthetic",
        target_layer=-1,
        score_dump_metadata={"sha256": "abc"},
    )

    path = tmp_path / "fusion.json"
    artifact.save_json(path)
    loaded = RankScoreFusionArtifact.load_json(path)
    expected_fused = artifact.score(scores)
    with patch("eigentruth.eval.score_fusion.torch.sort", side_effect=AssertionError("unexpected runtime sort")):
        fused = loaded.score(scores)
        flags = loaded.flags(scores)
    evaluation = calibrator.evaluate(
        labels=labels,
        scores=scores,
        directions={"truth_proj": "higher", "support_score": "lower"},
    )

    assert loaded == artifact
    assert loaded.signal_names() == ("truth_proj", "support_score")
    assert fused.tolist() == pytest.approx(expected_fused.tolist())
    assert loaded.threshold == pytest.approx(0.75)
    assert flags.tolist() == [False, False, False, True, True]
    assert evaluation["false_alarm"] == pytest.approx(0.25)
    assert evaluation["detection"] == pytest.approx(1.0)
    assert evaluation["auroc"] > 0.5


def test_rank_score_fusion_rejects_fractional_and_bool_labels():
    calibrator = RankScoreFusionCalibrator(alpha=0.4, method="max_rank")

    with pytest.raises(ValueError, match="labels"):
        calibrator.calibrate(labels=[0.0, 0.9, 1.0], scores={"maha": [0.0, 0.1, 1.0]})
    with pytest.raises(ValueError, match="bool"):
        calibrator.calibrate(labels=[False, 0, 1], scores={"maha": [0.0, 0.1, 1.0]})


def test_geometry_uncertainty_fusion_scores_joint_anomaly_higher():
    geometry = [0.9, 0.9, 0.2]
    uncertainty = [0.1, 0.9, 0.9]

    interaction = combine_geometry_uncertainty_scores(
        geometry,
        uncertainty,
        method="interaction",
        interaction_weight=2.0,
    )
    product = combine_geometry_uncertainty_scores(geometry, uncertainty, method="product")

    assert interaction[1] > interaction[0]
    assert interaction[1] > interaction[2]
    assert product.tolist() == pytest.approx([0.09, 0.81, 0.18])


def test_geometry_calibrated_anomaly_scores_uses_directional_rank_groups():
    calibration_scores = {
        "subspace_resid": [0.0, 1.0, 2.0, 3.0],
        "support_confidence": [10.0, 9.0, 8.0, 7.0],
    }
    scores = {
        "subspace_resid": [0.0, 3.5],
        "support_confidence": [10.0, 0.0],
    }

    fused = geometry_calibrated_anomaly_scores(
        calibration_scores=calibration_scores,
        scores=scores,
        geometry_signals=("subspace_resid",),
        uncertainty_signals=("support_confidence",),
        directions={"subspace_resid": "higher", "support_confidence": "lower"},
    )

    assert fused[1] > fused[0]
    assert fused.tolist() == pytest.approx([0.1875, 1.0])


def test_global_local_uncertainty_scores_gate_geometry_and_token_uncertainty():
    calibration_scores = {
        "hidden_geometry_entropy": [0.0, 1.0, 2.0, 3.0],
        "first_token_confidence": [10.0, 9.0, 8.0, 7.0],
    }
    scores = {
        "hidden_geometry_entropy": [0.0, 3.5],
        "first_token_confidence": [10.0, 0.0],
    }

    fused = global_local_uncertainty_scores(
        calibration_scores=calibration_scores,
        scores=scores,
        global_signals=("hidden_geometry_entropy",),
        local_signals=("first_token_confidence",),
        directions={
            "hidden_geometry_entropy": "higher",
            "first_token_confidence": "lower",
        },
    )

    assert fused.tolist() == pytest.approx([0.0625, 1.0])

    with pytest.raises(ValueError, match="must not overlap"):
        global_local_uncertainty_scores(
            calibration_scores=calibration_scores,
            scores=scores,
            global_signals=("hidden_geometry_entropy",),
            local_signals=("hidden_geometry_entropy",),
        )


def test_geometry_score_fusion_artifact_roundtrip_and_flags(tmp_path):
    labels = [0, 0, 0, 0, 1, 1]
    scores = {
        "subspace_resid": [0.0, 1.0, 2.0, 3.0, 6.0, 4.0],
        "support_confidence": [10.0, 9.0, 8.0, 7.0, 2.0, 0.0],
    }
    calibrator = GeometryScoreFusionCalibrator(alpha=0.4, interaction_weight=2.0)
    artifact = calibrator.calibrate(
        labels=labels,
        scores=scores,
        geometry_signals=("subspace_resid",),
        uncertainty_signals=("support_confidence",),
        directions={"subspace_resid": "higher", "support_confidence": "lower"},
        model_id="synthetic",
        target_layer=-2,
        score_dump_metadata={"sha256": "geometry"},
    )

    path = tmp_path / "geometry-fusion.json"
    artifact.save_json(path)
    loaded = GeometryScoreFusionArtifact.load_json(path)
    fused = loaded.score(scores)
    flags = loaded.flags(scores)
    evaluation = calibrator.evaluate(
        labels=labels,
        scores=scores,
        geometry_signals=("subspace_resid",),
        uncertainty_signals=("support_confidence",),
        directions={"subspace_resid": "higher", "support_confidence": "lower"},
    )

    assert loaded == artifact
    assert loaded.signal_names() == ("subspace_resid", "support_confidence")
    assert loaded.threshold == pytest.approx(0.65625)
    assert flags.tolist() == [False, False, False, True, True, True]
    assert fused[-1] > fused[0]
    assert evaluation["false_alarm"] == pytest.approx(0.25)
    assert evaluation["detection"] == pytest.approx(1.0)
    assert evaluation["auroc"] > 0.5


def test_geometry_score_fusion_rejects_overlapping_groups_and_bad_weights():
    calibrator = GeometryScoreFusionCalibrator(alpha=0.4)

    with pytest.raises(ValueError, match="overlap"):
        calibrator.calibrate(
            labels=[0, 1],
            scores={"score": [0.0, 1.0]},
            geometry_signals=("score",),
            uncertainty_signals=("score",),
        )
    with pytest.raises(ValueError, match="geometry_weight"):
        GeometryScoreFusionCalibrator(geometry_weight=-1.0)
