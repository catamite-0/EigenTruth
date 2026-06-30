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
    EvidenceAcquisitionCalibrationRecord,
    EvidenceAcquisitionCalibrationReport,
    EvidenceAcquisitionConformalCalibrator,
    EvidenceAcquisitionRiskMonitorReport,
    GeometryScoreFusionArtifact,
    GeometryScoreFusionCalibrator,
    MultipleTestingConformalArtifact,
    MultipleTestingConformalCalibrator,
    RankScoreFusionArtifact,
    RankScoreFusionCalibrator,
    SequentialConformalArtifact,
    SequentialConformalCalibrator,
    SteeringPolicyConfig,
    audit_evidence_acquisition_risk,
    evidence_acquisition_record_from_trace,
    evidence_acquisition_records_from_trace_feedback,
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


def test_evidence_acquisition_conformal_calibrator_reports_post_policy_gain(tmp_path):
    records = (
        EvidenceAcquisitionCalibrationRecord(pre_score=0.1, post_score=0.1, correct=True, action="answer"),
        EvidenceAcquisitionCalibrationRecord(pre_score=0.2, post_score=0.2, correct=True, action="answer"),
        EvidenceAcquisitionCalibrationRecord(pre_score=0.3, post_score=0.3, correct=True, action="answer"),
        EvidenceAcquisitionCalibrationRecord(pre_score=0.4, post_score=0.4, correct=True, action="answer"),
        EvidenceAcquisitionCalibrationRecord(pre_score=0.5, post_score=0.5, correct=True, action="answer"),
        EvidenceAcquisitionCalibrationRecord(pre_score=0.15, post_score=0.9, correct=False, action="acquire"),
        EvidenceAcquisitionCalibrationRecord(pre_score=0.25, post_score=1.1, correct=False, action="acquire"),
    )
    calibrator = EvidenceAcquisitionConformalCalibrator(alpha=0.4, score_name="policy_score")
    result = calibrator.calibrate(
        model_id="tiny-model",
        target_layer=-4,
        records=records,
        calibration_dataset_metadata={"fixture": "post-acquisition"},
        created_at="2026-06-30T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    report = result.report
    artifact = result.artifact
    assert report.naive_pre_threshold == pytest.approx(0.4)
    assert report.post_threshold == pytest.approx(0.4)
    assert report.n_acquired == 2
    assert report.acquisition_rate == pytest.approx(2 / 7)
    assert report.naive_pre_report is not None
    assert report.naive_pre_report.empirical_selective_accuracy == pytest.approx(4 / 6)
    assert report.post_acquisition_report.empirical_selective_accuracy == pytest.approx(1.0)
    assert report.selective_accuracy_delta == pytest.approx(1.0 - (4 / 6))
    assert report.metadata["post_acquisition_calibration"] is True
    assert report.metadata["calibration_scope"] == "post_acquisition_policy"

    score = artifact.get_score("policy_score")
    assert score.threshold == pytest.approx(report.post_threshold)
    assert score.conformal_alpha == pytest.approx(0.4)
    assert score.direction == "higher"
    assert artifact.calibration_dataset_metadata["fixture"] == "post-acquisition"
    assert artifact.calibration_dataset_metadata["post_acquisition_calibration"]["n_acquired"] == 2

    report_path = tmp_path / "evidence-acquisition-report.json"
    artifact_path = tmp_path / "evidence-acquisition-artifact.json"
    report.save_json(report_path)
    artifact.save_json(artifact_path)

    loaded_report = EvidenceAcquisitionCalibrationReport.from_dict(json.loads(report_path.read_text()))
    loaded_artifact = CalibrationArtifact.load_json(artifact_path)
    assert loaded_report.to_dict() == report.to_dict()
    assert loaded_artifact == artifact


def test_evidence_acquisition_conformal_calibrator_supports_lower_direction():
    records = (
        {"pre_score": 0.8, "post_score": 0.8, "correct": 1, "action": "answer"},
        {"pre_score": 0.7, "post_score": 0.7, "correct": 1, "action": "answer"},
        {"pre_score": 0.6, "post_score": 0.6, "correct": 1, "action": "answer"},
        {"pre_score": 0.5, "post_score": 0.5, "correct": 1, "action": "answer"},
        {"pre_score": 0.4, "post_score": 0.4, "correct": 1, "action": "answer"},
        {"pre_score": 0.65, "post_score": 0.1, "correct": 0, "action": "acquire"},
    )
    calibrator = EvidenceAcquisitionConformalCalibrator(
        alpha=0.4,
        score_name="support_policy_score",
        direction="lower",
    )
    result = calibrator.calibrate(
        model_id="tiny-model",
        target_layer=-4,
        records=records,
        created_at="2026-06-30T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    assert result.report.post_threshold == pytest.approx(0.5)
    assert result.report.post_acquisition_report.empirical_selective_accuracy == pytest.approx(1.0)
    assert result.artifact.get_score("support_policy_score").direction == "lower"
    assert result.artifact.get_score("support_policy_score").threshold == pytest.approx(0.5)


def test_evidence_acquisition_risk_monitor_passes_stable_feedback_stream():
    records = tuple(
        EvidenceAcquisitionCalibrationRecord(
            post_score=0.1 if index != 11 else 0.2,
            correct=index != 11,
            action="answer",
        )
        for index in range(20)
    )

    report = audit_evidence_acquisition_risk(
        records,
        threshold=0.5,
        target_error_rate=0.25,
        monitor_alpha=0.1,
        score_name="policy_score",
        checkpoints=(20,),
        metadata={"suite": "unit"},
    )
    roundtrip = EvidenceAcquisitionRiskMonitorReport.from_dict(report.to_dict())

    assert report.passed is True
    assert report.first_failed_checkpoint is None
    assert report.blocking_reasons == ()
    assert report.checks[0].accepted_count == 20
    assert report.checks[0].accepted_errors == 1
    assert report.checks[0].accepted_error_upper_bound < 0.25
    assert report.metadata["suite"] == "unit"
    assert roundtrip.to_dict() == report.to_dict()


def test_evidence_acquisition_risk_monitor_blocks_drifted_feedback_stream():
    records = tuple(
        EvidenceAcquisitionCalibrationRecord(
            post_score=0.1,
            correct=index >= 4,
            action="acquire" if index < 4 else "answer",
        )
        for index in range(10)
    )

    report = audit_evidence_acquisition_risk(
        records,
        threshold=0.5,
        target_error_rate=0.25,
        monitor_alpha=0.1,
        checkpoints=(10,),
    )

    assert report.passed is False
    assert report.first_failed_checkpoint == 10
    assert "accepted_error_upper_bound" in report.blocking_reasons[0]
    assert report.checks[0].accepted_errors == 4
    assert report.checks[0].n_acquired == 4


def test_evidence_acquisition_risk_monitor_respects_lower_direction_and_validates_inputs():
    records = (
        {"post_score": 0.9, "correct": 1, "action": "answer"},
        {"post_score": 0.8, "correct": 1, "action": "answer"},
        {"post_score": 0.7, "correct": 1, "action": "answer"},
        {"post_score": 0.1, "correct": 0, "action": "abstain"},
    )

    report = audit_evidence_acquisition_risk(
        records,
        threshold=0.5,
        target_error_rate=0.5,
        monitor_alpha=0.2,
        direction="lower",
        checkpoints=(4,),
    )

    assert report.passed is True
    assert report.checks[0].accepted_count == 3
    assert report.checks[0].accepted_errors == 0
    assert report.checks[0].n_abstained == 1
    with pytest.raises(ValueError, match="checkpoints"):
        audit_evidence_acquisition_risk(
            records,
            threshold=0.5,
            target_error_rate=0.5,
            checkpoints=(5,),
        )
    with pytest.raises(ValueError, match="direction"):
        audit_evidence_acquisition_risk(
            records,
            threshold=0.5,
            target_error_rate=0.5,
            direction="sideways",
        )
    with pytest.raises(ValueError, match="schedule"):
        EvidenceAcquisitionRiskMonitorReport.from_dict(
            {
                **report.to_dict(),
                "schedule": "surprise",
            }
        )


def test_evidence_acquisition_risk_monitor_dict_is_strict_json_ready_for_infinite_threshold():
    records = (
        {"post_score": 0.1, "correct": 1, "action": "answer"},
        {"post_score": 0.2, "correct": 1, "action": "answer"},
    )

    report = audit_evidence_acquisition_risk(
        records,
        threshold=math.inf,
        target_error_rate=0.5,
        monitor_alpha=0.2,
        checkpoints=(2,),
    )
    payload = report.to_dict()
    raw = json.dumps(payload, allow_nan=False)
    restored = EvidenceAcquisitionRiskMonitorReport.from_dict(json.loads(raw))

    assert payload["threshold"] == "inf"
    assert payload["checks"][0]["threshold"] == "inf"
    assert restored.threshold == math.inf


def test_evidence_acquisition_conformal_calibrator_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="alpha"):
        EvidenceAcquisitionConformalCalibrator(alpha=1.0)
    with pytest.raises(ValueError, match="direction"):
        EvidenceAcquisitionConformalCalibrator(direction="sideways")
    with pytest.raises(ValueError, match="non-empty"):
        EvidenceAcquisitionConformalCalibrator(score_name="")

    calibrator = EvidenceAcquisitionConformalCalibrator(alpha=0.4)
    with pytest.raises(ValueError, match="non-empty"):
        calibrator.report(())
    with pytest.raises(ValueError, match="correct"):
        calibrator.report(({"post_score": 0.1, "correct": 0},))
    with pytest.raises(ValueError, match="action"):
        EvidenceAcquisitionCalibrationRecord(post_score=0.1, correct=True, action="retrieve")
    with pytest.raises(ValueError, match="finite"):
        EvidenceAcquisitionCalibrationRecord(post_score=math.inf, correct=True)


def test_evidence_acquisition_record_from_trace_uses_post_policy_score():
    trace = {
        "request_id": "req-trace",
        "diagnostics": {"policy_score": 0.2},
        "risk_decision": {
            "action": "accept",
            "risk_level": "low",
            "diagnostics": {"policy_score": 0.85},
        },
        "actions": [{"action": "retrieve"}],
        "metadata": {
            "correct": True,
            "evidence_acquisition": {
                "pre_score": 0.2,
                "decision": {
                    "action": "acquire",
                    "metadata": {"post_acquisition_calibration_required": True},
                },
            },
        },
        "events": [
            {
                "event_type": "initial_risk_decision",
                "payload": {"diagnostics": {"policy_score": 0.2}},
            },
            {
                "event_type": "final_risk_decision",
                "payload": {"diagnostics": {"policy_score": 0.85}},
            },
        ],
    }

    record = evidence_acquisition_record_from_trace(trace, score_name="policy_score")

    assert record.record_id == "req-trace"
    assert record.pre_score == pytest.approx(0.2)
    assert record.post_score == pytest.approx(0.85)
    assert record.correct is True
    assert record.action == "acquire"
    assert record.acquired is True
    assert record.metadata["post_score_source"] == "risk_decision.diagnostics.policy_score"
    assert record.metadata["pre_score_source"] == "metadata.evidence_acquisition.pre_score"
    assert record.metadata["label_source"] == "metadata.correct"


def test_evidence_acquisition_records_from_trace_feedback_join_and_calibrate():
    traces = (
        {
            "request_id": "req-good",
            "diagnostics": {"policy_score": 0.2},
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "diagnostics": {"policy_score": 0.15},
            },
            "metadata": {"evidence_acquisition": {"decision": {"action": "answer"}}},
        },
        {
            "request_id": "req-acquired-bad",
            "diagnostics": {"policy_score": 0.1},
            "risk_decision": {
                "action": "accept",
                "risk_level": "low",
                "diagnostics": {"policy_score": 0.9},
            },
            "metadata": {"evidence_acquisition": {"decision": {"action": "acquire"}}},
        },
        {
            "request_id": "req-blocked-good",
            "diagnostics": {"policy_score": 0.3},
            "risk_decision": {
                "action": "abstain",
                "risk_level": "high",
                "diagnostics": {"policy_score": 0.8},
            },
            "metadata": {"evidence_acquisition": {"decision": {"action": "abstain"}}},
        },
    )
    feedback = (
        {"request_id": "req-good", "outcome": "correct", "feedback_source": "eval"},
        {"request_id": "req-acquired-bad", "outcome": "incorrect", "feedback_source": "eval"},
        {"request_id": "req-blocked-good", "outcome": "unnecessary_block", "feedback_source": "eval"},
    )

    records = evidence_acquisition_records_from_trace_feedback(
        traces,
        feedback,
        score_name="policy_score",
    )
    result = EvidenceAcquisitionConformalCalibrator(alpha=0.5, score_name="policy_score").calibrate(
        model_id="trace-model",
        target_layer=-1,
        records=records,
        created_at="2026-06-30T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    assert [record.correct for record in records] == [True, False, True]
    assert [record.action for record in records] == ["answer", "acquire", "abstain"]
    assert records[1].metadata["feedback_outcome"] == "incorrect"
    assert result.report.n_acquired == 1
    assert result.report.n_abstained == 1
    assert result.report.naive_pre_report is not None
    assert result.artifact.get_score("policy_score").threshold == pytest.approx(result.report.post_threshold)


def test_evidence_acquisition_trace_feedback_join_fails_closed_on_ambiguous_request():
    traces = (
        {
            "request_id": "duplicate",
            "risk_decision": {"diagnostics": {"policy_score": 0.1}},
            "metadata": {"correct": True},
        },
        {
            "request_id": "duplicate",
            "risk_decision": {"diagnostics": {"policy_score": 0.2}},
            "metadata": {"correct": True},
        },
    )
    feedback = ({"request_id": "duplicate", "outcome": "correct"},)

    with pytest.raises(ValueError, match="ambiguous_request_id"):
        evidence_acquisition_records_from_trace_feedback(
            traces,
            feedback,
            score_name="policy_score",
        )


def test_multiple_testing_conformal_artifact_roundtrip_and_runtime_decision(tmp_path):
    calibrator = MultipleTestingConformalCalibrator(alpha=0.3, method="by")
    artifact = calibrator.calibrate(
        model_id="tiny-model",
        target_layer=-4,
        calibration_scores={
            "support_score": list(range(100, 120)),
            "maha_last": list(range(20)),
        },
        directions={"support_score": "lower", "maha_last": "higher"},
        calibration_dataset_metadata={"name": "truthfulqa-mini"},
        created_at="2026-06-29T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    report = artifact.decide(
        {"support_score": 0.0, "maha_last": 5.0},
        metadata={"request_id": "req-1", "model_id": "spoofed"},
    )
    path = tmp_path / "multiple-testing-calibration.json"
    artifact.save_json(path)
    loaded = MultipleTestingConformalArtifact.load_json(path)
    loaded_report = loaded.decide(
        {"support_score": 0.0, "maha_last": 5.0},
        metadata={"request_id": "req-1", "model_id": "spoofed"},
    )

    assert loaded == artifact
    assert artifact.signal_names() == ("support_score", "maha_last")
    assert report.rejected is True
    assert report.rejected_signal_names == ("support_score",)
    assert report.metadata["request_id"] == "req-1"
    assert report.metadata["model_id"] == "tiny-model"
    assert report.metadata["runtime_metadata"]["model_id"] == "spoofed"
    assert loaded_report.to_dict() == report.to_dict()


def test_multiple_testing_conformal_calibrator_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="alpha"):
        MultipleTestingConformalCalibrator(alpha=1.0)
    with pytest.raises(ValueError, match="method"):
        MultipleTestingConformalCalibrator(method="sidak")

    calibrator = MultipleTestingConformalCalibrator(alpha=0.3)
    with pytest.raises(ValueError, match="at least one"):
        calibrator.calibrate(model_id="m", target_layer=0, calibration_scores={})
    with pytest.raises(ValueError, match="unknown"):
        calibrator.calibrate(
            model_id="m",
            target_layer=0,
            calibration_scores={"maha": [1.0, 2.0]},
            directions={"other": "higher"},
        )
    with pytest.raises(ValueError, match="direction"):
        calibrator.calibrate(
            model_id="m",
            target_layer=0,
            calibration_scores={"maha": [1.0, 2.0]},
            directions={"maha": "sideways"},
        )
    with pytest.raises(ValueError, match="bool"):
        calibrator.calibrate(model_id="m", target_layer=0, calibration_scores={"maha": [True, False]})
    with pytest.raises(ValueError, match="finite"):
        calibrator.calibrate(model_id="m", target_layer=0, calibration_scores={"maha": [1.0, math.inf]})


def test_sequential_conformal_artifact_roundtrip_and_runtime_sequence(tmp_path):
    calibrator = SequentialConformalCalibrator(alpha=0.5, schedule="linear")
    artifact = calibrator.calibrate(
        model_id="tiny-model",
        target_layer=-4,
        signal_name="support_score",
        calibration_scores=[10.0, 11.0, 12.0, 13.0],
        direction="lower",
        calibration_dataset_metadata={"name": "truthfulqa-mini"},
        created_at="2026-06-29T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    report = artifact.decide_sequence(
        [9.0, 12.0],
        metadata={"session_id": "s1", "signal_name": "spoofed"},
    )
    path = tmp_path / "sequential-calibration.json"
    artifact.save_json(path)
    loaded = SequentialConformalArtifact.load_json(path)
    loaded_report = loaded.decide_sequence(
        [9.0, 12.0],
        metadata={"session_id": "s1", "signal_name": "spoofed"},
    )

    assert loaded == artifact
    assert artifact.calibration_count == 4
    assert artifact.direction == "lower"
    assert artifact.schedule == "linear"
    assert report.rejected_steps == (1,)
    assert report.steps[0].score == pytest.approx(9.0)
    assert report.metadata["session_id"] == "s1"
    assert report.metadata["signal_name"] == "support_score"
    assert report.metadata["runtime_metadata"]["signal_name"] == "spoofed"
    assert loaded_report.to_dict() == report.to_dict()


def test_sequential_conformal_calibrator_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="alpha"):
        SequentialConformalCalibrator(alpha=1.0)
    with pytest.raises(ValueError, match="schedule"):
        SequentialConformalCalibrator(schedule="sidak")

    calibrator = SequentialConformalCalibrator(alpha=0.3)
    with pytest.raises(ValueError, match="direction"):
        calibrator.calibrate(
            model_id="m",
            target_layer=0,
            signal_name="score",
            calibration_scores=[1.0, 2.0],
            direction="sideways",
        )
    with pytest.raises(ValueError, match="bool"):
        calibrator.calibrate(
            model_id="m",
            target_layer=0,
            signal_name="score",
            calibration_scores=[True, False],
        )
    with pytest.raises(ValueError, match="finite"):
        calibrator.calibrate(
            model_id="m",
            target_layer=0,
            signal_name="score",
            calibration_scores=[1.0, math.inf],
        )


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
