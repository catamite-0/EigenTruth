import pytest

from eigentruth.calibration import (
    TrajectoryFusionDataset,
    calibrate_trajectory_fusion_from_report,
    trajectory_fusion_dataset_from_report,
)


def _trajectory_payload(score: float, *, layer: int, resolved_layer: int) -> dict:
    return {
        "convergence_score": score,
        "step_count": 4,
        "hidden_dim": 3,
        "path_length": 1.0,
        "direct_distance": 1.0,
        "path_efficiency": 1.0,
        "mean_step_distance": 0.4,
        "initial_step_distance": 0.6,
        "final_step_distance": 0.2,
        "convergence_ratio": 0.3,
        "step_distance_drop": 0.4,
        "log_decay_slope": -0.2,
        "koopman_rate": 0.8,
        "convergence_strength": 0.2,
        "decay_fraction": 1.0,
        "displacement_cv": 0.1,
        "metadata": {"layer": layer, "resolved_layer": resolved_layer},
    }


def _layer_sweep_report() -> dict:
    labels = [0, 0, 0, 1, 1]
    best_scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    weak_scores = [0.6, 0.5, 0.4, 0.3, 0.2]
    return {
        "workflow": "truthfulqa_forced_answer_trajectory_layer_sweep",
        "config": {"layers": [-1, -6]},
        "summary": {
            "status": "pass",
            "n_total": 5,
            "n_evaluated": 5,
            "n_skipped": 0,
            "best_layer": -6,
            "best_layer_key": "-6",
            "best_resolved_layer": 7,
            "trajectory_score_best_auroc": 1.0,
            "trajectory_score_direction_for_false": "higher",
        },
        "layer_summaries": [
            {
                "layer": -1,
                "layer_key": "-1",
                "resolved_layer": 12,
                "trajectory_score_best_auroc": 0.7,
                "trajectory_score_direction_for_false": "lower",
            },
            {
                "layer": -6,
                "layer_key": "-6",
                "resolved_layer": 7,
                "trajectory_score_best_auroc": 1.0,
                "trajectory_score_direction_for_false": "higher",
            },
        ],
        "records": [
            {
                "index": idx,
                "label": label,
                "nll_answer": 1.0 + idx,
                "trajectories": {
                    "-1": _trajectory_payload(weak_scores[idx], layer=-1, resolved_layer=12),
                    "-6": _trajectory_payload(best_scores[idx], layer=-6, resolved_layer=7),
                },
            }
            for idx, label in enumerate(labels)
        ],
        "metadata": {"model": "synthetic", "source_scores": {"path": "scores.json"}},
    }


def _single_layer_report() -> dict:
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]
    return {
        "workflow": "truthfulqa_forced_answer_trajectory",
        "config": {"layer": -1},
        "summary": {
            "status": "pass",
            "n_total": 4,
            "n_evaluated": 4,
            "n_skipped": 0,
            "trajectory_score_best_auroc": 1.0,
            "trajectory_score_direction_for_false": "lower",
        },
        "records": [
            {
                "index": idx,
                "label": label,
                "nll_answer": 2.0 + idx,
                "trajectory": _trajectory_payload(score, layer=-1, resolved_layer=12),
            }
            for idx, (label, score) in enumerate(zip(labels, scores, strict=True))
        ],
        "metadata": {"model": "single"},
    }


def test_trajectory_fusion_dataset_extracts_best_layer_and_calibrates():
    report = _layer_sweep_report()

    dataset = trajectory_fusion_dataset_from_report(report, include_nll_answer=True)
    artifact = dataset.calibrate(alpha=0.4, method="max_rank")
    fused = artifact.score(dataset.scores)

    assert dataset.labels == (0, 0, 0, 1, 1)
    assert dataset.scores["trajectory_convergence"] == pytest.approx((0.1, 0.2, 0.3, 0.8, 0.9))
    assert dataset.scores["nll_answer"] == pytest.approx((1.0, 2.0, 3.0, 4.0, 5.0))
    assert dataset.directions == {"trajectory_convergence": "higher", "nll_answer": "higher"}
    assert dataset.metadata["layer"] == -6
    assert dataset.metadata["resolved_layer"] == 7
    assert artifact.model_id == "synthetic"
    assert artifact.target_layer == 7
    assert fused[-1] > fused[0]
    assert fused[-2] > fused[1]


def test_trajectory_fusion_dataset_can_select_explicit_layer_direction():
    report = _layer_sweep_report()

    dataset = trajectory_fusion_dataset_from_report(report, layer=-1, signal_name="trajectory_layer_minus_1")

    assert dataset.scores["trajectory_layer_minus_1"] == pytest.approx((0.6, 0.5, 0.4, 0.3, 0.2))
    assert dataset.directions == {"trajectory_layer_minus_1": "lower"}
    assert dataset.metadata["layer_key"] == "-1"
    assert dataset.metadata["resolved_layer"] == 12


def test_trajectory_fusion_single_layer_preserves_lower_direction():
    report = _single_layer_report()

    dataset = trajectory_fusion_dataset_from_report(report)
    artifact = calibrate_trajectory_fusion_from_report(report, alpha=0.5)

    assert dataset.directions == {"trajectory_convergence": "lower"}
    assert dataset.scores["trajectory_convergence"] == pytest.approx((0.9, 0.8, 0.2, 0.1))
    assert artifact.signal_names() == ("trajectory_convergence",)
    assert artifact.signals[0].direction == "lower"
    assert artifact.target_layer == 12


def test_trajectory_fusion_dataset_roundtrip():
    dataset = trajectory_fusion_dataset_from_report(_layer_sweep_report())

    loaded = TrajectoryFusionDataset.from_dict(dataset.to_dict())

    assert loaded == dataset


def test_trajectory_fusion_rejects_missing_records_and_bad_layer():
    report = _layer_sweep_report()
    report["records"] = []

    with pytest.raises(ValueError, match="non-empty records"):
        trajectory_fusion_dataset_from_report(report)

    with pytest.raises(ValueError, match="layer summary"):
        trajectory_fusion_dataset_from_report(_layer_sweep_report(), layer=-99)
