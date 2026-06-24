"""Layer/score sweep calibration tests."""

import json

from eigentruth.calibration import LayerScoreSweepCalibrator, LayerScoreSweepReport


def _score_dump():
    return {
        "config": {"model": "tiny", "layer": -1, "offline": True},
        "labels": [0, 0, 0, 0, 1, 1, 1, 1],
        "scores": {
            "maha_last": [1.0, 2.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0],
            "truth_proj": [1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0],
            "subspace_resid": [0.1, 0.2, 0.1, 0.3, 2.0, 2.2, 2.4, 2.6],
        },
        "sweep_scores": {
            "-2": {
                "maha_last": [1.0, 1.5, 2.0, 2.5, 7.0, 8.0, 9.0, 10.0],
                "truth_proj": [0.5, 0.5, 1.0, 1.0, 2.0, 2.5, 3.0, 3.5],
                "subspace_resid": [0.0, 0.1, 0.0, 0.1, 5.0, 5.5, 6.0, 6.5],
            },
            "-1": {
                "maha_last": [1.0, 2.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0],
                "truth_proj": [1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0],
                "subspace_resid": [0.1, 0.2, 0.1, 0.3, 2.0, 2.2, 2.4, 2.6],
            },
        },
    }


def test_layer_score_sweep_report_selects_best_artifact(tmp_path):
    report = LayerScoreSweepCalibrator(alpha=0.4).calibrate_from_dump(
        _score_dump(),
        signals=("maha_last", "subspace_resid"),
        created_at="2026-06-16T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    best = report.best_score()
    artifact = report.best_artifact()

    assert best.layer == -2
    assert best.score_name == "maha_last"
    assert best.auroc == 1.0
    assert artifact.model_id == "tiny"
    assert artifact.target_layer == -2
    assert artifact.score_names() == ("maha_last",)
    assert artifact.get_score("maha_last").threshold == 2.0

    path = tmp_path / "sweep.json"
    report.save_json(path)
    loaded = LayerScoreSweepReport.load_json(path)

    assert loaded.best_score() == best
    assert json.loads(path.read_text())["best"]["score_name"] == "maha_last"


def test_layer_score_sweep_from_file_uses_validated_score_dump(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(_score_dump()), encoding="utf-8")

    report = LayerScoreSweepCalibrator(alpha=0.4).calibrate_from_file(path)

    assert report.scores_path == str(path)
    assert report.best_score().score_name == "maha_last"

    bad_path = tmp_path / "bad-scores.json"
    bad_path.write_text(
        json.dumps({
            "labels": [0, 1],
            "scores": {"maha_last": [0.1]},
        }),
        encoding="utf-8",
    )
    try:
        LayerScoreSweepCalibrator().calibrate_from_file(bad_path)
    except ValueError as exc:
        assert "length does not match labels" in str(exc)
    else:
        raise AssertionError("invalid score dump should be rejected")


def test_layer_score_sweep_supports_lower_is_anomalous_scores():
    dump = {
        "config": {"model": "tiny", "layer": 0},
        "labels": [0, 0, 0, 0, 1, 1, 1, 1],
        "scores": {"support": [4.0, 3.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]},
    }

    report = LayerScoreSweepCalibrator(alpha=0.4).calibrate_from_dump(
        dump,
        directions={"support": "lower"},
    )
    score = report.best_score()

    assert score.direction == "lower"
    assert score.auroc == 1.0
    assert score.threshold == 3.0
    assert score.false_alarm == 0.25
