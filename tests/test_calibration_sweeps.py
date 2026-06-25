"""Layer/score sweep calibration tests."""

import json

import pytest
import torch

import eigentruth.calibration.sweeps as sweep_module
from eigentruth.calibration import LayerScoreSweepCalibrator, LayerScoreSweepReport
from eigentruth.eval.conformal import directional_conformal_threshold, directional_trigger_rate
from eigentruth.eval.score_dump import ScoreDump, load_score_dump_layer_scores, write_score_dump_jsonl


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


def test_layer_score_sweep_rejects_fractional_labels():
    dump = _score_dump()
    dump["labels"] = [0.0, 0.9, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="label"):
        LayerScoreSweepCalibrator(alpha=0.4).calibrate_from_dump(dump)


def test_layer_score_sweep_parallel_matches_serial():
    serial = LayerScoreSweepCalibrator(alpha=0.4, max_workers=1).calibrate_from_dump(
        _score_dump(),
        signals=("maha_last", "subspace_resid"),
        created_at="2026-06-16T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )
    parallel = LayerScoreSweepCalibrator(alpha=0.4, max_workers=3).calibrate_from_dump(
        _score_dump(),
        signals=("maha_last", "subspace_resid"),
        created_at="2026-06-16T00:00:00+00:00",
        eigentruth_version="0.1.0",
    )

    assert parallel.score_results() == serial.score_results()
    assert [layer.layer for layer in parallel.layers] == [layer.layer for layer in serial.layers]
    assert parallel.best_score() == serial.best_score()
    assert parallel.metadata["sweep_max_workers"] == 3


def test_layer_score_sweep_prepares_each_score_once(monkeypatch):
    prepared = []
    original_prepare = sweep_module._prepare_sweep_score_tensor

    def count_prepare(scores, *, score_name):
        prepared.append(str(score_name))
        return original_prepare(scores, score_name=score_name)

    monkeypatch.setattr(sweep_module, "_prepare_sweep_score_tensor", count_prepare)

    report = LayerScoreSweepCalibrator(alpha=0.4).calibrate_from_dump(
        _score_dump(),
        signals=("maha_last", "subspace_resid"),
    )

    assert report.best_score().score_name == "maha_last"
    assert sorted(prepared) == ["maha_last", "maha_last", "subspace_resid", "subspace_resid"]


def test_layer_score_sweep_rejects_invalid_worker_count():
    try:
        LayerScoreSweepCalibrator(max_workers=0)
    except ValueError as exc:
        assert "max_workers" in str(exc)
    else:
        raise AssertionError("max_workers=0 should be rejected")


def test_layer_score_sweep_from_file_uses_validated_score_dump(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(_score_dump()), encoding="utf-8")

    original_to_mapping = ScoreDump.to_mapping

    def fail_to_mapping(self):
        raise AssertionError("calibrate_from_file should consume ScoreDump directly")

    ScoreDump.to_mapping = fail_to_mapping
    try:
        report = LayerScoreSweepCalibrator(alpha=0.4).calibrate_from_file(path)
    finally:
        ScoreDump.to_mapping = original_to_mapping

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


def test_layer_score_sweep_from_jsonl_file_reads_selected_signals(tmp_path, monkeypatch):
    dump = ScoreDump.from_mapping(_score_dump())
    manifest_path = tmp_path / "scores.manifest.json"
    write_score_dump_jsonl(dump, manifest_path)

    def fail_from_mapping(*args, **kwargs):
        raise AssertionError("JSONL sweep calibration should use layer-score loading")

    monkeypatch.setattr(ScoreDump, "from_mapping", fail_from_mapping)
    report = LayerScoreSweepCalibrator(alpha=0.4).calibrate_from_file(
        manifest_path,
        signals=("subspace_resid",),
    )

    assert report.scores_path == str(manifest_path)
    assert report.best_score().score_name == "subspace_resid"
    assert all(
        score.score_name == "subspace_resid"
        for layer in report.layers
        for score in layer.scores
    )


def test_layer_score_sweep_from_preloaded_layer_scores(tmp_path, monkeypatch):
    dump = ScoreDump.from_mapping(_score_dump())
    manifest_path = tmp_path / "scores.manifest.json"
    write_score_dump_jsonl(dump, manifest_path)
    layer_scores = load_score_dump_layer_scores(manifest_path, signals=("maha_last",))

    def fail_loader(*args, **kwargs):
        raise AssertionError("preloaded layer-score calibration should not read files")

    monkeypatch.setattr(sweep_module, "load_score_dump_layer_scores", fail_loader)
    report = LayerScoreSweepCalibrator(alpha=0.4).calibrate_from_layer_scores(
        layer_scores,
        signals=("maha_last",),
        scores_path=str(manifest_path),
    )

    assert report.scores_path == str(manifest_path)
    assert report.best_score().score_name == "maha_last"
    assert {score.score_name for score in report.score_results()} == {"maha_last"}


def test_layer_score_sweep_from_score_dump_public_api():
    score_dump = ScoreDump.from_mapping(_score_dump())

    report = LayerScoreSweepCalibrator(alpha=0.4).calibrate_from_score_dump(
        score_dump,
        scores_path="scores.json",
    )

    assert report.scores_path == "scores.json"
    assert report.best_score().layer == -2
    assert report.best_score().score_name == "maha_last"


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
    true_scores = torch.tensor([4.0, 3.0, 3.0, 2.0], dtype=torch.float64)
    false_scores = torch.tensor([1.0, 0.0, -1.0, -2.0], dtype=torch.float64)
    assert score.threshold == directional_conformal_threshold(true_scores, 0.4, "lower")
    assert score.false_alarm == directional_trigger_rate(true_scores, score.threshold, "lower")
    assert score.detection == directional_trigger_rate(false_scores, score.threshold, "lower")
