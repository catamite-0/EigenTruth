"""Smoke tests for benchmark reporting helpers."""

import importlib
import json
from types import SimpleNamespace

import pytest


def test_eval_conformal_run_respects_lower_direction(tmp_path):
    module = importlib.import_module("benchmarks.eval_conformal")
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps({
            "config": {"model": "tiny", "layer": 0},
            "labels": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            "scores": {"support": [10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3]},
        }),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        scores=str(scores_path),
        signal="support",
        signals=None,
        repeats=1,
        seed=0,
        json=None,
        save_calibration=None,
        save_sweep_report=None,
        save_best_calibration=None,
        best_by="auroc",
        artifact_alpha=0.10,
        direction="lower",
        model_id=None,
        model_revision=None,
        target_layer=None,
        created_at=None,
        commit_sha=None,
    )

    payload = module.run(args)
    report = payload["results"]["0.2"]["selective_report"]

    assert payload["config"]["direction"] == "lower"
    assert payload["results"]["0.2"]["threshold"] == pytest.approx(10.0)
    assert report["direction"] == "lower"
    assert report["false_alarm"] == pytest.approx(0.0)
    assert report["detection"] == pytest.approx(1.0)


def test_eval_truthfulqa_selective_reports_accept_score_directions():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    reports = module._selective_reports(
        {"support": [10.0, 11.0, 0.0, 1.0]},
        [0, 0, 1, 1],
        alpha=0.5,
        directions={"support": "lower"},
    )

    assert reports["support"]["threshold"] == pytest.approx(10.0)
    assert reports["support"]["direction"] == "lower"
    assert reports["support"]["false_alarm"] == pytest.approx(0.0)
    assert reports["support"]["detection"] == pytest.approx(1.0)


def test_eval_truthfulqa_exposes_internal_eigenscore_signal():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    assert "eigenscore" in module.SIGNALS
    assert module.DEFAULT_SCORE_DIRECTIONS["eigenscore"] == "higher"


def test_eval_truthfulqa_multisample_inside_signal_is_optional():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    disabled = SimpleNamespace(inside_samples=0)
    enabled = SimpleNamespace(inside_samples=3)

    assert module.INSIDE_SIGNAL not in module._enabled_signals(disabled)
    assert module.INSIDE_SIGNAL in module._enabled_signals(enabled)
    assert module.INSIDE_SIGNAL in module._sweep_signal_names(enabled)
    assert module.DEFAULT_SCORE_DIRECTIONS[module.INSIDE_SIGNAL] == "higher"


def test_eval_truthfulqa_candidate_verification_prompt_includes_context():
    module = importlib.import_module("benchmarks.eval_truthfulqa")
    stmt = module.Statement("What is the capital of France?", "Paris", 0)

    prompt = module._candidate_verification_prompt(stmt)

    assert "Question: What is the capital of France?" in prompt
    assert "Candidate answer: Paris" in prompt
    assert "factually correct" in prompt


def test_eval_truthfulqa_chunked_preserves_order():
    module = importlib.import_module("benchmarks.eval_truthfulqa")

    assert list(module._chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError, match="batch size"):
        list(module._chunked([1], 0))
