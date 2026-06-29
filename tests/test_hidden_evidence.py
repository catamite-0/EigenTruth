import json
import math
from types import SimpleNamespace

import pytest

from eigentruth.eval import (
    HiddenEvidenceCandidate,
    HiddenEvidenceSelectionPolicy,
    HiddenEvidenceSelectionReport,
    select_hidden_evidence,
    select_hidden_evidence_from_score_dump,
)
from eigentruth.registry import ArtifactRegistry


def _score_dump() -> dict:
    return {
        "config": {"model": "synthetic", "layer": -1},
        "labels": [0, 1, 1],
        "statements": [
            {"claim_id": "c1", "text": "Alpha is correct."},
            {"claim_id": "c2", "text": "Beta is unsupported."},
            {"claim_id": "c3", "text": "Gamma is refuted."},
        ],
        "scores": {
            "truth_proj": [0.1, 0.9, 0.2],
            "selfcheck_support_rate": [0.95, 0.2, 0.8],
        },
        "sweep_scores": {
            "-2": {
                "truth_proj": [0.2, 0.8, 0.3],
                "selfcheck_support_rate": [0.9, 0.1, 0.7],
            },
            "-1": {
                "truth_proj": [0.3, 0.7, 0.4],
                "selfcheck_support_rate": [0.85, 0.15, 0.75],
            },
        },
    }


def test_select_hidden_evidence_respects_direction_and_budgets():
    candidates = (
        HiddenEvidenceCandidate("r1", "support_rate", 0.1, direction="lower", layer=-1),
        HiddenEvidenceCandidate("r2", "support_rate", 0.6, direction="lower", layer=-1),
        HiddenEvidenceCandidate("r1", "truth_proj", 0.95, direction="higher", layer=-1),
        HiddenEvidenceCandidate("r3", "truth_proj", 0.7, direction="higher", layer=-1),
    )

    report = select_hidden_evidence(
        candidates,
        policy=HiddenEvidenceSelectionPolicy(max_items=3, max_per_record=1),
    )

    assert [item.record_id for item in report.selected] == ["r1", "r3", "r2"]
    lower_selection = next(item for item in report.selected if item.score_name == "support_rate")
    assert lower_selection.record_id == "r2"
    assert lower_selection.direction == "lower"
    assert lower_selection.anomaly_score == pytest.approx(0.5)
    assert report.dropped_counts["max_per_record"] == 1
    assert report.summary()["selected_record_count"] == 3


def test_select_hidden_evidence_from_score_dump_handles_sweep_and_lower_direction(tmp_path):
    report = select_hidden_evidence_from_score_dump(
        _score_dump(),
        score_names=("truth_proj", "selfcheck_support_rate"),
        sweep_score_names=("truth_proj", "selfcheck_support_rate"),
        directions={"selfcheck_support_rate": "lower"},
        policy=HiddenEvidenceSelectionPolicy(max_items=6, max_per_layer=2),
    )

    assert report.workflow == "hidden_evidence_selection"
    assert report.candidate_count == 18
    assert report.channel_count == 6
    assert len(report.selected) == 6
    assert report.summary()["selected_by_layer"] == {"-2": 2, "-1": 2, "primary": 2}
    assert any(
        item.record_id == "c2"
        and item.score_name == "selfcheck_support_rate"
        and item.direction == "lower"
        for item in report.selected
    )
    assert report.selected[0].metadata["text"] == "Beta is unsupported."

    path = tmp_path / "hidden-evidence.json"
    report.save_json(path)
    loaded = HiddenEvidenceSelectionReport.load_json(path)

    assert loaded == report
    json.dumps(report.to_dict())


def test_hidden_evidence_rejects_non_finite_candidate_scores():
    with pytest.raises(ValueError, match="score"):
        HiddenEvidenceCandidate("r1", "truth_proj", math.nan)
    with pytest.raises(ValueError, match="score"):
        select_hidden_evidence((
            {"record_id": "r1", "score_name": "truth_proj", "score": math.inf},
        ))


def test_select_hidden_evidence_cli_writes_report_and_registry(tmp_path):
    module = __import__("benchmarks.select_hidden_evidence", fromlist=["run"])
    scores_path = tmp_path / "scores.json"
    report_path = tmp_path / "hidden-evidence.json"
    registry_path = tmp_path / "registry.json"
    scores_path.write_text(json.dumps(_score_dump()), encoding="utf-8")

    payload = module.run(SimpleNamespace(
        scores=str(scores_path),
        json=str(report_path),
        signals="truth_proj,selfcheck_support_rate",
        sweep_signals="truth_proj,selfcheck_support_rate",
        no_primary=False,
        no_sweep=False,
        direction=("selfcheck_support_rate=lower",),
        max_items=5,
        max_per_record="2",
        max_per_layer="2",
        max_per_score=None,
        min_anomaly_score=None,
        registry=str(registry_path),
        register_name="synthetic-hidden-evidence",
        version="0.1",
        quiet=True,
    ))
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    registry = ArtifactRegistry.load_json(registry_path)
    record = registry.get("report:synthetic-hidden-evidence:0.1")

    assert saved == payload
    assert payload["workflow"] == "hidden_evidence_selection"
    assert payload["summary"]["selected_count"] == 5
    assert record.metadata["artifact_kind"] == "hidden_evidence_selection"
    assert record.metadata["summary"]["selected_count"] == 5
