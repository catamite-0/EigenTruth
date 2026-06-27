"""Tests for pathway intervention analysis helpers."""

import pytest
import torch

from eigentruth.intervention import (
    AttentionPathwayKnockoutReport,
    PathwayInterventionEffect,
    attention_pathway_knockout_report,
    knockout_attention_pathway,
    pathway_intervention_effect,
)


def test_knockout_attention_pathway_removes_prompt_flow_and_renormalizes():
    attention = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.4, 0.6, 0.0, 0.0],
                [0.50, 0.25, 0.25, 0.0],
                [0.10, 0.10, 0.20, 0.60],
            ]
        ],
        dtype=torch.float32,
    )

    knocked = knockout_attention_pathway(
        attention,
        pathway="prompt",
        prompt_start=0,
        answer_start=2,
        sequence_end=4,
    )

    assert knocked.shape == attention.shape
    assert torch.allclose(knocked[:, 2:4, 0:2], torch.zeros((1, 2, 2)))
    assert torch.allclose(knocked[:, 2:4, :].sum(dim=-1), torch.ones((1, 2)))
    assert knocked[0, 2, 2] == pytest.approx(1.0)
    assert knocked[0, 3, 2] == pytest.approx(0.25)
    assert knocked[0, 3, 3] == pytest.approx(0.75)


def test_attention_pathway_knockout_report_roundtrips_and_tracks_delta():
    attention = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.4, 0.6, 0.0, 0.0],
                [0.50, 0.25, 0.25, 0.0],
                [0.10, 0.10, 0.20, 0.60],
            ]
        ],
        dtype=torch.float32,
    )

    report = attention_pathway_knockout_report(
        attention,
        pathway="question",
        prompt_start=0,
        answer_start=2,
        sequence_end=4,
        metadata={"source": "unit"},
    )
    payload = report.to_dict()
    loaded = AttentionPathwayKnockoutReport.from_dict(payload)

    assert report.pathway == "prompt"
    assert report.baseline.prompt_flow_fraction == pytest.approx(0.475)
    assert report.intervened.prompt_flow_fraction == pytest.approx(0.0)
    assert report.removed_mass_fraction == pytest.approx(0.475)
    assert report.deltas["prompt_flow_fraction"] == pytest.approx(-0.475)
    assert report.deltas["answer_self_flow_fraction"] > 0.0
    assert payload["metadata"]["source"] == "unit"
    assert loaded.to_dict() == payload


def test_attention_pathway_knockout_report_supports_answer_pathway():
    attention = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.4, 0.6, 0.0, 0.0],
                [0.50, 0.25, 0.25, 0.0],
                [0.10, 0.10, 0.20, 0.60],
            ]
        ],
        dtype=torch.float32,
    )

    report = attention_pathway_knockout_report(
        attention,
        pathway="answer_anchored",
        prompt_start=0,
        answer_start=2,
        sequence_end=4,
    )

    assert report.pathway == "answer"
    assert report.baseline.answer_self_flow_fraction == pytest.approx(0.525)
    assert report.intervened.answer_self_flow_fraction == pytest.approx(0.0)
    assert report.deltas["prompt_flow_fraction"] > 0.0


def test_pathway_intervention_effect_respects_score_direction():
    higher = pathway_intervention_effect(
        "attn_prompt_flow_loss",
        baseline_score=0.8,
        intervened_score=0.3,
        direction="higher",
        metadata={"pathway": "prompt"},
    )
    lower = pathway_intervention_effect(
        "support_score",
        baseline_score=0.2,
        intervened_score=0.5,
        direction="lower",
    )

    assert higher.delta == pytest.approx(-0.5)
    assert higher.anomalous_delta == pytest.approx(-0.5)
    assert higher.risk_reduction == pytest.approx(0.5)
    assert higher.improved is True
    assert higher.metadata["pathway"] == "prompt"
    assert PathwayInterventionEffect.from_dict(higher.to_dict()).to_dict() == higher.to_dict()

    assert lower.delta == pytest.approx(0.3)
    assert lower.anomalous_delta == pytest.approx(-0.3)
    assert lower.risk_reduction == pytest.approx(0.3)
    assert lower.improved is True


def test_pathway_intervention_helpers_reject_invalid_inputs():
    with pytest.raises(ValueError, match="pathway"):
        knockout_attention_pathway(torch.ones(1, 3, 3), pathway="other", answer_start=1)
    with pytest.raises(ValueError, match="finite"):
        pathway_intervention_effect("bad", baseline_score=float("nan"), intervened_score=1.0)
    with pytest.raises(ValueError, match="direction"):
        pathway_intervention_effect("bad", baseline_score=1.0, intervened_score=2.0, direction="sideways")
