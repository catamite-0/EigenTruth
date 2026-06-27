"""Tests for pathway intervention analysis helpers."""

import pytest
import torch

from eigentruth.intervention import (
    ActivationInterventionSummary,
    ActivationPatchSummary,
    AttentionPathwayKnockoutReport,
    PathwayInterventionEffect,
    TemporaryActivationIntervention,
    TemporaryActivationPatch,
    apply_activation_intervention,
    apply_activation_patch,
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


def test_apply_activation_intervention_scales_answer_span_and_roundtrips_summary():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    out = apply_activation_intervention(
        hidden,
        sequence_lengths=(3, 2),
        answer_starts=(1, 1),
        span="answer",
        mode="scale",
        scale=0.5,
    )
    summary = ActivationInterventionSummary(
        layer_idx=0,
        span="answer",
        mode="scale",
        scale=0.5,
        sequence_lengths=(3, 2),
        answer_starts=(1, 1),
        affected_token_count=3,
    )

    assert torch.allclose(out[0, 0], hidden[0, 0])
    assert torch.allclose(out[0, 1:3], hidden[0, 1:3] * 0.5)
    assert torch.allclose(out[1, 1:2], hidden[1, 1:2] * 0.5)
    assert ActivationInterventionSummary.from_dict(summary.to_dict()).to_dict() == summary.to_dict()


def test_apply_activation_patch_copies_overlapping_answer_tokens_and_roundtrips_summary():
    target = torch.zeros(2, 4, 3)
    source = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    patched = apply_activation_patch(
        target,
        source,
        target_sequence_lengths=(4, 3),
        target_answer_starts=(1, 1),
        source_sequence_lengths=(4, 4),
        source_answer_starts=(2, 1),
        target_span="answer",
        source_span="answer",
    )
    summary = ActivationPatchSummary(
        layer_idx=0,
        target_span="answer",
        source_span="answer",
        alignment="left",
        target_sequence_lengths=(4, 3),
        target_answer_starts=(1, 1),
        source_sequence_lengths=(4, 4),
        source_answer_starts=(2, 1),
        copied_token_count=4,
    )

    assert torch.allclose(patched[0, 1:3], source[0, 2:4])
    assert torch.allclose(patched[0, 3], torch.zeros(3))
    assert torch.allclose(patched[1, 1:3], source[1, 1:3])
    assert ActivationPatchSummary.from_dict(summary.to_dict()).to_dict() == summary.to_dict()


def test_apply_activation_patch_right_aligns_when_spans_differ():
    target = torch.zeros(1, 5, 2)
    source = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)

    patched = apply_activation_patch(
        target,
        source,
        target_sequence_lengths=(5,),
        target_answer_starts=(1,),
        source_sequence_lengths=(4,),
        source_answer_starts=(2,),
        target_span="answer",
        source_span="answer",
        alignment="right",
    )

    assert torch.allclose(patched[0, 1:3], torch.zeros(2, 2))
    assert torch.allclose(patched[0, 3:5], source[0, 2:4])


def test_temporary_activation_intervention_modifies_layer_output():
    class Block(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

        def forward(self, x):
            return self.layers[0](x)

    model = Tiny()
    x = torch.ones(1, 3, 2)
    baseline = model(x)
    with TemporaryActivationIntervention(
        model,
        layer_idx=0,
        sequence_lengths=(3,),
        answer_starts=(1,),
        span="answer",
        mode="zero",
    ) as intervention:
        intervened = model(x)

    assert torch.allclose(baseline[:, 0, :], torch.full((1, 2), 2.0))
    assert torch.allclose(intervened[:, 0, :], torch.full((1, 2), 2.0))
    assert torch.allclose(intervened[:, 1:3, :], torch.zeros(1, 2, 2))
    assert intervention.summary.affected_token_count == 2


def test_temporary_activation_patch_modifies_layer_output():
    class Block(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

        def forward(self, x):
            return self.layers[0](x)

    model = Tiny()
    x = torch.ones(1, 3, 2)
    source_hidden = torch.tensor([[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]])

    with TemporaryActivationPatch(
        model,
        layer_idx=0,
        source_hidden=source_hidden,
        target_sequence_lengths=(3,),
        target_answer_starts=(1,),
        source_sequence_lengths=(3,),
        source_answer_starts=(1,),
        target_span="answer",
        source_span="answer",
    ) as patch:
        patched = model(x)

    assert torch.allclose(patched[:, 0, :], torch.full((1, 2), 2.0))
    assert torch.allclose(patched[:, 1:3, :], source_hidden[:, 1:3, :])
    assert patch.summary.copied_token_count == 2
