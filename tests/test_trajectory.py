import pytest
import torch

from eigentruth.core import (
    PromptAnswerPathwayMetrics,
    ResidualContributionProfile,
    TrajectoryMonitor,
    prompt_answer_pathway_metrics,
    residual_contribution_profile,
    trajectory_convergence_metrics,
)


def _convergent_trajectory() -> torch.Tensor:
    states = [torch.zeros(3)]
    current = states[0].clone()
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
        current = current + torch.tensor([scale, 0.0, 0.0])
        states.append(current.clone())
    return torch.stack(states)


def _wandering_trajectory() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )


def test_trajectory_convergence_metrics_detects_step_decay():
    convergent = trajectory_convergence_metrics(_convergent_trajectory(), metadata={"kind": "convergent"})
    wandering = trajectory_convergence_metrics(_wandering_trajectory(), metadata={"kind": "wandering"})
    roundtrip = type(convergent).from_dict(convergent.to_dict())

    assert convergent.step_count == 6
    assert convergent.hidden_dim == 3
    assert convergent.convergence_ratio < 0.1
    assert convergent.koopman_rate < 1.0
    assert convergent.convergence_strength > 0.0
    assert convergent.decay_fraction == pytest.approx(1.0)
    assert convergent.convergence_score > wandering.convergence_score
    assert roundtrip.to_dict() == convergent.to_dict()


def test_trajectory_monitor_records_report_summary():
    monitor = TrajectoryMonitor(metadata={"run": "unit"})
    monitor.record(_convergent_trajectory(), metadata={"id": "a"})
    monitor.record(_wandering_trajectory(), metadata={"id": "b"})
    report = monitor.to_report(metadata={"report": "trajectory"}).to_dict()

    assert len(monitor.trajectories) == 2
    assert report["workflow"] == "generation_trajectory_convergence"
    assert report["summary"]["n_trajectories"] == 2
    assert report["summary"]["mean_convergence_score"] > 0.0
    assert report["metadata"]["run"] == "unit"
    assert report["metadata"]["report"] == "trajectory"


def test_residual_contribution_profile_summarizes_layer_curve():
    profile = residual_contribution_profile(
        {-2: 0.25, -1: 0.75, 0: 0.0},
        layers=(-2, -1, 0),
        late_fraction=1 / 3,
        metadata={"kind": "unit"},
    )
    roundtrip = ResidualContributionProfile.from_dict(profile.to_dict())

    assert profile.layer_count == 3
    assert profile.total_contribution == pytest.approx(1.0)
    assert profile.mean_contribution == pytest.approx(1.0 / 3.0)
    assert profile.peak_contribution == pytest.approx(0.75)
    assert profile.peak_layer == -1
    assert profile.peak_position == pytest.approx(0.5)
    assert profile.layer_centroid == pytest.approx(0.375)
    assert profile.late_mass_fraction == pytest.approx(0.0)
    assert 0.0 <= profile.normalized_entropy <= 1.0
    assert profile.concentration == pytest.approx(1.0 - profile.normalized_entropy)
    assert profile.metadata["kind"] == "unit"
    assert roundtrip.to_dict() == profile.to_dict()


def test_residual_contribution_profile_rejects_invalid_values():
    with pytest.raises(ValueError, match="non-negative finite"):
        residual_contribution_profile({0: -1.0})
    with pytest.raises(ValueError, match="late_fraction"):
        residual_contribution_profile({0: 1.0}, late_fraction=0.0)


def test_prompt_answer_pathway_metrics_separates_anchor_movements():
    prompt = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    answer = torch.tensor([[2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [2.0, 6.0, 0.0]])
    metrics = prompt_answer_pathway_metrics(prompt, answer, metadata={"kind": "pathway"})
    roundtrip = PromptAnswerPathwayMetrics.from_dict(metrics.to_dict())

    assert metrics.prompt_token_count == 2
    assert metrics.answer_token_count == 3
    assert metrics.hidden_dim == 3
    assert metrics.prompt_answer_distance == pytest.approx(6.0 / (3.0 ** 0.5))
    assert metrics.prompt_answer_cosine_gap == pytest.approx(1.0 - 4.0 / ((2.0 ** 2) ** 0.5 * (40.0 ** 0.5)))
    assert metrics.answer_anchor_distance == pytest.approx(6.0 / (3.0 ** 0.5))
    assert metrics.answer_path_length == pytest.approx(6.0 / (3.0 ** 0.5))
    assert metrics.pathway_disagreement == pytest.approx(0.0)
    assert metrics.metadata["kind"] == "pathway"
    assert roundtrip.to_dict() == metrics.to_dict()


def test_prompt_answer_pathway_metrics_rejects_bad_shapes():
    with pytest.raises(ValueError, match="2D tensor"):
        prompt_answer_pathway_metrics(torch.randn(1, 2, 3), torch.randn(2, 3))
    with pytest.raises(ValueError, match="share hidden_dim"):
        prompt_answer_pathway_metrics(torch.randn(1, 3), torch.randn(2, 4))
    bad = torch.randn(1, 3)
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        prompt_answer_pathway_metrics(bad, torch.randn(1, 3))


def test_trajectory_convergence_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D tensor"):
        trajectory_convergence_metrics(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="at least three"):
        trajectory_convergence_metrics(torch.randn(2, 4))
    with pytest.raises(ValueError, match="finite"):
        bad = torch.randn(3, 4)
        bad[0, 0] = float("nan")
        trajectory_convergence_metrics(bad)
