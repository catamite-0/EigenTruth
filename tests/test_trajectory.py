import pytest
import torch

from eigentruth.core import TrajectoryMonitor, trajectory_convergence_metrics


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


def test_trajectory_convergence_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D tensor"):
        trajectory_convergence_metrics(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="at least three"):
        trajectory_convergence_metrics(torch.randn(2, 4))
    with pytest.raises(ValueError, match="finite"):
        bad = torch.randn(3, 4)
        bad[0, 0] = float("nan")
        trajectory_convergence_metrics(bad)
