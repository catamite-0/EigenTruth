import pytest
import torch

from eigentruth.training import (
    RepresentationTelemetryRecorder,
    RepresentationTelemetryReport,
    build_representation_manifold,
    representation_telemetry_snapshot,
)


def test_representation_telemetry_snapshot_reports_spectrum_and_drift():
    torch.manual_seed(123)
    baseline_states = torch.randn(96, 6)
    shifted_states = baseline_states.clone()
    shifted_states[:, 0] += 0.75
    shifted_states[:, 3:] *= 0.35
    baseline = build_representation_manifold(baseline_states, covariance_mode="shrinkage")

    snapshot, manifold = representation_telemetry_snapshot(
        shifted_states,
        step=3,
        layer=-2,
        baseline=baseline,
        covariance_mode="shrinkage",
    )
    payload = snapshot.to_dict()
    roundtrip = RepresentationTelemetryReport.from_dict({
        "snapshots": [payload],
        "summary": {"example": True},
    })

    assert manifold.is_ready()
    assert snapshot.step == 3
    assert snapshot.layer == -2
    assert snapshot.sample_count == 96
    assert snapshot.hidden_dim == 6
    assert snapshot.variance_trace > 0.0
    assert snapshot.effective_rank > 0.0
    assert snapshot.distance_to_baseline is not None
    assert snapshot.distance_to_baseline > 0.0
    assert snapshot.mean_shift_from_baseline == pytest.approx(0.75, rel=0.15)
    assert roundtrip.snapshots[0].to_dict() == payload


def test_representation_telemetry_recorder_uses_first_step_as_baseline():
    torch.manual_seed(456)
    step0 = {
        -2: torch.randn(80, 5),
        -1: torch.randn(80, 5) + 0.25,
    }
    step1 = {
        -2: step0[-2] + 0.05,
        -1: (step0[-1] * torch.tensor([1.0, 0.5, 0.25, 0.1, 0.05])) + 1.0,
    }
    recorder = RepresentationTelemetryRecorder(layers=(-2, -1), covariance_mode="shrinkage")

    first = recorder.record_step(0, step0)
    second = recorder.record_step(1, step1)
    report = recorder.to_report(metadata={"run": "synthetic"}).to_dict()

    assert [snapshot.distance_to_baseline for snapshot in first] == [0.0, 0.0]
    assert second[0].distance_to_baseline is not None
    assert second[1].distance_to_baseline is not None
    assert second[1].distance_to_baseline > second[0].distance_to_baseline
    assert report["summary"]["n_snapshots"] == 4
    assert report["summary"]["layers"] == [-2, -1]
    assert report["summary"]["steps"] == [0, 1]
    assert report["summary"]["final_by_layer"]["-1"]["distance_to_baseline"] == second[1].distance_to_baseline
    assert report["metadata"]["run"] == "synthetic"


def test_representation_telemetry_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D tensor"):
        build_representation_manifold(torch.randn(4))
    with pytest.raises(ValueError, match="at least two samples"):
        build_representation_manifold(torch.randn(1, 4))
    with pytest.raises(ValueError, match="finite"):
        bad = torch.randn(4, 3)
        bad[0, 0] = float("nan")
        build_representation_manifold(bad)
    with pytest.raises(ValueError, match="missing configured"):
        RepresentationTelemetryRecorder(layers=(-2,)).record_step(0, {-1: torch.randn(4, 3)})
