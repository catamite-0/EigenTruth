import json
from types import SimpleNamespace

import pytest
import torch

from eigentruth.training import (
    RepresentationTelemetryRecorder,
    RepresentationTelemetryReport,
    RepTelemetryCallback,
    build_representation_manifold,
    extract_hidden_state_matrices,
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


class _DummyHiddenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.5, -0.25, 0.75],
                    ]
                )
            )
        self.called_with_hidden_flags = False

    def forward(
        self,
        input_ids: torch.Tensor,
        shift: torch.Tensor | float = 0.0,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> SimpleNamespace:
        self.called_with_hidden_flags = bool(output_hidden_states and return_dict)
        inputs = torch.nn.functional.one_hot(input_ids, num_classes=3).float()
        hidden_1 = self.proj(inputs)
        hidden_2 = hidden_1 + torch.as_tensor(shift, dtype=torch.float32)
        return SimpleNamespace(hidden_states=(inputs, hidden_1, hidden_2))


def test_rep_telemetry_callback_captures_hidden_states_and_writes_report(tmp_path):
    model = _DummyHiddenModel()
    model.train()
    input_ids = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [0, 2, 1],
        ]
    )

    def batch_provider(state, **_kwargs):
        return {"input_ids": input_ids, "shift": torch.tensor(float(state.global_step) * 0.1)}

    callback = RepTelemetryCallback(
        batch_provider=batch_provider,
        layers=(-2, -1),
        capture_every_n_steps=2,
        report_path=tmp_path / "telemetry-report.json",
        metadata={"run": "dummy"},
    )

    callback.on_train_begin(None, SimpleNamespace(global_step=0, epoch=0.0), None, model=model)
    callback.on_step_end(None, SimpleNamespace(global_step=1, epoch=0.5), None, model=model)
    callback.on_step_end(None, SimpleNamespace(global_step=2, epoch=1.0), None, model=model)
    callback.on_train_end(None, SimpleNamespace(global_step=2, epoch=1.0), None, model=model)

    payload = json.loads((tmp_path / "telemetry-report.json").read_text(encoding="utf-8"))
    captured_events = [event for event in callback.events if event.status == "captured"]

    assert model.training is True
    assert model.called_with_hidden_flags is True
    assert len(callback.recorder.snapshots) == 4
    assert [event.event for event in captured_events] == ["train_begin", "step_end"]
    assert payload["summary"]["n_snapshots"] == 4
    assert payload["metadata"]["run"] == "dummy"
    assert payload["metadata"]["callback"]["type"] == "RepTelemetryCallback"
    assert payload["metadata"]["callback"]["events"][0]["status"] == "captured"
    assert callback.recorder.snapshots[-1].distance_to_baseline is not None


def test_rep_telemetry_callback_errors_are_recorded_by_default():
    callback = RepTelemetryCallback(batch_provider=None)
    snapshots = callback.capture("manual", model=_DummyHiddenModel(), state={"global_step": 7})

    assert snapshots == ()
    assert callback.events[-1].status == "error"
    assert callback.events[-1].step == 7
    assert "batch_provider" in callback.events[-1].reason


def test_rep_telemetry_callback_can_raise_on_error():
    callback = RepTelemetryCallback(batch_provider=None, raise_on_error=True)

    with pytest.raises(ValueError, match="batch_provider"):
        callback.capture("manual", model=_DummyHiddenModel(), state={"global_step": 7})


def test_extract_hidden_state_matrices_supports_pooling_modes():
    outputs = {
        "hidden_states": (
            torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            torch.ones(2, 3, 4),
        )
    }
    last = extract_hidden_state_matrices(outputs, layers=(0,), pooling="last_token")
    mean = extract_hidden_state_matrices(outputs, layers=(0,), pooling="mean_token")
    flat = extract_hidden_state_matrices(outputs, layers=(0,), pooling="flat_tokens")

    assert last[0].shape == (2, 4)
    assert last[0].tolist() == [[8.0, 9.0, 10.0, 11.0], [20.0, 21.0, 22.0, 23.0]]
    assert mean[0].shape == (2, 4)
    assert flat[0].shape == (6, 4)
    with pytest.raises(ValueError, match="layer 3"):
        extract_hidden_state_matrices(outputs, layers=(3,), pooling="last_token")
