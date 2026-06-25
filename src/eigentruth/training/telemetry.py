"""Dependency-free training-side representation telemetry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from eigentruth.core import COVARIANCE_MODES, TruthManifold


def _as_state_matrix(states: Tensor, *, name: str = "states") -> Tensor:
    matrix = torch.as_tensor(states, dtype=torch.float32).detach().cpu()
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor [sample_count, hidden_dim].")
    if int(matrix.shape[0]) < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if int(matrix.shape[1]) < 1:
        raise ValueError(f"{name} must have a non-empty hidden dimension.")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def build_representation_manifold(
    states: Tensor,
    *,
    covariance_mode: str = "shrinkage",
    covariance_low_rank: int = 16,
) -> TruthManifold:
    """Build a ``TruthManifold`` from one training telemetry state matrix."""
    if covariance_mode not in COVARIANCE_MODES:
        raise ValueError(f"covariance_mode must be one of {COVARIANCE_MODES}.")
    if int(covariance_low_rank) < 1:
        raise ValueError("covariance_low_rank must be >= 1.")
    matrix = _as_state_matrix(states)
    manifold = TruthManifold(covariance_mode=covariance_mode, covariance_low_rank=int(covariance_low_rank))
    manifold.update_many(matrix)
    return manifold


@dataclass(frozen=True)
class RepresentationTelemetrySnapshot:
    """One layer's representation telemetry at one training step."""

    step: int
    layer: int
    sample_count: int
    hidden_dim: int
    covariance_mode: str
    mean_norm: float
    variance_trace: float
    effective_rank: float
    participation_ratio: float
    stable_rank: float
    spike_count: int
    numerical_rank: int
    condition_number: float
    distance_to_baseline: float | None = None
    mean_shift_from_baseline: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready snapshot."""
        return {
            "step": int(self.step),
            "layer": int(self.layer),
            "sample_count": int(self.sample_count),
            "hidden_dim": int(self.hidden_dim),
            "covariance_mode": self.covariance_mode,
            "mean_norm": float(self.mean_norm),
            "variance_trace": float(self.variance_trace),
            "effective_rank": float(self.effective_rank),
            "participation_ratio": float(self.participation_ratio),
            "stable_rank": float(self.stable_rank),
            "spike_count": int(self.spike_count),
            "numerical_rank": int(self.numerical_rank),
            "condition_number": float(self.condition_number),
            "distance_to_baseline": (
                None if self.distance_to_baseline is None else float(self.distance_to_baseline)
            ),
            "mean_shift_from_baseline": (
                None if self.mean_shift_from_baseline is None else float(self.mean_shift_from_baseline)
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepresentationTelemetrySnapshot":
        """Build a snapshot from JSON-like data."""
        return cls(
            step=int(data["step"]),
            layer=int(data["layer"]),
            sample_count=int(data["sample_count"]),
            hidden_dim=int(data["hidden_dim"]),
            covariance_mode=str(data["covariance_mode"]),
            mean_norm=float(data["mean_norm"]),
            variance_trace=float(data["variance_trace"]),
            effective_rank=float(data["effective_rank"]),
            participation_ratio=float(data["participation_ratio"]),
            stable_rank=float(data["stable_rank"]),
            spike_count=int(data["spike_count"]),
            numerical_rank=int(data["numerical_rank"]),
            condition_number=float(data["condition_number"]),
            distance_to_baseline=(
                None if data.get("distance_to_baseline") is None else float(data["distance_to_baseline"])
            ),
            mean_shift_from_baseline=(
                None
                if data.get("mean_shift_from_baseline") is None
                else float(data["mean_shift_from_baseline"])
            ),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RepresentationTelemetryReport:
    """JSON-ready training telemetry report."""

    snapshots: tuple[RepresentationTelemetrySnapshot, ...]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "schema_version": int(self.schema_version),
            "workflow": "representation_training_telemetry",
            "summary": dict(self.summary),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepresentationTelemetryReport":
        """Build a telemetry report from JSON-like data."""
        return cls(
            snapshots=tuple(
                RepresentationTelemetrySnapshot.from_dict(snapshot)
                for snapshot in data.get("snapshots", ())
            ),
            summary=dict(data.get("summary") or {}),
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", 1)),
        )


def representation_telemetry_snapshot(
    states: Tensor,
    *,
    step: int,
    layer: int,
    baseline: TruthManifold | None = None,
    covariance_mode: str = "shrinkage",
    covariance_low_rank: int = 16,
    distance_covariance_mode: str = "model",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[RepresentationTelemetrySnapshot, TruthManifold]:
    """Summarize one layer's hidden states and optionally compare to a baseline."""
    manifold = build_representation_manifold(
        states,
        covariance_mode=covariance_mode,
        covariance_low_rank=covariance_low_rank,
    )
    spectrum = manifold.covariance_spectrum()
    variance_trace = float(spectrum.eigenvalues.sum().item())
    distance_to_baseline = None
    mean_shift = None
    if baseline is not None:
        if baseline.mean is None or manifold.mean is None:
            raise ValueError("baseline and current manifolds must have means.")
        distance = baseline.manifold_distance(
            manifold,
            covariance_mode=distance_covariance_mode,
        )
        distance_to_baseline = float(distance.detach().cpu().item())
        mean_shift = float(torch.norm(manifold.mean - baseline.mean).detach().cpu().item())

    snapshot = RepresentationTelemetrySnapshot(
        step=int(step),
        layer=int(layer),
        sample_count=int(manifold.n),
        hidden_dim=int(manifold.hidden_dim),
        covariance_mode=manifold.covariance_mode,
        mean_norm=float(torch.norm(manifold.mean).detach().cpu().item()) if manifold.mean is not None else 0.0,
        variance_trace=variance_trace,
        effective_rank=float(spectrum.effective_rank),
        participation_ratio=float(spectrum.participation_ratio),
        stable_rank=float(spectrum.stable_rank),
        spike_count=int(spectrum.spike_count),
        numerical_rank=int(spectrum.numerical_rank),
        condition_number=float(spectrum.condition_number),
        distance_to_baseline=distance_to_baseline,
        mean_shift_from_baseline=mean_shift,
        metadata=metadata or {},
    )
    return snapshot, manifold


@dataclass
class RepresentationTelemetryRecorder:
    """Callback-friendly recorder for per-layer training representation telemetry."""

    layers: tuple[int, ...] | None = None
    covariance_mode: str = "shrinkage"
    covariance_low_rank: int = 16
    distance_covariance_mode: str = "model"
    baseline_strategy: str = "first"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layers is not None:
            self.layers = tuple(int(layer) for layer in self.layers)
            if not self.layers:
                raise ValueError("layers must not be empty when provided.")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError("layers must not contain duplicates.")
        if self.covariance_mode not in COVARIANCE_MODES:
            raise ValueError(f"covariance_mode must be one of {COVARIANCE_MODES}.")
        if int(self.covariance_low_rank) < 1:
            raise ValueError("covariance_low_rank must be >= 1.")
        self.covariance_low_rank = int(self.covariance_low_rank)
        if self.baseline_strategy not in {"first", "manual", "none"}:
            raise ValueError("baseline_strategy must be one of: first, manual, none.")
        self._baselines: dict[int, TruthManifold] = {}
        self._snapshots: list[RepresentationTelemetrySnapshot] = []

    @property
    def snapshots(self) -> tuple[RepresentationTelemetrySnapshot, ...]:
        """Recorded snapshots in capture order."""
        return tuple(self._snapshots)

    @property
    def baselines(self) -> Mapping[int, TruthManifold]:
        """Current per-layer baseline manifolds."""
        return dict(self._baselines)

    def set_baseline(self, layer_states: Mapping[int, Tensor]) -> None:
        """Set manual per-layer baselines from state matrices."""
        for layer in self._resolve_layers(layer_states):
            self._baselines[int(layer)] = build_representation_manifold(
                layer_states[int(layer)],
                covariance_mode=self.covariance_mode,
                covariance_low_rank=self.covariance_low_rank,
            )

    def record_step(
        self,
        step: int,
        layer_states: Mapping[int, Tensor],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[RepresentationTelemetrySnapshot, ...]:
        """Record all configured layers for one training step."""
        snapshots = []
        for layer in self._resolve_layers(layer_states):
            baseline = self._baselines.get(layer)
            snapshot, manifold = representation_telemetry_snapshot(
                layer_states[layer],
                step=int(step),
                layer=layer,
                baseline=baseline,
                covariance_mode=self.covariance_mode,
                covariance_low_rank=self.covariance_low_rank,
                distance_covariance_mode=self.distance_covariance_mode,
                metadata=metadata or {},
            )
            if baseline is None and self.baseline_strategy == "first":
                self._baselines[layer] = manifold
                snapshot = RepresentationTelemetrySnapshot(
                    **{
                        **snapshot.to_dict(),
                        "distance_to_baseline": 0.0,
                        "mean_shift_from_baseline": 0.0,
                    }
                )
            snapshots.append(snapshot)
            self._snapshots.append(snapshot)
        return tuple(snapshots)

    def to_report(self, *, metadata: Mapping[str, Any] | None = None) -> RepresentationTelemetryReport:
        """Return a JSON-ready report over all recorded snapshots."""
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return RepresentationTelemetryReport(
            snapshots=tuple(self._snapshots),
            summary=_summarize_snapshots(self._snapshots),
            metadata=merged_metadata,
        )

    def _resolve_layers(self, layer_states: Mapping[int, Tensor]) -> tuple[int, ...]:
        available = {int(layer): states for layer, states in layer_states.items()}
        if not available:
            raise ValueError("layer_states must not be empty.")
        if self.layers is None:
            return tuple(sorted(available))
        missing = set(self.layers) - set(available)
        if missing:
            raise ValueError(f"layer_states is missing configured layer(s): {sorted(missing)}.")
        return self.layers


def _summarize_snapshots(snapshots: Sequence[RepresentationTelemetrySnapshot]) -> dict[str, Any]:
    if not snapshots:
        return {
            "n_snapshots": 0,
            "layers": [],
            "steps": [],
            "max_distance_to_baseline": None,
            "min_effective_rank": None,
            "final_by_layer": {},
        }
    layers = sorted({int(snapshot.layer) for snapshot in snapshots})
    steps = sorted({int(snapshot.step) for snapshot in snapshots})
    distances = [
        float(snapshot.distance_to_baseline)
        for snapshot in snapshots
        if snapshot.distance_to_baseline is not None
        and math.isfinite(float(snapshot.distance_to_baseline))
    ]
    effective_ranks = [
        float(snapshot.effective_rank)
        for snapshot in snapshots
        if math.isfinite(float(snapshot.effective_rank))
    ]
    final_by_layer = {}
    for layer in layers:
        layer_snapshots = [snapshot for snapshot in snapshots if int(snapshot.layer) == layer]
        final = max(layer_snapshots, key=lambda snapshot: int(snapshot.step))
        final_by_layer[str(layer)] = {
            "step": int(final.step),
            "effective_rank": float(final.effective_rank),
            "distance_to_baseline": final.distance_to_baseline,
            "mean_shift_from_baseline": final.mean_shift_from_baseline,
            "variance_trace": float(final.variance_trace),
        }
    return {
        "n_snapshots": len(snapshots),
        "layers": layers,
        "steps": steps,
        "max_distance_to_baseline": None if not distances else max(distances),
        "min_effective_rank": None if not effective_ranks else min(effective_ranks),
        "final_by_layer": final_by_layer,
    }
