"""Generation trajectory convergence diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryConvergenceMetrics:
    """JSON-ready convergence diagnostics for one hidden-state trajectory."""

    step_count: int
    hidden_dim: int
    path_length: float
    direct_distance: float
    path_efficiency: float
    mean_step_distance: float
    initial_step_distance: float
    final_step_distance: float
    convergence_ratio: float
    step_distance_drop: float
    log_decay_slope: float
    koopman_rate: float
    convergence_strength: float
    decay_fraction: float
    displacement_cv: float
    convergence_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metrics payload."""
        return {
            "step_count": int(self.step_count),
            "hidden_dim": int(self.hidden_dim),
            "path_length": float(self.path_length),
            "direct_distance": float(self.direct_distance),
            "path_efficiency": float(self.path_efficiency),
            "mean_step_distance": float(self.mean_step_distance),
            "initial_step_distance": float(self.initial_step_distance),
            "final_step_distance": float(self.final_step_distance),
            "convergence_ratio": float(self.convergence_ratio),
            "step_distance_drop": float(self.step_distance_drop),
            "log_decay_slope": float(self.log_decay_slope),
            "koopman_rate": float(self.koopman_rate),
            "convergence_strength": float(self.convergence_strength),
            "decay_fraction": float(self.decay_fraction),
            "displacement_cv": float(self.displacement_cv),
            "convergence_score": float(self.convergence_score),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrajectoryConvergenceMetrics":
        """Build metrics from a JSON-like payload."""
        return cls(
            step_count=int(data["step_count"]),
            hidden_dim=int(data["hidden_dim"]),
            path_length=float(data["path_length"]),
            direct_distance=float(data["direct_distance"]),
            path_efficiency=float(data["path_efficiency"]),
            mean_step_distance=float(data["mean_step_distance"]),
            initial_step_distance=float(data["initial_step_distance"]),
            final_step_distance=float(data["final_step_distance"]),
            convergence_ratio=float(data["convergence_ratio"]),
            step_distance_drop=float(data["step_distance_drop"]),
            log_decay_slope=float(data["log_decay_slope"]),
            koopman_rate=float(data["koopman_rate"]),
            convergence_strength=float(data["convergence_strength"]),
            decay_fraction=float(data["decay_fraction"]),
            displacement_cv=float(data["displacement_cv"]),
            convergence_score=float(data["convergence_score"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TrajectoryConvergenceReport:
    """JSON-ready report over one or more generation trajectories."""

    trajectories: tuple[TrajectoryConvergenceMetrics, ...]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "schema_version": int(self.schema_version),
            "workflow": "generation_trajectory_convergence",
            "summary": dict(self.summary),
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
            "metadata": dict(self.metadata),
        }


def trajectory_convergence_metrics(
    states: Tensor,
    *,
    eps: float = 1e-8,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryConvergenceMetrics:
    """Summarize convergence for one trajectory of per-token hidden states.

    ``states`` must be shaped ``[generation_step, hidden_dim]``. Consecutive
    step distances are treated as a generation trajectory; exponential decay in
    those distances produces a positive ``convergence_strength`` and a
    ``koopman_rate`` below one.
    """
    matrix = _as_trajectory_matrix(states)
    if float(eps) <= 0.0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be positive and finite.")
    displacements = matrix[1:] - matrix[:-1]
    step_distances = torch.norm(displacements, dim=-1).to(torch.float64)
    path_length = float(step_distances.sum().item())
    direct_distance = float(torch.norm(matrix[-1] - matrix[0]).item())
    initial_distance = float(step_distances[0].item())
    final_distance = float(step_distances[-1].item())
    mean_distance = float(step_distances.mean().item())
    path_efficiency = direct_distance / max(path_length, float(eps))
    convergence_ratio = final_distance / max(initial_distance, float(eps))
    step_drop = initial_distance - final_distance
    log_decay_slope = _log_decay_slope(step_distances, eps=float(eps))
    koopman_rate = float(math.exp(log_decay_slope))
    convergence_strength = max(0.0, -log_decay_slope)
    decay_fraction = _decay_fraction(step_distances, tolerance=float(eps))
    std = float(step_distances.std(unbiased=False).item()) if int(step_distances.numel()) > 1 else 0.0
    displacement_cv = std / max(mean_distance, float(eps))
    convergence_score = (
        convergence_strength
        + max(0.0, 1.0 - convergence_ratio)
        + decay_fraction
        + max(0.0, path_efficiency)
    )
    return TrajectoryConvergenceMetrics(
        step_count=int(matrix.shape[0]),
        hidden_dim=int(matrix.shape[1]),
        path_length=path_length,
        direct_distance=direct_distance,
        path_efficiency=path_efficiency,
        mean_step_distance=mean_distance,
        initial_step_distance=initial_distance,
        final_step_distance=final_distance,
        convergence_ratio=convergence_ratio,
        step_distance_drop=step_drop,
        log_decay_slope=log_decay_slope,
        koopman_rate=koopman_rate,
        convergence_strength=convergence_strength,
        decay_fraction=decay_fraction,
        displacement_cv=displacement_cv,
        convergence_score=float(convergence_score),
        metadata=metadata or {},
    )


@dataclass
class TrajectoryMonitor:
    """Recorder for model-free generation trajectory convergence diagnostics."""

    eps: float = 1e-8
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.eps) <= 0.0 or not math.isfinite(float(self.eps)):
            raise ValueError("eps must be positive and finite.")
        self.eps = float(self.eps)
        self._trajectories: list[TrajectoryConvergenceMetrics] = []

    @property
    def trajectories(self) -> tuple[TrajectoryConvergenceMetrics, ...]:
        """Recorded trajectory metrics in capture order."""
        return tuple(self._trajectories)

    def record(
        self,
        states: Tensor,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrajectoryConvergenceMetrics:
        """Record one hidden-state trajectory."""
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        metrics = trajectory_convergence_metrics(states, eps=self.eps, metadata=merged_metadata)
        self._trajectories.append(metrics)
        return metrics

    def to_report(self, *, metadata: Mapping[str, Any] | None = None) -> TrajectoryConvergenceReport:
        """Return a report over all recorded trajectories."""
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return TrajectoryConvergenceReport(
            trajectories=tuple(self._trajectories),
            summary=_summarize_trajectories(self._trajectories),
            metadata=merged_metadata,
        )


def _as_trajectory_matrix(states: Tensor) -> Tensor:
    matrix = torch.as_tensor(states, dtype=torch.float32).detach().cpu()
    if matrix.ndim != 2:
        raise ValueError("states must be a 2D tensor [generation_step, hidden_dim].")
    if int(matrix.shape[0]) < 3:
        raise ValueError("states must contain at least three generation steps.")
    if int(matrix.shape[1]) < 1:
        raise ValueError("states must have a non-empty hidden dimension.")
    if not torch.isfinite(matrix).all():
        raise ValueError("states must contain only finite values.")
    return matrix


def _log_decay_slope(step_distances: Tensor, *, eps: float) -> float:
    x = torch.arange(int(step_distances.numel()), dtype=torch.float64)
    y = torch.log(step_distances.clamp_min(float(eps)).to(torch.float64))
    centered_x = x - x.mean()
    denominator = (centered_x * centered_x).sum()
    if float(denominator.item()) <= float(eps):
        return 0.0
    centered_y = y - y.mean()
    slope = (centered_x * centered_y).sum() / denominator
    return float(slope.item())


def _decay_fraction(step_distances: Tensor, *, tolerance: float) -> float:
    if int(step_distances.numel()) <= 1:
        return 1.0
    left = step_distances[:-1]
    right = step_distances[1:]
    return float((right <= left + float(tolerance)).to(torch.float32).mean().item())


def _summarize_trajectories(trajectories: Sequence[TrajectoryConvergenceMetrics]) -> dict[str, Any]:
    if not trajectories:
        return {
            "n_trajectories": 0,
            "mean_convergence_score": None,
            "mean_convergence_strength": None,
            "mean_koopman_rate": None,
            "mean_decay_fraction": None,
        }
    return {
        "n_trajectories": len(trajectories),
        "mean_convergence_score": sum(t.convergence_score for t in trajectories) / len(trajectories),
        "mean_convergence_strength": sum(t.convergence_strength for t in trajectories) / len(trajectories),
        "mean_koopman_rate": sum(t.koopman_rate for t in trajectories) / len(trajectories),
        "mean_decay_fraction": sum(t.decay_fraction for t in trajectories) / len(trajectories),
    }
