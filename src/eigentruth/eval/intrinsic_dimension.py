"""Dependency-free intrinsic-dimension estimators for hidden-state samples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class IntrinsicDimensionReport:
    """JSON-ready TwoNN intrinsic-dimension estimate for one state matrix."""

    intrinsic_dimension: float
    estimator: str
    sample_count: int
    hidden_dim: int
    usable_count: int
    fit_count: int
    duplicate_count: int
    trim_fraction: float
    nearest_distance_mean: float
    second_nearest_distance_mean: float
    neighbor_ratio_mean: float
    neighbor_ratio_median: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable report."""
        return {
            "intrinsic_dimension": self.intrinsic_dimension,
            "estimator": self.estimator,
            "sample_count": self.sample_count,
            "hidden_dim": self.hidden_dim,
            "usable_count": self.usable_count,
            "fit_count": self.fit_count,
            "duplicate_count": self.duplicate_count,
            "trim_fraction": self.trim_fraction,
            "nearest_distance_mean": self.nearest_distance_mean,
            "second_nearest_distance_mean": self.second_nearest_distance_mean,
            "neighbor_ratio_mean": self.neighbor_ratio_mean,
            "neighbor_ratio_median": self.neighbor_ratio_median,
        }


def twonn_intrinsic_dimension(
    states: Tensor,
    *,
    trim_fraction: float = 0.05,
    eps: float = 1e-12,
) -> IntrinsicDimensionReport:
    """Estimate intrinsic dimension with the TwoNN nearest-neighbor estimator.

    The estimator uses the ratio between each point's second and first nearest
    neighbor distances. It fits ``-log(1 - F(mu)) = d * log(mu)`` through the
    origin, where ``mu = r2 / r1``. Distances are computed through a Gram matrix
    identity instead of ``torch.cdist`` so the function stays compatible with
    conservative CPU builds.
    """
    matrix = _as_state_matrix(states)
    trim = _validate_trim_fraction(trim_fraction)
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")
    sample_count, hidden_dim = int(matrix.shape[0]), int(matrix.shape[1])
    if sample_count < 3:
        raise ValueError("TwoNN intrinsic dimension requires at least three samples.")

    nearest = _two_nearest_distances(matrix)
    r1, r2 = nearest[:, 0], nearest[:, 1]
    duplicate_mask = r1 <= float(eps)
    valid = (~duplicate_mask) & torch.isfinite(r1) & torch.isfinite(r2) & (r2 >= r1)
    if int(valid.sum().item()) < 3:
        raise ValueError("TwoNN intrinsic dimension requires at least three non-duplicate samples.")

    valid_r1 = r1[valid].to(torch.float64)
    valid_r2 = r2[valid].to(torch.float64)
    ratios = (valid_r2 / valid_r1.clamp_min(float(eps))).sort().values
    ratios = ratios[torch.isfinite(ratios) & (ratios > 1.0 + float(eps))]
    if int(ratios.numel()) < 3:
        raise ValueError("TwoNN intrinsic dimension requires at least three usable neighbor ratios.")

    x = torch.log(ratios)
    ranks = torch.arange(1, ratios.numel() + 1, dtype=torch.float64, device=ratios.device)
    empirical_cdf = (ranks - 0.5) / float(ratios.numel())
    y = -torch.log1p(-empirical_cdf)
    fit_x, fit_y = _trim_twonn_axes(x, y, trim)
    denominator = (fit_x * fit_x).sum()
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= eps:
        raise ValueError("TwoNN intrinsic dimension is undefined for degenerate neighbor ratios.")
    slope = (fit_x * fit_y).sum() / denominator

    return IntrinsicDimensionReport(
        intrinsic_dimension=float(slope.item()),
        estimator="twonn",
        sample_count=sample_count,
        hidden_dim=hidden_dim,
        usable_count=int(ratios.numel()),
        fit_count=int(fit_x.numel()),
        duplicate_count=int(duplicate_mask.sum().item()),
        trim_fraction=trim,
        nearest_distance_mean=float(valid_r1.mean().item()),
        second_nearest_distance_mean=float(valid_r2.mean().item()),
        neighbor_ratio_mean=float(ratios.mean().item()),
        neighbor_ratio_median=float(ratios.median().item()),
    )


def intrinsic_dimension_profile(
    layer_states: Mapping[int, Tensor],
    *,
    trim_fraction: float = 0.05,
    eps: float = 1e-12,
) -> list[dict[str, float | int | str]]:
    """Return sorted per-layer TwoNN intrinsic-dimension reports."""
    normalized = {int(layer): states for layer, states in layer_states.items()}
    profile = []
    for layer in sorted(normalized):
        report = twonn_intrinsic_dimension(
            normalized[layer],
            trim_fraction=trim_fraction,
            eps=eps,
        )
        payload = report.to_dict()
        payload["layer"] = int(layer)
        profile.append(payload)
    return profile


def intrinsic_dimension_peak_layer(profile: Sequence[Mapping[str, object]]) -> int:
    """Return the layer with the largest finite intrinsic-dimension estimate."""
    if not profile:
        raise ValueError("profile must not be empty.")
    best_layer: int | None = None
    best_value = -math.inf
    for entry in profile:
        layer = int(entry["layer"])
        value = float(entry["intrinsic_dimension"])
        if not math.isfinite(value):
            continue
        if value > best_value:
            best_layer = layer
            best_value = value
    if best_layer is None:
        raise ValueError("profile does not contain a finite intrinsic-dimension estimate.")
    return best_layer


def _as_state_matrix(states: Tensor) -> Tensor:
    matrix = torch.as_tensor(states, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("states must be a non-empty matrix [sample_count, hidden_dim].")
    if not torch.isfinite(matrix).all():
        raise ValueError("states must contain only finite values.")
    return matrix.detach()


def _validate_trim_fraction(trim_fraction: float) -> float:
    trim = float(trim_fraction)
    if not math.isfinite(trim) or trim < 0.0 or trim >= 0.45:
        raise ValueError("trim_fraction must be finite and in [0.0, 0.45).")
    return trim


def _two_nearest_distances(states: Tensor) -> Tensor:
    squared_norm = (states * states).sum(dim=1, keepdim=True)
    distances_sq = squared_norm + squared_norm.T - 2.0 * (states @ states.T)
    distances = torch.sqrt(torch.clamp(distances_sq, min=0.0))
    distances.fill_diagonal_(float("inf"))
    return torch.topk(distances, k=2, largest=False).values


def _trim_twonn_axes(x: Tensor, y: Tensor, trim_fraction: float) -> tuple[Tensor, Tensor]:
    trim_count = int(math.floor(int(x.numel()) * trim_fraction))
    if trim_count <= 0:
        return x, y
    if int(x.numel()) - (2 * trim_count) < 3:
        return x, y
    return x[trim_count:-trim_count], y[trim_count:-trim_count]
