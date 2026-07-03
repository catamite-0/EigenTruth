"""Dependency-free helpers for calibrated diagnostic score fusion."""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import torch
from torch import Tensor

ArrayLike = Union[Tensor, Sequence[float]]

RANK_SCORE_FUSION_METHODS = ("max_rank", "mean_rank", "noisy_or_rank")
GEOMETRY_UNCERTAINTY_FUSION_METHODS = ("interaction", "product", "weighted_mean", "noisy_or")


def native_anomaly_scores(scores: ArrayLike, direction: str) -> Tensor:
    """Return scores with higher values meaning more anomalous."""
    if direction == "higher":
        return torch.as_tensor(scores, dtype=torch.float64).flatten()
    if direction == "lower":
        return -torch.as_tensor(scores, dtype=torch.float64).flatten()
    raise ValueError("direction must be 'higher' or 'lower'.")


def directional_rank_anomaly_scores(
    calibration_scores: ArrayLike,
    scores: ArrayLike,
    *,
    direction: str,
) -> Tensor:
    """Map native scores to empirical anomaly ranks in [0, 1].

    The calibration set is treated as the normal/reference population. Returned
    values are higher when a score is more extreme in the configured anomaly
    direction.
    """
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    calibration = torch.as_tensor(calibration_scores, dtype=torch.float64).flatten()
    scored = torch.as_tensor(scores, dtype=torch.float64).flatten()
    if calibration.numel() == 0:
        raise ValueError("calibration scores must be non-empty.")
    if not torch.isfinite(calibration).all() or not torch.isfinite(scored).all():
        raise ValueError("scores must be finite.")

    sorted_calibration, _ = torch.sort(calibration)
    return _directional_rank_anomaly_scores_from_sorted(
        sorted_calibration,
        scored,
        direction=direction,
    )


def _directional_rank_anomaly_scores_from_sorted(
    sorted_calibration_scores: ArrayLike,
    scores: ArrayLike,
    *,
    direction: str,
) -> Tensor:
    """Map scores to empirical anomaly ranks using pre-sorted calibration scores."""
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    sorted_calibration = torch.as_tensor(sorted_calibration_scores, dtype=torch.float64).flatten()
    scored = torch.as_tensor(scores, dtype=torch.float64).flatten()
    if sorted_calibration.numel() == 0:
        raise ValueError("calibration scores must be non-empty.")
    if not torch.isfinite(sorted_calibration).all() or not torch.isfinite(scored).all():
        raise ValueError("scores must be finite.")

    n = float(sorted_calibration.numel())
    if direction == "higher":
        counts = torch.searchsorted(sorted_calibration, scored, right=True).to(torch.float64)
    else:
        counts = (
            sorted_calibration.numel() - torch.searchsorted(sorted_calibration, scored, right=False)
        ).to(torch.float64)
    return counts / n


def combine_rank_anomaly_scores(rank_scores: Sequence[ArrayLike], method: str = "max_rank") -> Tensor:
    """Fuse one or more rank-normalized anomaly scores into one score."""
    if method not in RANK_SCORE_FUSION_METHODS:
        raise ValueError(f"method must be one of {RANK_SCORE_FUSION_METHODS}.")
    if not rank_scores:
        raise ValueError("at least one rank score is required.")
    tensors = [torch.as_tensor(score, dtype=torch.float64).flatten() for score in rank_scores]
    lengths = {score.numel() for score in tensors}
    if len(lengths) != 1:
        raise ValueError("rank scores must have the same length.")
    stacked = torch.stack(tensors, dim=0)
    if method == "max_rank":
        return stacked.max(dim=0).values
    if method == "mean_rank":
        return stacked.mean(dim=0)
    return 1.0 - torch.prod(1.0 - stacked.clamp(0.0, 1.0), dim=0)


def combine_geometry_uncertainty_scores(
    geometry_scores: ArrayLike,
    uncertainty_scores: ArrayLike,
    *,
    method: str = "interaction",
    geometry_weight: float = 1.0,
    uncertainty_weight: float = 1.0,
    interaction_weight: float = 1.0,
) -> Tensor:
    """Fuse rank-normalized geometry and uncertainty components.

    Inputs are expected to already live in anomaly-rank space where higher means
    more suspicious. The default interaction mode preserves single-channel
    evidence while adding an explicit geometry-by-uncertainty agreement term.
    """
    if method not in GEOMETRY_UNCERTAINTY_FUSION_METHODS:
        raise ValueError(f"method must be one of {GEOMETRY_UNCERTAINTY_FUSION_METHODS}.")
    geometry = torch.as_tensor(geometry_scores, dtype=torch.float64).flatten()
    uncertainty = torch.as_tensor(uncertainty_scores, dtype=torch.float64).flatten()
    if geometry.numel() != uncertainty.numel():
        raise ValueError("geometry_scores and uncertainty_scores must have the same length.")
    if not torch.isfinite(geometry).all() or not torch.isfinite(uncertainty).all():
        raise ValueError("scores must be finite.")
    geometry = geometry.clamp(0.0, 1.0)
    uncertainty = uncertainty.clamp(0.0, 1.0)
    geometry_weight = _non_negative_float(geometry_weight, name="geometry_weight")
    uncertainty_weight = _non_negative_float(uncertainty_weight, name="uncertainty_weight")
    interaction_weight = _non_negative_float(interaction_weight, name="interaction_weight")

    if method == "product":
        return geometry * uncertainty
    if method == "noisy_or":
        if geometry_weight + uncertainty_weight <= 0.0:
            raise ValueError("at least one score weight must be positive.")
        return 1.0 - (1.0 - geometry).pow(geometry_weight) * (1.0 - uncertainty).pow(uncertainty_weight)

    numerator = (geometry_weight * geometry) + (uncertainty_weight * uncertainty)
    denominator = geometry_weight + uncertainty_weight
    if method == "interaction":
        numerator = numerator + (interaction_weight * geometry * uncertainty)
        denominator = denominator + interaction_weight
    if denominator <= 0.0:
        raise ValueError("at least one score weight must be positive.")
    return (numerator / denominator).clamp(0.0, 1.0)


def geometry_calibrated_anomaly_scores(
    *,
    calibration_scores: Mapping[str, ArrayLike],
    scores: Mapping[str, ArrayLike],
    geometry_signals: Sequence[str],
    uncertainty_signals: Sequence[str],
    directions: Mapping[str, str] | None = None,
    geometry_method: str = "mean_rank",
    uncertainty_method: str = "mean_rank",
    fusion_method: str = "interaction",
    geometry_weight: float = 1.0,
    uncertainty_weight: float = 1.0,
    interaction_weight: float = 1.0,
) -> Tensor:
    """Return a rank-calibrated geometry-by-uncertainty anomaly score."""
    resolved_directions = {} if directions is None else dict(directions)
    geometry = _group_rank_scores(
        calibration_scores=calibration_scores,
        scores=scores,
        signals=geometry_signals,
        directions=resolved_directions,
        method=geometry_method,
    )
    uncertainty = _group_rank_scores(
        calibration_scores=calibration_scores,
        scores=scores,
        signals=uncertainty_signals,
        directions=resolved_directions,
        method=uncertainty_method,
    )
    return combine_geometry_uncertainty_scores(
        geometry,
        uncertainty,
        method=fusion_method,
        geometry_weight=geometry_weight,
        uncertainty_weight=uncertainty_weight,
        interaction_weight=interaction_weight,
    )


def global_local_uncertainty_scores(
    *,
    calibration_scores: Mapping[str, ArrayLike],
    scores: Mapping[str, ArrayLike],
    global_signals: Sequence[str],
    local_signals: Sequence[str],
    directions: Mapping[str, str] | None = None,
    global_method: str = "mean_rank",
    local_method: str = "mean_rank",
    gate_method: str = "product",
    global_weight: float = 1.0,
    local_weight: float = 1.0,
    interaction_weight: float = 1.0,
) -> Tensor:
    """Return a Global-Local Uncertainty (GLU) anomaly score.

    ``global_signals`` are hidden-state or representation-geometry signals.
    ``local_signals`` are token-level confidence/entropy signals. Both groups
    are rank-calibrated against normal calibration records, then fused with a
    multiplicative gate by default so high scores require both global geometry
    and local uncertainty evidence.
    """
    global_names = _unique_signal_names(global_signals, name="global_signals")
    local_names = _unique_signal_names(local_signals, name="local_signals")
    if set(global_names) & set(local_names):
        raise ValueError("global_signals and local_signals must not overlap.")
    return geometry_calibrated_anomaly_scores(
        calibration_scores=calibration_scores,
        scores=scores,
        geometry_signals=global_names,
        uncertainty_signals=local_names,
        directions=directions,
        geometry_method=global_method,
        uncertainty_method=local_method,
        fusion_method=gate_method,
        geometry_weight=global_weight,
        uncertainty_weight=local_weight,
        interaction_weight=interaction_weight,
    )


def _group_rank_scores(
    *,
    calibration_scores: Mapping[str, ArrayLike],
    scores: Mapping[str, ArrayLike],
    signals: Sequence[str],
    directions: Mapping[str, str],
    method: str,
) -> Tensor:
    signal_names = tuple(str(signal).strip() for signal in signals if str(signal).strip())
    if not signal_names:
        raise ValueError("at least one signal is required.")
    rank_scores = []
    for signal in signal_names:
        if signal not in calibration_scores:
            raise KeyError(signal)
        if signal not in scores:
            raise KeyError(signal)
        rank_scores.append(
            directional_rank_anomaly_scores(
                calibration_scores[signal],
                scores[signal],
                direction=str(directions.get(signal, "higher")),
            )
        )
    return combine_rank_anomaly_scores(rank_scores, method)


def _unique_signal_names(signals: Sequence[str], *, name: str) -> tuple[str, ...]:
    signal_names = tuple(str(signal).strip() for signal in signals if str(signal).strip())
    if not signal_names:
        raise ValueError(f"{name} must contain at least one signal.")
    if len(set(signal_names)) != len(signal_names):
        raise ValueError(f"{name} must contain unique signals.")
    return signal_names


def _non_negative_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number.")
    numeric = float(value)
    if not torch.isfinite(torch.tensor(numeric)) or numeric < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return numeric
