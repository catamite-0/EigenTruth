"""Dependency-free helpers for calibrated diagnostic score fusion."""

from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import Tensor

ArrayLike = Union[Tensor, Sequence[float]]

RANK_SCORE_FUSION_METHODS = ("max_rank", "mean_rank", "noisy_or_rank")


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
