"""Low-rank truth subspace utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class TruthSubspace:
    """Low-rank subspace fit to factual hidden states.

    The residual distance to this subspace is a natural extension of a single
    truth direction: states far from the factual subspace are treated as more
    anomalous, while contrastive projection remains available when false states
    are supplied.
    """

    mean: Optional[Tensor] = None
    basis: Optional[Tensor] = None  # [D, K], orthonormal columns
    rank: int = 0
    false_mean: Optional[Tensor] = None
    contrastive_direction: Optional[Tensor] = None

    @classmethod
    def fit(cls, states: Tensor, rank: int = 1) -> "TruthSubspace":
        """Fit a PCA subspace from factual states, shape [N, D]."""
        states = _as_state_matrix(states)
        if states.shape[0] < 2:
            raise ValueError("at least two factual states are required to fit a subspace.")
        if rank < 1:
            raise ValueError("rank must be >= 1.")
        mean = states.mean(dim=0).to(torch.float32)
        centered = states.to(torch.float32) - mean
        max_rank = min(rank, centered.shape[0], centered.shape[1])
        if max_rank < 1:
            raise ValueError("at least one state and one dimension are required.")
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:max_rank].T.contiguous()
        return cls(mean=mean, basis=basis, rank=max_rank)

    @classmethod
    def fit_contrastive(cls, true_states: Tensor, false_states: Tensor, rank: int = 1) -> "TruthSubspace":
        """Fit a factual subspace and a true-minus-false contrastive direction."""
        subspace = cls.fit(true_states, rank=rank)
        false_states = _as_state_matrix(false_states)
        false_mean = false_states.mean(dim=0).to(torch.float32)
        raw_direction = subspace.mean - false_mean
        direction = raw_direction / torch.norm(raw_direction).clamp(min=1e-8)
        subspace.false_mean = false_mean
        subspace.contrastive_direction = direction
        return subspace

    def is_ready(self) -> bool:
        """Return whether the subspace has fitted tensors."""
        return self.mean is not None and self.basis is not None and self.rank > 0

    def to(self, device: str | torch.device) -> "TruthSubspace":
        """Move tensors to a device in-place and return self."""
        device = torch.device(device)
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.basis is not None:
            self.basis = self.basis.to(device)
        if self.false_mean is not None:
            self.false_mean = self.false_mean.to(device)
        if self.contrastive_direction is not None:
            self.contrastive_direction = self.contrastive_direction.to(device)
        return self

    def coordinates(self, h: Tensor) -> Tensor:
        """Project states into subspace coordinates, shape [..., K]."""
        self._require_ready()
        delta = h.to(torch.float32) - self.mean.to(h.device)
        return delta @ self.basis.to(h.device)

    def project(self, h: Tensor) -> Tensor:
        """Return the Euclidean projection of states onto the fitted subspace."""
        self._require_ready()
        coords = self.coordinates(h)
        basis = self.basis.to(h.device)
        return self.mean.to(h.device) + coords @ basis.T

    def residual_distance(self, h: Tensor) -> Tensor:
        """Distance from states to the factual subspace; higher is more anomalous."""
        projected = self.project(h)
        return torch.norm(h.to(torch.float32) - projected.to(torch.float32), dim=-1)

    def truth_projection(self, h: Tensor) -> Tensor:
        """Projection onto the contrastive truth direction; lower is more suspicious."""
        if self.contrastive_direction is None:
            raise ValueError("contrastive_direction is not available; use fit_contrastive().")
        return h.to(torch.float32) @ self.contrastive_direction.to(h.device)

    def _require_ready(self) -> None:
        if not self.is_ready():
            raise RuntimeError("TruthSubspace is not fitted.")


def _as_state_matrix(states: Tensor) -> Tensor:
    states = torch.as_tensor(states, dtype=torch.float32)
    if states.ndim != 2:
        raise ValueError(f"expected states with shape [N, D], got {tuple(states.shape)}.")
    if states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("states must be non-empty.")
    return states
