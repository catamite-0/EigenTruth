"""Internal-state spectral uncertainty scores inspired by INSIDE/EigenScore."""

from __future__ import annotations

import torch
from torch import Tensor


def internal_eigenscore(embeddings: Tensor, *, alpha: float = 1e-3) -> Tensor:
    """Return a log-det spectral diversity score for hidden-state embeddings.

    `embeddings` has shape [K, D], where rows can be sampled response embeddings
    or a local sequence of token embeddings. The score follows the INSIDE
    EigenScore form: feature-center embeddings, form the K x K Gram covariance,
    add alpha * I for full rank, and average log eigenvalues. Higher values mean
    more internal-state diversity and are treated as more anomalous.
    """
    states = _as_embedding_matrix(embeddings)
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0.")
    if states.shape[0] < 2:
        return states.new_tensor(0.0)

    centered = states - states.mean(dim=1, keepdim=True)
    gram = centered @ centered.T
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    regularized = gram + float(alpha) * eye
    eigvals = torch.linalg.eigvalsh(regularized).clamp_min(torch.finfo(regularized.dtype).eps)
    return torch.log(eigvals).mean()


def spectral_effective_rank(embeddings: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Return the entropy effective rank of centered hidden-state embeddings."""
    states = _as_embedding_matrix(embeddings)
    if states.shape[0] < 2:
        return states.new_tensor(0.0)
    centered = states - states.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variances = singular_values.square()
    total = variances.sum()
    if float(total.item()) <= eps:
        return states.new_tensor(0.0)
    probs = variances / total.clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return torch.exp(entropy)


def _as_embedding_matrix(embeddings: Tensor) -> Tensor:
    states = torch.as_tensor(embeddings, dtype=torch.float32)
    if states.ndim != 2:
        raise ValueError(f"expected embeddings with shape [K, D], got {tuple(states.shape)}.")
    if states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("embeddings must be non-empty.")
    if not torch.isfinite(states).all():
        raise ValueError("embeddings must contain only finite values.")
    return states
