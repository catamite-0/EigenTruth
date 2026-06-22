"""Internal-state spectral uncertainty scores inspired by INSIDE/EigenScore."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def internal_eigenscore(embeddings: Tensor, *, alpha: float = 1e-3) -> Tensor:
    """Return a log-det spectral diversity score for hidden-state embeddings.

    `embeddings` has shape [K, D], where rows can be sampled response embeddings
    or a local sequence of token embeddings. The score follows the INSIDE
    EigenScore form: center each feature across the embedding set, form the K x K
    Gram covariance, add alpha * I for full rank, and average log eigenvalues.
    Higher values mean more internal-state diversity and are treated as more
    anomalous.
    """
    states = _as_embedding_matrix(embeddings)
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0.")
    if states.shape[0] < 2:
        return states.new_tensor(0.0)

    centered = states - states.mean(dim=0, keepdim=True)
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


def lexical_semantic_entropy(samples: Sequence[str], *, normalize: bool = True, eps: float = 1e-12) -> Tensor:
    """Return a dependency-free entropy proxy over normalized sampled texts.

    This is a lightweight semantic-entropy placeholder for benchmark plumbing: it
    clusters continuations by a lexical normalization key and computes Shannon
    entropy over those clusters. Higher values indicate more sampled-response
    disagreement.
    """
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")
    keys = [_normalized_sample_key(sample) for sample in samples]
    if len(keys) < 2:
        return torch.tensor(0.0)

    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    probs = torch.tensor(list(counts.values()), dtype=torch.float32) / float(len(keys))
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    if normalize:
        denominator = math.log(float(len(keys)))
        if denominator > eps:
            entropy = entropy / denominator
    return entropy


def _normalized_sample_key(sample: str) -> str:
    if not isinstance(sample, str):
        raise ValueError("samples must be strings.")
    tokens: list[str] = []
    current: list[str] = []
    for char in sample.casefold():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return " ".join(tokens) or "<empty>"


def _as_embedding_matrix(embeddings: Tensor) -> Tensor:
    states = torch.as_tensor(embeddings, dtype=torch.float32)
    if states.ndim != 2:
        raise ValueError(f"expected embeddings with shape [K, D], got {tuple(states.shape)}.")
    if states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("embeddings must be non-empty.")
    if not torch.isfinite(states).all():
        raise ValueError("embeddings must contain only finite values.")
    return states
