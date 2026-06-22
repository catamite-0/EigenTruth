"""Internal-state spectral uncertainty scores inspired by INSIDE/EigenScore."""

from __future__ import annotations

import math
from collections.abc import Hashable
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


def cluster_assignment_entropy(assignments: Sequence[object], *, normalize: bool = True, eps: float = 1e-12) -> Tensor:
    """Return Shannon entropy over externally supplied semantic cluster assignments."""
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")
    labels = tuple(assignments)
    if len(labels) < 2:
        return torch.tensor(0.0)

    counts: dict[Hashable, int] = {}
    for label in labels:
        if not isinstance(label, Hashable):
            raise ValueError("cluster assignments must be hashable.")
        counts[label] = counts.get(label, 0) + 1
    return _entropy_from_counts(tuple(counts.values()), len(labels), normalize=normalize, eps=eps)


def lexical_semantic_entropy(samples: Sequence[str], *, normalize: bool = True, eps: float = 1e-12) -> Tensor:
    """Return a dependency-free entropy proxy over normalized sampled texts.

    This is a lightweight semantic-entropy placeholder for benchmark plumbing: it
    clusters continuations by a lexical normalization key and computes Shannon
    entropy over those clusters. Higher values indicate more sampled-response
    disagreement.
    """
    keys = [_normalized_sample_key(sample) for sample in samples]
    return cluster_assignment_entropy(keys, normalize=normalize, eps=eps)


def embedding_semantic_entropy(
    embeddings: Tensor,
    *,
    similarity_threshold: float = 0.90,
    normalize: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Return entropy over cosine-similarity clusters of sampled response embeddings."""
    states = _as_embedding_matrix(embeddings)
    if not (-1.0 <= similarity_threshold <= 1.0):
        raise ValueError("similarity_threshold must be in [-1, 1].")
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")
    if states.shape[0] < 2:
        return states.new_tensor(0.0)

    norms = torch.linalg.vector_norm(states, dim=1, keepdim=True).clamp_min(eps)
    normalized = states / norms
    similarities = normalized @ normalized.T
    labels = _connected_similarity_clusters(similarities, float(similarity_threshold))
    entropy = cluster_assignment_entropy(labels, normalize=normalize, eps=eps)
    return states.new_tensor(float(entropy.item()))


def _entropy_from_counts(counts: Sequence[int], total: int, *, normalize: bool, eps: float) -> Tensor:
    probs = torch.tensor(list(counts), dtype=torch.float32) / float(total)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    if normalize:
        denominator = math.log(float(total))
        if denominator > eps:
            entropy = entropy / denominator
    return entropy


def _connected_similarity_clusters(similarities: Tensor, threshold: float) -> tuple[int, ...]:
    n = int(similarities.shape[0])
    parents = list(range(n))

    def find(idx: int) -> int:
        while parents[idx] != idx:
            parents[idx] = parents[parents[idx]]
            idx = parents[idx]
        return idx

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(n):
        for right in range(left + 1, n):
            if float(similarities[left, right].item()) >= threshold:
                union(left, right)

    roots: dict[int, int] = {}
    labels: list[int] = []
    for idx in range(n):
        root = find(idx)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])
    return tuple(labels)


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
