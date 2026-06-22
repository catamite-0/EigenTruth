"""Core geometry and representation primitives."""

from __future__ import annotations

from eigentruth.core.inside import internal_eigenscore, spectral_effective_rank
from eigentruth.core.math_engine import (
    TruthManifold,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,
    poincare_map,
    sherman_morrison_update,
)
from eigentruth.core.subspace import TruthSubspace

__all__ = [
    "TruthManifold",
    "TruthSubspace",
    "internal_eigenscore",
    "spectral_effective_rank",
    "hyperbolic_semantic_entropy",
    "mahalanobis_distance",
    "poincare_map",
    "sherman_morrison_update",
]
