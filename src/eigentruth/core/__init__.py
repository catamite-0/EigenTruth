"""Core geometry and representation primitives."""

from __future__ import annotations

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
    "hyperbolic_semantic_entropy",
    "mahalanobis_distance",
    "poincare_map",
    "sherman_morrison_update",
]
