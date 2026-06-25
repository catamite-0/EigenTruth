"""Core geometry and representation primitives."""

from __future__ import annotations

from eigentruth.core.inside import (
    cluster_assignment_entropy,
    embedding_semantic_entropy,
    internal_eigenscore,
    lexical_semantic_energy,
    lexical_semantic_entropy,
    semantic_energy_score,
    spectral_effective_rank,
)
from eigentruth.core.math_engine import (
    COVARIANCE_MODES,
    CovarianceSpectrum,
    TruthManifold,
    covariance_spectrum,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,
    poincare_map,
    sherman_morrison_update,
)
from eigentruth.core.subspace import TruthSubspace

__all__ = [
    "TruthManifold",
    "TruthSubspace",
    "COVARIANCE_MODES",
    "CovarianceSpectrum",
    "covariance_spectrum",
    "cluster_assignment_entropy",
    "embedding_semantic_entropy",
    "internal_eigenscore",
    "lexical_semantic_energy",
    "lexical_semantic_entropy",
    "semantic_energy_score",
    "spectral_effective_rank",
    "hyperbolic_semantic_entropy",
    "mahalanobis_distance",
    "poincare_map",
    "sherman_morrison_update",
]
