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
    covariance_shrinkage_intensity,
    covariance_spectrum,
    gaussian_wasserstein_distance,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,
    manifold_distance,
    manifold_wasserstein_distance,
    poincare_map,
    sherman_morrison_update,
)
from eigentruth.core.pre_generation import (
    ATTENTION_SOFT_TARGET_PROBE_SCHEMA_VERSION,
    AttentionSoftTargetProbeArtifact,
    load_attention_soft_target_probe,
    soft_error_rate_targets,
)
from eigentruth.core.subspace import TruthSubspace
from eigentruth.core.trajectory import (
    PromptAnswerPathwayMetrics,
    ResidualContributionProfile,
    TrajectoryConvergenceMetrics,
    TrajectoryConvergenceReport,
    TrajectoryMonitor,
    prompt_answer_pathway_metrics,
    residual_contribution_profile,
    trajectory_convergence_metrics,
)

RepresentationManifold = TruthManifold
RepresentationSubspace = TruthSubspace

__all__ = [
    "RepresentationManifold",
    "RepresentationSubspace",
    "TruthManifold",
    "TruthSubspace",
    "ATTENTION_SOFT_TARGET_PROBE_SCHEMA_VERSION",
    "AttentionSoftTargetProbeArtifact",
    "PromptAnswerPathwayMetrics",
    "ResidualContributionProfile",
    "TrajectoryConvergenceMetrics",
    "TrajectoryConvergenceReport",
    "TrajectoryMonitor",
    "prompt_answer_pathway_metrics",
    "residual_contribution_profile",
    "trajectory_convergence_metrics",
    "load_attention_soft_target_probe",
    "soft_error_rate_targets",
    "COVARIANCE_MODES",
    "CovarianceSpectrum",
    "covariance_shrinkage_intensity",
    "covariance_spectrum",
    "gaussian_wasserstein_distance",
    "cluster_assignment_entropy",
    "embedding_semantic_entropy",
    "internal_eigenscore",
    "lexical_semantic_energy",
    "lexical_semantic_entropy",
    "semantic_energy_score",
    "spectral_effective_rank",
    "hyperbolic_semantic_entropy",
    "mahalanobis_distance",
    "manifold_distance",
    "manifold_wasserstein_distance",
    "poincare_map",
    "sherman_morrison_update",
]
