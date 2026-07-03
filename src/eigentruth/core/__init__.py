"""Core geometry and representation primitives."""

from __future__ import annotations

from eigentruth.core.factuality_probe import (
    CLAIM_FACTUALITY_POOLING_MODES,
    CLAIM_FACTUALITY_PROBE_SCHEMA_VERSION,
    ClaimFactualityProbeArtifact,
    ClaimFactualityScore,
    claim_factuality_diagnostics,
    load_claim_factuality_probe,
    pool_claim_hidden_states,
)
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
    AttentionPathwayMetrics,
    PromptAnswerPathwayMetrics,
    ResidualContributionProfile,
    TrajectoryConvergenceMetrics,
    TrajectoryConvergenceReport,
    TrajectoryMonitor,
    attention_pathway_metrics,
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
    "CLAIM_FACTUALITY_POOLING_MODES",
    "CLAIM_FACTUALITY_PROBE_SCHEMA_VERSION",
    "ATTENTION_SOFT_TARGET_PROBE_SCHEMA_VERSION",
    "AttentionSoftTargetProbeArtifact",
    "ClaimFactualityProbeArtifact",
    "ClaimFactualityScore",
    "AttentionPathwayMetrics",
    "PromptAnswerPathwayMetrics",
    "ResidualContributionProfile",
    "TrajectoryConvergenceMetrics",
    "TrajectoryConvergenceReport",
    "TrajectoryMonitor",
    "attention_pathway_metrics",
    "prompt_answer_pathway_metrics",
    "residual_contribution_profile",
    "trajectory_convergence_metrics",
    "claim_factuality_diagnostics",
    "load_attention_soft_target_probe",
    "load_claim_factuality_probe",
    "pool_claim_hidden_states",
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
