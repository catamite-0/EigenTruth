"""EigenTruth — 基于几何动力学与表征工程的大模型表征诊断研究工具库。

Usage::

    from eigentruth.models.wrapper import EigenTruthWrapper

    monitor = EigenTruthWrapper(base_model, target_layer_idx=-10)
    monitor.warmup(fact_dataset, tokenizer)
    outputs = monitor.generate(**inputs, max_new_tokens=100)
"""

__version__ = "0.2.0"

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
from eigentruth.intervention.activation import (
    ActivationInterventionSummary,
    ActivationPatchSummary,
    TemporaryActivationIntervention,
    TemporaryActivationPatch,
    apply_activation_intervention,
    apply_activation_patch,
)
from eigentruth.intervention.hooks import TruthProbe
from eigentruth.intervention.multi_probe import (
    ConceptProbeConfig,
    ConceptProbeState,
    MultiConceptMonitor,
)
from eigentruth.intervention.pathways import (
    AttentionPathwayKnockoutReport,
    PathwayInterventionEffect,
    attention_pathway_knockout_report,
    knockout_attention_pathway,
    pathway_intervention_effect,
)
from eigentruth.models.wrapper import EigenTruthWrapper
from eigentruth.registry import ConceptArtifact, load_concept_artifact
from eigentruth.training import (
    RepresentationTelemetryRecorder,
    RepresentationTelemetryReport,
    RepresentationTelemetrySnapshot,
    RepTelemetryCallback,
    TelemetryCallbackEvent,
    build_representation_manifold,
    extract_hidden_state_matrices,
    representation_telemetry_snapshot,
)

RepresentationManifold = TruthManifold
RepresentationSubspace = TruthSubspace
RepresentationProbe = TruthProbe
RepresentationMonitor = EigenTruthWrapper

__all__ = [
    "EigenTruthWrapper",
    "RepresentationMonitor",
    "TruthProbe",
    "RepresentationProbe",
    "ConceptProbeConfig",
    "ConceptProbeState",
    "MultiConceptMonitor",
    "ActivationInterventionSummary",
    "ActivationPatchSummary",
    "AttentionPathwayKnockoutReport",
    "PathwayInterventionEffect",
    "TemporaryActivationIntervention",
    "TemporaryActivationPatch",
    "TruthManifold",
    "RepresentationManifold",
    "TruthSubspace",
    "RepresentationSubspace",
    "AttentionSoftTargetProbeArtifact",
    "ConceptArtifact",
    "AttentionPathwayMetrics",
    "PromptAnswerPathwayMetrics",
    "ResidualContributionProfile",
    "TrajectoryConvergenceMetrics",
    "TrajectoryConvergenceReport",
    "TrajectoryMonitor",
    "attention_pathway_metrics",
    "attention_pathway_knockout_report",
    "apply_activation_intervention",
    "apply_activation_patch",
    "knockout_attention_pathway",
    "pathway_intervention_effect",
    "prompt_answer_pathway_metrics",
    "residual_contribution_profile",
    "trajectory_convergence_metrics",
    "RepTelemetryCallback",
    "RepresentationTelemetryRecorder",
    "RepresentationTelemetryReport",
    "RepresentationTelemetrySnapshot",
    "TelemetryCallbackEvent",
    "build_representation_manifold",
    "extract_hidden_state_matrices",
    "representation_telemetry_snapshot",
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
    "load_attention_soft_target_probe",
    "load_concept_artifact",
    "semantic_energy_score",
    "soft_error_rate_targets",
    "spectral_effective_rank",
    "mahalanobis_distance",
    "manifold_distance",
    "manifold_wasserstein_distance",
    "poincare_map",
    "hyperbolic_semantic_entropy",
    "sherman_morrison_update",
    "__version__",
]
