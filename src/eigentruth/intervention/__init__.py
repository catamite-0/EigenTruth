"""Intervention hooks and multi-concept probe helpers."""

from eigentruth.intervention.activation import (
    ACTIVATION_INTERVENTION_MODES,
    ACTIVATION_INTERVENTION_SCHEMA_VERSION,
    ACTIVATION_INTERVENTION_SPANS,
    ACTIVATION_PATCH_ALIGNMENTS,
    ACTIVATION_PATCH_SCHEMA_VERSION,
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
    PATHWAY_INTERVENTION_SCHEMA_VERSION,
    AttentionPathwayKnockoutReport,
    PathwayInterventionEffect,
    attention_pathway_knockout_report,
    knockout_attention_pathway,
    pathway_intervention_effect,
)

RepresentationProbe = TruthProbe

__all__ = [
    "ConceptProbeConfig",
    "ConceptProbeState",
    "MultiConceptMonitor",
    "ACTIVATION_INTERVENTION_MODES",
    "ACTIVATION_INTERVENTION_SCHEMA_VERSION",
    "ACTIVATION_INTERVENTION_SPANS",
    "ACTIVATION_PATCH_ALIGNMENTS",
    "ACTIVATION_PATCH_SCHEMA_VERSION",
    "PATHWAY_INTERVENTION_SCHEMA_VERSION",
    "ActivationInterventionSummary",
    "ActivationPatchSummary",
    "AttentionPathwayKnockoutReport",
    "PathwayInterventionEffect",
    "RepresentationProbe",
    "TemporaryActivationIntervention",
    "TemporaryActivationPatch",
    "TruthProbe",
    "apply_activation_intervention",
    "apply_activation_patch",
    "attention_pathway_knockout_report",
    "knockout_attention_pathway",
    "pathway_intervention_effect",
]
