"""Intervention hooks and multi-concept probe helpers."""

from eigentruth.intervention.activation import (
    ACTIVATION_INTERVENTION_MODES,
    ACTIVATION_INTERVENTION_SCHEMA_VERSION,
    ACTIVATION_INTERVENTION_SPANS,
    ActivationInterventionSummary,
    TemporaryActivationIntervention,
    apply_activation_intervention,
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
    "PATHWAY_INTERVENTION_SCHEMA_VERSION",
    "ActivationInterventionSummary",
    "AttentionPathwayKnockoutReport",
    "PathwayInterventionEffect",
    "RepresentationProbe",
    "TemporaryActivationIntervention",
    "TruthProbe",
    "apply_activation_intervention",
    "attention_pathway_knockout_report",
    "knockout_attention_pathway",
    "pathway_intervention_effect",
]
