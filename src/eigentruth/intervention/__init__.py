"""Intervention hooks and multi-concept probe helpers."""

from eigentruth.intervention.hooks import TruthProbe
from eigentruth.intervention.multi_probe import (
    ConceptProbeConfig,
    ConceptProbeState,
    MultiConceptMonitor,
)

RepresentationProbe = TruthProbe

__all__ = [
    "ConceptProbeConfig",
    "ConceptProbeState",
    "MultiConceptMonitor",
    "RepresentationProbe",
    "TruthProbe",
]
