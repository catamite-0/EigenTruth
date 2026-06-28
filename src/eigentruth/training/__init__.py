"""Training-side representation telemetry utilities."""

from __future__ import annotations

from eigentruth.training.callbacks import (
    RepTelemetryCallback,
    TelemetryCallbackEvent,
    extract_hidden_state_matrices,
)
from eigentruth.training.telemetry import (
    RepresentationTelemetryRecorder,
    RepresentationTelemetryReport,
    RepresentationTelemetrySnapshot,
    build_representation_manifold,
    representation_telemetry_snapshot,
)

__all__ = [
    "RepTelemetryCallback",
    "RepresentationTelemetryRecorder",
    "RepresentationTelemetryReport",
    "RepresentationTelemetrySnapshot",
    "TelemetryCallbackEvent",
    "build_representation_manifold",
    "extract_hidden_state_matrices",
    "representation_telemetry_snapshot",
]
