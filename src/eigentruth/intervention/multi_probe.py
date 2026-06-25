"""Multi-concept probe attachment helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch.nn as nn

from eigentruth.core.math_engine import TruthManifold
from eigentruth.intervention.hooks import TruthProbe
from eigentruth.json_utils import to_jsonable
from eigentruth.registry.concepts import ConceptArtifact


@dataclass(frozen=True)
class ConceptProbeConfig:
    """Configuration for one concept probe attached to one layer."""

    name: str
    layer_idx: int
    manifold: TruthManifold
    threshold: float = 15.0
    steering_lambda: float = 0.0
    hse_window_size: int = 20
    curvature: float = 1.0
    custom_layer_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_artifact(
        cls,
        artifact: ConceptArtifact,
        *,
        threshold: float = 15.0,
        steering_lambda: float = 0.0,
        hse_window_size: int = 20,
        curvature: float = 1.0,
        custom_layer_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ConceptProbeConfig":
        """Build a probe config from a saved concept artifact."""
        merged_metadata: dict[str, Any] = dict(artifact.metadata)
        if metadata is not None:
            merged_metadata.update(dict(metadata))
        merged_metadata.setdefault("concept_version", artifact.version)
        if artifact.description is not None:
            merged_metadata.setdefault("description", artifact.description)
        return cls(
            name=artifact.name,
            layer_idx=artifact.layer_idx,
            manifold=artifact.manifold,
            threshold=threshold,
            steering_lambda=steering_lambda,
            hse_window_size=hse_window_size,
            curvature=curvature,
            custom_layer_path=custom_layer_path,
            metadata=merged_metadata,
        )

    def build_probe(self) -> TruthProbe:
        """Construct the underlying single-concept probe."""
        return TruthProbe(
            manifold=self.manifold,
            steering_lambda=self.steering_lambda,
            threshold=self.threshold,
            hse_window_size=self.hse_window_size,
            curvature=self.curvature,
        )


@dataclass(frozen=True)
class ConceptProbeState:
    """Serializable runtime diagnostics for one concept probe."""

    name: str
    layer_idx: int
    last_mahalanobis_distance: float
    last_hse: float
    threshold: float
    steering_lambda: float
    probe_active: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "name": self.name,
            "layer_idx": self.layer_idx,
            "last_mahalanobis_distance": self.last_mahalanobis_distance,
            "last_hse": self.last_hse,
            "threshold": self.threshold,
            "steering_lambda": self.steering_lambda,
            "probe_active": self.probe_active,
            "metadata": to_jsonable(self.metadata),
        }


class MultiConceptMonitor:
    """Attach and summarize several concept probes on one model."""

    def __init__(self, configs: Sequence[ConceptProbeConfig]) -> None:
        if not configs:
            raise ValueError("MultiConceptMonitor requires at least one ConceptProbeConfig.")
        names = [config.name for config in configs]
        if len(set(names)) != len(names):
            raise ValueError("Concept probe names must be unique.")
        self.configs = tuple(configs)
        self._probes: dict[str, TruthProbe] = {}

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Sequence[ConceptArtifact],
        *,
        threshold: float = 15.0,
        steering_lambda: float = 0.0,
        hse_window_size: int = 20,
        curvature: float = 1.0,
        custom_layer_path: str | None = None,
    ) -> "MultiConceptMonitor":
        """Build a monitor from saved concept artifacts using shared probe defaults."""
        return cls(
            [
                ConceptProbeConfig.from_artifact(
                    artifact,
                    threshold=threshold,
                    steering_lambda=steering_lambda,
                    hse_window_size=hse_window_size,
                    curvature=curvature,
                    custom_layer_path=custom_layer_path,
                )
                for artifact in artifacts
            ]
        )

    @property
    def is_active(self) -> bool:
        """Whether all configured probes are currently attached."""
        return bool(self._probes) and all(probe.is_active for probe in self._probes.values())

    def register(self, model: nn.Module) -> None:
        """Attach all probes to the target model."""
        self.remove()
        registered: dict[str, TruthProbe] = {}
        try:
            for config in self.configs:
                probe = config.build_probe()
                probe.register(
                    model,
                    config.layer_idx,
                    custom_layer_path=config.custom_layer_path,
                )
                registered[config.name] = probe
        except Exception:
            for probe in registered.values():
                probe.remove()
            raise
        self._probes = registered

    def remove(self) -> None:
        """Detach all active probes."""
        for probe in self._probes.values():
            probe.remove()
        self._probes = {}

    def reset_history(self) -> None:
        """Clear all probe HSE histories."""
        for probe in self._probes.values():
            probe.reset_history()

    def states(self) -> tuple[ConceptProbeState, ...]:
        """Return per-concept runtime states."""
        states: list[ConceptProbeState] = []
        for config in self.configs:
            probe = self._probes.get(config.name)
            states.append(
                ConceptProbeState(
                    name=config.name,
                    layer_idx=config.layer_idx,
                    last_mahalanobis_distance=0.0 if probe is None else probe.last_distance,
                    last_hse=0.0 if probe is None else probe.last_hse,
                    threshold=config.threshold,
                    steering_lambda=config.steering_lambda,
                    probe_active=False if probe is None else probe.is_active,
                    metadata=config.metadata,
                )
            )
        return tuple(states)

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe aggregate diagnostics for all concept probes."""
        states = self.states()
        return {
            "concept_count": len(states),
            "is_active": self.is_active,
            "concepts": {state.name: state.to_dict() for state in states},
        }


__all__ = [
    "ConceptProbeConfig",
    "ConceptProbeState",
    "MultiConceptMonitor",
]
