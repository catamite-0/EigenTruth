"""Versioned concept artifacts for reusable EigenTruth probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from eigentruth.core.math_engine import TruthManifold
from eigentruth.json_utils import to_jsonable

CONCEPT_ARTIFACT_SCHEMA_VERSION = 1


def _manifold_state_dict(manifold: TruthManifold) -> dict[str, Any]:
    return {
        "format": 2,
        "mean": manifold.mean,
        "_M2": manifold._M2,
        "_M2_diag": manifold._M2_diag,
        "n": manifold.n,
        "hidden_dim": manifold.hidden_dim,
        "ridge_lambda": manifold.ridge_lambda,
        "covariance_mode": manifold.covariance_mode,
        "covariance_low_rank": manifold.covariance_low_rank,
        "false_mean": manifold.false_mean,
        "contrastive_direction": manifold.contrastive_direction,
    }


def _manifold_from_state_dict(state: Mapping[str, Any]) -> TruthManifold:
    manifold = TruthManifold(
        covariance_mode=str(state.get("covariance_mode", "full")),
        covariance_low_rank=int(state.get("covariance_low_rank", 16)),
    )
    manifold.mean = state["mean"]
    manifold._M2 = state.get("_M2")
    manifold._M2_diag = state.get("_M2_diag")
    manifold.n = int(state["n"])
    manifold.hidden_dim = int(state["hidden_dim"])
    manifold.ridge_lambda = float(state.get("ridge_lambda", 0.1))
    manifold.false_mean = state.get("false_mean")
    manifold.contrastive_direction = state.get("contrastive_direction")
    manifold._dirty = True
    if manifold.mean is not None:
        manifold._device = manifold.mean.device
    return manifold


@dataclass(frozen=True)
class ConceptArtifact:
    """A saved, versioned concept manifold bound to one model layer."""

    name: str
    version: str
    layer_idx: int
    manifold: TruthManifold
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CONCEPT_ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe artifact metadata without embedding tensor payloads."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "layer_idx": self.layer_idx,
            "description": self.description,
            "metadata": to_jsonable(self.metadata),
            "manifold": {
                "ready": self.manifold.is_ready(),
                "n": self.manifold.n,
                "hidden_dim": self.manifold.hidden_dim,
                "covariance_mode": self.manifold.covariance_mode,
                "covariance_low_rank": self.manifold.covariance_low_rank,
                "has_false_mean": self.manifold.false_mean is not None,
                "has_contrastive_direction": self.manifold.contrastive_direction is not None,
            },
        }

    def save(self, path: str | Path) -> None:
        """Save the concept artifact as one torch payload."""
        if not self.manifold.is_ready():
            raise ValueError("ConceptArtifact requires a ready TruthManifold before saving.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "layer_idx": self.layer_idx,
            "description": self.description,
            "metadata": to_jsonable(self.metadata),
            "manifold": _manifold_state_dict(self.manifold),
        }
        torch.save(payload, output_path)

    @classmethod
    def load(cls, path: str | Path) -> "ConceptArtifact":
        """Load a concept artifact saved by :meth:`save`."""
        payload = torch.load(path, weights_only=True)
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != CONCEPT_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported concept artifact schema_version={schema_version}; "
                f"expected {CONCEPT_ARTIFACT_SCHEMA_VERSION}."
            )
        return cls(
            name=str(payload["name"]),
            version=str(payload["version"]),
            layer_idx=int(payload["layer_idx"]),
            description=payload.get("description"),
            metadata=dict(payload.get("metadata", {})),
            manifold=_manifold_from_state_dict(payload["manifold"]),
            schema_version=schema_version,
        )


def load_concept_artifact(path: str | Path) -> ConceptArtifact:
    """Load a saved concept artifact."""
    return ConceptArtifact.load(path)


__all__ = [
    "CONCEPT_ARTIFACT_SCHEMA_VERSION",
    "ConceptArtifact",
    "load_concept_artifact",
]
