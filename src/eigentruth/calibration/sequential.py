"""Runtime artifacts for sequential conformal alpha spending."""

from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from eigentruth import __version__
from eigentruth.eval.conformal import SequentialConformalReport, sequential_conformal_monitor
from eigentruth.json_utils import strict_json_dumps

ArrayLike = torch.Tensor | Sequence[float]


@dataclass(frozen=True)
class SequentialConformalArtifact:
    """Deployable conformal artifact for session or batch alpha spending."""

    model_id: str
    target_layer: int
    signal_name: str
    calibration_scores: tuple[float, ...]
    alpha: float
    eigentruth_version: str
    direction: str = "higher"
    schedule: str = "harmonic"
    model_revision: Optional[str] = None
    warmup_dataset_metadata: Mapping[str, Any] = field(default_factory=dict)
    calibration_dataset_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    commit_sha: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        signal_name = str(self.signal_name)
        if not signal_name:
            raise ValueError("signal_name must be non-empty.")
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "target_layer", int(self.target_layer))
        object.__setattr__(self, "signal_name", signal_name)
        object.__setattr__(
            self,
            "calibration_scores",
            _finite_score_tuple(self.calibration_scores, name="calibration_scores"),
        )
        object.__setattr__(self, "alpha", _alpha_float(self.alpha))
        object.__setattr__(self, "direction", _normalize_direction(self.direction))
        object.__setattr__(self, "schedule", _normalize_schedule(self.schedule))
        object.__setattr__(self, "eigentruth_version", str(self.eigentruth_version))
        object.__setattr__(self, "warmup_dataset_metadata", dict(self.warmup_dataset_metadata))
        object.__setattr__(self, "calibration_dataset_metadata", dict(self.calibration_dataset_metadata))
        if self.model_revision is not None:
            object.__setattr__(self, "model_revision", str(self.model_revision))
        if self.created_at is not None:
            object.__setattr__(self, "created_at", str(self.created_at))
        if self.commit_sha is not None:
            object.__setattr__(self, "commit_sha", str(self.commit_sha))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def calibration_count(self) -> int:
        """Return the number of stored calibration scores."""
        return len(self.calibration_scores)

    def decide_sequence(
        self,
        scores: ArrayLike,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SequentialConformalReport:
        """Score a runtime sequence with the stored alpha-spending configuration."""
        artifact_metadata: dict[str, Any] = {
            "artifact_type": "sequential_conformal",
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "target_layer": self.target_layer,
            "signal_name": self.signal_name,
            "direction": self.direction,
            "schedule": self.schedule,
            "eigentruth_version": self.eigentruth_version,
        }
        if metadata is not None:
            runtime_metadata = dict(metadata)
            artifact_metadata["runtime_metadata"] = runtime_metadata
            for key, value in runtime_metadata.items():
                if key not in artifact_metadata:
                    artifact_metadata[key] = value
        return sequential_conformal_monitor(
            self.calibration_scores,
            scores,
            alpha=self.alpha,
            direction=self.direction,
            schedule=self.schedule,
            metadata=artifact_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact payload."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "target_layer": self.target_layer,
            "signal_name": self.signal_name,
            "direction": self.direction,
            "alpha": self.alpha,
            "schedule": self.schedule,
            "calibration_count": self.calibration_count,
            "calibration_scores": list(self.calibration_scores),
            "eigentruth_version": self.eigentruth_version,
            "warmup_dataset_metadata": dict(self.warmup_dataset_metadata),
            "calibration_dataset_metadata": dict(self.calibration_dataset_metadata),
            "created_at": self.created_at,
            "commit_sha": self.commit_sha,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SequentialConformalArtifact":
        """Build an artifact from JSON-like data."""
        return cls(
            model_id=str(data["model_id"]),
            model_revision=None if data.get("model_revision") is None else str(data["model_revision"]),
            target_layer=int(data["target_layer"]),
            signal_name=str(data["signal_name"]),
            direction=str(data.get("direction", "higher")),
            alpha=data["alpha"],
            schedule=str(data.get("schedule", "harmonic")),
            calibration_scores=tuple(data["calibration_scores"]),
            eigentruth_version=str(data["eigentruth_version"]),
            warmup_dataset_metadata=dict(data.get("warmup_dataset_metadata", {})),
            calibration_dataset_metadata=dict(data.get("calibration_dataset_metadata", {})),
            created_at=None if data.get("created_at") is None else str(data["created_at"]),
            commit_sha=None if data.get("commit_sha") is None else str(data["commit_sha"]),
            schema_version=int(data.get("schema_version", 1)),
        )

    def save_json(self, path: str | Path) -> None:
        """Save artifact metadata as UTF-8 JSON."""
        Path(path).write_text(strict_json_dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "SequentialConformalArtifact":
        """Load artifact metadata from UTF-8 JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class SequentialConformalCalibrator:
    """Build sequential conformal artifacts from held-out normal scores."""

    alpha: float = 0.1
    schedule: str = "harmonic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _alpha_float(self.alpha))
        object.__setattr__(self, "schedule", _normalize_schedule(self.schedule))

    def calibrate(
        self,
        *,
        model_id: str,
        target_layer: int,
        signal_name: str,
        calibration_scores: ArrayLike,
        direction: str = "higher",
        model_revision: Optional[str] = None,
        warmup_dataset_metadata: Optional[Mapping[str, Any]] = None,
        calibration_dataset_metadata: Optional[Mapping[str, Any]] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
    ) -> SequentialConformalArtifact:
        """Create a sequential conformal artifact for one model/layer/signal setting."""
        metadata = dict(calibration_dataset_metadata or {})
        scores = _finite_score_tuple(calibration_scores, name="calibration_scores")
        metadata["sequential_conformal"] = {
            "alpha": self.alpha,
            "schedule": self.schedule,
            "signal_name": str(signal_name),
            "direction": _normalize_direction(direction),
            "n_calibration": len(scores),
        }
        return SequentialConformalArtifact(
            model_id=model_id,
            model_revision=model_revision,
            target_layer=target_layer,
            signal_name=signal_name,
            calibration_scores=scores,
            alpha=self.alpha,
            direction=direction,
            schedule=self.schedule,
            eigentruth_version=eigentruth_version,
            warmup_dataset_metadata=warmup_dataset_metadata or {},
            calibration_dataset_metadata=metadata,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            commit_sha=commit_sha,
        )


def _normalize_direction(value: object) -> str:
    direction = str(value)
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    return direction


def _normalize_schedule(value: object) -> str:
    schedule = str(value).strip().lower().replace("_", "-")
    aliases = {
        "linear": "linear",
        "equal": "linear",
        "harmonic": "harmonic",
        "front-loaded": "harmonic",
        "geometric": "geometric",
        "halving": "geometric",
    }
    if schedule not in aliases:
        raise ValueError("schedule must be 'linear', 'harmonic', or 'geometric'.")
    return aliases[schedule]


def _alpha_float(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("alpha must be in (0, 1).")
    alpha = float(value)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    return alpha


def _finite_score_tuple(values: ArrayLike, *, name: str) -> tuple[float, ...]:
    if _contains_bool(values):
        raise ValueError(f"{name} must be numeric and must not contain bool values.")
    try:
        tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values.")
    return tuple(float(value) for value in tensor.tolist())


def _contains_bool(values: object) -> bool:
    if isinstance(values, bool):
        return True
    if isinstance(values, torch.Tensor):
        return values.dtype == torch.bool
    if isinstance(values, (str, bytes)):
        return False
    if isinstance(values, SequenceABC):
        return any(_contains_bool(value) for value in values)
    return False
