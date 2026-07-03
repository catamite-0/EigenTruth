"""Runtime artifacts for conformal multiple-testing diagnostics."""

from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from eigentruth import __version__
from eigentruth.eval.conformal import MultipleTestingHallucinationReport, multiple_testing_conformal_report
from eigentruth.json_utils import strict_json_dumps

ArrayLike = torch.Tensor | Sequence[float]


@dataclass(frozen=True)
class MultipleTestingConformalSignal:
    """Held-out calibration scores for one conformal signal."""

    name: str
    calibration_scores: tuple[float, ...]
    direction: str = "higher"

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ValueError("signal name must be non-empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", _normalize_direction(self.direction))
        object.__setattr__(
            self,
            "calibration_scores",
            _finite_score_tuple(self.calibration_scores, name=f"calibration_scores.{name}"),
        )

    @property
    def calibration_count(self) -> int:
        """Return the number of calibration scores stored for this signal."""
        return len(self.calibration_scores)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable signal payload."""
        return {
            "name": self.name,
            "direction": self.direction,
            "calibration_count": self.calibration_count,
            "calibration_scores": list(self.calibration_scores),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MultipleTestingConformalSignal":
        """Build a signal from JSON-like data."""
        return cls(
            name=str(data["name"]),
            direction=str(data.get("direction", "higher")),
            calibration_scores=tuple(data["calibration_scores"]),
        )


@dataclass(frozen=True)
class MultipleTestingConformalArtifact:
    """Deployable conformal multiple-testing calibration artifact.

    Unlike a scalar threshold artifact, this stores the held-out calibration
    distribution for each signal so runtime p-values and global rejection rules
    remain exactly reproducible.
    """

    model_id: str
    target_layer: int
    signals: tuple[MultipleTestingConformalSignal, ...]
    alpha: float
    eigentruth_version: str
    method: str = "by"
    model_revision: Optional[str] = None
    warmup_dataset_metadata: Mapping[str, Any] = field(default_factory=dict)
    calibration_dataset_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    commit_sha: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        signals = tuple(self.signals)
        if not signals:
            raise ValueError("signals must be non-empty.")
        names = tuple(signal.name for signal in signals)
        if len(set(names)) != len(names):
            raise ValueError("signal names must be unique.")
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "target_layer", int(self.target_layer))
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "alpha", _alpha_float(self.alpha))
        object.__setattr__(self, "method", _normalize_method(self.method))
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

    def signal_names(self) -> tuple[str, ...]:
        """Return signal names in artifact order."""
        return tuple(signal.name for signal in self.signals)

    def get_signal(self, name: str) -> MultipleTestingConformalSignal:
        """Return one signal config by name."""
        for signal in self.signals:
            if signal.name == name:
                return signal
        raise KeyError(name)

    def decide(
        self,
        scores: Mapping[str, float],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MultipleTestingHallucinationReport:
        """Score one runtime item with the stored multiple-testing calibration."""
        artifact_metadata: dict[str, Any] = {
            "artifact_type": "multiple_testing_conformal",
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "target_layer": self.target_layer,
            "eigentruth_version": self.eigentruth_version,
            "signal_names": list(self.signal_names()),
        }
        if metadata is not None:
            runtime_metadata = dict(metadata)
            artifact_metadata["runtime_metadata"] = runtime_metadata
            for key, value in runtime_metadata.items():
                if key not in artifact_metadata:
                    artifact_metadata[key] = value
        return multiple_testing_conformal_report(
            {signal.name: signal.calibration_scores for signal in self.signals},
            scores,
            alpha=self.alpha,
            directions={signal.name: signal.direction for signal in self.signals},
            method=self.method,
            metadata=artifact_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact payload."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "target_layer": self.target_layer,
            "alpha": self.alpha,
            "method": self.method,
            "signals": [signal.to_dict() for signal in self.signals],
            "eigentruth_version": self.eigentruth_version,
            "warmup_dataset_metadata": dict(self.warmup_dataset_metadata),
            "calibration_dataset_metadata": dict(self.calibration_dataset_metadata),
            "created_at": self.created_at,
            "commit_sha": self.commit_sha,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MultipleTestingConformalArtifact":
        """Build an artifact from JSON-like data."""
        raw_signals = data.get("signals", ())
        if not isinstance(raw_signals, SequenceABC) or isinstance(raw_signals, (str, bytes)):
            raise ValueError("signals must be a sequence.")
        return cls(
            model_id=str(data["model_id"]),
            model_revision=None if data.get("model_revision") is None else str(data["model_revision"]),
            target_layer=int(data["target_layer"]),
            signals=tuple(MultipleTestingConformalSignal.from_dict(signal) for signal in raw_signals),
            alpha=data["alpha"],
            method=str(data.get("method", "by")),
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
    def load_json(cls, path: str | Path) -> "MultipleTestingConformalArtifact":
        """Load artifact metadata from UTF-8 JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class MultipleTestingConformalCalibrator:
    """Build runtime multiple-testing artifacts from held-out normal scores."""

    alpha: float = 0.1
    method: str = "by"

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _alpha_float(self.alpha))
        object.__setattr__(self, "method", _normalize_method(self.method))

    def calibrate(
        self,
        *,
        model_id: str,
        target_layer: int,
        calibration_scores: Mapping[str, ArrayLike],
        directions: Optional[Mapping[str, str]] = None,
        model_revision: Optional[str] = None,
        warmup_dataset_metadata: Optional[Mapping[str, Any]] = None,
        calibration_dataset_metadata: Optional[Mapping[str, Any]] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
    ) -> MultipleTestingConformalArtifact:
        """Create a multi-signal conformal artifact for one model/layer setting."""
        if len(calibration_scores) == 0:
            raise ValueError("calibration_scores must contain at least one score family.")
        signal_names = tuple(str(name) for name in calibration_scores)
        if len(set(signal_names)) != len(signal_names):
            raise ValueError("score names must be unique after string conversion.")
        raw_directions = {} if directions is None else {str(name): str(value) for name, value in directions.items()}
        extra_directions = sorted(set(raw_directions.keys()) - set(signal_names))
        if extra_directions:
            raise ValueError(f"directions contains unknown signals: {extra_directions}.")

        signals = tuple(
            MultipleTestingConformalSignal(
                name=str(name),
                calibration_scores=scores,
                direction=raw_directions.get(str(name), "higher"),
            )
            for name, scores in calibration_scores.items()
        )
        metadata = dict(calibration_dataset_metadata or {})
        metadata["multiple_testing_conformal"] = {
            "alpha": self.alpha,
            "method": self.method,
            "signals": [
                {
                    "name": signal.name,
                    "direction": signal.direction,
                    "n_calibration": signal.calibration_count,
                }
                for signal in signals
            ],
        }

        return MultipleTestingConformalArtifact(
            model_id=model_id,
            model_revision=model_revision,
            target_layer=target_layer,
            signals=signals,
            alpha=self.alpha,
            method=self.method,
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


def _normalize_method(value: object) -> str:
    method = str(value).strip().lower().replace("_", "-")
    aliases = {
        "by": "by",
        "benjamini-yekutieli": "by",
        "bh": "bh",
        "benjamini-hochberg": "bh",
        "bonferroni": "bonferroni",
    }
    if method not in aliases:
        raise ValueError("method must be 'by', 'bh', or 'bonferroni'.")
    return aliases[method]


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
