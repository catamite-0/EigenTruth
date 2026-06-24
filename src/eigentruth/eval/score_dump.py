"""Validated score-dump utilities for model-free benchmark reuse."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class ScoreDump:
    """A validated ``eval_truthfulqa.py --dump-scores`` payload."""

    labels: tuple[int, ...]
    scores: Mapping[str, tuple[float, ...]]
    config: Mapping[str, Any] = field(default_factory=dict)
    sweep_scores: Mapping[str, Mapping[str, tuple[float, ...]]] = field(default_factory=dict)
    statements: tuple[Mapping[str, Any], ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        allow_empty: bool = False,
        allow_missing_scores: bool = False,
        require_statements: bool = False,
    ) -> "ScoreDump":
        """Build and validate a score dump from a JSON-like mapping."""
        if not isinstance(payload, Mapping):
            raise ValueError("score dump must be a JSON object.")
        labels = _coerce_labels(payload.get("labels"), allow_empty=allow_empty)
        scores = _coerce_score_mapping(
            payload.get("scores"),
            n_labels=len(labels),
            name="scores",
            allow_missing=allow_missing_scores,
        )
        if not scores and not allow_missing_scores:
            raise ValueError("score dump must contain at least one score family.")
        sweep_scores = _coerce_sweep_scores(payload.get("sweep_scores", {}), n_labels=len(labels))
        statements = _coerce_statements(
            payload.get("statements", ()),
            n_labels=len(labels),
            require_statements=require_statements,
        )
        extras = {
            str(key): value
            for key, value in payload.items()
            if key not in {"config", "labels", "scores", "sweep_scores", "statements"}
        }
        return cls(
            labels=labels,
            scores=scores,
            config=dict(_mapping(payload.get("config"))),
            sweep_scores=sweep_scores,
            statements=statements,
            extras=extras,
        )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        *,
        allow_empty: bool = False,
        allow_missing_scores: bool = False,
        require_statements: bool = False,
    ) -> "ScoreDump":
        """Load and validate a score dump from a UTF-8 JSON file."""
        return cls.from_mapping(
            json.loads(Path(path).read_text(encoding="utf-8")),
            allow_empty=allow_empty,
            allow_missing_scores=allow_missing_scores,
            require_statements=require_statements,
        )

    @property
    def n_total(self) -> int:
        """Return the number of scored records."""
        return len(self.labels)

    @property
    def n_true(self) -> int:
        """Return the number of normal/true labels."""
        return sum(1 for label in self.labels if label == 0)

    @property
    def n_false(self) -> int:
        """Return the number of anomalous/false labels."""
        return sum(1 for label in self.labels if label == 1)

    def require_scores(self, names: Sequence[str], *, primary_only: bool = True) -> None:
        """Raise if required score names are missing."""
        available = set(self.scores)
        if not primary_only:
            for layer_scores in self.sweep_scores.values():
                available.update(layer_scores)
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"score dump is missing requested score(s): {missing}.")

    def signal_names(self, *, include_sweep: bool = True) -> tuple[str, ...]:
        """Return sorted score names available in this dump."""
        names = set(self.scores)
        if include_sweep:
            for layer_scores in self.sweep_scores.values():
                names.update(layer_scores)
        return tuple(sorted(names))

    def summary(self) -> dict[str, Any]:
        """Return compact JSON metadata for registry/report provenance."""
        return {
            "n_total": self.n_total,
            "n_true": self.n_true,
            "n_false": self.n_false,
            "score_count": len(self.scores),
            "score_names": self.signal_names(include_sweep=False),
            "sweep_layer_count": len(self.sweep_scores),
            "sweep_layers": tuple(sorted(self.sweep_scores, key=_layer_sort_key)),
            "sweep_score_count": sum(len(layer_scores) for layer_scores in self.sweep_scores.values()),
            "sweep_score_names": tuple(sorted({
                name
                for layer_scores in self.sweep_scores.values()
                for name in layer_scores
            })),
            "all_signal_names": self.signal_names(include_sweep=True),
            "has_statements": bool(self.statements),
            "statement_count": len(self.statements),
            "model": self.config.get("model"),
            "layer": self.config.get("layer"),
        }

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable score dump mapping."""
        payload = dict(self.extras)
        payload.update({
            "config": dict(self.config),
            "labels": list(self.labels),
            "scores": {name: list(values) for name, values in self.scores.items()},
        })
        if self.statements:
            payload["statements"] = [dict(statement) for statement in self.statements]
        if self.sweep_scores:
            payload["sweep_scores"] = {
                str(layer): {name: list(values) for name, values in layer_scores.items()}
                for layer, layer_scores in self.sweep_scores.items()
            }
        return payload


def load_score_dump(
    path: str | Path,
    *,
    required_scores: Sequence[str] = (),
    allow_empty: bool = False,
    allow_missing_scores: bool = False,
    require_statements: bool = False,
    primary_only: bool = True,
) -> ScoreDump:
    """Load a validated score dump and optionally require score names."""
    dump = ScoreDump.load_json(
        path,
        allow_empty=allow_empty,
        allow_missing_scores=allow_missing_scores,
        require_statements=require_statements,
    )
    dump.require_scores(tuple(required_scores), primary_only=primary_only)
    return dump


def score_dump_file_metadata(
    path: str | Path,
    dump: ScoreDump | None = None,
    *,
    cache: MutableMapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a stable local fingerprint plus optional parsed score-dump summary."""
    score_path = Path(path)
    fingerprint = _cached_file_fingerprint(score_path, cache) if score_path.is_file() else {
        "sha256": None,
        "size_bytes": None,
    }
    metadata: dict[str, Any] = {
        "path": str(score_path),
        "exists": score_path.exists(),
        "kind": "file" if score_path.is_file() else ("missing" if not score_path.exists() else "other"),
        **fingerprint,
    }
    if dump is not None:
        metadata["summary"] = dump.summary()
    return metadata


def _coerce_labels(value: Any, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("score dump labels must be a list.")
    labels = tuple(int(label) for label in value)
    if not labels and not allow_empty:
        raise ValueError("score dump labels must be non-empty.")
    invalid = [label for label in labels if label not in {0, 1}]
    if invalid:
        raise ValueError("score dump labels must be binary values in {0, 1}.")
    return labels


def _coerce_score_mapping(
    value: Any,
    *,
    n_labels: int,
    name: str,
    allow_missing: bool = False,
) -> dict[str, tuple[float, ...]]:
    if value is None and allow_missing:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"score dump {name} must be an object.")
    scores: dict[str, tuple[float, ...]] = {}
    for score_name, raw_values in value.items():
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes, bytearray)):
            raise ValueError(f"score {score_name!r} must be a list.")
        values = tuple(float(item) for item in raw_values)
        if len(values) != n_labels:
            raise ValueError(
                f"score {score_name!r} length does not match labels "
                f"({len(values)} scores vs {n_labels} labels)."
            )
        scores[str(score_name)] = values
    return scores


def _coerce_sweep_scores(value: Any, *, n_labels: int) -> dict[str, Mapping[str, tuple[float, ...]]]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("score dump sweep_scores must be an object.")
    sweep_scores = {}
    for layer, layer_scores in value.items():
        sweep_scores[str(layer)] = _coerce_score_mapping(
            layer_scores,
            n_labels=n_labels,
            name=f"sweep_scores[{layer!r}]",
        )
    return sweep_scores


def _coerce_statements(
    value: Any,
    *,
    n_labels: int,
    require_statements: bool,
) -> tuple[Mapping[str, Any], ...]:
    if value in (None, ()):
        if require_statements:
            raise ValueError("score dump statements are required.")
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("score dump statements must be a list.")
    statements = tuple(dict(_mapping(item)) for item in value)
    if len(statements) != n_labels:
        raise ValueError(
            f"score dump statements length does not match labels "
            f"({len(statements)} statements vs {n_labels} labels)."
        )
    return statements


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _layer_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _cached_file_fingerprint(
    path: Path,
    cache: MutableMapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    stat = path.stat()
    if cache is None:
        return {"sha256": _sha256_file(path), "size_bytes": stat.st_size}
    cache_key = f"{path.resolve(strict=False)}:{stat.st_size}:{stat.st_mtime_ns}"
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached)
    fingerprint = {"sha256": _sha256_file(path), "size_bytes": stat.st_size}
    cache[cache_key] = dict(fingerprint)
    return fingerprint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
