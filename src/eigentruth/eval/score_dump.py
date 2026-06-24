"""Validated score-dump utilities for model-free benchmark reuse."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

JSONL_FORMAT = "eigentruth.score_dump.jsonl"


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


@dataclass(frozen=True)
class ScoreDumpColumns:
    """Selected primary score columns loaded from a score dump."""

    labels: tuple[int, ...]
    scores: Mapping[str, tuple[float, ...]]
    config: Mapping[str, Any] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)
    source_format: str = "json"

    @property
    def n_total(self) -> int:
        """Return the number of records in the selected view."""
        return len(self.labels)

    def require_scores(self, names: Sequence[str]) -> None:
        """Raise if required score columns are missing from this view."""
        missing = [name for name in names if name not in self.scores]
        if missing:
            raise ValueError(f"score dump is missing requested score(s): {missing}.")


@dataclass(frozen=True)
class ScoreDumpStatementScores:
    """Selected primary score columns plus optional statement metadata."""

    labels: tuple[int, ...]
    scores: Mapping[str, tuple[float, ...]]
    statements: tuple[Mapping[str, Any], ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)
    source_format: str = "json"

    @property
    def n_total(self) -> int:
        """Return the number of records in the selected statement view."""
        return len(self.labels)

    def require_scores(self, names: Sequence[str]) -> None:
        """Raise if required score columns are missing from this view."""
        missing = [name for name in names if name not in self.scores]
        if missing:
            raise ValueError(f"score dump is missing requested score(s): {missing}.")


@dataclass(frozen=True)
class ScoreDumpLayerScores:
    """Selected score columns grouped by layer."""

    labels: tuple[int, ...]
    layer_scores: Mapping[int, Mapping[str, tuple[float, ...]]]
    score_sources: Mapping[int, Mapping[str, str]] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)
    source_format: str = "json"

    @property
    def n_total(self) -> int:
        """Return the number of records in the selected layer view."""
        return len(self.labels)


@dataclass(frozen=True)
class ScoreDumpRecord:
    """One streaming row from a JSONL score dump."""

    label: int
    scores: Mapping[str, float] = field(default_factory=dict)
    sweep_scores: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    statement: Mapping[str, Any] | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        score_names: Sequence[str] | None = None,
        sweep_score_names: Mapping[str, Sequence[str]] | None = None,
        allow_missing_scores: bool = False,
        require_statement: bool = False,
    ) -> "ScoreDumpRecord":
        """Build and validate one JSONL score-dump record."""
        if not isinstance(payload, Mapping):
            raise ValueError("score dump JSONL record must be a JSON object.")
        try:
            label = int(payload.get("label"))
        except (TypeError, ValueError) as exc:
            raise ValueError("score dump JSONL record label must be an integer.") from exc
        if label not in {0, 1}:
            raise ValueError("score dump JSONL record label must be binary values in {0, 1}.")
        scores = _coerce_record_scores(
            payload.get("scores"),
            score_names=score_names,
            allow_missing_scores=allow_missing_scores,
        )
        sweep_scores = _coerce_record_sweep_scores(
            payload.get("sweep_scores", {}),
            sweep_score_names=sweep_score_names,
        )
        raw_statement = payload.get("statement")
        if raw_statement is None and require_statement:
            raise ValueError("score dump JSONL record statement is required.")
        statement = None if raw_statement is None else dict(_required_mapping(raw_statement, "statement"))
        extras = {
            str(key): value
            for key, value in payload.items()
            if key not in {"label", "scores", "sweep_scores", "statement"}
        }
        return cls(
            label=label,
            scores=scores,
            sweep_scores=sweep_scores,
            statement=statement,
            extras=extras,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable row mapping."""
        payload = dict(self.extras)
        payload.update({
            "label": self.label,
            "scores": {name: float(value) for name, value in self.scores.items()},
        })
        if self.sweep_scores:
            payload["sweep_scores"] = {
                str(layer): {name: float(value) for name, value in layer_scores.items()}
                for layer, layer_scores in self.sweep_scores.items()
            }
        if self.statement is not None:
            payload["statement"] = dict(self.statement)
        return payload


@dataclass(frozen=True)
class ScoreDumpJsonlManifest:
    """Manifest for a streaming JSONL score dump."""

    records_path: str
    config: Mapping[str, Any] = field(default_factory=dict)
    score_names: tuple[str, ...] = ()
    sweep_score_names: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    n_total: int | None = None
    has_statements: bool = False
    extras: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    format: str = JSONL_FORMAT

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ScoreDumpJsonlManifest":
        """Build and validate a JSONL manifest from a JSON-like mapping."""
        if not isinstance(payload, Mapping):
            raise ValueError("score dump JSONL manifest must be a JSON object.")
        if payload.get("format") != JSONL_FORMAT:
            raise ValueError(f"score dump JSONL manifest format must be {JSONL_FORMAT!r}.")
        records_path = payload.get("records_path")
        if not isinstance(records_path, str) or not records_path:
            raise ValueError("score dump JSONL manifest records_path must be a non-empty string.")
        n_total = payload.get("n_total")
        parsed_n_total = None if n_total is None else int(n_total)
        if parsed_n_total is not None and parsed_n_total < 0:
            raise ValueError("score dump JSONL manifest n_total must be non-negative.")
        extras_payload = payload.get("extras", {})
        extras = dict(_required_mapping(extras_payload, "extras")) if extras_payload is not None else {}
        return cls(
            records_path=records_path,
            config=dict(_mapping(payload.get("config"))),
            score_names=_coerce_name_tuple(payload.get("score_names", ()), name="score_names"),
            sweep_score_names=_coerce_manifest_sweep_score_names(payload.get("sweep_scores", {})),
            n_total=parsed_n_total,
            has_statements=bool(payload.get("has_statements", False)),
            extras=extras,
            schema_version=int(payload.get("schema_version", 1)),
            format=str(payload.get("format")),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "ScoreDumpJsonlManifest":
        """Load a JSONL manifest from UTF-8 JSON."""
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_score_dump(
        cls,
        dump: ScoreDump,
        *,
        records_path: str,
    ) -> "ScoreDumpJsonlManifest":
        """Build a manifest for an existing in-memory score dump."""
        return cls(
            records_path=records_path,
            config=dict(dump.config),
            score_names=tuple(dump.scores),
            sweep_score_names={
                str(layer): tuple(layer_scores)
                for layer, layer_scores in dump.sweep_scores.items()
            },
            n_total=dump.n_total,
            has_statements=bool(dump.statements),
            extras=dict(dump.extras),
        )

    def records_file(self, manifest_path: str | Path) -> Path:
        """Resolve the records JSONL path relative to the manifest file."""
        records = Path(self.records_path)
        if records.is_absolute():
            return records
        return Path(manifest_path).parent / records

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest mapping."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "format": self.format,
            "records_path": self.records_path,
            "config": dict(self.config),
            "score_names": list(self.score_names),
            "sweep_scores": {
                str(layer): list(score_names)
                for layer, score_names in self.sweep_score_names.items()
            },
            "n_total": self.n_total,
            "has_statements": self.has_statements,
        }
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload

    def save_json(self, path: str | Path) -> None:
        """Save the manifest as UTF-8 JSON."""
        Path(path).write_text(json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    dump = _load_score_dump_path(
        Path(path),
        allow_empty=allow_empty,
        allow_missing_scores=allow_missing_scores,
        require_statements=require_statements,
    )
    dump.require_scores(tuple(required_scores), primary_only=primary_only)
    return dump


def load_score_dump_columns(
    path: str | Path,
    score_names: Sequence[str],
    *,
    allow_empty: bool = False,
    cache: MutableMapping[str, Any] | None = None,
) -> ScoreDumpColumns:
    """Load selected primary score columns without materializing unused JSONL columns."""
    score_path = Path(path)
    requested = tuple(str(name) for name in score_names)
    if not requested:
        raise ValueError("at least one score name is required.")
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    if _is_jsonl_manifest_payload(payload):
        manifest = ScoreDumpJsonlManifest.from_mapping(payload)
        cache_key = _jsonl_view_cache_key(
            score_path,
            manifest,
            view="columns",
            options=(requested, allow_empty),
        )
        cached = _score_dump_view_cache_get(cache, cache_key, ScoreDumpColumns)
        if cached is not None:
            return cached
        columns = _load_score_dump_jsonl_columns(
            score_path,
            manifest,
            score_names=requested,
            allow_empty=allow_empty,
        )
        _score_dump_view_cache_set(cache, cache_key, columns)
        _score_dump_jsonl_summary_cache_set(cache, score_path, manifest, columns.summary)
        return columns
    dump = ScoreDump.from_mapping(payload, allow_empty=allow_empty)
    dump.require_scores(requested)
    return ScoreDumpColumns(
        labels=dump.labels,
        scores={name: dump.scores[name] for name in requested},
        config=dict(dump.config),
        summary=dump.summary(),
    )


def load_score_dump_statement_scores(
    path: str | Path,
    score_names: Sequence[str],
    *,
    allow_empty: bool = False,
    require_statements: bool = False,
    cache: MutableMapping[str, Any] | None = None,
) -> ScoreDumpStatementScores:
    """Load selected primary scores plus statement metadata when present."""
    score_path = Path(path)
    requested = tuple(str(name) for name in score_names)
    if not requested:
        raise ValueError("at least one score name is required.")
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    if _is_jsonl_manifest_payload(payload):
        manifest = ScoreDumpJsonlManifest.from_mapping(payload)
        cache_key = _jsonl_view_cache_key(
            score_path,
            manifest,
            view="statement_scores",
            options=(requested, allow_empty, require_statements),
        )
        cached = _score_dump_view_cache_get(cache, cache_key, ScoreDumpStatementScores)
        if cached is not None:
            return cached
        statement_scores = _load_score_dump_jsonl_statement_scores(
            score_path,
            manifest,
            score_names=requested,
            allow_empty=allow_empty,
            require_statements=require_statements,
        )
        _score_dump_view_cache_set(cache, cache_key, statement_scores)
        _score_dump_jsonl_summary_cache_set(cache, score_path, manifest, statement_scores.summary)
        return statement_scores
    dump = ScoreDump.from_mapping(
        payload,
        allow_empty=allow_empty,
        require_statements=require_statements,
    )
    dump.require_scores(requested)
    return ScoreDumpStatementScores(
        labels=dump.labels,
        scores={name: dump.scores[name] for name in requested},
        statements=dump.statements,
        config=dict(dump.config),
        summary=dump.summary(),
    )


def load_score_dump_layer_scores(
    path: str | Path,
    *,
    signals: Sequence[str] | None = None,
    allow_empty: bool = False,
    cache: MutableMapping[str, Any] | None = None,
) -> ScoreDumpLayerScores:
    """Load score columns grouped by layer, filtering JSONL inputs by signal name."""
    score_path = Path(path)
    selected = None if signals is None else {str(signal) for signal in signals}
    if selected is not None and not selected:
        raise ValueError("signals must contain at least one signal name when provided.")
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    if _is_jsonl_manifest_payload(payload):
        manifest = ScoreDumpJsonlManifest.from_mapping(payload)
        cache_key = _jsonl_view_cache_key(
            score_path,
            manifest,
            view="layer_scores",
            options=(tuple(sorted(selected)) if selected is not None else None, allow_empty),
        )
        cached = _score_dump_view_cache_get(cache, cache_key, ScoreDumpLayerScores)
        if cached is not None:
            return cached
        layer_scores = _load_score_dump_jsonl_layer_scores(
            score_path,
            manifest,
            signals=selected,
            allow_empty=allow_empty,
        )
        _score_dump_view_cache_set(cache, cache_key, layer_scores)
        _score_dump_jsonl_summary_cache_set(cache, score_path, manifest, layer_scores.summary)
        return layer_scores
    dump = ScoreDump.from_mapping(payload, allow_empty=allow_empty)
    return _score_dump_layer_scores_from_score_dump(dump, signals=selected)


def iter_score_dump_jsonl_records(
    manifest_path: str | Path,
    *,
    allow_missing_scores: bool = False,
    require_statements: bool = False,
) -> Iterator[ScoreDumpRecord]:
    """Iterate validated JSONL score-dump records from a manifest."""
    manifest = ScoreDumpJsonlManifest.load_json(manifest_path)
    yield from _iter_score_dump_jsonl_records(
        manifest_path=Path(manifest_path),
        manifest=manifest,
        allow_missing_scores=allow_missing_scores,
        require_statements=require_statements,
    )


def write_score_dump_jsonl(
    dump: ScoreDump,
    manifest_path: str | Path,
    *,
    records_path: str | Path | None = None,
    record_extra_names: Sequence[str] = (),
) -> ScoreDumpJsonlManifest:
    """Write an in-memory score dump as JSONL records plus a manifest.

    ``record_extra_names`` moves length-matched dump extras into each JSONL row
    instead of keeping large per-record arrays in the manifest.
    """
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    if records_path is None:
        records_file = manifest_file.with_suffix(".records.jsonl")
    else:
        records_file = Path(records_path)
    if records_path is not None and not records_file.is_absolute():
        records_file = manifest_file.parent / records_file
    records_file.parent.mkdir(parents=True, exist_ok=True)

    relative_records_path = _manifest_records_path(manifest_file, records_file)
    record_extra_columns = _record_extra_columns(
        dump.extras,
        record_extra_names=record_extra_names,
        n_total=dump.n_total,
    )
    manifest_extras = {
        name: value
        for name, value in dump.extras.items()
        if name not in record_extra_columns
    }
    manifest = ScoreDumpJsonlManifest(
        records_path=relative_records_path,
        config=dict(dump.config),
        score_names=tuple(dump.scores),
        sweep_score_names={
            str(layer): tuple(layer_scores)
            for layer, layer_scores in dump.sweep_scores.items()
        },
        n_total=dump.n_total,
        has_statements=bool(dump.statements),
        extras=manifest_extras,
    )
    with records_file.open("w", encoding="utf-8") as stream:
        for index, label in enumerate(dump.labels):
            record = ScoreDumpRecord(
                label=label,
                scores={name: values[index] for name, values in dump.scores.items()},
                sweep_scores={
                    str(layer): {name: values[index] for name, values in layer_scores.items()}
                    for layer, layer_scores in dump.sweep_scores.items()
                },
                statement=dump.statements[index] if dump.statements else None,
                extras={name: values[index] for name, values in record_extra_columns.items()},
            )
            stream.write(json.dumps(record.to_mapping(), sort_keys=True) + "\n")
    manifest.save_json(manifest_file)
    return manifest


def score_dump_file_metadata(
    path: str | Path,
    dump: ScoreDump | None = None,
    *,
    cache: MutableMapping[str, Any] | None = None,
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
    manifest = _metadata_jsonl_manifest(score_path)
    if manifest is not None:
        records_file = manifest.records_file(score_path)
        records_fingerprint = (
            _cached_file_fingerprint(records_file, cache)
            if records_file.is_file()
            else {"sha256": None, "size_bytes": None}
        )
        metadata.update({
            "source_format": JSONL_FORMAT,
            "records": {
                "path": str(records_file),
                "exists": records_file.exists(),
                "kind": (
                    "file"
                    if records_file.is_file()
                    else ("missing" if not records_file.exists() else "other")
                ),
                **records_fingerprint,
            },
        })
        if dump is None and records_file.is_file():
            metadata["summary"] = _cached_jsonl_manifest_summary(
                score_path,
                manifest,
                cache,
            )
    if dump is not None:
        metadata["summary"] = dump.summary()
    return metadata


def _load_score_dump_path(
    path: Path,
    *,
    allow_empty: bool,
    allow_missing_scores: bool,
    require_statements: bool,
) -> ScoreDump:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _is_jsonl_manifest_payload(payload):
        return _load_score_dump_jsonl_manifest(
            path,
            ScoreDumpJsonlManifest.from_mapping(payload),
            allow_empty=allow_empty,
            allow_missing_scores=allow_missing_scores,
            require_statements=require_statements,
        )
    return ScoreDump.from_mapping(
        payload,
        allow_empty=allow_empty,
        allow_missing_scores=allow_missing_scores,
        require_statements=require_statements,
    )


def _is_jsonl_manifest_payload(payload: Any) -> bool:
    return isinstance(payload, Mapping) and payload.get("format") == JSONL_FORMAT


def _metadata_jsonl_manifest(path: Path) -> ScoreDumpJsonlManifest | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            head = stream.read(64 * 1024)
        if JSONL_FORMAT not in head:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _is_jsonl_manifest_payload(payload):
            return None
        return ScoreDumpJsonlManifest.from_mapping(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _load_score_dump_jsonl_manifest(
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    *,
    allow_empty: bool,
    allow_missing_scores: bool,
    require_statements: bool,
) -> ScoreDump:
    labels: list[int] = []
    scores = {name: [] for name in manifest.score_names}
    sweep_scores = {
        str(layer): {name: [] for name in score_names}
        for layer, score_names in manifest.sweep_score_names.items()
    }
    statements: list[Mapping[str, Any]] = []
    record_extras: dict[str, list[Any]] = {}
    saw_statement = False
    missing_statement = False

    for record in _iter_score_dump_jsonl_records(
        manifest_path=manifest_path,
        manifest=manifest,
        allow_missing_scores=allow_missing_scores,
        require_statements=require_statements,
    ):
        labels.append(record.label)
        for name in manifest.score_names:
            scores[name].append(record.scores[name])
        for layer, layer_score_names in manifest.sweep_score_names.items():
            for name in layer_score_names:
                sweep_scores[str(layer)][name].append(record.sweep_scores[str(layer)][name])
        if record.statement is None:
            missing_statement = True
        else:
            saw_statement = True
            statements.append(record.statement)
        index = len(labels) - 1
        for values in record_extras.values():
            values.append(None)
        for name, value in record.extras.items():
            values = record_extras.setdefault(name, [None] * (index + 1))
            values[index] = value

    if saw_statement and missing_statement:
        raise ValueError("score dump JSONL records must either all include statements or none do.")

    payload: dict[str, Any] = dict(manifest.extras)
    payload.update(record_extras)
    payload.update({
        "config": dict(manifest.config),
        "labels": labels,
        "scores": scores,
        "sweep_scores": sweep_scores,
    })
    if saw_statement:
        payload["statements"] = statements
    return ScoreDump.from_mapping(
        payload,
        allow_empty=allow_empty,
        allow_missing_scores=allow_missing_scores,
        require_statements=require_statements,
    )


def _load_score_dump_jsonl_columns(
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    *,
    score_names: Sequence[str],
    allow_empty: bool,
) -> ScoreDumpColumns:
    missing = [name for name in score_names if name not in manifest.score_names]
    if missing:
        raise ValueError(f"score dump is missing requested score(s): {missing}.")

    labels: list[int] = []
    scores = {name: [] for name in score_names}
    for label, record_scores, _ in _iter_score_dump_jsonl_selected_records(
        manifest_path=manifest_path,
        manifest=manifest,
        score_names=score_names,
        sweep_score_names={},
    ):
        labels.append(label)
        for name in score_names:
            scores[name].append(record_scores[name])

    if not labels and not allow_empty:
        raise ValueError("score dump labels must be non-empty.")
    label_tuple = tuple(labels)
    return ScoreDumpColumns(
        labels=label_tuple,
        scores={name: tuple(values) for name, values in scores.items()},
        config=dict(manifest.config),
        summary=_jsonl_manifest_summary(manifest, labels=label_tuple),
        source_format=JSONL_FORMAT,
    )


def _load_score_dump_jsonl_statement_scores(
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    *,
    score_names: Sequence[str],
    allow_empty: bool,
    require_statements: bool,
) -> ScoreDumpStatementScores:
    missing = [name for name in score_names if name not in manifest.score_names]
    if missing:
        raise ValueError(f"score dump is missing requested score(s): {missing}.")

    labels: list[int] = []
    scores = {name: [] for name in score_names}
    statements: list[Mapping[str, Any]] = []
    saw_statement = False
    missing_statement = False
    for label, record_scores, statement in _iter_score_dump_jsonl_selected_statement_records(
        manifest_path=manifest_path,
        manifest=manifest,
        score_names=score_names,
        require_statements=require_statements,
    ):
        labels.append(label)
        for name in score_names:
            scores[name].append(record_scores[name])
        if statement is None:
            missing_statement = True
        else:
            saw_statement = True
            statements.append(statement)

    if saw_statement and missing_statement:
        raise ValueError("score dump JSONL records must either all include statements or none do.")
    if not labels and not allow_empty:
        raise ValueError("score dump labels must be non-empty.")

    label_tuple = tuple(labels)
    return ScoreDumpStatementScores(
        labels=label_tuple,
        scores={name: tuple(values) for name, values in scores.items()},
        statements=tuple(statements) if saw_statement else (),
        config=dict(manifest.config),
        summary=_jsonl_manifest_summary(manifest, labels=label_tuple),
        source_format=JSONL_FORMAT,
    )


def _load_score_dump_jsonl_layer_scores(
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    *,
    signals: set[str] | None,
    allow_empty: bool,
) -> ScoreDumpLayerScores:
    primary_names = tuple(
        name for name in manifest.score_names
        if signals is None or name in signals
    )
    sweep_names = {
        str(layer): tuple(name for name in score_names if signals is None or name in signals)
        for layer, score_names in manifest.sweep_score_names.items()
    }
    sweep_names = {layer: names for layer, names in sweep_names.items() if names}
    if signals is not None and not primary_names and not sweep_names:
        raise ValueError("no matching score signals were found in the score dump.")

    labels: list[int] = []
    primary_scores = {name: [] for name in primary_names}
    sweep_scores = {
        str(layer): {name: [] for name in score_names}
        for layer, score_names in sweep_names.items()
    }
    for label, record_scores, record_sweep_scores in _iter_score_dump_jsonl_selected_records(
        manifest_path=manifest_path,
        manifest=manifest,
        score_names=primary_names,
        sweep_score_names=sweep_names,
    ):
        labels.append(label)
        for name in primary_names:
            primary_scores[name].append(record_scores[name])
        for layer, layer_score_names in sweep_names.items():
            for name in layer_score_names:
                sweep_scores[str(layer)][name].append(record_sweep_scores[str(layer)][name])

    if not labels and not allow_empty:
        raise ValueError("score dump labels must be non-empty.")

    primary_layer = int(manifest.config.get("layer", 0))
    layer_scores: dict[int, dict[str, tuple[float, ...]]] = {}
    score_sources: dict[int, dict[str, str]] = {}
    if primary_scores:
        layer_scores[primary_layer] = {name: tuple(values) for name, values in primary_scores.items()}
        score_sources[primary_layer] = {name: "scores" for name in primary_scores}
    for layer_key, layer_score_values in sweep_scores.items():
        layer = int(layer_key)
        layer_scores.setdefault(layer, {}).update({
            name: tuple(values)
            for name, values in layer_score_values.items()
        })
        score_sources.setdefault(layer, {}).update({
            name: "sweep_scores"
            for name in layer_score_values
        })

    label_tuple = tuple(labels)
    return ScoreDumpLayerScores(
        labels=label_tuple,
        layer_scores=layer_scores,
        score_sources=score_sources,
        config=dict(manifest.config),
        summary=_jsonl_manifest_summary(manifest, labels=label_tuple),
        source_format=JSONL_FORMAT,
    )


def _load_score_dump_jsonl_labels(
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
) -> tuple[int, ...]:
    labels: list[int] = []
    records_file = manifest.records_file(manifest_path)
    count = 0
    with records_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("record must be a JSON object.")
                labels.append(_coerce_record_label(payload.get("label")))
            except Exception as exc:
                raise ValueError(
                    f"invalid score dump JSONL record at {records_file}:{line_number}: {exc}"
                ) from exc
            count += 1
    if manifest.n_total is not None and count != manifest.n_total:
        raise ValueError(
            f"score dump JSONL record count does not match manifest "
            f"({count} records vs {manifest.n_total} expected)."
        )
    return tuple(labels)


def _score_dump_layer_scores_from_score_dump(
    dump: ScoreDump,
    *,
    signals: set[str] | None,
) -> ScoreDumpLayerScores:
    primary_layer = int(dump.config.get("layer", 0))
    layer_scores: dict[int, dict[str, tuple[float, ...]]] = {}
    score_sources: dict[int, dict[str, str]] = {}
    selected_primary = {
        name: values
        for name, values in dump.scores.items()
        if signals is None or name in signals
    }
    if selected_primary:
        layer_scores[primary_layer] = dict(selected_primary)
        score_sources[primary_layer] = {name: "scores" for name in selected_primary}
    for layer_key, raw_layer_scores in dump.sweep_scores.items():
        layer = int(layer_key)
        selected_scores = {
            name: values
            for name, values in raw_layer_scores.items()
            if signals is None or name in signals
        }
        if not selected_scores:
            continue
        layer_scores.setdefault(layer, {}).update(selected_scores)
        score_sources.setdefault(layer, {}).update({
            name: "sweep_scores"
            for name in selected_scores
        })
    if signals is not None and not layer_scores:
        raise ValueError("no matching score signals were found in the score dump.")
    return ScoreDumpLayerScores(
        labels=dump.labels,
        layer_scores=layer_scores,
        score_sources=score_sources,
        config=dict(dump.config),
        summary=dump.summary(),
    )


def _jsonl_manifest_summary(
    manifest: ScoreDumpJsonlManifest,
    *,
    labels: Sequence[int],
) -> dict[str, Any]:
    sweep_layers = tuple(sorted((str(layer) for layer in manifest.sweep_score_names), key=_layer_sort_key))
    sweep_score_names = tuple(sorted({
        name
        for layer_scores in manifest.sweep_score_names.values()
        for name in layer_scores
    }))
    score_names = tuple(manifest.score_names)
    return {
        "n_total": len(labels),
        "n_true": sum(1 for label in labels if label == 0),
        "n_false": sum(1 for label in labels if label == 1),
        "score_count": len(score_names),
        "score_names": score_names,
        "sweep_layer_count": len(manifest.sweep_score_names),
        "sweep_layers": sweep_layers,
        "sweep_score_count": sum(len(score_names) for score_names in manifest.sweep_score_names.values()),
        "sweep_score_names": sweep_score_names,
        "all_signal_names": tuple(sorted(set(score_names).union(sweep_score_names))),
        "has_statements": bool(manifest.has_statements),
        "statement_count": len(labels) if manifest.has_statements else 0,
        "model": manifest.config.get("model"),
        "layer": manifest.config.get("layer"),
    }


def _cached_jsonl_manifest_summary(
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    cache: MutableMapping[str, Any] | None,
) -> dict[str, Any]:
    if cache is None:
        return _jsonl_manifest_summary(
            manifest,
            labels=_load_score_dump_jsonl_labels(manifest_path, manifest),
        )

    cache_key = _jsonl_summary_cache_key(manifest_path, manifest)
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached)
    summary = _jsonl_manifest_summary(
        manifest,
        labels=_load_score_dump_jsonl_labels(manifest_path, manifest),
    )
    cache[cache_key] = dict(summary)
    return summary


def _score_dump_jsonl_summary_cache_set(
    cache: MutableMapping[str, Any] | None,
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    summary: Mapping[str, Any],
) -> None:
    if cache is not None:
        cache[_jsonl_summary_cache_key(manifest_path, manifest)] = dict(summary)


def _jsonl_summary_cache_key(
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
) -> str:
    records_file = manifest.records_file(manifest_path)
    payload = {
        "manifest_path": str(manifest_path.resolve(strict=False)),
        "manifest_signature": _file_cache_signature(manifest_path),
        "records_path": str(records_file.resolve(strict=False)),
        "records_signature": _file_cache_signature(records_file),
        "view": "summary",
    }
    return "score-dump-jsonl-summary-v1:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _jsonl_view_cache_key(
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    *,
    view: str,
    options: Sequence[Any],
) -> str:
    records_file = manifest.records_file(manifest_path)
    payload = {
        "manifest_path": str(manifest_path.resolve(strict=False)),
        "manifest_signature": _file_cache_signature(manifest_path),
        "records_path": str(records_file.resolve(strict=False)),
        "records_signature": _file_cache_signature(records_file),
        "score_names": tuple(manifest.score_names),
        "sweep_score_names": {
            str(layer): tuple(score_names)
            for layer, score_names in manifest.sweep_score_names.items()
        },
        "n_total": manifest.n_total,
        "has_statements": manifest.has_statements,
        "view": view,
        "options": tuple(options),
    }
    return "score-dump-jsonl-view-v1:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _score_dump_view_cache_get(
    cache: MutableMapping[str, Any] | None,
    key: str,
    expected_type: type,
) -> Any | None:
    if cache is None:
        return None
    cached = cache.get(key)
    if isinstance(cached, expected_type):
        return cached
    return None


def _score_dump_view_cache_set(
    cache: MutableMapping[str, Any] | None,
    key: str,
    value: Any,
) -> None:
    if cache is not None:
        cache[key] = value


def _file_cache_signature(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"exists": False}
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _iter_score_dump_jsonl_records(
    *,
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    allow_missing_scores: bool,
    require_statements: bool,
) -> Iterator[ScoreDumpRecord]:
    records_file = manifest.records_file(manifest_path)
    count = 0
    with records_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = ScoreDumpRecord.from_mapping(
                    payload,
                    score_names=manifest.score_names,
                    sweep_score_names=manifest.sweep_score_names,
                    allow_missing_scores=allow_missing_scores,
                    require_statement=require_statements or manifest.has_statements,
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid score dump JSONL record at {records_file}:{line_number}: {exc}"
                ) from exc
            count += 1
            yield record
    if manifest.n_total is not None and count != manifest.n_total:
        raise ValueError(
            f"score dump JSONL record count does not match manifest "
            f"({count} records vs {manifest.n_total} expected)."
        )


def _iter_score_dump_jsonl_selected_records(
    *,
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    score_names: Sequence[str],
    sweep_score_names: Mapping[str, Sequence[str]],
) -> Iterator[tuple[int, dict[str, float], dict[str, dict[str, float]]]]:
    records_file = manifest.records_file(manifest_path)
    selected_score_names = tuple(str(name) for name in score_names)
    selected_sweep_score_names = {
        str(layer): tuple(str(name) for name in names)
        for layer, names in sweep_score_names.items()
        if names
    }
    count = 0
    with records_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("record must be a JSON object.")
                label = _coerce_record_label(payload.get("label"))
                scores = _selected_record_scores(
                    payload.get("scores"),
                    score_names=selected_score_names,
                )
                sweep_scores = _selected_record_sweep_scores(
                    payload.get("sweep_scores", {}),
                    sweep_score_names=selected_sweep_score_names,
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid score dump JSONL record at {records_file}:{line_number}: {exc}"
                ) from exc
            count += 1
            yield label, scores, sweep_scores
    if manifest.n_total is not None and count != manifest.n_total:
        raise ValueError(
            f"score dump JSONL record count does not match manifest "
            f"({count} records vs {manifest.n_total} expected)."
        )


def _iter_score_dump_jsonl_selected_statement_records(
    *,
    manifest_path: Path,
    manifest: ScoreDumpJsonlManifest,
    score_names: Sequence[str],
    require_statements: bool,
) -> Iterator[tuple[int, dict[str, float], Mapping[str, Any] | None]]:
    records_file = manifest.records_file(manifest_path)
    selected_score_names = tuple(str(name) for name in score_names)
    count = 0
    require_statement = require_statements or manifest.has_statements
    with records_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("record must be a JSON object.")
                label = _coerce_record_label(payload.get("label"))
                scores = _selected_record_scores(
                    payload.get("scores"),
                    score_names=selected_score_names,
                )
                statement = _selected_record_statement(
                    payload.get("statement"),
                    require_statement=require_statement,
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid score dump JSONL record at {records_file}:{line_number}: {exc}"
                ) from exc
            count += 1
            yield label, scores, statement
    if manifest.n_total is not None and count != manifest.n_total:
        raise ValueError(
            f"score dump JSONL record count does not match manifest "
            f"({count} records vs {manifest.n_total} expected)."
        )


def _coerce_name_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"score dump JSONL manifest {name} must be a list.")
    names = tuple(str(item) for item in value)
    if any(not item for item in names):
        raise ValueError(f"score dump JSONL manifest {name} cannot contain empty names.")
    if len(set(names)) != len(names):
        raise ValueError(f"score dump JSONL manifest {name} cannot contain duplicates.")
    return names


def _coerce_manifest_sweep_score_names(value: Any) -> dict[str, tuple[str, ...]]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("score dump JSONL manifest sweep_scores must be an object.")
    return {
        str(layer): _coerce_name_tuple(score_names, name=f"sweep_scores[{layer!r}]")
        for layer, score_names in value.items()
    }


def _coerce_record_label(value: Any) -> int:
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("score dump JSONL record label must be an integer.") from exc
    if label not in {0, 1}:
        raise ValueError("score dump JSONL record label must be binary values in {0, 1}.")
    return label


def _selected_record_scores(
    value: Any,
    *,
    score_names: Sequence[str],
) -> dict[str, float]:
    if not score_names:
        return {}
    raw_scores = {
        str(name): raw_value
        for name, raw_value in _required_mapping(value, "scores").items()
    }
    missing = sorted(set(score_names) - set(raw_scores))
    if missing:
        raise ValueError(f"score dump JSONL record is missing score(s): {missing}.")
    return {name: float(raw_scores[name]) for name in score_names}


def _selected_record_sweep_scores(
    value: Any,
    *,
    sweep_score_names: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    if not sweep_score_names:
        return {}
    raw_sweep_scores = {
        str(layer): layer_scores
        for layer, layer_scores in _required_mapping(value, "sweep_scores").items()
    }
    missing_layers = sorted(set(sweep_score_names) - set(raw_sweep_scores))
    if missing_layers:
        raise ValueError(f"score dump JSONL record is missing sweep layer(s): {missing_layers}.")

    selected: dict[str, dict[str, float]] = {}
    for layer, expected_names in sweep_score_names.items():
        raw_layer_scores = {
            str(name): raw_value
            for name, raw_value in _required_mapping(
                raw_sweep_scores[str(layer)],
                f"sweep_scores[{str(layer)!r}]",
            ).items()
        }
        missing_scores = sorted(set(expected_names) - set(raw_layer_scores))
        if missing_scores:
            raise ValueError(
                f"score dump JSONL record is missing sweep score(s) for layer {str(layer)!r}: "
                f"{missing_scores}."
            )
        selected[str(layer)] = {
            name: float(raw_layer_scores[name])
            for name in expected_names
        }
    return selected


def _selected_record_statement(
    value: Any,
    *,
    require_statement: bool,
) -> Mapping[str, Any] | None:
    if value is None:
        if require_statement:
            raise ValueError("score dump JSONL record statement is required.")
        return None
    return dict(_required_mapping(value, "statement"))


def _coerce_record_scores(
    value: Any,
    *,
    score_names: Sequence[str] | None,
    allow_missing_scores: bool,
) -> dict[str, float]:
    if value is None:
        if allow_missing_scores and not score_names:
            return {}
        raise ValueError("score dump JSONL record scores must be an object.")
    if not isinstance(value, Mapping):
        raise ValueError("score dump JSONL record scores must be an object.")
    raw_scores = {str(name): raw_value for name, raw_value in value.items()}
    if score_names is not None:
        expected = set(score_names)
        actual = set(raw_scores)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            raise ValueError(f"score dump JSONL record is missing score(s): {missing}.")
        if extra:
            raise ValueError(f"score dump JSONL record has unexpected score(s): {extra}.")
    return {name: float(raw_scores[name]) for name in raw_scores}


def _coerce_record_sweep_scores(
    value: Any,
    *,
    sweep_score_names: Mapping[str, Sequence[str]] | None,
) -> dict[str, Mapping[str, float]]:
    if value in (None, {}):
        raw_sweep_scores: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw_sweep_scores = {str(layer): layer_scores for layer, layer_scores in value.items()}
    else:
        raise ValueError("score dump JSONL record sweep_scores must be an object.")

    if sweep_score_names is None:
        return {
            str(layer): {
                str(name): float(raw_value)
                for name, raw_value in _required_mapping(layer_scores, f"sweep_scores[{layer!r}]").items()
            }
            for layer, layer_scores in raw_sweep_scores.items()
        }

    expected_layers = {str(layer) for layer in sweep_score_names}
    actual_layers = set(raw_sweep_scores)
    missing_layers = sorted(expected_layers - actual_layers)
    extra_layers = sorted(actual_layers - expected_layers)
    if missing_layers:
        raise ValueError(f"score dump JSONL record is missing sweep layer(s): {missing_layers}.")
    if extra_layers:
        raise ValueError(f"score dump JSONL record has unexpected sweep layer(s): {extra_layers}.")

    sweep_scores: dict[str, Mapping[str, float]] = {}
    for layer, expected_names in sweep_score_names.items():
        layer_key = str(layer)
        raw_layer_scores = {
            str(name): raw_value
            for name, raw_value in _required_mapping(
                raw_sweep_scores[layer_key],
                f"sweep_scores[{layer_key!r}]",
            ).items()
        }
        expected = set(expected_names)
        actual = set(raw_layer_scores)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            raise ValueError(
                f"score dump JSONL record is missing sweep score(s) for layer {layer_key!r}: {missing}."
            )
        if extra:
            raise ValueError(
                f"score dump JSONL record has unexpected sweep score(s) for layer {layer_key!r}: {extra}."
            )
        sweep_scores[layer_key] = {
            name: float(raw_layer_scores[name])
            for name in raw_layer_scores
        }
    return sweep_scores


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"score dump JSONL record {name} must be an object.")
    return value


def _record_extra_columns(
    extras: Mapping[str, Any],
    *,
    record_extra_names: Sequence[str],
    n_total: int,
) -> dict[str, tuple[Any, ...]]:
    columns: dict[str, tuple[Any, ...]] = {}
    for raw_name in record_extra_names:
        name = str(raw_name)
        if name not in extras:
            raise ValueError(f"record extra {name!r} is missing from score dump extras.")
        values = extras[name]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ValueError(f"record extra {name!r} must be a list with one value per record.")
        column = tuple(values)
        if len(column) != n_total:
            raise ValueError(
                f"record extra {name!r} length does not match labels "
                f"({len(column)} values vs {n_total} labels)."
            )
        columns[name] = column
    return columns


def _manifest_records_path(manifest_file: Path, records_file: Path) -> str:
    try:
        return str(records_file.resolve().relative_to(manifest_file.parent.resolve()))
    except ValueError:
        return str(records_file)


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
    cache: MutableMapping[str, Any] | None,
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
