"""Budgeted hidden-evidence selection from diagnostic score dumps."""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.eval.score_dump import ScoreDump, load_score_dump
from eigentruth.json_utils import strict_json_dumps, to_jsonable

DEFAULT_STATEMENT_METADATA_KEYS = (
    "record_id",
    "claim_id",
    "statement_id",
    "question_id",
    "id",
    "question",
    "answer",
    "text",
)


@dataclass(frozen=True)
class HiddenEvidenceCandidate:
    """One hidden-state or diagnostic-score evidence candidate."""

    record_id: str
    score_name: str
    score: float
    direction: str = "higher"
    layer: str | int | None = None
    source: str = "candidate"
    record_index: int | None = None
    evidence_ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        record_id = str(self.record_id).strip()
        score_name = str(self.score_name).strip()
        if not record_id:
            raise ValueError("record_id must be non-empty.")
        if not score_name:
            raise ValueError("score_name must be non-empty.")
        score = _finite_float(self.score, name="score")
        direction = _direction(self.direction)
        layer = _optional_non_empty_str(self.layer)
        source = str(self.source).strip() or "candidate"
        record_index = _optional_non_negative_int(self.record_index, name="record_index")
        evidence_ref = (
            _candidate_evidence_ref(
                source=source,
                layer=layer,
                score_name=score_name,
                record_id=record_id,
            )
            if self.evidence_ref is None
            else str(self.evidence_ref).strip()
        )
        if not evidence_ref:
            raise ValueError("evidence_ref must be non-empty.")
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "score_name", score_name)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "record_index", record_index)
        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def channel_key(self) -> str:
        """Return the normalization channel for this candidate."""
        layer = self.layer if self.layer is not None else "primary"
        return f"{self.source}:{layer}:{self.score_name}"

    def native_anomaly_value(self) -> float:
        """Return a direction-aligned score where larger is more anomalous."""
        return self.score if self.direction == "higher" else -self.score

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "record_id": self.record_id,
            "record_index": self.record_index,
            "score_name": self.score_name,
            "score": self.score,
            "direction": self.direction,
            "layer": self.layer,
            "source": self.source,
            "evidence_ref": self.evidence_ref,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HiddenEvidenceCandidate":
        """Build a candidate from JSON-like data."""
        return cls(
            record_id=str(data["record_id"]),
            record_index=None if data.get("record_index") is None else int(data["record_index"]),
            score_name=str(data["score_name"]),
            score=data["score"],
            direction=str(data.get("direction", "higher")),
            layer=data.get("layer"),
            source=str(data.get("source", "candidate")),
            evidence_ref=None if data.get("evidence_ref") is None else str(data["evidence_ref"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class HiddenEvidenceSelectionPolicy:
    """Budget policy for sparse hidden-evidence selection."""

    max_items: int = 32
    max_per_record: int | None = 4
    max_per_layer: int | None = None
    max_per_score: int | None = None
    min_anomaly_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_items", _positive_int(self.max_items, name="max_items"))
        object.__setattr__(
            self,
            "max_per_record",
            _optional_positive_int(self.max_per_record, name="max_per_record"),
        )
        object.__setattr__(
            self,
            "max_per_layer",
            _optional_positive_int(self.max_per_layer, name="max_per_layer"),
        )
        object.__setattr__(
            self,
            "max_per_score",
            _optional_positive_int(self.max_per_score, name="max_per_score"),
        )
        if self.min_anomaly_score is not None:
            min_anomaly_score = _finite_float(self.min_anomaly_score, name="min_anomaly_score")
            if not (0.0 <= min_anomaly_score <= 1.0):
                raise ValueError("min_anomaly_score must be in [0, 1].")
            object.__setattr__(self, "min_anomaly_score", min_anomaly_score)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "max_items": self.max_items,
            "max_per_record": self.max_per_record,
            "max_per_layer": self.max_per_layer,
            "max_per_score": self.max_per_score,
            "min_anomaly_score": self.min_anomaly_score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HiddenEvidenceSelectionPolicy":
        """Build a policy from JSON-like data."""
        return cls(
            max_items=data.get("max_items", 32),
            max_per_record=data.get("max_per_record", 4),
            max_per_layer=data.get("max_per_layer"),
            max_per_score=data.get("max_per_score"),
            min_anomaly_score=data.get("min_anomaly_score"),
        )


@dataclass(frozen=True)
class HiddenEvidenceSelection:
    """One selected evidence item with channel-normalized anomaly rank."""

    record_id: str
    score_name: str
    score: float
    direction: str
    anomaly_score: float
    rank: int
    global_rank: int
    channel_rank: int
    channel_size: int
    layer: str | None = None
    source: str = "candidate"
    record_index: int | None = None
    evidence_ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        record_id = str(self.record_id).strip()
        score_name = str(self.score_name).strip()
        if not record_id:
            raise ValueError("record_id must be non-empty.")
        if not score_name:
            raise ValueError("score_name must be non-empty.")
        score = _finite_float(self.score, name="score")
        anomaly_score = _finite_float(self.anomaly_score, name="anomaly_score")
        if not (0.0 <= anomaly_score <= 1.0):
            raise ValueError("anomaly_score must be in [0, 1].")
        rank = _positive_int(self.rank, name="rank")
        global_rank = _positive_int(self.global_rank, name="global_rank")
        channel_rank = _positive_int(self.channel_rank, name="channel_rank")
        channel_size = _positive_int(self.channel_size, name="channel_size")
        if channel_rank > channel_size:
            raise ValueError("channel_rank must be <= channel_size.")
        layer = _optional_non_empty_str(self.layer)
        source = str(self.source).strip() or "candidate"
        record_index = _optional_non_negative_int(self.record_index, name="record_index")
        evidence_ref = (
            _candidate_evidence_ref(
                source=source,
                layer=layer,
                score_name=score_name,
                record_id=record_id,
            )
            if self.evidence_ref is None
            else str(self.evidence_ref).strip()
        )
        if not evidence_ref:
            raise ValueError("evidence_ref must be non-empty.")
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "score_name", score_name)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "direction", _direction(self.direction))
        object.__setattr__(self, "anomaly_score", anomaly_score)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "global_rank", global_rank)
        object.__setattr__(self, "channel_rank", channel_rank)
        object.__setattr__(self, "channel_size", channel_size)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "record_index", record_index)
        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_candidate(
        cls,
        candidate: HiddenEvidenceCandidate,
        *,
        anomaly_score: float,
        rank: int,
        global_rank: int,
        channel_rank: int,
        channel_size: int,
    ) -> "HiddenEvidenceSelection":
        """Build a selected item from a candidate plus ranking metadata."""
        return cls(
            record_id=candidate.record_id,
            record_index=candidate.record_index,
            score_name=candidate.score_name,
            score=candidate.score,
            direction=candidate.direction,
            anomaly_score=anomaly_score,
            rank=rank,
            global_rank=global_rank,
            channel_rank=channel_rank,
            channel_size=channel_size,
            layer=candidate.layer,
            source=candidate.source,
            evidence_ref=candidate.evidence_ref,
            metadata=candidate.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "record_id": self.record_id,
            "record_index": self.record_index,
            "score_name": self.score_name,
            "score": self.score,
            "direction": self.direction,
            "anomaly_score": self.anomaly_score,
            "rank": self.rank,
            "global_rank": self.global_rank,
            "channel_rank": self.channel_rank,
            "channel_size": self.channel_size,
            "layer": self.layer,
            "source": self.source,
            "evidence_ref": self.evidence_ref,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HiddenEvidenceSelection":
        """Build a selected item from JSON-like data."""
        return cls(
            record_id=str(data["record_id"]),
            record_index=None if data.get("record_index") is None else int(data["record_index"]),
            score_name=str(data["score_name"]),
            score=data["score"],
            direction=str(data.get("direction", "higher")),
            anomaly_score=data["anomaly_score"],
            rank=data["rank"],
            global_rank=data.get("global_rank", data["rank"]),
            channel_rank=data["channel_rank"],
            channel_size=data["channel_size"],
            layer=data.get("layer"),
            source=str(data.get("source", "candidate")),
            evidence_ref=None if data.get("evidence_ref") is None else str(data["evidence_ref"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class HiddenEvidenceSelectionReport:
    """Versioned report for sparse hidden-evidence selection."""

    selected: tuple[HiddenEvidenceSelection, ...]
    policy: HiddenEvidenceSelectionPolicy = field(default_factory=HiddenEvidenceSelectionPolicy)
    candidate_count: int = 0
    channel_count: int = 0
    dropped_counts: Mapping[str, int] = field(default_factory=dict)
    source_summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    workflow: str = "hidden_evidence_selection"
    status: str = "complete"

    def __post_init__(self) -> None:
        selected = tuple(_selection_from_any(item) for item in self.selected)
        policy = (
            self.policy
            if isinstance(self.policy, HiddenEvidenceSelectionPolicy)
            else HiddenEvidenceSelectionPolicy.from_dict(self.policy)  # type: ignore[arg-type]
        )
        candidate_count = _non_negative_int(self.candidate_count, name="candidate_count")
        channel_count = _non_negative_int(self.channel_count, name="channel_count")
        dropped_counts = {
            str(key): _non_negative_int(value, name=f"dropped_counts.{key}")
            for key, value in self.dropped_counts.items()
        }
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "channel_count", channel_count)
        object.__setattr__(self, "dropped_counts", dropped_counts)
        object.__setattr__(self, "source_summary", _json_canonical_mapping(self.source_summary))
        object.__setattr__(self, "metadata", _json_canonical_mapping(self.metadata))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "workflow", str(self.workflow))
        object.__setattr__(self, "status", str(self.status))

    def summary(self) -> dict[str, Any]:
        """Return compact metadata for trace or registry records."""
        selected = self.selected
        selected_by_layer: dict[str, int] = {}
        selected_by_score: dict[str, int] = {}
        for item in selected:
            layer = item.layer if item.layer is not None else "primary"
            selected_by_layer[layer] = selected_by_layer.get(layer, 0) + 1
            selected_by_score[item.score_name] = selected_by_score.get(item.score_name, 0) + 1
        anomaly_scores = [item.anomaly_score for item in selected]
        return {
            "workflow": self.workflow,
            "status": self.status,
            "candidate_count": self.candidate_count,
            "channel_count": self.channel_count,
            "selected_count": len(selected),
            "selected_record_count": len({item.record_id for item in selected}),
            "selected_layer_count": len(selected_by_layer),
            "selected_score_count": len(selected_by_score),
            "selected_by_layer": dict(sorted(selected_by_layer.items(), key=lambda item: _layer_sort_key(item[0]))),
            "selected_by_score": dict(sorted(selected_by_score.items())),
            "max_anomaly_score": max(anomaly_scores) if anomaly_scores else None,
            "min_selected_anomaly_score": min(anomaly_scores) if anomaly_scores else None,
            "budget_exhausted": bool(self.dropped_counts.get("max_items", 0)),
            "dropped_counts": dict(self.dropped_counts),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "workflow": self.workflow,
            "status": self.status,
            "policy": self.policy.to_dict(),
            "candidate_count": self.candidate_count,
            "channel_count": self.channel_count,
            "dropped_counts": dict(self.dropped_counts),
            "source_summary": to_jsonable(self.source_summary),
            "selected": [item.to_dict() for item in self.selected],
            "summary": self.summary(),
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HiddenEvidenceSelectionReport":
        """Build a report from JSON-like data."""
        return cls(
            selected=tuple(HiddenEvidenceSelection.from_dict(item) for item in data.get("selected", ())),
            policy=HiddenEvidenceSelectionPolicy.from_dict(data.get("policy", {})),
            candidate_count=int(data.get("candidate_count", 0)),
            channel_count=int(data.get("channel_count", 0)),
            dropped_counts=dict(data.get("dropped_counts", {})),
            source_summary=dict(data.get("source_summary", {})),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", 1)),
            workflow=str(data.get("workflow", "hidden_evidence_selection")),
            status=str(data.get("status", "complete")),
        )

    def save_json(self, path: str | Path) -> None:
        """Save the report as UTF-8 JSON."""
        Path(path).write_text(strict_json_dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "HiddenEvidenceSelectionReport":
        """Load a report from UTF-8 JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class _RankedCandidate:
    candidate: HiddenEvidenceCandidate
    anomaly_score: float
    native_anomaly_value: float
    channel_rank: int
    channel_size: int


def select_hidden_evidence(
    candidates: Sequence[HiddenEvidenceCandidate | Mapping[str, Any]],
    *,
    policy: HiddenEvidenceSelectionPolicy | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    source_summary: Mapping[str, Any] | None = None,
) -> HiddenEvidenceSelectionReport:
    """Select a sparse, budgeted hidden-evidence report from candidates.

    Scores are rank-normalized within each ``source/layer/score`` channel so
    heterogeneous diagnostics can be compared without assuming shared units.
    """
    resolved_policy = _policy(policy)
    parsed_candidates = tuple(_candidate_from_any(candidate) for candidate in candidates)
    if not parsed_candidates:
        raise ValueError("candidates must be non-empty.")
    ranked = _rank_candidates(parsed_candidates)
    ranked.sort(key=_global_rank_sort_key)

    selected: list[HiddenEvidenceSelection] = []
    record_counts: dict[str, int] = {}
    layer_counts: dict[str, int] = {}
    score_counts: dict[str, int] = {}
    dropped = {
        "min_anomaly_score": 0,
        "max_items": 0,
        "max_per_record": 0,
        "max_per_layer": 0,
        "max_per_score": 0,
    }
    for global_rank, item in enumerate(ranked, start=1):
        candidate = item.candidate
        if (
            resolved_policy.min_anomaly_score is not None
            and item.anomaly_score < resolved_policy.min_anomaly_score
        ):
            dropped["min_anomaly_score"] += 1
            continue
        if len(selected) >= resolved_policy.max_items:
            dropped["max_items"] += 1
            continue
        layer_key = candidate.layer if candidate.layer is not None else "primary"
        if (
            resolved_policy.max_per_record is not None
            and record_counts.get(candidate.record_id, 0) >= resolved_policy.max_per_record
        ):
            dropped["max_per_record"] += 1
            continue
        if (
            resolved_policy.max_per_layer is not None
            and layer_counts.get(layer_key, 0) >= resolved_policy.max_per_layer
        ):
            dropped["max_per_layer"] += 1
            continue
        if (
            resolved_policy.max_per_score is not None
            and score_counts.get(candidate.score_name, 0) >= resolved_policy.max_per_score
        ):
            dropped["max_per_score"] += 1
            continue
        selected.append(
            HiddenEvidenceSelection.from_candidate(
                candidate,
                anomaly_score=item.anomaly_score,
                rank=len(selected) + 1,
                global_rank=global_rank,
                channel_rank=item.channel_rank,
                channel_size=item.channel_size,
            )
        )
        record_counts[candidate.record_id] = record_counts.get(candidate.record_id, 0) + 1
        layer_counts[layer_key] = layer_counts.get(layer_key, 0) + 1
        score_counts[candidate.score_name] = score_counts.get(candidate.score_name, 0) + 1

    return HiddenEvidenceSelectionReport(
        selected=tuple(selected),
        policy=resolved_policy,
        candidate_count=len(parsed_candidates),
        channel_count=len({candidate.channel_key() for candidate in parsed_candidates}),
        dropped_counts={key: count for key, count in dropped.items() if count},
        source_summary={} if source_summary is None else dict(source_summary),
        metadata={} if metadata is None else dict(metadata),
    )


def select_hidden_evidence_from_score_dump(
    dump: ScoreDump | Mapping[str, Any] | str | Path,
    *,
    score_names: Sequence[str] | None = None,
    sweep_score_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    include_primary: bool = True,
    include_sweep: bool = True,
    directions: Mapping[str, str] | None = None,
    policy: HiddenEvidenceSelectionPolicy | Mapping[str, Any] | None = None,
    statement_metadata_keys: Sequence[str] = DEFAULT_STATEMENT_METADATA_KEYS,
    metadata: Mapping[str, Any] | None = None,
) -> HiddenEvidenceSelectionReport:
    """Select hidden evidence directly from a validated score dump."""
    score_dump = _score_dump_from_any(dump)
    candidates = hidden_evidence_candidates_from_score_dump(
        score_dump,
        score_names=score_names,
        sweep_score_names=sweep_score_names,
        include_primary=include_primary,
        include_sweep=include_sweep,
        directions={} if directions is None else directions,
        statement_metadata_keys=statement_metadata_keys,
    )
    source_summary = score_dump.summary()
    source_summary.update({
        "include_primary": bool(include_primary),
        "include_sweep": bool(include_sweep),
        "requested_score_names": None if score_names is None else tuple(score_names),
        "requested_sweep_score_names": _requested_sweep_score_names_summary(sweep_score_names),
    })
    return select_hidden_evidence(
        candidates,
        policy=policy,
        metadata={} if metadata is None else metadata,
        source_summary=source_summary,
    )


def hidden_evidence_candidates_from_score_dump(
    dump: ScoreDump | Mapping[str, Any],
    *,
    score_names: Sequence[str] | None = None,
    sweep_score_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    include_primary: bool = True,
    include_sweep: bool = True,
    directions: Mapping[str, str] | None = None,
    statement_metadata_keys: Sequence[str] = DEFAULT_STATEMENT_METADATA_KEYS,
) -> tuple[HiddenEvidenceCandidate, ...]:
    """Build hidden-evidence candidates from primary and sweep score columns."""
    score_dump = dump if isinstance(dump, ScoreDump) else ScoreDump.from_mapping(dump)
    resolved_directions = {} if directions is None else dict(directions)
    candidates: list[HiddenEvidenceCandidate] = []
    if include_primary:
        selected_scores = _selected_names(score_dump.scores, score_names)
        for score_name in selected_scores:
            direction = _score_direction(score_name, layer=None, directions=resolved_directions)
            for index, score in enumerate(score_dump.scores[score_name]):
                candidates.append(
                    _candidate_from_score_dump_row(
                        score_dump,
                        index=index,
                        score_name=score_name,
                        score=score,
                        direction=direction,
                        layer=None,
                        source="primary",
                        statement_metadata_keys=statement_metadata_keys,
                    )
                )
    if include_sweep:
        for layer, layer_scores in sorted(
            score_dump.sweep_scores.items(),
            key=lambda item: _layer_sort_key(str(item[0])),
        ):
            selected_sweep_scores = _selected_sweep_names(layer, layer_scores, sweep_score_names)
            for score_name in selected_sweep_scores:
                direction = _score_direction(score_name, layer=str(layer), directions=resolved_directions)
                for index, score in enumerate(layer_scores[score_name]):
                    candidates.append(
                        _candidate_from_score_dump_row(
                            score_dump,
                            index=index,
                            score_name=score_name,
                            score=score,
                            direction=direction,
                            layer=str(layer),
                            source="sweep",
                            statement_metadata_keys=statement_metadata_keys,
                        )
                    )
    if not candidates:
        raise ValueError("score dump did not produce any hidden-evidence candidates.")
    return tuple(candidates)


def _rank_candidates(candidates: Sequence[HiddenEvidenceCandidate]) -> list[_RankedCandidate]:
    channels: dict[str, list[HiddenEvidenceCandidate]] = {}
    for candidate in candidates:
        channels.setdefault(candidate.channel_key(), []).append(candidate)

    ranked: list[_RankedCandidate] = []
    for channel_candidates in channels.values():
        native_values = sorted(candidate.native_anomaly_value() for candidate in channel_candidates)
        channel_size = len(channel_candidates)
        channel_order = {
            id(candidate): rank
            for rank, candidate in enumerate(
                sorted(channel_candidates, key=_channel_rank_sort_key),
                start=1,
            )
        }
        for candidate in channel_candidates:
            native_value = candidate.native_anomaly_value()
            anomaly_score = bisect.bisect_right(native_values, native_value) / channel_size
            ranked.append(
                _RankedCandidate(
                    candidate=candidate,
                    anomaly_score=anomaly_score,
                    native_anomaly_value=native_value,
                    channel_rank=channel_order[id(candidate)],
                    channel_size=channel_size,
                )
            )
    return ranked


def _candidate_from_score_dump_row(
    dump: ScoreDump,
    *,
    index: int,
    score_name: str,
    score: float,
    direction: str,
    layer: str | None,
    source: str,
    statement_metadata_keys: Sequence[str],
) -> HiddenEvidenceCandidate:
    statement = dump.statements[index] if index < len(dump.statements) else {}
    metadata = _statement_metadata(
        statement,
        keys=statement_metadata_keys,
    )
    metadata["label"] = dump.labels[index]
    return HiddenEvidenceCandidate(
        record_id=_record_id(statement, index),
        record_index=index,
        score_name=score_name,
        score=score,
        direction=direction,
        layer=layer,
        source=source,
        metadata=metadata,
    )


def _candidate_from_any(candidate: HiddenEvidenceCandidate | Mapping[str, Any]) -> HiddenEvidenceCandidate:
    if isinstance(candidate, HiddenEvidenceCandidate):
        return candidate
    if isinstance(candidate, Mapping):
        return HiddenEvidenceCandidate.from_dict(candidate)
    raise ValueError("candidate must be HiddenEvidenceCandidate or mapping.")


def _selection_from_any(selection: HiddenEvidenceSelection | Mapping[str, Any]) -> HiddenEvidenceSelection:
    if isinstance(selection, HiddenEvidenceSelection):
        return selection
    if isinstance(selection, Mapping):
        return HiddenEvidenceSelection.from_dict(selection)
    raise ValueError("selection must be HiddenEvidenceSelection or mapping.")


def _score_dump_from_any(dump: ScoreDump | Mapping[str, Any] | str | Path) -> ScoreDump:
    if isinstance(dump, ScoreDump):
        return dump
    if isinstance(dump, Mapping):
        return ScoreDump.from_mapping(dump)
    return load_score_dump(Path(dump))


def _selected_names(scores: Mapping[str, Sequence[float]], names: Sequence[str] | None) -> tuple[str, ...]:
    if names is None:
        return tuple(sorted(scores))
    selected = tuple(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))
    missing = [name for name in selected if name not in scores]
    if missing:
        raise ValueError(f"score dump is missing requested score(s): {missing}.")
    return selected


def _selected_sweep_names(
    layer: str,
    scores: Mapping[str, Sequence[float]],
    names: Sequence[str] | Mapping[str, Sequence[str]] | None,
) -> tuple[str, ...]:
    if names is None:
        return tuple(sorted(scores))
    if isinstance(names, Mapping):
        layer_names = names.get(str(layer), ())
        if not layer_names:
            return ()
        return _selected_names(scores, layer_names)
    return _selected_names(scores, names)


def _requested_sweep_score_names_summary(
    names: Sequence[str] | Mapping[str, Sequence[str]] | None,
) -> Any:
    if names is None:
        return None
    if isinstance(names, Mapping):
        return {str(layer): tuple(score_names) for layer, score_names in names.items()}
    return tuple(names)


def _score_direction(score_name: str, *, layer: str | None, directions: Mapping[str, Any]) -> str:
    keys = (
        f"{layer}:{score_name}" if layer is not None else "",
        score_name,
    )
    for key in keys:
        if key and key in directions:
            return _direction(directions[key])
    return "higher"


def _record_id(statement: Mapping[str, Any], index: int) -> str:
    for key in ("record_id", "claim_id", "statement_id", "id", "question_id"):
        value = statement.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return f"record:{index}"


def _statement_metadata(statement: Mapping[str, Any], *, keys: Sequence[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in keys:
        key_text = str(key).strip()
        if key_text and key_text in statement:
            metadata[key_text] = statement[key_text]
    return metadata


def _candidate_evidence_ref(
    *,
    source: str,
    layer: str | None,
    score_name: str,
    record_id: str,
) -> str:
    layer_ref = "primary" if layer is None else f"layer:{layer}"
    return f"{source}:{layer_ref}:{score_name}:{record_id}"


def _policy(
    policy: HiddenEvidenceSelectionPolicy | Mapping[str, Any] | None,
) -> HiddenEvidenceSelectionPolicy:
    if policy is None:
        return HiddenEvidenceSelectionPolicy()
    if isinstance(policy, HiddenEvidenceSelectionPolicy):
        return policy
    return HiddenEvidenceSelectionPolicy.from_dict(policy)


def _json_canonical_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(strict_json_dumps(dict(values)))


def _global_rank_sort_key(item: _RankedCandidate) -> tuple[Any, ...]:
    candidate = item.candidate
    layer = candidate.layer if candidate.layer is not None else "primary"
    record_index = candidate.record_index if candidate.record_index is not None else 10**12
    return (
        -item.anomaly_score,
        -item.native_anomaly_value,
        _layer_sort_key(layer),
        candidate.score_name,
        record_index,
        candidate.record_id,
    )


def _channel_rank_sort_key(candidate: HiddenEvidenceCandidate) -> tuple[Any, ...]:
    layer = candidate.layer if candidate.layer is not None else "primary"
    record_index = candidate.record_index if candidate.record_index is not None else 10**12
    return (
        -candidate.native_anomaly_value(),
        record_index,
        _layer_sort_key(layer),
        candidate.score_name,
        candidate.record_id,
    )


def _layer_sort_key(value: str) -> tuple[int, int | str]:
    if value == "primary":
        return (-1, -10**12)
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _direction(value: Any) -> str:
    direction = str(value).strip()
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    return direction


def _optional_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if numeric <= 0:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name=name)


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, not bool.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _optional_non_negative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name=name)
