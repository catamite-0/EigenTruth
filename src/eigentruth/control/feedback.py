"""Post-hoc feedback records for product factuality traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from eigentruth.control.trace import ProductTrace


class FeedbackOutcome(str, Enum):
    """Normalized post-hoc outcome labels for product feedback."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    UNSUPPORTED = "unsupported"
    UNNECESSARY_BLOCK = "unnecessary_block"
    APPROPRIATE_BLOCK = "appropriate_block"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProductFeedbackRecord:
    """One post-hoc feedback item linked to a ProductTrace request or claim."""

    request_id: str
    outcome: FeedbackOutcome | str
    trace_fingerprint: str | None = None
    claim_id: str | None = None
    feedback_source: str = "manual"
    corrected_text: str | None = None
    evidence_refs: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        request_id = _non_empty_string(self.request_id, name="request_id")
        feedback_source = _non_empty_string(self.feedback_source, name="feedback_source")
        outcome = _coerce_outcome(self.outcome)
        evidence_refs = tuple(str(item) for item in self.evidence_refs)
        schema_version = int(self.schema_version)
        if schema_version < 1:
            raise ValueError("schema_version must be positive.")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "feedback_source", feedback_source)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "trace_fingerprint", _optional_string(self.trace_fingerprint))
        object.__setattr__(self, "claim_id", _optional_string(self.claim_id))
        object.__setattr__(self, "corrected_text", _optional_string(self.corrected_text))
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "created_at", _optional_string(self.created_at))
        object.__setattr__(self, "schema_version", schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "trace_fingerprint": self.trace_fingerprint,
            "claim_id": self.claim_id,
            "outcome": self.outcome.value,
            "feedback_source": self.feedback_source,
            "corrected_text": self.corrected_text,
            "evidence_refs": list(self.evidence_refs),
            "metadata": _jsonable(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProductFeedbackRecord":
        """Build a feedback record from JSON-like data."""
        return cls(
            request_id=str(data["request_id"]),
            trace_fingerprint=None if data.get("trace_fingerprint") is None else str(data["trace_fingerprint"]),
            claim_id=None if data.get("claim_id") is None else str(data["claim_id"]),
            outcome=str(data.get("outcome", FeedbackOutcome.UNKNOWN.value)),
            feedback_source=str(data.get("feedback_source", "manual")),
            corrected_text=None if data.get("corrected_text") is None else str(data["corrected_text"]),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", ())),
            metadata=dict(data.get("metadata", {})),
            created_at=None if data.get("created_at") is None else str(data["created_at"]),
            schema_version=int(data.get("schema_version", 1)),
        )


@dataclass(frozen=True)
class ProductFeedbackStore:
    """JSONL-backed local store for ProductFeedbackRecord items."""

    path: str | Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def append(self, record: ProductFeedbackRecord | Mapping[str, Any]) -> None:
        """Append one feedback record as one JSONL line."""
        item = _feedback_record(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(_feedback_json_line(item))

    def extend(self, records: Sequence[ProductFeedbackRecord | Mapping[str, Any]]) -> None:
        """Append several feedback records as JSONL lines."""
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(_feedback_json_line(_feedback_record(record)))

    def read_all(self) -> tuple[ProductFeedbackRecord, ...]:
        """Read all feedback records currently stored in the JSONL file."""
        return tuple(iter_feedback_jsonl(self.path))


def product_trace_fingerprint(trace: ProductTrace | Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a ProductTrace payload."""
    payload = trace.to_dict() if isinstance(trace, ProductTrace) else dict(trace)
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def iter_feedback_jsonl(path: str | Path) -> Iterator[ProductFeedbackRecord]:
    """Yield feedback records from a JSONL file."""
    feedback_path = Path(path)
    with feedback_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("feedback JSONL row must be an object.")
                yield ProductFeedbackRecord.from_dict(payload)
            except Exception as exc:
                raise ValueError(
                    f"invalid ProductFeedbackRecord at {feedback_path}:{line_number}: {exc}"
                ) from exc


def load_feedback_jsonl(paths: str | Path | Sequence[str | Path]) -> tuple[ProductFeedbackRecord, ...]:
    """Load feedback records from one or more JSONL files."""
    if isinstance(paths, str | Path):
        feedback_paths = (paths,)
    else:
        feedback_paths = tuple(paths)
    records: list[ProductFeedbackRecord] = []
    for path in feedback_paths:
        records.extend(iter_feedback_jsonl(path))
    return tuple(records)


def write_feedback_jsonl(
    path: str | Path,
    records: Sequence[ProductFeedbackRecord | Mapping[str, Any]],
    *,
    append: bool = False,
) -> None:
    """Write feedback records to a JSONL file."""
    feedback_path = Path(path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with feedback_path.open(mode, encoding="utf-8") as stream:
        for record in records:
            stream.write(_feedback_json_line(_feedback_record(record)))


def _feedback_json_line(record: ProductFeedbackRecord) -> str:
    return json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _feedback_record(record: ProductFeedbackRecord | Mapping[str, Any]) -> ProductFeedbackRecord:
    if isinstance(record, ProductFeedbackRecord):
        return record
    return ProductFeedbackRecord.from_dict(record)


def _coerce_outcome(value: FeedbackOutcome | str) -> FeedbackOutcome:
    if isinstance(value, FeedbackOutcome):
        return value
    try:
        return FeedbackOutcome(str(value))
    except ValueError as exc:
        choices = ", ".join(outcome.value for outcome in FeedbackOutcome)
        raise ValueError(f"outcome must be one of: {choices}.") from exc


def _non_empty_string(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_jsonable(item) for item in value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
