"""Evidence memory and correction-buffer primitives for EigenTruth."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from eigentruth.json_utils import strict_json_dumps, to_jsonable

EVIDENCE_STANCES = frozenset({"support", "contradict", "neutral", "unresolved"})
VERIFIED_TRAINING_STATUSES = frozenset(
    {
        "approved",
        "human_verified",
        "pass",
        "passed",
        "promote",
        "strong_verified",
        "verified",
    }
)


@dataclass(frozen=True)
class EvidenceRecord:
    """A source-backed evidence row tied to one claim-like statement."""

    record_id: str
    claim: str
    evidence_text: str
    stance: str
    source: str | None = None
    credibility: float | None = None
    corrected_claim: str | None = None
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        record_id = str(self.record_id).strip()
        claim = str(self.claim).strip()
        evidence_text = str(self.evidence_text).strip()
        stance = str(self.stance).strip().lower()
        if not record_id:
            raise ValueError("EvidenceRecord.record_id must be non-empty.")
        if not claim:
            raise ValueError("EvidenceRecord.claim must be non-empty.")
        if not evidence_text:
            raise ValueError("EvidenceRecord.evidence_text must be non-empty.")
        if stance not in EVIDENCE_STANCES:
            allowed = ", ".join(sorted(EVIDENCE_STANCES))
            raise ValueError(f"EvidenceRecord.stance must be one of: {allowed}.")
        if self.credibility is not None:
            credibility = float(self.credibility)
            if not math.isfinite(credibility) or credibility < 0.0 or credibility > 1.0:
                raise ValueError("EvidenceRecord.credibility must be finite and in [0, 1].")
            object.__setattr__(self, "credibility", credibility)
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "claim", claim)
        object.__setattr__(self, "evidence_text", evidence_text)
        object.__setattr__(self, "stance", stance)
        object.__setattr__(self, "source", None if self.source is None else str(self.source))
        object.__setattr__(
            self,
            "corrected_claim",
            None if self.corrected_claim is None else str(self.corrected_claim).strip() or None,
        )
        object.__setattr__(self, "source_provenance", dict(self.source_provenance))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "claim": self.claim,
            "evidence_text": self.evidence_text,
            "stance": self.stance,
            "source": self.source,
            "credibility": self.credibility,
            "corrected_claim": self.corrected_claim,
            "source_provenance": to_jsonable(self.source_provenance),
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            record_id=str(data.get("record_id", "")),
            claim=str(data.get("claim", "")),
            evidence_text=str(data.get("evidence_text", "")),
            stance=str(data.get("stance", "")),
            source=None if data.get("source") is None else str(data.get("source")),
            credibility=None if data.get("credibility") is None else float(data["credibility"]),
            corrected_claim=None
            if data.get("corrected_claim") is None
            else str(data.get("corrected_claim")),
            source_provenance=_mapping(data.get("source_provenance")),
            metadata=_mapping(data.get("metadata")),
        )


class TruthMemory:
    """Small dependency-free evidence ledger with deterministic lexical search."""

    def __init__(self, records: Iterable[EvidenceRecord | Mapping[str, Any]] = ()) -> None:
        self._records: list[EvidenceRecord] = []
        for record in records:
            self.add(record)

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records)

    def add(self, record: EvidenceRecord | Mapping[str, Any]) -> EvidenceRecord:
        evidence = _coerce_evidence_record(record)
        self._records.append(evidence)
        return evidence

    def search(
        self,
        query: str,
        *,
        stance: str | Sequence[str] | None = None,
        limit: int | None = 5,
        min_score: float = 0.0,
    ) -> tuple[EvidenceRecord, ...]:
        """Return records ranked by simple token overlap with the query."""
        allowed_stances = _stance_filter(stance)
        scored: list[tuple[float, int, EvidenceRecord]] = []
        for index, record in enumerate(self._records):
            if allowed_stances is not None and record.stance not in allowed_stances:
                continue
            score = _lexical_overlap(query, record.claim)
            if score <= 0.0:
                score = _lexical_overlap(query, record.evidence_text) * 0.5
            if score >= min_score:
                scored.append((score, index, record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        records = [record for _, _, record in scored]
        if limit is None:
            return tuple(records)
        return tuple(records[: max(0, int(limit))])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "record_count": len(self._records),
            "records": [record.to_dict() for record in self._records],
        }

    def write_jsonl(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(strict_json_dumps(record.to_dict(), sort_keys=True) + "\n" for record in self._records),
            encoding="utf-8",
        )
        return output

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "TruthMemory":
        return cls(_read_jsonl_records(path, EvidenceRecord.from_dict))


@dataclass(frozen=True)
class CorrectionRecord:
    """A verified or candidate self-revision event."""

    record_id: str
    prompt: str
    initial_answer: str
    revised_answer: str
    claims: Sequence[str] = ()
    evidence_records: Sequence[EvidenceRecord | Mapping[str, Any]] = ()
    revision_trace: Mapping[str, Any] = field(default_factory=dict)
    verifier_status: str = "unverified"
    correction_success: bool = False
    failure_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.record_id).strip():
            raise ValueError("CorrectionRecord.record_id must be non-empty.")
        if not str(self.prompt).strip():
            raise ValueError("CorrectionRecord.prompt must be non-empty.")
        object.__setattr__(self, "record_id", str(self.record_id).strip())
        object.__setattr__(self, "prompt", str(self.prompt))
        object.__setattr__(self, "initial_answer", str(self.initial_answer))
        object.__setattr__(self, "revised_answer", str(self.revised_answer))
        object.__setattr__(self, "claims", tuple(str(claim) for claim in self.claims))
        object.__setattr__(
            self,
            "evidence_records",
            tuple(_coerce_evidence_record(record) for record in self.evidence_records),
        )
        object.__setattr__(self, "revision_trace", dict(self.revision_trace))
        object.__setattr__(self, "verifier_status", str(self.verifier_status).strip().lower())
        object.__setattr__(self, "correction_success", bool(self.correction_success))
        object.__setattr__(
            self,
            "failure_type",
            None if self.failure_type is None else str(self.failure_type).strip() or None,
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_training_verified(self) -> bool:
        return self.correction_success and self.verifier_status in VERIFIED_TRAINING_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "prompt": self.prompt,
            "initial_answer": self.initial_answer,
            "revised_answer": self.revised_answer,
            "claims": tuple(self.claims),
            "evidence_records": tuple(record.to_dict() for record in self.evidence_records),
            "revision_trace": to_jsonable(self.revision_trace),
            "verifier_status": self.verifier_status,
            "correction_success": self.correction_success,
            "failure_type": self.failure_type,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorrectionRecord":
        return cls(
            record_id=str(data.get("record_id", "")),
            prompt=str(data.get("prompt", "")),
            initial_answer=str(data.get("initial_answer", "")),
            revised_answer=str(data.get("revised_answer", "")),
            claims=tuple(str(claim) for claim in _sequence(data.get("claims"))),
            evidence_records=tuple(
                _coerce_evidence_record(record)
                for record in _sequence(data.get("evidence_records"))
                if isinstance(record, Mapping)
            ),
            revision_trace=_mapping(data.get("revision_trace")),
            verifier_status=str(data.get("verifier_status", "unverified")),
            correction_success=bool(data.get("correction_success", False)),
            failure_type=None if data.get("failure_type") is None else str(data.get("failure_type")),
            metadata=_mapping(data.get("metadata")),
        )


class CorrectionBuffer:
    """Append-only correction-event buffer with safe training export filters."""

    def __init__(self, records: Iterable[CorrectionRecord | Mapping[str, Any]] = ()) -> None:
        self._records: list[CorrectionRecord] = []
        for record in records:
            self.add(record)

    @property
    def records(self) -> tuple[CorrectionRecord, ...]:
        return tuple(self._records)

    def add(self, record: CorrectionRecord | Mapping[str, Any]) -> CorrectionRecord:
        correction = record if isinstance(record, CorrectionRecord) else CorrectionRecord.from_dict(record)
        self._records.append(correction)
        return correction

    def verified_records(self) -> tuple[CorrectionRecord, ...]:
        return tuple(record for record in self._records if record.is_training_verified)

    def training_records(self, *, format: str = "sft") -> tuple[dict[str, Any], ...]:
        fmt = str(format).strip().lower()
        if fmt not in {"sft", "dpo"}:
            raise ValueError("training export format must be 'sft' or 'dpo'.")
        return tuple(_training_payload(record, format=fmt) for record in self.verified_records())

    def write_jsonl(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(strict_json_dumps(record.to_dict(), sort_keys=True) + "\n" for record in self._records),
            encoding="utf-8",
        )
        return output

    def write_training_jsonl(self, path: str | Path, *, format: str = "sft") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = (
            strict_json_dumps(record, sort_keys=True) + "\n"
            for record in self.training_records(format=format)
        )
        output.write_text(
            "".join(lines),
            encoding="utf-8",
        )
        return output

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "CorrectionBuffer":
        return cls(_read_jsonl_records(path, CorrectionRecord.from_dict))


def _training_payload(record: CorrectionRecord, *, format: str) -> dict[str, Any]:
    evidence = "\n".join(
        f"- [{item.stance}] {item.evidence_text}" for item in record.evidence_records
    )
    prompt = record.prompt if not evidence else f"{record.prompt}\n\nEvidence:\n{evidence}"
    metadata = {
        "record_id": record.record_id,
        "verifier_status": record.verifier_status,
        "failure_type": record.failure_type,
        "claim_count": len(record.claims),
    }
    if format == "dpo":
        return {
            "prompt": prompt,
            "chosen": record.revised_answer,
            "rejected": record.initial_answer,
            "metadata": metadata,
        }
    return {
        "messages": (
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": record.revised_answer},
        ),
        "metadata": metadata,
    }


def _coerce_evidence_record(record: EvidenceRecord | Mapping[str, Any]) -> EvidenceRecord:
    if isinstance(record, EvidenceRecord):
        return record
    if isinstance(record, Mapping):
        return EvidenceRecord.from_dict(record)
    raise TypeError("evidence record must be an EvidenceRecord or mapping.")


def _read_jsonl_records(path: str | Path, loader: Any) -> tuple[Any, ...]:
    records: list[Any] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL row {line_number} must be an object.")
        records.append(loader(payload))
    return tuple(records)


def _lexical_overlap(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    if left_tokens == right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(value: str) -> tuple[str, ...]:
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", value.lower())
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", value)
    return tuple(ascii_tokens + cjk_tokens)


def _stance_filter(stance: str | Sequence[str] | None) -> frozenset[str] | None:
    if stance is None:
        return None
    if isinstance(stance, str):
        values = (stance,)
    else:
        values = tuple(str(item) for item in stance)
    normalized = frozenset(value.strip().lower() for value in values)
    unknown = normalized - EVIDENCE_STANCES
    if unknown:
        allowed = ", ".join(sorted(EVIDENCE_STANCES))
        raise ValueError(f"stance filter contains unknown values {sorted(unknown)}; allowed: {allowed}.")
    return normalized


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


__all__ = [
    "CorrectionBuffer",
    "CorrectionRecord",
    "EVIDENCE_STANCES",
    "EvidenceRecord",
    "TruthMemory",
    "VERIFIED_TRAINING_STATUSES",
]
