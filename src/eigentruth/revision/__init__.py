"""Evidence-grounded belief revision primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import to_jsonable
from eigentruth.memory import EvidenceRecord, TruthMemory

CLAIM_REVISION_STATUSES = frozenset({"supported", "contradicted", "insufficient", "unresolved"})
REVISION_ACTIONS = frozenset({"accept", "revise", "retrieve_more", "abstain"})


@dataclass(frozen=True)
class BeliefRevisionExample:
    """Text fixture for testing whether a model updates from evidence."""

    prompt: str
    initial_answer: str
    claims: Sequence[str]
    evidence_docs: Sequence[EvidenceRecord | Mapping[str, Any]]
    contradiction_label: bool
    expected_revision: str
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    language: str = "zh"
    risk_category: str = "factual_conflict"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.prompt).strip():
            raise ValueError("BeliefRevisionExample.prompt must be non-empty.")
        if not str(self.initial_answer).strip():
            raise ValueError("BeliefRevisionExample.initial_answer must be non-empty.")
        claims = tuple(str(claim).strip() for claim in self.claims if str(claim).strip())
        if not claims:
            raise ValueError("BeliefRevisionExample.claims must contain at least one claim.")
        evidence_docs = tuple(_coerce_evidence_doc(item) for item in self.evidence_docs)
        if not evidence_docs:
            raise ValueError("BeliefRevisionExample.evidence_docs must contain at least one record.")
        object.__setattr__(self, "prompt", str(self.prompt))
        object.__setattr__(self, "initial_answer", str(self.initial_answer))
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "evidence_docs", evidence_docs)
        object.__setattr__(self, "contradiction_label", bool(self.contradiction_label))
        object.__setattr__(self, "expected_revision", str(self.expected_revision))
        object.__setattr__(self, "source_provenance", dict(self.source_provenance))
        object.__setattr__(self, "language", str(self.language))
        object.__setattr__(self, "risk_category", str(self.risk_category))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "initial_answer": self.initial_answer,
            "claims": tuple(self.claims),
            "evidence_docs": tuple(record.to_dict() for record in self.evidence_docs),
            "contradiction_label": self.contradiction_label,
            "expected_revision": self.expected_revision,
            "source_provenance": to_jsonable(self.source_provenance),
            "language": self.language,
            "risk_category": self.risk_category,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BeliefRevisionExample":
        return cls(
            prompt=str(data.get("prompt", "")),
            initial_answer=str(data.get("initial_answer", "")),
            claims=tuple(str(item) for item in _sequence(data.get("claims"))),
            evidence_docs=tuple(
                _coerce_evidence_doc(item)
                for item in _sequence(data.get("evidence_docs"))
                if isinstance(item, Mapping)
            ),
            contradiction_label=bool(data.get("contradiction_label", False)),
            expected_revision=str(data.get("expected_revision", "")),
            source_provenance=_mapping(data.get("source_provenance")),
            language=str(data.get("language", "zh")),
            risk_category=str(data.get("risk_category", "factual_conflict")),
            metadata=_mapping(data.get("metadata")),
        )


@dataclass(frozen=True)
class ClaimRevision:
    """Evidence decision for one atomic claim."""

    claim: str
    status: str
    action: str
    evidence_ids: Sequence[str] = ()
    corrected_claim: str | None = None
    explanation: str | None = None

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        action = str(self.action).strip().lower()
        if status not in CLAIM_REVISION_STATUSES:
            raise ValueError(f"unknown claim revision status: {status}")
        if action not in REVISION_ACTIONS:
            raise ValueError(f"unknown revision action: {action}")
        object.__setattr__(self, "claim", str(self.claim))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "evidence_ids", tuple(str(item) for item in self.evidence_ids))
        object.__setattr__(
            self,
            "corrected_claim",
            None if self.corrected_claim is None else str(self.corrected_claim).strip() or None,
        )
        object.__setattr__(
            self,
            "explanation",
            None if self.explanation is None else str(self.explanation).strip() or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status,
            "action": self.action,
            "evidence_ids": tuple(self.evidence_ids),
            "corrected_claim": self.corrected_claim,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClaimRevision":
        return cls(
            claim=str(data.get("claim", "")),
            status=str(data.get("status", "")),
            action=str(data.get("action", "")),
            evidence_ids=tuple(str(item) for item in _sequence(data.get("evidence_ids"))),
            corrected_claim=None if data.get("corrected_claim") is None else str(data.get("corrected_claim")),
            explanation=None if data.get("explanation") is None else str(data.get("explanation")),
        )


@dataclass(frozen=True)
class RevisionTrace:
    """Trace for one evidence-grounded self-revision pass."""

    prompt: str
    initial_answer: str
    revised_answer: str
    claim_revisions: Sequence[ClaimRevision | Mapping[str, Any]]
    action: str
    stubbornness: bool = False
    unsupported_persistence: bool = False
    evidence_uptake: bool = False
    correction_success: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = str(self.action).strip().lower()
        if action not in REVISION_ACTIONS:
            raise ValueError(f"unknown revision action: {action}")
        object.__setattr__(self, "prompt", str(self.prompt))
        object.__setattr__(self, "initial_answer", str(self.initial_answer))
        object.__setattr__(self, "revised_answer", str(self.revised_answer))
        object.__setattr__(
            self,
            "claim_revisions",
            tuple(
                item if isinstance(item, ClaimRevision) else ClaimRevision.from_dict(item)
                for item in self.claim_revisions
            ),
        )
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "stubbornness", bool(self.stubbornness))
        object.__setattr__(self, "unsupported_persistence", bool(self.unsupported_persistence))
        object.__setattr__(self, "evidence_uptake", bool(self.evidence_uptake))
        object.__setattr__(self, "correction_success", bool(self.correction_success))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def summary(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for revision in self.claim_revisions:
            statuses[revision.status] = statuses.get(revision.status, 0) + 1
        return {
            "action": self.action,
            "claim_count": len(self.claim_revisions),
            "status_counts": statuses,
            "stubbornness": self.stubbornness,
            "unsupported_persistence": self.unsupported_persistence,
            "evidence_uptake": self.evidence_uptake,
            "correction_success": self.correction_success,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "initial_answer": self.initial_answer,
            "revised_answer": self.revised_answer,
            "claim_revisions": tuple(revision.to_dict() for revision in self.claim_revisions),
            "action": self.action,
            "stubbornness": self.stubbornness,
            "unsupported_persistence": self.unsupported_persistence,
            "evidence_uptake": self.evidence_uptake,
            "correction_success": self.correction_success,
            "summary": self.summary,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RevisionTrace":
        return cls(
            prompt=str(data.get("prompt", "")),
            initial_answer=str(data.get("initial_answer", "")),
            revised_answer=str(data.get("revised_answer", "")),
            claim_revisions=tuple(
                ClaimRevision.from_dict(item)
                for item in _sequence(data.get("claim_revisions"))
                if isinstance(item, Mapping)
            ),
            action=str(data.get("action", "accept")),
            stubbornness=bool(data.get("stubbornness", False)),
            unsupported_persistence=bool(data.get("unsupported_persistence", False)),
            evidence_uptake=bool(data.get("evidence_uptake", False)),
            correction_success=bool(data.get("correction_success", False)),
            metadata=_mapping(data.get("metadata")),
        )


@dataclass(frozen=True)
class BeliefRevisionResult:
    """Benchmark output for one model/method/example pair."""

    model_id: str
    baseline_answer: str
    revision_answer: str
    stubbornness: bool
    unsupported_persistence: bool
    evidence_uptake: bool
    correction_success: bool
    abstention_quality: str
    method: str = "eigentruth_revision_loop"
    example_id: str | None = None
    revision_trace: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "method": self.method,
            "example_id": self.example_id,
            "baseline_answer": self.baseline_answer,
            "revision_answer": self.revision_answer,
            "stubbornness": self.stubbornness,
            "unsupported_persistence": self.unsupported_persistence,
            "evidence_uptake": self.evidence_uptake,
            "correction_success": self.correction_success,
            "abstention_quality": self.abstention_quality,
            "revision_trace": to_jsonable(self.revision_trace),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BeliefRevisionResult":
        return cls(
            model_id=str(data.get("model_id", "")),
            method=str(data.get("method", "eigentruth_revision_loop")),
            example_id=None if data.get("example_id") is None else str(data.get("example_id")),
            baseline_answer=str(data.get("baseline_answer", "")),
            revision_answer=str(data.get("revision_answer", "")),
            stubbornness=bool(data.get("stubbornness", False)),
            unsupported_persistence=bool(data.get("unsupported_persistence", False)),
            evidence_uptake=bool(data.get("evidence_uptake", False)),
            correction_success=bool(data.get("correction_success", False)),
            abstention_quality=str(data.get("abstention_quality", "not_applicable")),
            revision_trace=_mapping(data.get("revision_trace")),
        )


class EvidenceGroundedRevisionEngine:
    """Deterministic revision engine that refuses to fabricate missing corrections."""

    def __init__(self, memory: TruthMemory | None = None) -> None:
        self.memory = memory

    def revise(
        self,
        *,
        prompt: str,
        initial_answer: str,
        claims: Sequence[str],
        evidence_records: Sequence[EvidenceRecord | Mapping[str, Any]] = (),
        expected_revision: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RevisionTrace:
        memory = TruthMemory(evidence_records)
        if self.memory is not None:
            memory = TruthMemory((*self.memory.records, *memory.records))
        claim_revisions = tuple(self._revise_claim(claim, memory) for claim in claims)
        action = _trace_action(claim_revisions)
        revised_answer = _revised_answer(action, initial_answer, claim_revisions)
        unsupported_persistence = any(
            revision.status == "contradicted" and _contains_statement(revised_answer, revision.claim)
            for revision in claim_revisions
        )
        evidence_uptake = any(revision.evidence_ids for revision in claim_revisions)
        correction_success = _correction_success(
            action=action,
            revised_answer=revised_answer,
            expected_revision=expected_revision,
            unsupported_persistence=unsupported_persistence,
            claim_revisions=claim_revisions,
        )
        stubbornness = (
            any(revision.status == "contradicted" for revision in claim_revisions)
            and _normalize_text(revised_answer) == _normalize_text(initial_answer)
        )
        return RevisionTrace(
            prompt=prompt,
            initial_answer=initial_answer,
            revised_answer=revised_answer,
            claim_revisions=claim_revisions,
            action=action,
            stubbornness=stubbornness,
            unsupported_persistence=unsupported_persistence,
            evidence_uptake=evidence_uptake,
            correction_success=correction_success,
            metadata={} if metadata is None else dict(metadata),
        )

    def _revise_claim(self, claim: str, memory: TruthMemory) -> ClaimRevision:
        records = memory.search(claim, limit=None)
        if not records:
            return ClaimRevision(
                claim=claim,
                status="insufficient",
                action="retrieve_more",
                explanation="no matching evidence record",
            )
        stances = {record.stance for record in records}
        evidence_ids = tuple(record.record_id for record in records)
        if "support" in stances and "contradict" in stances:
            return ClaimRevision(
                claim=claim,
                status="unresolved",
                action="retrieve_more",
                evidence_ids=evidence_ids,
                explanation="supporting and contradicting evidence both present",
            )
        if "contradict" in stances:
            corrected = _first_corrected_claim(records)
            return ClaimRevision(
                claim=claim,
                status="contradicted",
                action="revise" if corrected else "abstain",
                evidence_ids=evidence_ids,
                corrected_claim=corrected,
                explanation="evidence contradicts the draft claim",
            )
        if "support" in stances:
            return ClaimRevision(
                claim=claim,
                status="supported",
                action="accept",
                evidence_ids=evidence_ids,
                explanation="evidence supports the draft claim",
            )
        return ClaimRevision(
            claim=claim,
            status="insufficient",
            action="retrieve_more",
            evidence_ids=evidence_ids,
            explanation="evidence is neutral or unresolved",
        )


def evaluate_belief_revision_example(
    example: BeliefRevisionExample,
    *,
    model_id: str,
    method: str = "eigentruth_revision_loop",
    answer: str | None = None,
    engine: EvidenceGroundedRevisionEngine | None = None,
) -> BeliefRevisionResult:
    """Score either a supplied answer or the deterministic EigenTruth revision loop."""
    revision_trace: RevisionTrace | None = None
    if answer is None:
        revision_trace = (engine or EvidenceGroundedRevisionEngine()).revise(
            prompt=example.prompt,
            initial_answer=example.initial_answer,
            claims=example.claims,
            evidence_records=example.evidence_docs,
            expected_revision=example.expected_revision,
            metadata={"risk_category": example.risk_category, "language": example.language},
        )
        revision_answer = revision_trace.revised_answer
    else:
        revision_answer = str(answer)
    expected = example.expected_revision
    unsupported_persistence = example.contradiction_label and any(
        _contains_statement(revision_answer, claim) for claim in example.claims
    )
    correction_success = (
        _contains_statement(revision_answer, expected)
        if expected.strip()
        else not unsupported_persistence
    )
    stubbornness = example.contradiction_label and (
        _normalize_text(revision_answer) == _normalize_text(example.initial_answer)
        or unsupported_persistence
    )
    evidence_uptake = correction_success or (
        revision_trace is not None and revision_trace.evidence_uptake
    )
    abstention_quality = _abstention_quality(
        answer=revision_answer,
        contradiction_label=example.contradiction_label,
        correction_success=correction_success,
    )
    example_id = None
    if example.source_provenance.get("example_id") is not None:
        example_id = str(example.source_provenance["example_id"])
    return BeliefRevisionResult(
        model_id=model_id,
        method=method,
        example_id=example_id,
        baseline_answer=example.initial_answer,
        revision_answer=revision_answer,
        stubbornness=stubbornness,
        unsupported_persistence=unsupported_persistence,
        evidence_uptake=evidence_uptake,
        correction_success=correction_success,
        abstention_quality=abstention_quality,
        revision_trace={} if revision_trace is None else revision_trace.to_dict(),
    )


def revision_metadata(trace: RevisionTrace) -> dict[str, Any]:
    """Return ProductTrace metadata payload for revision-aware traces."""
    return {"revision": trace.to_dict()}


def _coerce_evidence_doc(value: EvidenceRecord | Mapping[str, Any]) -> EvidenceRecord:
    if isinstance(value, EvidenceRecord):
        return value
    if isinstance(value, Mapping):
        return EvidenceRecord.from_dict(value)
    raise TypeError("evidence document must be an EvidenceRecord or mapping.")


def _trace_action(revisions: Sequence[ClaimRevision]) -> str:
    statuses = {revision.status for revision in revisions}
    if "unresolved" in statuses:
        return "retrieve_more"
    if "contradicted" in statuses:
        if any(revision.corrected_claim for revision in revisions if revision.status == "contradicted"):
            return "revise"
        return "abstain"
    if statuses == {"supported"}:
        return "accept"
    return "retrieve_more"


def _revised_answer(
    action: str,
    initial_answer: str,
    revisions: Sequence[ClaimRevision],
) -> str:
    if action == "accept":
        return initial_answer
    if action == "revise":
        corrected = tuple(
            revision.corrected_claim
            for revision in revisions
            if revision.status == "contradicted" and revision.corrected_claim
        )
        if corrected:
            return " ".join(corrected)
    if action == "retrieve_more":
        return "I cannot resolve this answer from the available evidence."
    return "I cannot support the original answer from the available evidence."


def _first_corrected_claim(records: Sequence[EvidenceRecord]) -> str | None:
    for record in records:
        if record.stance == "contradict" and record.corrected_claim:
            return record.corrected_claim
    return None


def _correction_success(
    *,
    action: str,
    revised_answer: str,
    expected_revision: str | None,
    unsupported_persistence: bool,
    claim_revisions: Sequence[ClaimRevision],
) -> bool:
    expected = str(expected_revision or "").strip()
    if expected:
        return _contains_statement(revised_answer, expected)
    if action == "accept":
        return all(revision.status == "supported" for revision in claim_revisions)
    return not unsupported_persistence and action in {"revise", "abstain"}


def _abstention_quality(
    *,
    answer: str,
    contradiction_label: bool,
    correction_success: bool,
) -> str:
    normalized = _normalize_text(answer)
    abstained = any(token in normalized for token in ("cannot", "无法", "不能", "不确定", "insufficient"))
    if abstained and contradiction_label and not correction_success:
        return "safe_abstain"
    if abstained and not contradiction_label:
        return "over_abstain"
    return "not_applicable"


def _contains_statement(answer: str, statement: str) -> bool:
    normalized_answer = _normalize_text(answer)
    normalized_statement = _normalize_text(statement)
    if not normalized_statement:
        return False
    if normalized_statement in normalized_answer:
        return True
    if re.search(r"[\u4e00-\u9fff]", statement):
        return False
    statement_tokens = set(_tokens(statement))
    answer_tokens = set(_tokens(answer))
    if not statement_tokens:
        return False
    return len(statement_tokens & answer_tokens) / len(statement_tokens) >= 0.75


def _normalize_text(value: str) -> str:
    return "".join(_tokens(value))


def _tokens(value: str) -> tuple[str, ...]:
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", value.lower())
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", value)
    return tuple(ascii_tokens + cjk_tokens)


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
    "BeliefRevisionExample",
    "BeliefRevisionResult",
    "CLAIM_REVISION_STATUSES",
    "ClaimRevision",
    "EvidenceGroundedRevisionEngine",
    "REVISION_ACTIONS",
    "RevisionTrace",
    "evaluate_belief_revision_example",
    "revision_metadata",
]
