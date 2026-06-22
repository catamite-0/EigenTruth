"""Dependency-free structured question/answer verifier adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.verify import Claim, VerificationResult, VerificationStatus, normalize_claim_text


@dataclass(frozen=True)
class QuestionAnswerFact:
    """One structured question/answer fact."""

    question: str
    answer: str
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must be non-empty.")
        if not self.answer.strip():
            raise ValueError("answer must be non-empty.")

    def to_evidence(self) -> str:
        """Return a compact evidence string."""
        text = f"{self.question.strip()} {self.answer.strip()}"
        if self.source:
            return f"{self.source}: {text}"
        return text

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "QuestionAnswerFact":
        """Build a QA fact from a JSON-like mapping."""
        metadata = dict(data.get("metadata", {}))
        question = data.get("question", metadata.get("question"))
        answer = data.get("answer", metadata.get("answer"))
        if question is None or answer is None:
            raise ValueError("QA fact mapping must contain question and answer fields.")
        source = data.get("source")
        return cls(
            question=str(question),
            answer=str(answer),
            source=None if source is None else str(source),
            metadata=metadata,
        )


@dataclass(frozen=True)
class QuestionAnswerVerifier:
    """Structured verifier over known correct answers for exact questions.

    This adapter models a database/domain-state lookup. If the claim supplies a
    question and candidate answer, and the database has correct answer(s) for the
    same question, matching answers are supported while non-matching answers are
    refuted. It does not use labels.
    """

    facts: Sequence[QuestionAnswerFact | Mapping[str, Any]]

    def __post_init__(self) -> None:
        facts = tuple(_coerce_fact(item) for item in self.facts)
        index: dict[str, tuple[QuestionAnswerFact, ...]] = {}
        for fact in facts:
            key = normalize_claim_text(fact.question)
            index[key] = (*index.get(key, ()), fact)
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "_index", index)

    @classmethod
    def from_corpus(cls, corpus: Mapping[str, Any]) -> "QuestionAnswerVerifier":
        """Build from a corpus JSON object with documents/records."""
        raw_documents = corpus.get("documents", corpus.get("records", ()))
        facts = []
        for item in raw_documents:
            if not isinstance(item, Mapping):
                continue
            try:
                facts.append(QuestionAnswerFact.from_mapping(item))
            except ValueError:
                continue
        if not facts:
            raise ValueError("QA corpus does not contain any structured question/answer facts.")
        return cls(facts)

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim against structured QA facts."""
        question, answer = _claim_question_answer(claim, context)
        if not question or not answer:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.2,
                explanation="claim did not provide structured question and answer fields",
                metadata={"verifier": "structured_qa", "decision_rule": "missing_question_or_answer"},
            )

        question_key = normalize_claim_text(question)
        candidates = self._index.get(question_key, ())
        if not candidates:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.25,
                explanation="no structured answer facts found for question",
                metadata={
                    "verifier": "structured_qa",
                    "decision_rule": "question_not_found",
                    "question_key": question_key,
                },
            )

        answer_key = normalize_claim_text(answer)
        for fact in candidates:
            if normalize_claim_text(fact.answer) == answer_key:
                return VerificationResult(
                    status=VerificationStatus.SUPPORTED,
                    confidence=0.95,
                    evidence=(fact.to_evidence(),),
                    explanation="candidate answer matches structured correct answer",
                    metadata={
                        "verifier": "structured_qa",
                        "decision_rule": "answer_match",
                        "question_key": question_key,
                        "n_known_answers": len(candidates),
                    },
                )

        evidence = tuple(fact.to_evidence() for fact in candidates)
        return VerificationResult(
            status=VerificationStatus.REFUTED,
            confidence=0.9,
            evidence=evidence,
            explanation="question has known correct answer(s), but candidate answer does not match",
            metadata={
                "verifier": "structured_qa",
                "decision_rule": "answer_mismatch",
                "question_key": question_key,
                "n_known_answers": len(candidates),
            },
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def _coerce_fact(value: QuestionAnswerFact | Mapping[str, Any]) -> QuestionAnswerFact:
    if isinstance(value, QuestionAnswerFact):
        return value
    return QuestionAnswerFact.from_mapping(value)


def _claim_question_answer(
    claim: Claim,
    context: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    statement = {}
    if context is not None and isinstance(context.get("statement"), Mapping):
        statement = dict(context["statement"])
    metadata_statement = claim.metadata.get("statement") if isinstance(claim.metadata, Mapping) else None
    if isinstance(metadata_statement, Mapping):
        statement = {**statement, **dict(metadata_statement)}
    question = claim.metadata.get("question") if isinstance(claim.metadata, Mapping) else None
    answer = claim.metadata.get("answer") if isinstance(claim.metadata, Mapping) else None
    return (
        _optional_text(question if question is not None else statement.get("question")),
        _optional_text(answer if answer is not None else statement.get("answer")),
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
