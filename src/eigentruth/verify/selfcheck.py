"""Dependency-free self-consistency verifier for sampled responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus
from eigentruth.verify.rules import normalize_claim_text

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "false",
    "incorrect",
    "wrong",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "cannot",
    "can't",
    "不是",
    "没有",
    "并非",
    "错误",
    "不正确",
}


class _Sample(NamedTuple):
    text: str
    source: str | None
    metadata: Mapping[str, Any]


class _SampleDecision(NamedTuple):
    status: VerificationStatus
    overlap: float
    reason: str
    source: str | None


@dataclass(frozen=True)
class SelfConsistencyVerifier:
    """Verify claims against sampled responses from the same generator.

    This is a lightweight FactSelfCheck-style adapter: the caller supplies
    alternative generations, and the verifier measures whether they repeatedly
    support or contradict each claim. It deliberately does not call an LLM or
    retriever, which keeps the core verification path dependency-free.
    """

    samples: Sequence[str | Mapping[str, Any]] = ()
    min_samples: int = 2
    min_overlap: float = 0.65
    support_threshold: float = 0.60
    refute_threshold: float = 0.50
    context_keys: Sequence[str] = ("selfcheck_samples", "sampled_responses", "samples")

    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        if not (0.0 <= self.min_overlap <= 1.0):
            raise ValueError("min_overlap must be in [0, 1].")
        if not (0.0 <= self.support_threshold <= 1.0):
            raise ValueError("support_threshold must be in [0, 1].")
        if not (0.0 <= self.refute_threshold <= 1.0):
            raise ValueError("refute_threshold must be in [0, 1].")
        object.__setattr__(self, "samples", tuple(_coerce_sample(item) for item in self.samples))
        object.__setattr__(self, "context_keys", tuple(str(key) for key in self.context_keys))

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim against configured and context-provided samples."""
        samples = self._samples_from_context(context)
        if len(samples) < self.min_samples:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=1.0,
                explanation="self-consistency verifier needs more sampled responses",
                metadata={
                    "verifier": "self_consistency",
                    "sample_count": len(samples),
                    "min_samples": self.min_samples,
                    "decision_rule": "too_few_samples",
                },
            )

        claim_tokens = _tokens(claim.text)
        if not claim_tokens:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=0.0,
                explanation="self-consistency verifier found no lexical tokens in claim",
                metadata={
                    "verifier": "self_consistency",
                    "sample_count": len(samples),
                    "decision_rule": "empty_claim_tokens",
                },
            )

        decisions = tuple(
            _judge_sample(claim, claim_tokens, sample, min_overlap=self.min_overlap)
            for sample in samples
        )
        support_count = sum(1 for item in decisions if item.status is VerificationStatus.SUPPORTED)
        refute_count = sum(1 for item in decisions if item.status is VerificationStatus.REFUTED)
        insufficient_count = len(decisions) - support_count - refute_count
        support_rate = support_count / len(decisions)
        refute_rate = refute_count / len(decisions)
        best_overlap = max((item.overlap for item in decisions), default=0.0)
        metadata = {
            "verifier": "self_consistency",
            "sample_count": len(samples),
            "support_count": support_count,
            "refute_count": refute_count,
            "insufficient_count": insufficient_count,
            "support_rate": support_rate,
            "refute_rate": refute_rate,
            "best_overlap": best_overlap,
            "min_overlap": self.min_overlap,
            "support_threshold": self.support_threshold,
            "refute_threshold": self.refute_threshold,
            "sample_decisions": tuple(_decision_to_dict(item) for item in decisions),
        }
        if refute_rate >= self.refute_threshold and refute_count > 0:
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=min(0.95, 0.5 + 0.45 * refute_rate),
                evidence=_evidence_labels(samples, decisions, VerificationStatus.REFUTED),
                explanation="sampled responses consistently contradicted claim",
                metadata={**metadata, "decision_rule": "refute_rate"},
            )
        if support_rate >= self.support_threshold and support_count > 0:
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=min(0.95, 0.5 + 0.45 * support_rate),
                evidence=_evidence_labels(samples, decisions, VerificationStatus.SUPPORTED),
                explanation="sampled responses consistently supported claim",
                metadata={**metadata, "decision_rule": "support_rate"},
            )
        return VerificationResult(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            confidence=max(0.2, 0.5 * max(support_rate, refute_rate, best_overlap)),
            evidence=_evidence_labels(samples, decisions, VerificationStatus.SUPPORTED),
            explanation="sampled responses did not reach a consistency threshold",
            metadata={**metadata, "decision_rule": "mixed_or_low_overlap"},
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)

    def _samples_from_context(self, context: Mapping[str, Any] | None) -> tuple[_Sample, ...]:
        samples = list(self.samples)
        if context is not None:
            for key in self.context_keys:
                if key in context:
                    samples.extend(_coerce_samples(context[key]))
        return tuple(samples)


def _judge_sample(
    claim: Claim,
    claim_tokens: Sequence[str],
    sample: _Sample,
    *,
    min_overlap: float,
) -> _SampleDecision:
    sample_tokens = _tokens(sample.text)
    if not sample_tokens:
        return _SampleDecision(VerificationStatus.INSUFFICIENT_EVIDENCE, 0.0, "empty_sample_tokens", sample.source)
    exact = normalize_claim_text(claim.text) in normalize_claim_text(sample.text)
    overlap = _token_overlap(claim_tokens, sample_tokens)
    if exact:
        return _SampleDecision(VerificationStatus.SUPPORTED, 1.0, "exact_containment", sample.source)
    if overlap < min_overlap:
        return _SampleDecision(VerificationStatus.INSUFFICIENT_EVIDENCE, overlap, "low_overlap", sample.source)
    if _has_negation(claim_tokens) != _has_negation(sample_tokens):
        return _SampleDecision(VerificationStatus.REFUTED, overlap, "negation_mismatch", sample.source)
    if _number_mismatch(claim.text, sample.text):
        return _SampleDecision(VerificationStatus.REFUTED, overlap, "number_mismatch", sample.source)
    return _SampleDecision(VerificationStatus.SUPPORTED, overlap, "token_overlap", sample.source)


def _coerce_samples(value: Any) -> tuple[_Sample, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        return (_coerce_sample(value),)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_coerce_sample(item) for item in value)
    raise ValueError("selfcheck samples must be strings, mappings, or sequences of those values.")


def _coerce_sample(value: str | Mapping[str, Any]) -> _Sample:
    if isinstance(value, str):
        text = value
        source = None
        metadata: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw_text = value.get("text", value.get("content", value.get("response")))
        if raw_text is None:
            raise ValueError("selfcheck sample mapping must contain 'text', 'content', or 'response'.")
        text = str(raw_text)
        raw_source = value.get("source")
        source = None if raw_source is None else str(raw_source)
        raw_metadata = value.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    else:
        raise ValueError("selfcheck samples must be strings or mappings.")
    if not text.strip():
        raise ValueError("selfcheck sample text must be non-empty.")
    return _Sample(text=text, source=source, metadata=metadata)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(text))


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).replace(",", "") for match in _NUMBER_RE.finditer(text))


def _number_mismatch(claim_text: str, sample_text: str) -> bool:
    claim_numbers = set(_numbers(claim_text))
    sample_numbers = set(_numbers(sample_text))
    return bool(claim_numbers and sample_numbers and claim_numbers != sample_numbers)


def _token_overlap(claim_tokens: Sequence[str], sample_tokens: Sequence[str]) -> float:
    if not claim_tokens:
        return 0.0
    sample_set = set(sample_tokens)
    if not sample_set:
        return 0.0
    return sum(1 for token in claim_tokens if token in sample_set) / len(claim_tokens)


def _has_negation(tokens: Sequence[str]) -> bool:
    token_set = set(tokens)
    return any(token in token_set for token in _NEGATION_TOKENS)


def _decision_to_dict(decision: _SampleDecision) -> dict[str, Any]:
    return {
        "status": decision.status.value,
        "overlap": decision.overlap,
        "reason": decision.reason,
        "source": decision.source,
    }


def _evidence_labels(
    samples: Sequence[_Sample],
    decisions: Sequence[_SampleDecision],
    status: VerificationStatus,
    *,
    max_items: int = 3,
    max_chars: int = 180,
) -> tuple[str, ...]:
    labels = []
    for sample, decision in zip(samples, decisions):
        if decision.status is not status:
            continue
        snippet = " ".join(sample.text.split())
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3].rstrip() + "..."
        labels.append(f"{sample.source}: {snippet}" if sample.source else snippet)
        if len(labels) >= max_items:
            break
    return tuple(labels)
