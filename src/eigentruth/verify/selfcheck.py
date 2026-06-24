"""Dependency-free self-consistency verifier for sampled responses."""

from __future__ import annotations

import math
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
    early_stop: bool = False
    max_samples: int | None = None
    context_keys: Sequence[str] = ("selfcheck_samples", "sampled_responses", "samples")

    def __post_init__(self) -> None:
        min_samples = _positive_int(self.min_samples, name="min_samples")
        min_overlap = _unit_float(self.min_overlap, name="min_overlap")
        support_threshold = _unit_float(self.support_threshold, name="support_threshold")
        refute_threshold = _unit_float(self.refute_threshold, name="refute_threshold")
        early_stop = _strict_bool(self.early_stop, name="early_stop")
        object.__setattr__(self, "min_samples", min_samples)
        object.__setattr__(self, "min_overlap", min_overlap)
        object.__setattr__(self, "support_threshold", support_threshold)
        object.__setattr__(self, "refute_threshold", refute_threshold)
        object.__setattr__(self, "early_stop", early_stop)
        if self.max_samples is not None:
            max_samples = _positive_int(self.max_samples, name="max_samples")
            if max_samples < min_samples:
                raise ValueError("max_samples must be >= min_samples when set.")
            object.__setattr__(self, "max_samples", max_samples)
        object.__setattr__(self, "samples", tuple(_coerce_sample(item) for item in self.samples))
        object.__setattr__(self, "context_keys", tuple(str(key) for key in self.context_keys))

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim against configured and context-provided samples."""
        samples = self._samples_from_context(context)
        available_sample_count = len(samples)
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        if len(samples) < self.min_samples:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=1.0,
                explanation="self-consistency verifier needs more sampled responses",
                metadata={
                    "verifier": "self_consistency",
                    "sample_count": len(samples),
                    "available_sample_count": available_sample_count,
                    "processed_sample_count": 0,
                    "skipped_sample_count": 0,
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
                    "available_sample_count": available_sample_count,
                    "processed_sample_count": 0,
                    "skipped_sample_count": 0,
                    "decision_rule": "empty_claim_tokens",
                },
            )

        decisions, early_stop_reason = self._judge_samples(
            claim,
            claim_tokens,
            samples,
        )
        support_count = sum(1 for item in decisions if item.status is VerificationStatus.SUPPORTED)
        refute_count = sum(1 for item in decisions if item.status is VerificationStatus.REFUTED)
        insufficient_count = len(decisions) - support_count - refute_count
        skipped_sample_count = len(samples) - len(decisions)
        rate_denominator = len(samples) if early_stop_reason is not None else len(decisions)
        support_rate = support_count / rate_denominator
        refute_rate = refute_count / rate_denominator
        processed_support_rate = support_count / len(decisions)
        processed_refute_rate = refute_count / len(decisions)
        best_overlap = max((item.overlap for item in decisions), default=0.0)
        metadata = {
            "verifier": "self_consistency",
            "sample_count": len(samples),
            "available_sample_count": available_sample_count,
            "processed_sample_count": len(decisions),
            "skipped_sample_count": skipped_sample_count,
            "max_samples": self.max_samples,
            "support_count": support_count,
            "refute_count": refute_count,
            "insufficient_count": insufficient_count,
            "support_rate": support_rate,
            "refute_rate": refute_rate,
            "processed_support_rate": processed_support_rate,
            "processed_refute_rate": processed_refute_rate,
            "best_overlap": best_overlap,
            "min_overlap": self.min_overlap,
            "support_threshold": self.support_threshold,
            "refute_threshold": self.refute_threshold,
            "early_stop_enabled": self.early_stop,
            "early_stop": early_stop_reason is not None,
            "early_stop_reason": early_stop_reason,
            "sample_decisions_truncated": skipped_sample_count > 0,
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

    def sample_budget_status(
        self,
        claim: Claim,
        samples: Sequence[str | Mapping[str, Any]] | None = None,
        *,
        total_samples: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return whether current samples fix the final threshold outcome.

        This is useful during generation: callers can pass the samples generated
        so far plus the planned total sample budget, and stop generating once
        no future samples can change the self-consistency decision.
        """
        if samples is None:
            current_samples = self._samples_from_context(context)
        else:
            current_samples = _coerce_samples(samples)
        planned_total = self.max_samples if total_samples is None else int(total_samples)
        if planned_total is None:
            planned_total = len(current_samples)
        if planned_total < len(current_samples):
            raise ValueError("total_samples must be >= current sample count.")

        if len(current_samples) < self.min_samples:
            return {
                "can_stop": False,
                "reason": None,
                "sample_count": len(current_samples),
                "total_samples": planned_total,
                "remaining_samples": planned_total - len(current_samples),
                "min_samples": self.min_samples,
                "decision_rule": "too_few_samples",
            }
        claim_tokens = _tokens(claim.text)
        if not claim_tokens:
            return {
                "can_stop": False,
                "reason": None,
                "sample_count": len(current_samples),
                "total_samples": planned_total,
                "remaining_samples": planned_total - len(current_samples),
                "decision_rule": "empty_claim_tokens",
            }
        decisions = tuple(
            _judge_sample(claim, claim_tokens, sample, min_overlap=self.min_overlap)
            for sample in current_samples
        )
        support_count = sum(1 for item in decisions if item.status is VerificationStatus.SUPPORTED)
        refute_count = sum(1 for item in decisions if item.status is VerificationStatus.REFUTED)
        reason = _early_stop_reason(
            decisions,
            total_samples=planned_total,
            support_threshold=self.support_threshold,
            refute_threshold=self.refute_threshold,
        )
        return {
            "can_stop": reason is not None,
            "reason": reason,
            "sample_count": len(current_samples),
            "total_samples": planned_total,
            "remaining_samples": planned_total - len(current_samples),
            "support_count": support_count,
            "refute_count": refute_count,
            "insufficient_count": len(decisions) - support_count - refute_count,
            "support_rate_lower_bound": support_count / planned_total if planned_total else 0.0,
            "refute_rate_lower_bound": refute_count / planned_total if planned_total else 0.0,
            "support_threshold": self.support_threshold,
            "refute_threshold": self.refute_threshold,
        }

    def _samples_from_context(self, context: Mapping[str, Any] | None) -> tuple[_Sample, ...]:
        samples = list(self.samples)
        if context is not None:
            for key in self.context_keys:
                if key in context:
                    samples.extend(_coerce_samples(context[key]))
        return tuple(samples)

    def _judge_samples(
        self,
        claim: Claim,
        claim_tokens: Sequence[str],
        samples: Sequence[_Sample],
    ) -> tuple[tuple[_SampleDecision, ...], str | None]:
        decisions = []
        for sample in samples:
            decisions.append(_judge_sample(claim, claim_tokens, sample, min_overlap=self.min_overlap))
            if not self.early_stop or len(decisions) < self.min_samples:
                continue
            reason = _early_stop_reason(
                decisions,
                total_samples=len(samples),
                support_threshold=self.support_threshold,
                refute_threshold=self.refute_threshold,
            )
            if reason is not None:
                return tuple(decisions), reason
        return tuple(decisions), None


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


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        stripped = value.strip()
        signless = stripped[1:] if stripped[:1] in {"+", "-"} else stripped
        if not signless or not signless.isdecimal():
            raise ValueError(f"{name} must be a positive integer.")
        parsed = int(stripped)
    else:
        raise ValueError(f"{name} must be a positive integer.")
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _unit_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number in [0, 1], not bool.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1].") from exc
    if not math.isfinite(parsed) or not (0.0 <= parsed <= 1.0):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return parsed


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or boolean string.")


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


def _early_stop_reason(
    decisions: Sequence[_SampleDecision],
    *,
    total_samples: int,
    support_threshold: float,
    refute_threshold: float,
) -> str | None:
    """Return a deterministic rate-bound early-stop reason, if one is available."""
    processed = len(decisions)
    remaining = total_samples - processed
    if remaining <= 0:
        return None
    support_count = sum(1 for item in decisions if item.status is VerificationStatus.SUPPORTED)
    refute_count = sum(1 for item in decisions if item.status is VerificationStatus.REFUTED)
    if _threshold_is_guaranteed(refute_count, total_samples, refute_threshold):
        return "refute_threshold_guaranteed"

    refute_possible = _threshold_can_still_be_met(
        refute_count,
        remaining,
        total_samples,
        refute_threshold,
    )
    if (
        not refute_possible
        and _threshold_is_guaranteed(support_count, total_samples, support_threshold)
    ):
        return "support_threshold_guaranteed"

    support_possible = _threshold_can_still_be_met(
        support_count,
        remaining,
        total_samples,
        support_threshold,
    )
    if not refute_possible and not support_possible:
        return "thresholds_unreachable"
    return None


def _threshold_is_guaranteed(count: int, total: int, threshold: float) -> bool:
    return count > 0 and count / total >= threshold


def _threshold_can_still_be_met(count: int, remaining: int, total: int, threshold: float) -> bool:
    return (count + remaining) > 0 and (count + remaining) / total >= threshold


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
