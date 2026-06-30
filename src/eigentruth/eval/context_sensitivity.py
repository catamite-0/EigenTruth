"""Evidence context-sensitivity scoring from paired token log-probabilities.

The helpers in this module intentionally do not call an LLM.  They consume
precomputed token log-probabilities with and without evidence context, then
summarize whether the evidence made generated tokens less likely.  That makes
the primitive usable with retrieval/verifier/world-model adapters without
adding a mandatory model dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import strict_json_dumps, to_jsonable


@dataclass(frozen=True)
class ContextSensitivityToken:
    """One generated token with baseline and evidence-conditioned logprobs."""

    token: str
    baseline_logprob: float
    context_logprob: float
    claim_id: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        baseline_logprob = _log_probability(self.baseline_logprob, name="baseline_logprob")
        context_logprob = _log_probability(self.context_logprob, name="context_logprob")
        claim_id = _optional_non_empty_str(self.claim_id)
        span_start = _optional_non_negative_int(self.span_start, name="span_start")
        span_end = _optional_non_negative_int(self.span_end, name="span_end")
        if span_start is not None and span_end is not None and span_end < span_start:
            raise ValueError("span_end must be >= span_start.")
        object.__setattr__(self, "token", str(self.token))
        object.__setattr__(self, "baseline_logprob", baseline_logprob)
        object.__setattr__(self, "context_logprob", context_logprob)
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "span_start", span_start)
        object.__setattr__(self, "span_end", span_end)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextSensitivityToken":
        """Build a token input from a JSON-like mapping."""
        baseline = data.get("baseline_logprob", data.get("no_context_logprob"))
        context = data.get("context_logprob", data.get("evidence_logprob"))
        if baseline is None:
            raise ValueError("token input must include baseline_logprob.")
        if context is None:
            raise ValueError("token input must include context_logprob.")
        return cls(
            token=str(data.get("token", "")),
            baseline_logprob=baseline,
            context_logprob=context,
            claim_id=None if data.get("claim_id") is None else str(data["claim_id"]),
            span_start=data.get("span_start"),
            span_end=data.get("span_end"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable token payload."""
        return {
            "token": self.token,
            "baseline_logprob": self.baseline_logprob,
            "context_logprob": self.context_logprob,
            "claim_id": self.claim_id,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class ContextSensitivityTokenScore:
    """One token's evidence-sensitivity diagnostics."""

    index: int
    token: str
    baseline_logprob: float
    context_logprob: float
    logprob_delta: float
    unsupported_context_shift: float
    context_sensitivity_ratio: float
    flagged: bool
    reasons: tuple[str, ...] = ()
    claim_id: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index = _non_negative_int(self.index, name="index")
        baseline_logprob = _log_probability(self.baseline_logprob, name="baseline_logprob")
        context_logprob = _log_probability(self.context_logprob, name="context_logprob")
        logprob_delta = _finite_float(self.logprob_delta, name="logprob_delta")
        unsupported_context_shift = _non_negative_float(
            self.unsupported_context_shift,
            name="unsupported_context_shift",
        )
        ratio = _finite_float(self.context_sensitivity_ratio, name="context_sensitivity_ratio")
        claim_id = _optional_non_empty_str(self.claim_id)
        span_start = _optional_non_negative_int(self.span_start, name="span_start")
        span_end = _optional_non_negative_int(self.span_end, name="span_end")
        if span_start is not None and span_end is not None and span_end < span_start:
            raise ValueError("span_end must be >= span_start.")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "token", str(self.token))
        object.__setattr__(self, "baseline_logprob", baseline_logprob)
        object.__setattr__(self, "context_logprob", context_logprob)
        object.__setattr__(self, "logprob_delta", logprob_delta)
        object.__setattr__(self, "unsupported_context_shift", unsupported_context_shift)
        object.__setattr__(self, "context_sensitivity_ratio", ratio)
        object.__setattr__(self, "flagged", bool(self.flagged))
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "span_start", span_start)
        object.__setattr__(self, "span_end", span_end)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable score payload."""
        return {
            "index": self.index,
            "token": self.token,
            "baseline_logprob": self.baseline_logprob,
            "context_logprob": self.context_logprob,
            "logprob_delta": self.logprob_delta,
            "unsupported_context_shift": self.unsupported_context_shift,
            "context_sensitivity_ratio": self.context_sensitivity_ratio,
            "flagged": self.flagged,
            "reasons": list(self.reasons),
            "claim_id": self.claim_id,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class ContextSensitivityReport:
    """JSON-ready report for token and claim-level context sensitivity."""

    token_scores: tuple[ContextSensitivityTokenScore, ...]
    ratio_threshold: float
    shift_threshold: float
    min_abs_delta: float = 0.0
    summary: Mapping[str, Any] = field(default_factory=dict)
    claim_summaries: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        token_scores = tuple(
            score
            if isinstance(score, ContextSensitivityTokenScore)
            else ContextSensitivityTokenScore(**dict(score))  # type: ignore[arg-type]
            for score in self.token_scores
        )
        object.__setattr__(self, "token_scores", token_scores)
        object.__setattr__(
            self,
            "ratio_threshold",
            _positive_float(self.ratio_threshold, name="ratio_threshold"),
        )
        object.__setattr__(
            self,
            "shift_threshold",
            _non_negative_float(self.shift_threshold, name="shift_threshold"),
        )
        object.__setattr__(
            self,
            "min_abs_delta",
            _non_negative_float(self.min_abs_delta, name="min_abs_delta"),
        )
        object.__setattr__(self, "summary", dict(self.summary))
        object.__setattr__(
            self,
            "claim_summaries",
            {str(key): dict(value) for key, value in self.claim_summaries.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def flagged_token_count(self) -> int:
        """Return the number of flagged tokens."""
        return sum(1 for score in self.token_scores if score.flagged)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report payload."""
        return {
            "workflow": "context_sensitivity_scoring",
            "ratio_threshold": self.ratio_threshold,
            "shift_threshold": self.shift_threshold,
            "min_abs_delta": self.min_abs_delta,
            "summary": to_jsonable(self.summary),
            "claim_summaries": to_jsonable(self.claim_summaries),
            "token_scores": [score.to_dict() for score in self.token_scores],
            "metadata": to_jsonable(self.metadata),
        }

    def save_json(self, path: str | Path) -> None:
        """Write the report as deterministic JSON."""
        Path(path).write_text(strict_json_dumps(self.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextSensitivityReport":
        """Build a report from a JSON-like payload."""
        return cls(
            token_scores=tuple(ContextSensitivityTokenScore(**dict(item)) for item in data.get("token_scores", ())),
            ratio_threshold=data.get("ratio_threshold", 1.25),
            shift_threshold=data.get("shift_threshold", 0.0),
            min_abs_delta=data.get("min_abs_delta", 0.0),
            summary=dict(data.get("summary", {})),
            claim_summaries={
                str(key): dict(value)
                for key, value in dict(data.get("claim_summaries", {})).items()
            },
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "ContextSensitivityReport":
        """Load a report from JSON."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("context-sensitivity report JSON must be an object.")
        return cls.from_dict(payload)


def context_logprob_delta(context_logprob: float, baseline_logprob: float) -> float:
    """Return ``context_logprob - baseline_logprob``.

    Positive values mean the evidence context increased token likelihood;
    negative values mean the generated token became less likely under evidence.
    """
    return _log_probability(context_logprob, name="context_logprob") - _log_probability(
        baseline_logprob,
        name="baseline_logprob",
    )


def unsupported_context_shift(context_logprob: float, baseline_logprob: float) -> float:
    """Return positive evidence-against-token shift in logprob space."""
    delta = context_logprob_delta(context_logprob, baseline_logprob)
    return max(0.0, -delta)


def context_sensitivity_ratio(context_logprob: float, baseline_logprob: float, *, eps: float = 1e-9) -> float:
    """Return a REFIND-style logprob ratio for evidence sensitivity.

    With normal negative log-probabilities, values above 1.0 mean the evidence
    context made the token less likely than the no-context baseline.  ``eps``
    only guards exact-zero denominators.
    """
    context = _log_probability(context_logprob, name="context_logprob")
    baseline = _log_probability(baseline_logprob, name="baseline_logprob")
    eps_value = _positive_float(eps, name="eps")
    denominator = baseline if abs(baseline) >= eps_value else -eps_value
    return context / denominator


def score_context_sensitivity(
    tokens: Sequence[ContextSensitivityToken | Mapping[str, Any]],
    *,
    ratio_threshold: float = 1.25,
    shift_threshold: float = 0.25,
    min_abs_delta: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> ContextSensitivityReport:
    """Score evidence context sensitivity for paired token logprob inputs.

    Tokens are flagged when the context-sensitivity ratio is at least
    ``ratio_threshold`` or the positive unsupported shift is greater than
    ``shift_threshold``, after the absolute logprob delta exceeds
    ``min_abs_delta``.
    """
    ratio_threshold = _positive_float(ratio_threshold, name="ratio_threshold")
    shift_threshold = _non_negative_float(shift_threshold, name="shift_threshold")
    min_abs_delta = _non_negative_float(min_abs_delta, name="min_abs_delta")
    normalized_tokens = tuple(_token_from_input(token) for token in tokens)
    token_scores = tuple(
        _score_token(
            token,
            index=index,
            ratio_threshold=ratio_threshold,
            shift_threshold=shift_threshold,
            min_abs_delta=min_abs_delta,
        )
        for index, token in enumerate(normalized_tokens)
    )
    return ContextSensitivityReport(
        token_scores=token_scores,
        ratio_threshold=ratio_threshold,
        shift_threshold=shift_threshold,
        min_abs_delta=min_abs_delta,
        summary=_summary(token_scores),
        claim_summaries=_claim_summaries(token_scores),
        metadata=dict(metadata or {}),
    )


def _score_token(
    token: ContextSensitivityToken,
    *,
    index: int,
    ratio_threshold: float,
    shift_threshold: float,
    min_abs_delta: float,
) -> ContextSensitivityTokenScore:
    delta = context_logprob_delta(token.context_logprob, token.baseline_logprob)
    shift = max(0.0, -delta)
    ratio = context_sensitivity_ratio(token.context_logprob, token.baseline_logprob)
    magnitude_ok = abs(delta) >= min_abs_delta
    reasons: list[str] = []
    if magnitude_ok and ratio >= ratio_threshold:
        reasons.append("context_sensitivity_ratio")
    if magnitude_ok and shift > shift_threshold:
        reasons.append("unsupported_context_shift")
    return ContextSensitivityTokenScore(
        index=index,
        token=token.token,
        baseline_logprob=token.baseline_logprob,
        context_logprob=token.context_logprob,
        logprob_delta=delta,
        unsupported_context_shift=shift,
        context_sensitivity_ratio=ratio,
        flagged=bool(reasons),
        reasons=tuple(reasons),
        claim_id=token.claim_id,
        span_start=token.span_start,
        span_end=token.span_end,
        metadata=token.metadata,
    )


def _summary(scores: Sequence[ContextSensitivityTokenScore]) -> dict[str, Any]:
    count = len(scores)
    flagged = sum(1 for score in scores if score.flagged)
    supported = sum(1 for score in scores if score.logprob_delta >= 0.0)
    shifts = [score.unsupported_context_shift for score in scores]
    deltas = [score.logprob_delta for score in scores]
    ratios = [score.context_sensitivity_ratio for score in scores]
    return {
        "token_count": count,
        "flagged_token_count": flagged,
        "flagged_rate": _rate(flagged, count),
        "supported_token_count": supported,
        "supported_rate": _rate(supported, count),
        "max_unsupported_context_shift": max(shifts) if shifts else 0.0,
        "mean_unsupported_context_shift": _mean(shifts),
        "min_logprob_delta": min(deltas) if deltas else 0.0,
        "mean_logprob_delta": _mean(deltas),
        "max_context_sensitivity_ratio": max(ratios) if ratios else 0.0,
    }


def _claim_summaries(scores: Sequence[ContextSensitivityTokenScore]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ContextSensitivityTokenScore]] = {}
    for score in scores:
        if score.claim_id is None:
            continue
        grouped.setdefault(score.claim_id, []).append(score)
    summaries: dict[str, dict[str, Any]] = {}
    for claim_id, claim_scores in sorted(grouped.items()):
        summary = _summary(claim_scores)
        reasons = sorted({reason for score in claim_scores for reason in score.reasons})
        summary["reasons"] = reasons
        summaries[claim_id] = summary
    return summaries


def _token_from_input(value: ContextSensitivityToken | Mapping[str, Any]) -> ContextSensitivityToken:
    if isinstance(value, ContextSensitivityToken):
        return value
    if isinstance(value, Mapping):
        return ContextSensitivityToken.from_dict(value)
    raise ValueError("tokens must contain ContextSensitivityToken objects or mappings.")


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _optional_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _log_probability(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric > 0.0:
        raise ValueError(f"{name} must be a log probability <= 0.")
    return numeric


def _positive_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _non_negative_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


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
