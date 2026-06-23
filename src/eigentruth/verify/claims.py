"""Lightweight claim extraction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from eigentruth.verify.protocols import Claim

_SENTENCE_RE = re.compile(r"[^.!?。！？]+[.!?。！？]?")
_NUMBER_RE = re.compile(r"\d")
_CITATION_RE = re.compile(r"https?://|www\.|\[[0-9]+\]|\([A-Za-z][A-Za-z .-]+,?\s+\d{4}\)")
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|without|cannot|can't|isn't|aren't|wasn't|weren't|false|incorrect|wrong)\b"
    r"|不是|没有|並非|并非|无|錯誤|错误|不正确",
    re.IGNORECASE,
)
_TIME_SENSITIVE_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|current|currently|latest|now|recent|recently|this year|last year|next year|"
    r"as of|截至|目前|现在|今天|昨天|明天|最新|最近|今年|去年|明年)\b|\b20\d{2}\b",
    re.IGNORECASE,
)
_CALC_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_CALC_EXPRESSION_RE = r"[-+*/().%\d\s]+[+*/%-][-+*/().%\d\s]*"
_SYMBOLIC_CALCULATION_RE = re.compile(
    rf"(?P<expression>{_CALC_EXPRESSION_RE})\s*(?:=|equals|is)\s*(?P<expected>{_CALC_NUMBER_RE})",
    re.IGNORECASE,
)
_LABELED_CALCULATION_RE = re.compile(
    rf"\b(?:expression|expr|calculation|calculate)\s*[:=]\s*"
    rf"(?P<expression>{_CALC_EXPRESSION_RE})\s*(?:[,;]\s*)?"
    rf"(?:expected|result|answer)\s*[:=]\s*(?P<expected>{_CALC_NUMBER_RE})",
    re.IGNORECASE,
)
_WORD_OPERATOR_CALCULATION_RE = re.compile(
    rf"(?P<left>{_CALC_NUMBER_RE})\s+"
    r"(?P<operator>plus|minus|times|multiplied\s+by|divided\s+by|over)\s+"
    rf"(?P<right>{_CALC_NUMBER_RE})\s*(?:=|equals|is)\s*(?P<expected>{_CALC_NUMBER_RE})",
    re.IGNORECASE,
)
_WORD_OPERATORS = {
    "plus": "+",
    "minus": "-",
    "times": "*",
    "multiplied by": "*",
    "divided by": "/",
    "over": "/",
}


@runtime_checkable
class ClaimExtractor(Protocol):
    """Interface for pluggable claim extractors."""

    def extract(self, text: str, *, min_chars: int = 3) -> Sequence[Claim]:
        """Return claim candidates from text."""
        ...


@dataclass(frozen=True)
class SentenceClaimExtractor:
    """Dependency-free sentence-level claim extractor."""

    extractor_name: str = "sentence_split"

    def extract(self, text: str, *, min_chars: int = 3) -> tuple[Claim, ...]:
        """Split text into simple sentence-level atomic claim candidates."""
        claims = []
        for idx, match in enumerate(_SENTENCE_RE.finditer(text), start=1):
            claim_text = match.group(0).strip()
            if len(claim_text) < min_chars:
                continue
            metadata: dict[str, Any] = {
                "extractor": self.extractor_name,
                "source_index": idx,
                "features": claim_features(claim_text),
            }
            calculation = extract_calculation(claim_text)
            if calculation is not None:
                metadata["calculation"] = calculation
            claims.append(
                Claim(
                    text=claim_text,
                    claim_id=f"c{len(claims) + 1}",
                    span=(match.start(), match.end()),
                    metadata=metadata,
                )
            )
        return tuple(claims)


def extract_claims(
    text: str,
    *,
    min_chars: int = 3,
    extractor: ClaimExtractor | None = None,
) -> tuple[Claim, ...]:
    """Extract sentence-level claim candidates with lightweight metadata.

    This keeps the existing dependency-free splitter as the default while making
    the extraction step swappable for stronger extractors later.
    """
    active_extractor = SentenceClaimExtractor() if extractor is None else extractor
    return tuple(active_extractor.extract(text, min_chars=min_chars))


def claim_features(text: str) -> dict[str, bool]:
    """Return simple rule-based metadata flags for a claim."""
    return {
        "has_number": bool(_NUMBER_RE.search(text)),
        "has_citation": bool(_CITATION_RE.search(text)),
        "has_negation": bool(_NEGATION_RE.search(text)),
        "is_time_sensitive": bool(_TIME_SENSITIVE_RE.search(text)),
        "has_calculation": extract_calculation(text) is not None,
    }


def extract_calculation(text: str) -> dict[str, Any] | None:
    """Extract deterministic arithmetic metadata from simple calculator claims."""
    symbolic = _SYMBOLIC_CALCULATION_RE.search(text)
    if symbolic is not None:
        return _calculation_payload(
            symbolic.group("expression"),
            symbolic.group("expected"),
            parser="symbolic",
        )

    labeled = _LABELED_CALCULATION_RE.search(text)
    if labeled is not None:
        return _calculation_payload(
            labeled.group("expression"),
            labeled.group("expected"),
            parser="labeled",
        )

    word_operator = _WORD_OPERATOR_CALCULATION_RE.search(text)
    if word_operator is not None:
        operator = _normalize_operator(word_operator.group("operator"))
        expression = f"{word_operator.group('left')} {operator} {word_operator.group('right')}"
        return _calculation_payload(
            expression,
            word_operator.group("expected"),
            parser="word_operator",
        )
    return None


def _calculation_payload(expression: str, expected: str, *, parser: str) -> dict[str, Any] | None:
    expression = expression.strip(" \t\r\n,;:")
    if not expression:
        return None
    try:
        expected_value = float(expected)
    except ValueError:
        return None
    return {
        "expression": expression,
        "expected": expected_value,
        "source": "claim_text",
        "parser": parser,
    }


def _normalize_operator(value: str) -> str:
    collapsed = " ".join(value.lower().split())
    return _WORD_OPERATORS[collapsed]
