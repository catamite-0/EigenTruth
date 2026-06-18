"""Lightweight claim extraction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

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
            claims.append(
                Claim(
                    text=claim_text,
                    claim_id=f"c{len(claims) + 1}",
                    span=(match.start(), match.end()),
                    metadata={
                        "extractor": self.extractor_name,
                        "source_index": idx,
                        "features": claim_features(claim_text),
                    },
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
    }
