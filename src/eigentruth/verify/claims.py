"""Lightweight claim extraction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

from eigentruth.verify.protocols import Claim
from eigentruth.verify.search_planning import clean_candidate, extract_entity_candidates

if TYPE_CHECKING:
    from eigentruth.verify.triples import ClaimTripleExtractor

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
_QUOTED_ENTITY_RE = re.compile(r"[\"'“”‘’](?P<span>[^\"'“”‘’]{2,80})[\"'“”‘’]")
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
_ENTITY_HINT_STOPWORDS = {
    "a",
    "an",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "no",
    "not",
    "on",
    "that",
    "the",
    "these",
    "this",
    "those",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
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
    include_triples: bool = False
    triple_metadata_key: str = "claim_triples"
    require_triple_audit: bool = False

    def extract(self, text: str, *, min_chars: int = 3) -> tuple[Claim, ...]:
        """Split text into simple sentence-level atomic claim candidates."""
        claims = []
        for idx, match in enumerate(_SENTENCE_RE.finditer(text), start=1):
            claim_text = match.group(0).strip()
            if len(claim_text) < min_chars:
                continue
            features = claim_features(claim_text)
            metadata: dict[str, Any] = {
                "extractor": self.extractor_name,
                "source_index": idx,
                "features": features,
            }
            entity_candidates = claim_entity_candidates(claim_text)
            if entity_candidates:
                metadata["entity_candidates"] = entity_candidates
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
        extracted = tuple(claims)
        if not self.include_triples:
            return extracted
        return enrich_claims_with_triples(
            extracted,
            metadata_key=self.triple_metadata_key,
            require_triple_audit=self.require_triple_audit,
        )


def extract_claims(
    text: str,
    *,
    min_chars: int = 3,
    extractor: ClaimExtractor | None = None,
    include_triples: bool = False,
    triple_metadata_key: str = "claim_triples",
    require_triple_audit: bool = False,
) -> tuple[Claim, ...]:
    """Extract sentence-level claim candidates with lightweight metadata.

    This keeps the existing dependency-free splitter as the default while making
    the extraction step swappable for stronger extractors later.
    """
    active_extractor = SentenceClaimExtractor() if extractor is None else extractor
    claims = tuple(active_extractor.extract(text, min_chars=min_chars))
    if not include_triples:
        return claims
    return enrich_claims_with_triples(
        claims,
        metadata_key=triple_metadata_key,
        require_triple_audit=require_triple_audit,
    )


def enrich_claims_with_triples(
    claims: Sequence[Claim],
    *,
    triple_extractor: ClaimTripleExtractor | None = None,
    metadata_key: str = "claim_triples",
    require_triple_audit: bool = False,
    replace_existing: bool = False,
) -> tuple[Claim, ...]:
    """Attach rule-based fact triples to claim metadata when they can be parsed."""
    metadata_key = str(metadata_key).strip()
    if not metadata_key:
        raise ValueError("metadata_key must be non-empty.")
    from eigentruth.verify.triples import RuleBasedTripleExtractor

    active_extractor = RuleBasedTripleExtractor() if triple_extractor is None else triple_extractor
    enriched = []
    for claim in claims:
        metadata = dict(claim.metadata) if isinstance(claim.metadata, Mapping) else {}
        if metadata.get(metadata_key) and not replace_existing:
            if require_triple_audit and metadata.get("requires_triple_audit") is not True:
                metadata["requires_triple_audit"] = True
                enriched.append(
                    Claim(
                        text=claim.text,
                        claim_id=claim.claim_id,
                        span=claim.span,
                        metadata=metadata,
                    )
                )
            else:
                enriched.append(claim)
            continue
        extraction_metadata = dict(metadata)
        if replace_existing:
            extraction_metadata.pop(metadata_key, None)
            extraction_metadata.pop("triples", None)
            extraction_metadata.pop("claim_triples", None)
        extraction_claim = Claim(
            text=claim.text,
            claim_id=claim.claim_id,
            span=claim.span,
            metadata=extraction_metadata,
        )
        triples = tuple(active_extractor.extract(extraction_claim))
        if not triples:
            enriched.append(claim)
            continue
        metadata[metadata_key] = tuple(triple.to_dict() for triple in triples)
        if require_triple_audit:
            metadata["requires_triple_audit"] = True
        enriched.append(
            Claim(
                text=claim.text,
                claim_id=claim.claim_id,
                span=claim.span,
                metadata=metadata,
            )
        )
    return tuple(enriched)


def claim_entity_candidates(text: str, *, max_items: int = 4) -> tuple[str, ...]:
    """Return conservative entity-like surface candidates from claim text."""
    max_items = max(int(max_items), 0)
    if max_items == 0:
        return ()

    quoted = (
        clean_candidate(match.group("span"))
        for match in _QUOTED_ENTITY_RE.finditer(str(text))
    )
    planned = extract_entity_candidates(str(text), max_items=max(max_items * 2, max_items))
    candidates: list[str] = []
    for candidate in quoted:
        if _valid_quoted_entity_candidate(candidate):
            candidates.append(candidate)
    for candidate in planned:
        if _looks_like_entity_surface(candidate):
            candidates.append(candidate)
    return tuple(dict.fromkeys(candidates))[:max_items]


def claim_features(text: str) -> dict[str, bool]:
    """Return simple rule-based metadata flags for a claim."""
    return {
        "has_number": bool(_NUMBER_RE.search(text)),
        "has_citation": bool(_CITATION_RE.search(text)),
        "has_negation": bool(_NEGATION_RE.search(text)),
        "is_time_sensitive": bool(_TIME_SENSITIVE_RE.search(text)),
        "has_calculation": extract_calculation(text) is not None,
        "has_named_entity_hint": bool(claim_entity_candidates(text, max_items=1)),
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


def _valid_quoted_entity_candidate(candidate: str) -> bool:
    candidate = clean_candidate(candidate)
    return len(candidate) >= 2 and candidate.casefold() not in _ENTITY_HINT_STOPWORDS


def _looks_like_entity_surface(candidate: str) -> bool:
    candidate = clean_candidate(candidate)
    if not candidate:
        return False
    if candidate.casefold() in _ENTITY_HINT_STOPWORDS:
        return False
    if len(candidate) < 3 and not candidate.isupper():
        return False
    return any(char.isupper() for char in candidate)
