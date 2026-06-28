"""Dependency-free citation/search query planning helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import to_jsonable

QUERY_PLAN_STRATEGIES = (
    "question",
    "queue_query",
    "question_and_query",
    "claim_entity",
)

_CAPITALIZED_SPAN_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*")
_QUOTED_SPAN_RE = re.compile(r"[\"'“”‘’](?P<span>[^\"'“”‘’]{2,80})[\"'“”‘’]")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.'-]*|\d+(?:\.\d+)?")
_PUNCT_RE = re.compile(r"\s+([?.!,;:])")
_QUESTION_MARK_RE = re.compile(r"\?+$")
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "ever",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "many",
    "much",
    "no",
    "not",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "true",
    "was",
    "were",
    "what",
    "what's",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}
_LEADING_ENTITY_WORDS = {
    "A",
    "An",
    "The",
    "This",
    "That",
    "These",
    "Those",
    "What",
    "Which",
    "Who",
}
_QUESTION_TYPE_HINTS = {
    "definition": ("definition",),
    "person": ("person", "biography"),
    "location": ("location",),
    "quantity": ("population", "statistics"),
    "temporal": ("date", "history"),
    "method": ("method",),
    "causal": ("cause",),
}


@dataclass(frozen=True)
class CitationSearchQueryPlan:
    """A sanitized query plan for external citation/source discovery."""

    query: str
    alternate_queries: Sequence[str] = ()
    strategy: str = "question"
    entity_candidates: Sequence[str] = ()
    keyword_terms: Sequence[str] = ()
    removed_phrase_hashes: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        query = clean_search_query(self.query)
        if not query:
            raise ValueError("query plan primary query must be non-empty.")
        alternates = tuple(
            item
            for item in _unique_queries(clean_search_query(value) for value in self.alternate_queries)
            if item and item.casefold() != query.casefold()
        )
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "alternate_queries", alternates)
        object.__setattr__(self, "strategy", str(self.strategy).strip() or "question")
        object.__setattr__(self, "entity_candidates", tuple(_unique(
            clean_candidate(value) for value in self.entity_candidates
        )))
        object.__setattr__(self, "keyword_terms", tuple(_unique(
            clean_candidate(value) for value in self.keyword_terms
        )))
        object.__setattr__(self, "removed_phrase_hashes", tuple(str(item) for item in self.removed_phrase_hashes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def variants(self) -> tuple[str, ...]:
        """Return primary query plus alternates."""
        return (self.query, *tuple(self.alternate_queries))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe plan payload without sensitive phrase values."""
        return {
            "query": self.query,
            "alternate_queries": tuple(self.alternate_queries),
            "strategy": self.strategy,
            "entity_candidates": tuple(self.entity_candidates),
            "keyword_terms": tuple(self.keyword_terms),
            "removed_phrase_hashes": tuple(self.removed_phrase_hashes),
            "metadata": to_jsonable(dict(self.metadata)),
        }


def plan_citation_search_query(
    *,
    question: str,
    candidate_query: str = "",
    question_type: str = "",
    disallowed_phrases: Sequence[str] = (),
    strategy: str = "question",
    max_alternate_queries: int = 3,
    max_entity_candidates: int = 4,
    max_keyword_terms: int = 8,
) -> CitationSearchQueryPlan:
    """Build a sanitized citation search query plan.

    ``candidate_query`` is treated as untrusted internal queue text because it
    may contain model answers. Phrases supplied through ``disallowed_phrases``
    are removed from that candidate before it is allowed across the external
    adapter boundary. The original question is kept intact because it is the
    user/source prompt, not generated answer content.
    """
    strategy = str(strategy).strip() or "question"
    if strategy not in QUERY_PLAN_STRATEGIES:
        raise ValueError(f"strategy must be one of: {', '.join(QUERY_PLAN_STRATEGIES)}.")
    if int(max_alternate_queries) < 0:
        raise ValueError("max_alternate_queries cannot be negative.")
    if int(max_entity_candidates) <= 0:
        raise ValueError("max_entity_candidates must be positive.")
    if int(max_keyword_terms) <= 0:
        raise ValueError("max_keyword_terms must be positive.")

    question_clean = clean_search_query(question)
    candidate_clean, removed_hashes = sanitize_search_query(
        candidate_query,
        disallowed_phrases=disallowed_phrases,
    )
    entities = extract_entity_candidates(question_clean, max_items=int(max_entity_candidates))
    keyword_terms = extract_keyword_terms(question_clean, max_items=int(max_keyword_terms))
    variants = _variants_for_strategy(
        strategy,
        question=question_clean,
        candidate_query=candidate_clean,
        question_type=question_type,
        entities=entities,
        keyword_terms=keyword_terms,
    )
    if not variants:
        raise ValueError("citation search query plan has no usable query variant.")
    return CitationSearchQueryPlan(
        query=variants[0],
        alternate_queries=variants[1: int(max_alternate_queries) + 1],
        strategy=strategy,
        entity_candidates=entities,
        keyword_terms=keyword_terms,
        removed_phrase_hashes=removed_hashes,
        metadata={
            "question_type": str(question_type).strip(),
            "candidate_query_sanitized": bool(candidate_clean),
            "removed_disallowed_phrase_count": len(removed_hashes),
        },
    )


def sanitize_search_query(
    value: str,
    *,
    disallowed_phrases: Sequence[str] = (),
) -> tuple[str, tuple[str, ...]]:
    """Remove disallowed phrases from untrusted internal query text."""
    query = clean_search_query(value)
    removed_hashes: list[str] = []
    for phrase in disallowed_phrases:
        phrase_clean = clean_search_query(phrase)
        if not _removable_phrase(phrase_clean):
            continue
        next_query, removed = _remove_phrase(query, phrase_clean)
        if removed:
            query = next_query
            removed_hashes.append(_sha256_text(phrase_clean))
    return clean_search_query(query), tuple(removed_hashes)


def clean_search_query(value: Any) -> str:
    """Normalize external search query text."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    text = _PUNCT_RE.sub(r"\1", text)
    return text.strip(" \t\r\n,;:")


def clean_candidate(value: Any) -> str:
    """Normalize one candidate entity/keyword phrase."""
    return clean_search_query(value).strip(" \t\r\n?.!,;:\"'()[]{}")


def extract_entity_candidates(question: str, *, max_items: int = 4) -> tuple[str, ...]:
    """Extract simple entity-like spans from a question."""
    candidates: list[str] = []
    for match in _QUOTED_SPAN_RE.finditer(question):
        candidates.append(clean_candidate(match.group("span")))
    for match in _CAPITALIZED_SPAN_RE.finditer(question):
        candidates.append(_strip_leading_entity_words(clean_candidate(match.group(0))))
    keyword_phrase = " ".join(extract_keyword_terms(question, max_items=4))
    if keyword_phrase:
        candidates.append(keyword_phrase)
    return tuple(item for item in _unique(candidates) if _valid_candidate(item))[: int(max_items)]


def extract_keyword_terms(question: str, *, max_items: int = 8) -> tuple[str, ...]:
    """Return non-stopword keyword terms from a question."""
    terms = [
        clean_candidate(match.group(0))
        for match in _TOKEN_RE.finditer(question)
        if match.group(0).casefold() not in _STOPWORDS
    ]
    return tuple(item for item in _unique(terms) if _valid_candidate(item))[: int(max_items)]


def _variants_for_strategy(
    strategy: str,
    *,
    question: str,
    candidate_query: str,
    question_type: str,
    entities: Sequence[str],
    keyword_terms: Sequence[str],
) -> tuple[str, ...]:
    if strategy == "question":
        return tuple(_usable_variants((question, _keyword_query(keyword_terms))))
    if strategy == "queue_query":
        return tuple(_usable_variants((candidate_query, question, _keyword_query(keyword_terms))))
    if strategy == "question_and_query":
        return tuple(_usable_variants((
            _combine_query(question, candidate_query),
            candidate_query,
            question,
            _keyword_query(keyword_terms),
        )))
    focused = _focused_query(question_type=question_type, entities=entities, keyword_terms=keyword_terms)
    return tuple(_usable_variants((
        focused,
        _keyword_query(keyword_terms),
        question,
        candidate_query,
    )))


def _focused_query(
    *,
    question_type: str,
    entities: Sequence[str],
    keyword_terms: Sequence[str],
) -> str:
    entity = next((item for item in entities if item), "")
    keywords = tuple(keyword_terms)
    parts = []
    if entity:
        parts.append(entity)
    existing_terms = set(_tokens(" ".join(parts)))
    for term in keywords:
        term_key = term.casefold()
        if term_key in existing_terms:
            continue
        parts.append(term)
        existing_terms.update(_tokens(term))
    if not parts:
        return ""
    hint = _question_type_hint(question_type, keyword_terms=keywords)
    if hint and hint.casefold() not in existing_terms:
        parts.append(hint)
    return clean_search_query(" ".join(parts[:10]))


def _question_type_hint(question_type: str, *, keyword_terms: Sequence[str]) -> str:
    lowered_terms = {term.casefold() for term in keyword_terms}
    if "founded" in lowered_terms or "founder" in lowered_terms:
        return "founder"
    if "population" in lowered_terms:
        return "population"
    hints = _QUESTION_TYPE_HINTS.get(str(question_type).strip(), ())
    return hints[0] if hints else ""


def _keyword_query(keyword_terms: Sequence[str]) -> str:
    return clean_search_query(" ".join(keyword_terms))


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(value))


def _combine_query(question: str, candidate_query: str) -> str:
    if not question:
        return candidate_query
    if not candidate_query:
        return question
    question_no_mark = _QUESTION_MARK_RE.sub("", question).casefold()
    candidate_folded = candidate_query.casefold()
    if question_no_mark and question_no_mark in candidate_folded:
        return candidate_query
    return clean_search_query(f"{question} {candidate_query}")


def _usable_variants(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(item for item in _unique_queries(clean_search_query(value) for value in values) if item)


def _remove_phrase(query: str, phrase: str) -> tuple[str, bool]:
    next_query = query
    removed = False
    for variant in _phrase_variants(phrase):
        pattern = re.compile(re.escape(variant), re.IGNORECASE)
        next_query, count = pattern.subn(" ", next_query)
        removed = removed or bool(count)
    return clean_search_query(next_query), removed


def _phrase_variants(phrase: str) -> tuple[str, ...]:
    variants = [phrase]
    terminal_stripped = phrase.rstrip("?.!,;:")
    if terminal_stripped != phrase:
        variants.append(terminal_stripped)
    quote_stripped = terminal_stripped.strip("\"'“”‘’")
    if quote_stripped != terminal_stripped:
        variants.append(quote_stripped)
    return tuple(item for item in _unique_exact_queries(variants) if _removable_phrase(item))


def _removable_phrase(value: str) -> bool:
    tokens = tuple(_TOKEN_RE.finditer(value))
    if len(value) < 3 or not tokens:
        return False
    lowered = value.casefold()
    return lowered not in _STOPWORDS


def _strip_leading_entity_words(value: str) -> str:
    parts = value.split()
    while len(parts) > 1 and parts[0] in _LEADING_ENTITY_WORDS:
        parts.pop(0)
    return clean_candidate(" ".join(parts))


def _valid_candidate(value: str) -> bool:
    if not value:
        return False
    tokens = tuple(_TOKEN_RE.finditer(value))
    if not tokens:
        return False
    return any(token.group(0).casefold() not in _STOPWORDS for token in tokens)


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean_candidate(value)
        folded = item.casefold()
        if not item or folded in seen:
            continue
        result.append(item)
        seen.add(folded)
    return tuple(result)


def _unique_queries(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean_search_query(value)
        folded = _query_dedupe_key(item)
        if not item or folded in seen:
            continue
        result.append(item)
        seen.add(folded)
    return tuple(result)


def _query_dedupe_key(value: str) -> str:
    return clean_search_query(value).rstrip("?.!,;:").casefold()


def _unique_exact_queries(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean_search_query(value)
        folded = item.casefold()
        if not item or folded in seen:
            continue
        result.append(item)
        seen.add(folded)
    return tuple(result)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
