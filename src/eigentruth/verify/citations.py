"""Dependency-free citation integrity checks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import to_jsonable
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus

_BRACKET_REF_RE = re.compile(r"\[(?P<label>[A-Za-z0-9_.:-]+)\]")
_DOI_RE = re.compile(r"\b(?:doi:\s*)?(?P<doi>10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"\b(?:arxiv:\s*)?(?P<arxiv>(?:\d{4}\.\d{4,5})(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7})\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"\bhttps?://[^\s)\],;]+", re.IGNORECASE)
_AUTHOR_YEAR_RE = re.compile(r"\((?P<author>[A-Z][A-Za-z .'-]+),?\s+(?P<year>19\d{2}|20\d{2})\)")


@dataclass(frozen=True)
class CitationRecord:
    """Trusted citation metadata used to audit cited claims."""

    citation_id: str
    title: str | None = None
    authors: Sequence[str] = ()
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        citation_id = str(self.citation_id).strip()
        if not citation_id:
            raise ValueError("citation_id must be non-empty.")
        object.__setattr__(self, "citation_id", citation_id)
        object.__setattr__(self, "title", _optional_non_empty_string(self.title))
        object.__setattr__(self, "authors", tuple(_non_empty_strings(self.authors)))
        object.__setattr__(self, "year", _optional_year(self.year))
        object.__setattr__(self, "doi", _normalize_doi(self.doi))
        object.__setattr__(self, "arxiv_id", _normalize_arxiv_id(self.arxiv_id))
        object.__setattr__(self, "url", _normalize_url(self.url))
        object.__setattr__(self, "source", _optional_non_empty_string(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CitationRecord":
        """Build a record from a JSON-like mapping."""
        raw_id = data.get("citation_id", data.get("id", data.get("ref", data.get("label"))))
        raw_arxiv = data.get("arxiv_id", data.get("arxiv"))
        if raw_id is None:
            raise ValueError("citation record mapping must contain citation_id/id/ref/label.")
        return cls(
            citation_id=str(raw_id),
            title=None if data.get("title") is None else str(data.get("title")),
            authors=tuple(_as_sequence(data.get("authors", ()))),
            year=None if data.get("year") is None else int(data.get("year")),
            doi=None if data.get("doi") is None else str(data.get("doi")),
            arxiv_id=None if raw_arxiv is None else str(raw_arxiv),
            url=None if data.get("url") is None else str(data.get("url")),
            source=None if data.get("source") is None else str(data.get("source")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "citation_id": self.citation_id,
            "title": self.title,
            "authors": tuple(self.authors),
            "year": self.year,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "source": self.source,
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class CitationVerifier:
    """Verify citation references against a trusted local catalog.

    The verifier is intentionally dependency-free and network-free. It catches
    common citation hallucination modes such as unresolved references and
    metadata drift in DOI, arXiv id, URL, year, title, or author fields.
    """

    records: Sequence[CitationRecord | Mapping[str, Any]] = ()
    min_title_token_overlap: float = 0.65

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.min_title_token_overlap) <= 1.0):
            raise ValueError("min_title_token_overlap must be in [0, 1].")
        records = tuple(_citation_record(record) for record in self.records)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "min_title_token_overlap", float(self.min_title_token_overlap))

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim's citation references."""
        catalog = _CitationCatalog((*self.records, *_context_records(context)))
        references = extract_citation_references(claim, context=context)
        if not references:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=1.0,
                explanation="claim does not contain citation references",
                metadata={"verifier": "citation", "decision_rule": "no_citation_reference"},
            )
        if not catalog.records:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.5,
                explanation="citation catalog is empty",
                metadata={
                    "verifier": "citation",
                    "decision_rule": "empty_catalog",
                    "references": tuple(references),
                },
            )

        audits = []
        unresolved = []
        mismatches = []
        matched_records = []
        matched_record_ids = set()
        for reference in references:
            record = catalog.resolve(reference)
            if record is None:
                unresolved.append(reference)
                audits.append({
                    "reference": reference,
                    "status": "unresolved",
                })
                continue
            if record.citation_id not in matched_record_ids:
                matched_records.append(record)
                matched_record_ids.add(record.citation_id)
            comparison = _compare_reference_to_record(
                reference,
                record,
                min_title_token_overlap=self.min_title_token_overlap,
            )
            audits.append(comparison)
            if comparison["mismatches"]:
                mismatches.append(comparison)

        metadata = {
            "verifier": "citation",
            "decision_rule": "citation_catalog_match",
            "references": tuple(to_jsonable(dict(item)) for item in references),
            "audits": tuple(to_jsonable(dict(item)) for item in audits),
            "matched_citation_ids": tuple(record.citation_id for record in matched_records),
            "catalog_size": len(catalog.records),
        }
        if mismatches:
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                confidence=0.95,
                evidence=tuple(_record_evidence(item) for item in matched_records),
                explanation="citation metadata does not match trusted catalog",
                metadata={**metadata, "mismatch_count": len(mismatches)},
            )
        if unresolved:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.6,
                evidence=tuple(_record_evidence(item) for item in matched_records),
                explanation="one or more citation references were not found in trusted catalog",
                metadata={**metadata, "unresolved_count": len(unresolved)},
            )
        return VerificationResult(
            status=VerificationStatus.SUPPORTED,
            confidence=0.9,
            evidence=tuple(_record_evidence(item) for item in matched_records),
            explanation="citation references match trusted catalog",
            metadata=metadata,
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def extract_citation_references(
    claim: Claim | str,
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Extract citation references from claim metadata and text."""
    del context
    text = claim if isinstance(claim, str) else claim.text
    metadata = {} if isinstance(claim, str) else _claim_metadata(claim)
    references: list[dict[str, Any]] = []
    for item in _metadata_reference_items(metadata):
        normalized = _normalize_reference(item, source="claim_metadata")
        if normalized:
            references.append(normalized)
    for match in _BRACKET_REF_RE.finditer(text):
        references.append({"citation_id": match.group("label"), "source": "claim_text.bracket"})
    for match in _DOI_RE.finditer(text):
        references.append({"doi": _normalize_doi(match.group("doi")), "source": "claim_text.doi"})
    for match in _ARXIV_RE.finditer(text):
        references.append({"arxiv_id": _normalize_arxiv_id(match.group("arxiv")), "source": "claim_text.arxiv"})
    for match in _URL_RE.finditer(text):
        references.append({"url": _normalize_url(match.group(0)), "source": "claim_text.url"})
    for match in _AUTHOR_YEAR_RE.finditer(text):
        references.append({
            "author": " ".join(match.group("author").split()),
            "year": int(match.group("year")),
            "source": "claim_text.author_year",
        })
    return tuple(_dedupe_references(references))


class _CitationCatalog:
    def __init__(self, records: Sequence[CitationRecord]) -> None:
        self.records = tuple(records)
        self.by_id = {record.citation_id.lower(): record for record in self.records}
        self.by_doi = {
            record.doi: record
            for record in self.records
            if record.doi is not None
        }
        self.by_arxiv = {
            record.arxiv_id: record
            for record in self.records
            if record.arxiv_id is not None
        }
        self.by_url = {
            record.url: record
            for record in self.records
            if record.url is not None
        }

    def resolve(self, reference: Mapping[str, Any]) -> CitationRecord | None:
        raw_id = reference.get("citation_id", reference.get("id", reference.get("ref", reference.get("label"))))
        if raw_id is not None:
            record = self.by_id.get(str(raw_id).strip().lower())
            if record is not None:
                return record
        doi = _normalize_doi(reference.get("doi"))
        if doi is not None and doi in self.by_doi:
            return self.by_doi[doi]
        arxiv_id = _normalize_arxiv_id(reference.get("arxiv_id", reference.get("arxiv")))
        if arxiv_id is not None and arxiv_id in self.by_arxiv:
            return self.by_arxiv[arxiv_id]
        url = _normalize_url(reference.get("url"))
        if url is not None and url in self.by_url:
            return self.by_url[url]
        author = _optional_non_empty_string(reference.get("author"))
        year = _optional_year(reference.get("year"))
        if author is not None and year is not None:
            author_norm = _normalize_author(author)
            for record in self.records:
                if record.year == year and any(author_norm in _normalize_author(item) for item in record.authors):
                    return record
        return None


def _compare_reference_to_record(
    reference: Mapping[str, Any],
    record: CitationRecord,
    *,
    min_title_token_overlap: float,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for key, normalizer in (
        ("doi", _normalize_doi),
        ("arxiv_id", _normalize_arxiv_id),
        ("url", _normalize_url),
    ):
        claimed = normalizer(reference.get(key))
        expected = getattr(record, key)
        if claimed is not None and expected is not None and claimed != expected:
            mismatches.append({"field": key, "claimed": claimed, "expected": expected})
    claimed_year = _optional_year(reference.get("year"))
    if claimed_year is not None and record.year is not None and claimed_year != record.year:
        mismatches.append({"field": "year", "claimed": claimed_year, "expected": record.year})
    claimed_title = _optional_non_empty_string(reference.get("title"))
    if claimed_title is not None and record.title is not None:
        overlap = _token_overlap(claimed_title, record.title)
        if overlap < min_title_token_overlap:
            mismatches.append({
                "field": "title",
                "claimed": claimed_title,
                "expected": record.title,
                "token_overlap": overlap,
                "minimum": min_title_token_overlap,
            })
    claimed_author = _optional_non_empty_string(reference.get("author"))
    if claimed_author is not None and record.authors:
        claimed_author_norm = _normalize_author(claimed_author)
        if not any(claimed_author_norm in _normalize_author(item) for item in record.authors):
            mismatches.append({
                "field": "author",
                "claimed": claimed_author,
                "expected": tuple(record.authors),
            })
    return {
        "reference": to_jsonable(dict(reference)),
        "record": record.to_dict(),
        "status": "mismatch" if mismatches else "matched",
        "mismatches": tuple(mismatches),
    }


def _record_evidence(record: CitationRecord) -> str:
    parts = [record.citation_id]
    if record.title:
        parts.append(record.title)
    if record.year is not None:
        parts.append(str(record.year))
    if record.doi:
        parts.append(f"doi:{record.doi}")
    if record.arxiv_id:
        parts.append(f"arxiv:{record.arxiv_id}")
    if record.url:
        parts.append(record.url)
    return " | ".join(parts)


def _metadata_reference_items(metadata: Mapping[str, Any]) -> tuple[Any, ...]:
    items: list[Any] = []
    for key in ("citation", "citations", "citation_check", "citation_checks"):
        value = metadata.get(key)
        if value is not None:
            items.extend(_as_sequence(value))
    return tuple(items)


def _normalize_reference(value: Any, *, source: str) -> dict[str, Any] | None:
    if isinstance(value, str):
        stripped = value.strip()
        return {"citation_id": stripped, "source": source} if stripped else None
    if not isinstance(value, Mapping):
        return None
    reference = dict(value)
    reference_keys = (
        "citation_id",
        "id",
        "ref",
        "label",
        "doi",
        "arxiv",
        "arxiv_id",
        "url",
        "author",
        "year",
        "title",
    )
    if not any(key in reference for key in reference_keys):
        return None
    raw_id = reference.get("citation_id", reference.get("id", reference.get("ref", reference.get("label"))))
    normalized: dict[str, Any] = {}
    if raw_id is not None and str(raw_id).strip():
        normalized["citation_id"] = str(raw_id).strip()
    if (doi := _normalize_doi(reference.get("doi"))) is not None:
        normalized["doi"] = doi
    if (arxiv_id := _normalize_arxiv_id(reference.get("arxiv_id", reference.get("arxiv")))) is not None:
        normalized["arxiv_id"] = arxiv_id
    if (url := _normalize_url(reference.get("url"))) is not None:
        normalized["url"] = url
    if (author := _optional_non_empty_string(reference.get("author"))) is not None:
        normalized["author"] = author
    if (year := _optional_year(reference.get("year"))) is not None:
        normalized["year"] = year
    if (title := _optional_non_empty_string(reference.get("title"))) is not None:
        normalized["title"] = title
    normalized["source"] = str(reference.get("source", source))
    return normalized or None


def _dedupe_references(references: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    seen = set()
    deduped = []
    for reference in references:
        key = tuple(sorted((str(k), repr(v)) for k, v in reference.items() if k != "source"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(reference))
    return tuple(deduped)


def _context_records(context: Mapping[str, Any] | None) -> tuple[CitationRecord, ...]:
    if not isinstance(context, Mapping):
        return ()
    raw_records: list[Any] = []
    for key in ("citation_catalog", "citations", "citation_records"):
        value = context.get(key)
        if value is not None:
            if isinstance(value, Mapping):
                raw_records.extend(_records_from_mapping(value))
            else:
                raw_records.extend(_as_sequence(value))
    return tuple(_citation_record(item) for item in raw_records)


def _records_from_mapping(value: Mapping[str, Any]) -> tuple[Any, ...]:
    if any(key in value for key in ("citation_id", "id", "ref", "label")):
        return (value,)
    records = []
    for key, item in value.items():
        if isinstance(item, Mapping):
            records.append({"citation_id": key, **dict(item)})
        else:
            records.append({"citation_id": key, "title": str(item)})
    return tuple(records)


def _citation_record(value: CitationRecord | Mapping[str, Any]) -> CitationRecord:
    if isinstance(value, CitationRecord):
        return value
    if isinstance(value, Mapping):
        return CitationRecord.from_mapping(value)
    raise ValueError("citation records must be CitationRecord objects or mappings.")


def _claim_metadata(claim: Claim) -> Mapping[str, Any]:
    return claim.metadata if isinstance(claim.metadata, Mapping) else {}


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _optional_non_empty_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _non_empty_strings(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(item for value in values if (item := str(value).strip()))


def _optional_year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        year = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("citation year must be an integer.") from exc
    if year < 1000 or year > 9999:
        raise ValueError("citation year must be a four-digit year.")
    return year


def _normalize_doi(value: Any) -> str | None:
    text = _optional_non_empty_string(value)
    if text is None:
        return None
    text = text.lower()
    if text.startswith("doi:"):
        text = text[4:].strip()
    return text.rstrip(".,;)")


def _normalize_arxiv_id(value: Any) -> str | None:
    text = _optional_non_empty_string(value)
    if text is None:
        return None
    text = text.lower()
    if text.startswith("arxiv:"):
        text = text[6:].strip()
    return text.rstrip(".,;)")


def _normalize_url(value: Any) -> str | None:
    text = _optional_non_empty_string(value)
    if text is None:
        return None
    return text.rstrip(".,;)")


def _normalize_author(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _token_overlap(a: str, b: str) -> float:
    left = set(re.findall(r"[a-z0-9]+", a.lower()))
    right = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)
