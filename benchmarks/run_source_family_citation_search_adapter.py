"""Run a dependency-free local source-family citation/search adapter.

The adapter consumes sanitized citation/search request JSONL with optional
``source_family_plan`` fields and ranks caller-supplied local source catalog
documents by lexical overlap plus source-family compatibility. It does not fetch
network content and does not make verifier decisions. Returned result JSONL must
still pass the citation-search evidence workflow before it can become release
evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

PROVIDER = "source_family_catalog"
DEFAULT_SOURCE_FAMILY = "reference"
RESERVED_CATALOG_FIELDS = {
    "answer",
    "claim_id",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "record_index",
    "row_index",
    "score_label",
    "source_index",
    "target_id",
}
TEXT_FIELDS = ("text", "content", "document", "body", "snippet", "summary", "abstract")
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
OFFICIAL_FAMILIES = {"official", "official_statistics", "domain_specific"}
FALLBACK_FAMILIES = {"encyclopedic", "reference"}


@dataclass(frozen=True)
class SourceCatalogDocument:
    """One local source-family catalog document."""

    text: str
    title: str = ""
    source: str = ""
    url: str = ""
    provider: str = PROVIDER
    source_family: str = DEFAULT_SOURCE_FAMILY
    published_at: str = ""
    timestamp: str = ""
    base_score: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("source catalog document text must be non-empty.")
        base_score = float(self.base_score)
        if not (0.0 <= base_score <= 1.0):
            raise ValueError("source catalog document base_score must be in [0, 1].")
        object.__setattr__(self, "text", _clean(self.text))
        object.__setattr__(self, "title", _clean(self.title))
        object.__setattr__(self, "source", _clean(self.source))
        object.__setattr__(self, "url", _clean(self.url))
        object.__setattr__(self, "provider", _clean(self.provider) or PROVIDER)
        object.__setattr__(self, "source_family", _normalize_family(self.source_family))
        object.__setattr__(self, "published_at", _clean(self.published_at))
        object.__setattr__(self, "timestamp", _clean(self.timestamp))
        object.__setattr__(self, "base_score", base_score)
        object.__setattr__(self, "metadata", dict(self.metadata))


def run_source_family_citation_search_adapter(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_catalog_paths: Sequence[str | Path],
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    max_results: int = 3,
    max_query_variants: int = 3,
    min_text_overlap: float = 0.05,
    diversify_source_families: bool = False,
    default_source_family: str = DEFAULT_SOURCE_FAMILY,
    compact_json: bool = True,
    fail_on_empty: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank local source-family catalog documents for sanitized requests."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if not source_catalog_paths:
        raise ValueError("source_catalog_paths must contain at least one path.")
    if max_results <= 0:
        raise ValueError("max_results must be positive.")
    if max_query_variants <= 0:
        raise ValueError("max_query_variants must be positive.")
    if not (0.0 <= min_text_overlap <= 1.0):
        raise ValueError("min_text_overlap must be in [0, 1].")

    requests = _load_jsonl(input_path)
    catalog = _load_source_catalogs(
        source_catalog_paths,
        default_source_family=default_source_family,
    )
    rows = tuple(
        _rank_request(
            request,
            catalog=catalog,
            max_results=int(max_results),
            max_query_variants=int(max_query_variants),
            min_text_overlap=float(min_text_overlap),
            diversify_source_families=bool(diversify_source_families),
        )
        for request in requests
    )
    _write_jsonl(output_path, rows, compact=compact_json)
    summary = _summary(rows, catalog=catalog)
    payload = {
        "schema_version": 1,
        "workflow": "source_family_citation_search_adapter",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_catalog_paths": [str(path) for path in source_catalog_paths],
        "config": {
            "max_results": int(max_results),
            "max_query_variants": int(max_query_variants),
            "min_text_overlap": float(min_text_overlap),
            "diversify_source_families": bool(diversify_source_families),
            "default_source_family": _normalize_family(default_source_family),
        },
        "summary": summary,
        "metadata": dict(metadata or {}),
    }
    if report_json_path is not None:
        _write_json(report_json_path, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        artifacts: dict[str, str | Path] = {
            "adapter_requests": Path(input_path),
            "adapter_results": Path(output_path),
        }
        if report_json_path is not None:
            artifacts["adapter_report"] = Path(report_json_path)
        for idx, path in enumerate(source_catalog_paths, start=1):
            artifacts[f"source_catalog_{idx}"] = Path(path)
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": payload["workflow"],
                "request_count": summary["request_count"],
                "source_document_count": summary["source_document_count"],
                "request_with_results_count": summary["request_with_results_count"],
                "request_without_results_count": summary["request_without_results_count"],
                "request_coverage": summary["request_coverage"],
                "result_count": summary["result_count"],
                **dict(metadata or {}),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)
        payload["artifact_manifest"] = str(manifest_path)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_json_path or output_path,
            metadata={
                "workflow": payload["workflow"],
                "request_count": summary["request_count"],
                "source_document_count": summary["source_document_count"],
                "request_with_results_count": summary["request_with_results_count"],
                "request_without_results_count": summary["request_without_results_count"],
                "request_coverage": summary["request_coverage"],
                "result_count": summary["result_count"],
                "artifact_manifest": payload.get("artifact_manifest"),
                **dict(metadata or {}),
            },
        ).save_json()
    if fail_on_empty and summary["request_with_results_count"] == 0:
        raise SystemExit(1)
    return payload


def _rank_request(
    request: Mapping[str, Any],
    *,
    catalog: Sequence[SourceCatalogDocument],
    max_results: int,
    max_query_variants: int,
    min_text_overlap: float,
    diversify_source_families: bool,
) -> dict[str, Any]:
    request_id = _clean(request.get("request_id"))
    query_variants = _request_query_variants(request, max_items=max_query_variants)
    query = query_variants[0] if query_variants else ""
    if not request_id:
        return _error_row("", query, "missing_request_id")
    if not query_variants:
        return _error_row(request_id, "", "missing_query")
    plan = _source_family_plan(request)
    preferred_families = _preferred_families(plan)
    query_hint_tokens = _tokens(" ".join(_string_sequence(plan.get("query_hints", ()))))
    scored: list[tuple[float, dict[str, Any]]] = []
    for document in catalog:
        score = _score_document(
            document,
            query_variants=query_variants,
            query_hint_tokens=query_hint_tokens,
            preferred_families=preferred_families,
            freshness_required=bool(plan.get("freshness_required")),
            official_source_preferred=bool(plan.get("official_source_preferred")),
        )
        if score["text_overlap"] < min_text_overlap:
            continue
        scored.append((float(score["score"]), _result_from_document(document, score=score)))
    scored.sort(key=lambda item: (
        item[0],
        _family_priority(item[1].get("source_family"), preferred_families),
        -int(item[1].get("rank", 999999)),
    ), reverse=True)
    selected = (
        _select_family_diverse_results(scored, max_results=max_results, preferred_families=preferred_families)
        if diversify_source_families
        else tuple(result for _, result in scored[:max_results])
    )
    results = tuple(
        {**result, "rank": rank}
        for rank, result in enumerate(selected, start=1)
    )
    return {
        "request_id": request_id,
        "results": results,
        "metadata": {
            "provider": PROVIDER,
            "query": query,
            "query_variants": query_variants,
            "searched_query_variant_count": len(query_variants),
            "preferred_source_families": preferred_families,
            "freshness_required": bool(plan.get("freshness_required")),
            "official_source_preferred": bool(plan.get("official_source_preferred")),
            "catalog_document_count": len(catalog),
            "diversify_source_families": bool(diversify_source_families),
        },
    }


def _select_family_diverse_results(
    scored: Sequence[tuple[float, dict[str, Any]]],
    *,
    max_results: int,
    preferred_families: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    if max_results <= 0:
        return ()
    selected: list[dict[str, Any]] = []
    selected_keys: set[int] = set()

    def add_first_for_family(family: str) -> None:
        if len(selected) >= max_results:
            return
        normalized = _normalize_family(family)
        for index, (_score, result) in enumerate(scored):
            if index in selected_keys:
                continue
            if _normalize_family(result.get("source_family")) != normalized:
                continue
            selected.append(result)
            selected_keys.add(index)
            return

    family_order = (
        *tuple(family for family in preferred_families if _normalize_family(family) not in FALLBACK_FAMILIES),
        *tuple(family for family in preferred_families if _normalize_family(family) in FALLBACK_FAMILIES),
    )
    for family in family_order:
        add_first_for_family(family)
    for index, (_score, result) in enumerate(scored):
        if len(selected) >= max_results:
            break
        if index in selected_keys:
            continue
        family = _normalize_family(result.get("source_family"))
        if any(_normalize_family(existing.get("source_family")) == family for existing in selected):
            continue
        selected.append(result)
        selected_keys.add(index)
    for index, (_score, result) in enumerate(scored):
        if len(selected) >= max_results:
            break
        if index in selected_keys:
            continue
        selected.append(result)
        selected_keys.add(index)
    return tuple(selected[:max_results])


def _score_document(
    document: SourceCatalogDocument,
    *,
    query_variants: Sequence[str],
    query_hint_tokens: Sequence[str],
    preferred_families: Sequence[str],
    freshness_required: bool,
    official_source_preferred: bool,
) -> dict[str, Any]:
    document_tokens = _tokens(" ".join((document.title, document.text)))
    best_overlap = 0.0
    matched_query = ""
    matched_query_index = 0
    for index, query in enumerate(query_variants, start=1):
        overlap = _token_overlap(_tokens(query), document_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            matched_query = query
            matched_query_index = index
    hint_overlap = _token_overlap(query_hint_tokens, document_tokens)
    family_match = document.source_family in set(preferred_families)
    official_match = (
        document.source_family in OFFICIAL_FAMILIES
        or _bool_metadata(document.metadata.get("official_source"))
        or _bool_metadata(document.metadata.get("trusted_source"))
    )
    freshness_match = bool(document.published_at or document.timestamp)
    score = (
        0.72 * best_overlap
        + 0.12 * hint_overlap
        + 0.10 * float(family_match)
        + 0.04 * float(official_source_preferred and official_match)
        + 0.02 * float(freshness_required and freshness_match)
    )
    score *= document.base_score
    return {
        "score": max(0.0, min(1.0, score)),
        "text_overlap": best_overlap,
        "query_hint_overlap": hint_overlap,
        "family_match": family_match,
        "official_match": official_match,
        "freshness_match": freshness_match,
        "matched_query": matched_query,
        "matched_query_index": matched_query_index,
    }


def _result_from_document(document: SourceCatalogDocument, *, score: Mapping[str, Any]) -> dict[str, Any]:
    snippet = _snippet(document)
    return {
        "title": document.title,
        "snippet": snippet,
        "text": document.text,
        "url": document.url,
        "source": document.source or document.url or f"{PROVIDER}:{document.source_family}",
        "provider": document.provider,
        "rank": 0,
        "score": float(score["score"]),
        "text_overlap": float(score["text_overlap"]),
        "query_hint_overlap": float(score["query_hint_overlap"]),
        "family_match": bool(score["family_match"]),
        "official_match": bool(score["official_match"]),
        "freshness_match": bool(score["freshness_match"]),
        "matched_query": str(score["matched_query"]),
        "matched_query_index": int(score["matched_query_index"]),
        "source_family": document.source_family,
        "source_family_confidence": 1.0 if score["family_match"] else 0.5,
        "published_at": document.published_at or document.timestamp,
        "metadata": dict(document.metadata),
    }


def _load_source_catalogs(
    source_catalog_paths: Sequence[str | Path],
    *,
    default_source_family: str,
) -> tuple[SourceCatalogDocument, ...]:
    documents: list[SourceCatalogDocument] = []
    for path in tuple(Path(item) for item in source_catalog_paths):
        documents.extend(_load_source_catalog(path, default_source_family=default_source_family))
    if not documents:
        raise ValueError("source catalog is empty.")
    return tuple(documents)


def _load_source_catalog(path: Path, *, default_source_family: str) -> tuple[SourceCatalogDocument, ...]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            raw_documents = (
                payload.get("documents")
                or payload.get("records")
                or payload.get("source_documents")
                or payload.get("results")
                or ()
            )
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            raw_documents = payload
        else:
            raise ValueError(f"{path} must contain a JSON object or list.")
        return tuple(
            _coerce_catalog_document(
                item,
                source_default=f"{path}:{idx}",
                default_source_family=default_source_family,
            )
            for idx, item in enumerate(_mapping_sequence(raw_documents), start=1)
        )
    if suffix == ".jsonl":
        documents: list[SourceCatalogDocument] = []
        with path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{path}:{idx} must contain a JSON object.")
                documents.append(_coerce_catalog_document(
                    payload,
                    source_default=f"{path}:{idx}",
                    default_source_family=default_source_family,
                ))
        return tuple(documents)
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if chunk.strip()]
    if len(chunks) == 1:
        chunks = [line.strip() for line in chunks[0].splitlines() if line.strip()]
    return tuple(
        SourceCatalogDocument(
            text=chunk,
            source=f"{path}#{idx}",
            source_family=default_source_family,
            metadata={"source_path": str(path), "source_format": "text"},
        )
        for idx, chunk in enumerate(chunks, start=1)
    )


def _coerce_catalog_document(
    item: Mapping[str, Any],
    *,
    source_default: str,
    default_source_family: str,
) -> SourceCatalogDocument:
    _reject_reserved_fields(item, source=source_default)
    metadata = dict(_mapping(item.get("metadata")))
    _reject_reserved_fields(metadata, source=source_default)
    for key, value in item.items():
        if key not in {
            "text",
            "content",
            "document",
            "body",
            "snippet",
            "summary",
            "abstract",
            "title",
            "source",
            "url",
            "href",
            "provider",
            "source_provider",
            "source_family",
            "source_family_name",
            "published_at",
            "publication_date",
            "timestamp",
            "retrieved_at",
            "score",
            "base_score",
            "metadata",
        } and key not in metadata:
            metadata[str(key)] = value
    text = _first_nonempty(item, TEXT_FIELDS)
    if not text:
        raise ValueError(f"source catalog document {source_default!r} has no text.")
    return SourceCatalogDocument(
        text=text,
        title=_clean(item.get("title")),
        source=_clean(item.get("source")) or source_default,
        url=_clean(item.get("url") or item.get("href")),
        provider=_clean(item.get("provider") or item.get("source_provider")) or PROVIDER,
        source_family=_clean(item.get("source_family") or item.get("source_family_name")) or default_source_family,
        published_at=_clean(item.get("published_at") or item.get("publication_date")),
        timestamp=_clean(item.get("timestamp") or item.get("retrieved_at")),
        base_score=float(item.get("base_score", item.get("score", 1.0))),
        metadata=metadata,
    )


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    catalog: Sequence[SourceCatalogDocument],
) -> dict[str, Any]:
    result_counts = [len(_result_items(row.get("results"))) for row in rows]
    request_count = len(rows)
    request_with_results_count = sum(1 for count in result_counts if count > 0)
    request_without_results_ids = tuple(
        _clean(row.get("request_id"))
        for row, count in zip(rows, result_counts)
        if count == 0 and _clean(row.get("request_id"))
    )
    families = Counter(document.source_family for document in catalog)
    result_families = Counter(
        str(result.get("source_family"))
        for row in rows
        for result in _result_items(row.get("results"))
        if result.get("source_family")
    )
    providers = Counter(document.provider for document in catalog)
    result_providers = Counter(
        str(result.get("provider"))
        for row in rows
        for result in _result_items(row.get("results"))
        if result.get("provider")
    )
    return {
        "request_count": request_count,
        "source_document_count": len(catalog),
        "request_error_count": sum(1 for row in rows if row.get("error")),
        "request_with_results_count": request_with_results_count,
        "request_without_results_count": sum(1 for count in result_counts if count == 0),
        "request_without_results_ids": request_without_results_ids,
        "request_coverage": 1.0 if request_count == 0 else request_with_results_count / request_count,
        "result_count": sum(result_counts),
        "catalog_source_family_counts": _sorted_counter(families),
        "result_source_family_counts": _sorted_counter(result_families),
        "catalog_provider_counts": _sorted_counter(providers),
        "result_provider_counts": _sorted_counter(result_providers),
    }


def _source_family_plan(request: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = request.get("source_family_plan")
    if isinstance(plan, Mapping):
        return plan
    metadata = _mapping(request.get("metadata"))
    return {
        "families": _string_sequence(metadata.get("preferred_source_families", ())),
        "freshness_required": bool(metadata.get("freshness_required")),
        "official_source_preferred": bool(metadata.get("official_source_preferred")),
        "query_hints": (),
    }


def _preferred_families(plan: Mapping[str, Any]) -> tuple[str, ...]:
    families = tuple(_normalize_family(item) for item in _string_sequence(plan.get("families", ())) if item)
    return families or (DEFAULT_SOURCE_FAMILY,)


def _family_priority(value: Any, preferred_families: Sequence[str]) -> int:
    family = _normalize_family(value)
    try:
        return len(preferred_families) - tuple(preferred_families).index(family)
    except ValueError:
        return 0


def _request_query_variants(request: Mapping[str, Any], *, max_items: int) -> tuple[str, ...]:
    values: list[str] = [_clean(request.get("query"))]
    raw_alternates = request.get("alternate_queries", ())
    if isinstance(raw_alternates, Sequence) and not isinstance(raw_alternates, (str, bytes, bytearray)):
        values.extend(_clean(item) for item in raw_alternates)
    seen: set[str] = set()
    variants: list[str] = []
    for value in values:
        if not value:
            continue
        folded = value.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        variants.append(value)
        if len(variants) >= int(max_items):
            break
    return tuple(variants)


def _token_overlap(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
    query_set = set(query_tokens)
    if not query_set:
        return 0.0
    document_set = set(document_tokens)
    if not document_set:
        return 0.0
    return len(query_set & document_set) / len(query_set)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN_RE.finditer(value))


def _error_row(request_id: str, query: str, error: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "results": (),
        "error": error,
        "metadata": {"query": query, "provider": PROVIDER},
    }


def _snippet(document: SourceCatalogDocument, *, limit: int = 240) -> str:
    text = document.text
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _first_nonempty(item: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field_name in fields:
        value = _clean(item.get(field_name))
        if value:
            return value
    return ""


def _reject_reserved_fields(item: Mapping[str, Any], *, source: str) -> None:
    reserved = sorted(set(str(key) for key in item) & RESERVED_CATALOG_FIELDS)
    if reserved:
        raise ValueError(f"source catalog document {source!r} contains reserved fields: {', '.join(reserved)}")


def _result_items(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _bool_metadata(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "on"}
    return False


def _normalize_family(value: Any) -> str:
    family = _clean(value).casefold().replace("-", "_").replace(" ", "_")
    return family or DEFAULT_SOURCE_FAMILY


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


def _load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            rows.append(dict(payload))
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-catalog", action="append", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--max-query-variants", type=int, default=3)
    parser.add_argument("--min-text-overlap", type=float, default=0.05)
    parser.add_argument("--diversify-source-families", action="store_true")
    parser.add_argument("--default-source-family", default=DEFAULT_SOURCE_FAMILY)
    parser.add_argument("--pretty-json", action="store_true")
    parser.add_argument("--fail-on-empty", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    args = parser.parse_args(argv)
    payload = run_source_family_citation_search_adapter(
        input_path=args.input,
        output_path=args.output,
        source_catalog_paths=tuple(args.source_catalog or ()),
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        max_results=args.max_results,
        max_query_variants=args.max_query_variants,
        min_text_overlap=args.min_text_overlap,
        diversify_source_families=bool(args.diversify_source_families),
        default_source_family=args.default_source_family,
        compact_json=not bool(args.pretty_json),
        fail_on_empty=bool(args.fail_on_empty),
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "source_family_citation_search_adapter_ok "
        f"requests={summary['request_count']} "
        f"source_docs={summary['source_document_count']} "
        f"with_results={summary['request_with_results_count']} "
        f"coverage={summary['request_coverage']:.3f} "
        f"results={summary['result_count']} "
        f"errors={summary['request_error_count']}"
    )


if __name__ == "__main__":
    main()
