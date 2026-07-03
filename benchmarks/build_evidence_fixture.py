"""Build local retrieval evidence fixtures for verifier-ensemble benchmarks.

This is a no-network bridge from statement-bearing score dumps to
``eval_verifier_ensemble.py`` claim fixtures. It retrieves evidence from local
JSON/JSONL/text corpora using dependency-free local retrievers and writes one
fixture record per score row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eigentruth.adapters import (
    InMemoryRetriever,
    ProvenanceFilteredRetriever,
    RetrievalHit,
    RetrievalQuery,
    SQLiteFTSRetriever,
    plan_triple_slot_retrieval,
)
from eigentruth.eval.score_dump import (
    ScoreDump,
    score_dump_file_metadata,
)
from eigentruth.eval.score_dump import (
    load_score_dump as _load_validated_score_dump,
)
from eigentruth.json_utils import strict_json_dumps
from eigentruth.registry import fingerprint_path
from eigentruth.verify.protocols import Claim
from eigentruth.verify.search_planning import SOURCE_FAMILY_NAMES, plan_citation_search_query

RETRIEVER_BACKENDS = ("memory", "sqlite_fts", "auto")
SOURCE_FAMILY_FILTERS = ("off", "planned", "planned_rerank")
QUERY_FIELDS = (
    "text",
    "answer",
    "question",
    "question_answer",
    "citation_question",
    "citation_entity",
    "triple_slot",
)
CITATION_QUERY_FIELDS = ("citation_question", "citation_entity")
_SOURCE_FAMILY_FILTER_FETCH_MULTIPLIER = 20
_SOURCE_FAMILY_COMPATIBILITY = {
    "official": ("official_statistics",),
    "official_statistics": ("official",),
    "reference": ("encyclopedic",),
    "encyclopedic": ("reference",),
}
_SOURCE_PREFIX_FAMILY_HINTS = (
    ("worldbank:", "official_statistics"),
    ("wikidata:", "reference"),
    ("wikipedia:", "encyclopedic"),
    ("openalex:", "scholarly"),
    ("official:", "official"),
    ("news:", "news"),
    ("qa:", "reference"),
)
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_ANSWER_CAPITALIZED_SPAN_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*")
_ANSWER_QUOTED_SPAN_RE = re.compile(r"[\"'“”‘’](?P<span>[^\"'“”‘’]{2,80})[\"'“”‘’]")
_SLOT_NORMALIZE_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")
_STRUCTURED_ENTITY_METADATA_KEYS = (
    "country_name",
    "subject_label",
    "subject",
    "entity",
    "entity_name",
    "organization_name",
    "location_name",
)
_STRUCTURED_PROPERTY_METADATA_KEYS = (
    "indicator_name",
    "statement_property_label",
    "property_label",
    "property",
    "indicator",
)
_STRUCTURED_CODE_METADATA_KEYS = (
    "country_code_iso3",
    "country_code_iso2",
    "subject_qid",
)
_SLOT_ALIASES = {
    "united states": ("united states of america", "usa", "us", "u s", "america"),
    "united kingdom": ("uk", "u k", "great britain", "britain"),
    "russian federation": ("russia",),
    "korea rep": ("south korea", "republic of korea"),
    "iran islamic rep": ("iran",),
    "egypt arab rep": ("egypt",),
    "bahamas the": ("bahamas", "the bahamas"),
    "gambia the": ("gambia", "the gambia"),
    "hong kong sar china": ("hong kong",),
    "macao sar china": ("macao", "macau"),
}
_RERANK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}
_ANSWER_SLOT_HINT_BLOCKLIST = {
    "am",
    "answer",
    "country",
    "data",
    "he",
    "i",
    "it",
    "it s",
    "no",
    "none",
    "nothing",
    "nowhere",
    "population",
    "she",
    "sibling",
    "siblings",
    "statistics",
    "there",
    "there s",
    "the country",
    "total",
    "unknown",
    "yes",
}


def load_score_dump(path: Path) -> dict[str, Any]:
    """Load and validate a statement-bearing score dump."""
    return _load_validated_score_dump(
        path,
        allow_missing_scores=True,
        require_statements=True,
    ).to_mapping()


def load_corpus(paths: Sequence[Path]) -> tuple[RetrievalHit, ...]:
    """Load local evidence documents from JSON, JSONL, or text files."""
    documents: list[RetrievalHit] = []
    for path in paths:
        if path.suffix.lower() == ".json":
            documents.extend(_documents_from_json(path))
        elif path.suffix.lower() == ".jsonl":
            documents.extend(_documents_from_jsonl(path))
        else:
            documents.extend(_documents_from_text(path))
    if not documents:
        raise ValueError("corpus must contain at least one non-empty document.")
    return tuple(documents)


def build_evidence_fixture(
    dump: Mapping[str, Any],
    corpus_documents: Sequence[RetrievalHit | Mapping[str, Any] | str],
    *,
    retriever_min_overlap: float = 0.2,
    retrieval_limit: int = 5,
    query_field: str = "text",
    retriever_backend: str = "memory",
    retriever_index_path: str | Path | None = None,
    include_label_metadata: bool = True,
    require_retrieval_source: bool = False,
    allowed_retrieval_source_prefixes: Sequence[str] = (),
    denied_retrieval_source_prefixes: Sequence[str] = (),
    min_retrieval_score: float = 0.0,
    required_retrieval_metadata: Mapping[str, Any] | None = None,
    max_retrieval_hits_per_source: int | None = None,
    source_family_filter: str = "off",
    source_binding_queue: Mapping[str, Any] | None = None,
    use_precomputed_retrieval_hits: bool = False,
) -> dict[str, Any]:
    """Build a claim/evidence fixture using only local retrieval over claim text."""
    if retrieval_limit <= 0:
        raise ValueError("retrieval_limit must be positive.")
    if query_field not in QUERY_FIELDS:
        raise ValueError(f"query_field must be one of: {', '.join(QUERY_FIELDS)}.")
    if retriever_backend not in RETRIEVER_BACKENDS:
        raise ValueError(f"retriever_backend must be one of: {', '.join(RETRIEVER_BACKENDS)}.")
    if retriever_backend == "memory" and retriever_index_path is not None:
        raise ValueError("retriever_index_path is only supported with sqlite_fts or auto backends.")
    source_family_filter = _source_family_filter(source_family_filter)
    labels = tuple(int(label) for label in dump.get("labels", ()))
    statements = tuple(dict(statement) for statement in dump.get("statements", ()))
    if len(labels) != len(statements):
        raise ValueError("labels and statements must have the same length.")

    documents = tuple(corpus_documents)
    source_binding_index = _source_binding_index(source_binding_queue)
    source_binding_document_cache: dict[tuple[str, ...], tuple[RetrievalHit | Mapping[str, Any] | str, ...]] = {}
    source_binding_retriever_cache: dict[tuple[str, ...], tuple[Any, Mapping[str, Any]]] = {}
    index_path = None if retriever_index_path is None else Path(retriever_index_path)
    provenance_filter = _retrieval_provenance_filter_config(
        require_source=require_retrieval_source,
        allowed_source_prefixes=allowed_retrieval_source_prefixes,
        denied_source_prefixes=denied_retrieval_source_prefixes,
        min_score=min_retrieval_score,
        required_metadata=required_retrieval_metadata,
        max_hits_per_source=max_retrieval_hits_per_source,
    )
    retriever, retriever_info = _build_retriever(
        documents,
        min_overlap=retriever_min_overlap,
        backend=retriever_backend,
        index_path=index_path,
        provenance_filter=provenance_filter,
    )
    records = []
    total_hits = 0
    total_candidate_hits = 0
    total_source_family_filtered_hits = 0
    source_bound_record_count = 0
    source_bound_hit_record_count = 0
    source_binding_fallback_count = 0
    for idx, (label, statement) in enumerate(zip(labels, statements), start=1):
        claim_text = _statement_text(statement)
        claim_id = str(statement.get("claim_id") or f"c{idx}")
        source_binding_keys = _statement_source_binding_keys(
            idx - 1,
            statement,
            source_binding_index=source_binding_index,
        )
        retrieval_queries, query_metadata = _retrieval_query_bundle(
            statement,
            query_field=query_field,
            claim_id=claim_id,
        )
        query_text = " | ".join(query.query for query in retrieval_queries)
        candidate_limit = _retrieval_candidate_limit(
            retrieval_limit,
            source_family_filter=source_family_filter,
        )
        raw_candidate_hits, source_binding_metadata = _retrieve_with_optional_source_binding(
            retrieval_queries,
            retriever=retriever,
            documents=documents,
            binding_keys=source_binding_keys,
            candidate_limit=candidate_limit,
            min_overlap=retriever_min_overlap,
            provenance_filter=provenance_filter,
            document_cache=source_binding_document_cache,
            retriever_cache=source_binding_retriever_cache,
        )
        if source_binding_metadata["requested"]:
            source_bound_record_count += 1
        if source_binding_metadata["mode"] == "exact":
            source_bound_hit_record_count += 1
        if source_binding_metadata["fallback"]:
            source_binding_fallback_count += 1
        candidate_hits, duplicate_candidate_hits = (
            _deduplicate_retrieval_hits(raw_candidate_hits)
            if len(retrieval_queries) > 1
            else (raw_candidate_hits, 0)
        )
        hits, source_family_filter_metadata = _apply_source_family_filter(
            candidate_hits,
            statement=statement,
            query_field=query_field,
            source_family_filter=source_family_filter,
            limit=retrieval_limit,
        )
        total_candidate_hits += len(candidate_hits)
        total_hits += len(hits)
        total_source_family_filtered_hits += int(source_family_filter_metadata.get("dropped_hit_count", 0))
        claim_metadata = dict(statement.get("metadata", {}))
        triple_slot_plan = query_metadata.get("triple_slot_plan")
        if isinstance(triple_slot_plan, Mapping) and triple_slot_plan.get("triples"):
            claim_metadata.setdefault("claim_triples", tuple(triple_slot_plan.get("triples", ())))
            claim_metadata.setdefault("triple_slot_query_plan", triple_slot_plan)

        record_metadata: dict[str, Any] = {
            "index": idx - 1,
            "statement": statement,
            "retrieval": {
                "n_hits": len(hits),
                "n_candidate_hits": len(candidate_hits),
                "retriever": retriever_info["type"],
                "requested_backend": retriever_info["requested_backend"],
                "actual_backend": retriever_info["actual_backend"],
                "fallback_reason": retriever_info.get("fallback_reason"),
                "requested_index_path": retriever_info.get("requested_index_path"),
                "actual_index_path": retriever_info.get("actual_index_path"),
                "index_reused": retriever_info.get("index_reused"),
                "min_overlap": retriever_min_overlap,
                "limit": retrieval_limit,
                "query_field": query_field,
                "query": query_text,
                "queries": tuple(query.to_dict() for query in retrieval_queries),
                "query_count": len(retrieval_queries),
                "raw_candidate_hit_count": len(raw_candidate_hits),
                "duplicate_candidate_hit_count": duplicate_candidate_hits,
                "query_plan": query_metadata,
                "provenance_filter": retriever_info.get("provenance_filter"),
                "source_binding": source_binding_metadata,
                "source_family_filter": source_family_filter_metadata,
                "use_precomputed_hits": bool(use_precomputed_retrieval_hits),
            },
        }
        if include_label_metadata:
            record_metadata["score_label"] = label
        records.append({
            "claim": claim_text,
            "claim_id": claim_id,
            "claim_metadata": claim_metadata,
            "retrieval_documents": [hit.to_dict() for hit in hits],
            "metadata": record_metadata,
        })

    return {
        "schema_version": 1,
        "fixture_type": "local_retrieval_evidence",
        "description": (
            "Evidence fixture built by local token-overlap retrieval over a supplied corpus. "
            "Labels are optional audit metadata only; retrieval uses claim text only."
        ),
        "label_usage": {
            "labels_used_for_retrieval": False,
            "labels_copied_to_record_metadata": include_label_metadata,
        },
        "retriever": {
            **retriever_info,
            "min_overlap": retriever_min_overlap,
            "limit": retrieval_limit,
            "query_field": query_field,
            "n_corpus_documents": len(documents),
            "source_family_filter": source_family_filter,
            "source_binding": _source_binding_retriever_summary(source_binding_index),
            "use_precomputed_hits": bool(use_precomputed_retrieval_hits),
        },
        "summary": {
            "n_records": len(records),
            "records_with_hits": sum(1 for record in records if record["retrieval_documents"]),
            "total_hits": total_hits,
            "total_candidate_hits": total_candidate_hits,
            "source_family_filtered_hits": total_source_family_filtered_hits,
            "source_bound_record_count": source_bound_record_count,
            "source_bound_hit_record_count": source_bound_hit_record_count,
            "source_binding_fallback_count": source_binding_fallback_count,
            "average_hits_per_record": float(total_hits) / len(records) if records else 0.0,
        },
        "records": records,
    }


def build_evidence_input_provenance(
    *,
    scores_path: str | Path,
    corpus_paths: Sequence[str | Path],
    score_dump: ScoreDump | Mapping[str, Any] | None = None,
    retriever_backend: str = "memory",
    retriever_index_path: str | Path | None = None,
    retriever_min_overlap: float = 0.2,
    retrieval_limit: int = 5,
    query_field: str = "text",
    include_label_metadata: bool = True,
    require_retrieval_source: bool = False,
    allowed_retrieval_source_prefixes: Sequence[str] = (),
    denied_retrieval_source_prefixes: Sequence[str] = (),
    min_retrieval_score: float = 0.0,
    required_retrieval_metadata: Mapping[str, Any] | None = None,
    max_retrieval_hits_per_source: int | None = None,
    source_family_filter: str = "off",
    source_binding_queue_path: str | Path | None = None,
    use_precomputed_retrieval_hits: bool = False,
) -> dict[str, Any]:
    """Build input fingerprints and builder settings for fixture reproducibility."""
    score_dump_obj = _coerce_score_dump_for_metadata(score_dump)
    index_path = None if retriever_index_path is None else Path(retriever_index_path)
    source_family_filter = _source_family_filter(source_family_filter)
    return {
        "schema_version": 1,
        "builder": "build_evidence_fixture",
        "score_dump": score_dump_file_metadata(scores_path, score_dump_obj),
        "corpora": [fingerprint_path(path).to_dict() for path in corpus_paths],
        "retriever_index": None if index_path is None else fingerprint_path(index_path).to_dict(),
        "source_binding_queue": (
            None if source_binding_queue_path is None else fingerprint_path(source_binding_queue_path).to_dict()
        ),
        "config": {
            "retriever_backend": retriever_backend,
            "retriever_min_overlap": float(retriever_min_overlap),
            "retrieval_limit": int(retrieval_limit),
            "query_field": query_field,
            "include_label_metadata": bool(include_label_metadata),
            "provenance_filter": _retrieval_provenance_filter_config(
                require_source=require_retrieval_source,
                allowed_source_prefixes=allowed_retrieval_source_prefixes,
                denied_source_prefixes=denied_retrieval_source_prefixes,
                min_score=min_retrieval_score,
                required_metadata=required_retrieval_metadata,
                max_hits_per_source=max_retrieval_hits_per_source,
            ),
            "source_family_filter": source_family_filter,
            "source_binding_queue_path": None if source_binding_queue_path is None else str(source_binding_queue_path),
            "use_precomputed_retrieval_hits": bool(use_precomputed_retrieval_hits),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    scores_path = Path(args.scores)
    corpus_paths = tuple(Path(path) for path in args.corpus)
    validated_dump = _load_validated_score_dump(
        scores_path,
        allow_missing_scores=True,
        require_statements=True,
    )
    dump = validated_dump.to_mapping()
    corpus = load_corpus(corpus_paths)
    source_binding_queue_path = getattr(args, "source_binding_queue", None)
    source_binding_queue = None if source_binding_queue_path is None else _load_json_object(source_binding_queue_path)
    include_label_metadata = not bool(args.omit_label_metadata)
    fixture = build_evidence_fixture(
        dump,
        corpus,
        retriever_min_overlap=args.retriever_min_overlap,
        retrieval_limit=args.retrieval_limit,
        query_field=args.query_field,
        retriever_backend=args.retriever_backend,
        retriever_index_path=args.retriever_index_path,
        include_label_metadata=include_label_metadata,
        require_retrieval_source=bool(getattr(args, "require_retrieval_source", False)),
        allowed_retrieval_source_prefixes=_parse_csv(getattr(args, "allowed_retrieval_source_prefix", None)),
        denied_retrieval_source_prefixes=_parse_csv(getattr(args, "denied_retrieval_source_prefix", None)),
        min_retrieval_score=float(getattr(args, "min_retrieval_score", 0.0)),
        required_retrieval_metadata=_parse_key_values(getattr(args, "required_retrieval_metadata", None)),
        max_retrieval_hits_per_source=getattr(args, "max_retrieval_hits_per_source", None),
        source_family_filter=getattr(args, "source_family_filter", "off"),
        source_binding_queue=source_binding_queue,
        use_precomputed_retrieval_hits=bool(getattr(args, "use_precomputed_retrieval_hits", False)),
    )
    fixture["input_provenance"] = build_evidence_input_provenance(
        scores_path=scores_path,
        corpus_paths=corpus_paths,
        score_dump=validated_dump,
        retriever_backend=args.retriever_backend,
        retriever_index_path=args.retriever_index_path,
        retriever_min_overlap=args.retriever_min_overlap,
        retrieval_limit=args.retrieval_limit,
        query_field=args.query_field,
        include_label_metadata=include_label_metadata,
        require_retrieval_source=bool(getattr(args, "require_retrieval_source", False)),
        allowed_retrieval_source_prefixes=_parse_csv(getattr(args, "allowed_retrieval_source_prefix", None)),
        denied_retrieval_source_prefixes=_parse_csv(getattr(args, "denied_retrieval_source_prefix", None)),
        min_retrieval_score=float(getattr(args, "min_retrieval_score", 0.0)),
        required_retrieval_metadata=_parse_key_values(getattr(args, "required_retrieval_metadata", None)),
        max_retrieval_hits_per_source=getattr(args, "max_retrieval_hits_per_source", None),
        source_family_filter=getattr(args, "source_family_filter", "off"),
        source_binding_queue_path=source_binding_queue_path,
        use_precomputed_retrieval_hits=bool(getattr(args, "use_precomputed_retrieval_hits", False)),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)
    summary = fixture["summary"]
    print(
        f"Wrote local evidence fixture to {output_path} "
        f"({summary['records_with_hits']}/{summary['n_records']} records with hits)"
    )
    return fixture


def _statement_text(statement: Mapping[str, Any]) -> str:
    text = str(statement.get("claim") or statement.get("text") or statement.get("answer") or "").strip()
    if not text:
        raise ValueError("statement record is missing claim/text/answer.")
    return text


def _coerce_score_dump_for_metadata(score_dump: ScoreDump | Mapping[str, Any] | None) -> ScoreDump | None:
    if score_dump is None:
        return None
    if isinstance(score_dump, ScoreDump):
        return score_dump
    return ScoreDump.from_mapping(
        score_dump,
        allow_missing_scores=True,
        require_statements=True,
    )


def _build_retriever(
    documents: Sequence[RetrievalHit | Mapping[str, Any] | str],
    *,
    min_overlap: float,
    backend: str,
    index_path: Path | None,
    provenance_filter: Mapping[str, Any] | None,
):
    requested_index_path = None if index_path is None else str(index_path)
    if backend == "memory":
        retriever = InMemoryRetriever(documents, min_overlap=min_overlap)
        retriever_info = {
            "type": type(retriever).__name__,
            "requested_backend": backend,
            "actual_backend": "memory",
            "fallback_reason": None,
            "requested_index_path": requested_index_path,
            "actual_index_path": None,
            "index_reused": False,
        }
        return _maybe_wrap_retriever(retriever, retriever_info, provenance_filter=provenance_filter)
    retriever = SQLiteFTSRetriever(documents, min_overlap=min_overlap, index_path=index_path)
    if retriever.available:
        retriever_info = {
            "type": type(retriever).__name__,
            "requested_backend": backend,
            "actual_backend": "sqlite_fts",
            "fallback_reason": None,
            "requested_index_path": requested_index_path,
            "actual_index_path": None if retriever.index_path is None else str(retriever.index_path),
            "index_reused": retriever.index_reused,
            "document_fingerprint": retriever.document_fingerprint,
        }
        return _maybe_wrap_retriever(retriever, retriever_info, provenance_filter=provenance_filter)
    if backend == "sqlite_fts":
        retriever_info = {
            "type": "InMemoryRetriever",
            "requested_backend": backend,
            "actual_backend": "memory",
            "fallback_reason": retriever.fallback_reason,
            "requested_index_path": requested_index_path,
            "actual_index_path": None,
            "index_reused": False,
            "document_fingerprint": retriever.document_fingerprint,
        }
        return _maybe_wrap_retriever(retriever, retriever_info, provenance_filter=provenance_filter)
    fallback = InMemoryRetriever(documents, min_overlap=min_overlap)
    retriever_info = {
        "type": type(fallback).__name__,
        "requested_backend": backend,
        "actual_backend": "memory",
        "fallback_reason": retriever.fallback_reason,
        "requested_index_path": requested_index_path,
        "actual_index_path": None,
        "index_reused": False,
        "document_fingerprint": retriever.document_fingerprint,
    }
    return _maybe_wrap_retriever(fallback, retriever_info, provenance_filter=provenance_filter)


def _maybe_wrap_retriever(
    retriever,
    retriever_info: Mapping[str, Any],
    *,
    provenance_filter: Mapping[str, Any] | None,
):
    info = dict(retriever_info)
    if not provenance_filter or not _provenance_filter_enabled(provenance_filter):
        info["provenance_filter"] = None
        return retriever, info
    wrapped = ProvenanceFilteredRetriever(
        retriever,
        min_score=float(provenance_filter["min_score"]),
        require_source=bool(provenance_filter["require_source"]),
        allowed_source_prefixes=tuple(provenance_filter["allowed_source_prefixes"]),
        denied_source_prefixes=tuple(provenance_filter["denied_source_prefixes"]),
        required_metadata=dict(provenance_filter["required_metadata"]),
        max_hits_per_source=provenance_filter["max_hits_per_source"],
    )
    info["wrapped_type"] = info.get("type")
    info["type"] = type(wrapped).__name__
    info["provenance_filter"] = dict(provenance_filter)
    return wrapped, info


def _retrieval_provenance_filter_config(
    *,
    require_source: bool,
    allowed_source_prefixes: Sequence[str],
    denied_source_prefixes: Sequence[str],
    min_score: float,
    required_metadata: Mapping[str, Any] | None,
    max_hits_per_source: int | None,
) -> dict[str, Any]:
    score = float(min_score)
    if not (0.0 <= score <= 1.0):
        raise ValueError("min_retrieval_score must be in [0, 1].")
    if max_hits_per_source is not None:
        max_hits_per_source = int(max_hits_per_source)
        if max_hits_per_source <= 0:
            raise ValueError("max_retrieval_hits_per_source must be positive when set.")
    return {
        "require_source": bool(require_source),
        "allowed_source_prefixes": _clean_string_tuple(
            allowed_source_prefixes,
            name="allowed_retrieval_source_prefixes",
        ),
        "denied_source_prefixes": _clean_string_tuple(
            denied_source_prefixes,
            name="denied_retrieval_source_prefixes",
        ),
        "min_score": score,
        "required_metadata": dict(required_metadata or {}),
        "max_hits_per_source": max_hits_per_source,
    }


def _provenance_filter_enabled(config: Mapping[str, Any]) -> bool:
    return (
        bool(config.get("require_source"))
        or bool(config.get("allowed_source_prefixes"))
        or bool(config.get("denied_source_prefixes"))
        or float(config.get("min_score", 0.0)) > 0.0
        or bool(config.get("required_metadata"))
        or config.get("max_hits_per_source") is not None
    )


def _source_binding_index(queue_report: Mapping[str, Any] | None) -> dict[int, tuple[str, ...]]:
    if queue_report is None:
        return {}
    requests = _mapping_sequence(queue_report.get("adapter_requests", ()))
    by_record: dict[int, set[str]] = {}
    for request in requests:
        record_index = _request_record_index(request)
        if record_index is None:
            continue
        keys = _request_source_binding_keys(request)
        if keys:
            by_record.setdefault(record_index, set()).update(keys)
    return {
        record_index: tuple(sorted(keys))
        for record_index, keys in sorted(by_record.items())
        if keys
    }


def _statement_source_binding_keys(
    record_index: int,
    statement: Mapping[str, Any],
    *,
    source_binding_index: Mapping[int, Sequence[str]],
) -> tuple[str, ...]:
    explicit_keys = _source_binding_keys_from_mapping(statement)
    metadata = statement.get("metadata")
    if isinstance(metadata, Mapping):
        explicit_keys.update(_source_binding_keys_from_mapping(metadata))
    if explicit_keys:
        return tuple(sorted(explicit_keys))
    return tuple(source_binding_index.get(int(record_index), ()))


def _request_source_binding_keys(request: Mapping[str, Any]) -> set[str]:
    keys = _source_binding_keys_from_mapping(request)
    metadata = request.get("metadata")
    if isinstance(metadata, Mapping):
        keys.update(_source_binding_keys_from_mapping(metadata))
    keys.add(_sha256_json(_minimal_source_request_fingerprint(request)))
    return {key for key in keys if key}


def _request_record_index(request: Mapping[str, Any]) -> int | None:
    for key in ("record_index", "source_index", "row_index"):
        value = _optional_int(request.get(key))
        if value is not None and value >= 0:
            return value
    for key in ("target_id", "record_id", "source_request_id", "queue_id", "request_id"):
        match = re.search(r"(?:record|row|source)[-_:]?(\d+)", str(request.get(key) or ""), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _minimal_source_request_fingerprint(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queue_id": request.get("queue_id"),
        "source_request_id": request.get("source_request_id"),
        "adapter_family": request.get("adapter_family"),
        "request_type": request.get("request_type"),
        "question": request.get("question"),
        "query": request.get("query"),
    }


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(strict_json_dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _retrieve_with_optional_source_binding(
    queries: Sequence[RetrievalQuery],
    *,
    retriever,
    documents: Sequence[RetrievalHit | Mapping[str, Any] | str],
    binding_keys: Sequence[str],
    candidate_limit: int,
    min_overlap: float,
    provenance_filter: Mapping[str, Any] | None,
    document_cache: dict[tuple[str, ...], tuple[RetrievalHit | Mapping[str, Any] | str, ...]],
    retriever_cache: dict[tuple[str, ...], tuple[Any, Mapping[str, Any]]],
) -> tuple[tuple[RetrievalHit, ...], dict[str, Any]]:
    key_tuple = tuple(sorted(dict.fromkeys(str(key).strip() for key in binding_keys if str(key).strip())))
    metadata: dict[str, Any] = {
        "requested": bool(key_tuple),
        "mode": "none",
        "fallback": False,
        "binding_key_count": len(key_tuple),
        "bound_document_count": 0,
        "bound_candidate_hit_count": 0,
        "fallback_candidate_hit_count": 0,
    }
    if not key_tuple:
        return _retrieve_candidates(queries, retriever=retriever, limit=candidate_limit), metadata

    bound_documents = document_cache.get(key_tuple)
    if bound_documents is None:
        bound_documents = _source_bound_documents(documents, key_tuple)
        document_cache[key_tuple] = bound_documents
    metadata["bound_document_count"] = len(bound_documents)
    if bound_documents:
        cached = retriever_cache.get(key_tuple)
        if cached is None:
            cached = _build_retriever(
                bound_documents,
                min_overlap=float(min_overlap),
                backend="memory",
                index_path=None,
                provenance_filter=provenance_filter,
            )
            retriever_cache[key_tuple] = cached
        bound_retriever, bound_info = cached
        bound_hits = _retrieve_candidates(queries, retriever=bound_retriever, limit=candidate_limit)
        metadata["bound_candidate_hit_count"] = len(bound_hits)
        metadata["bound_actual_backend"] = bound_info.get("actual_backend")
        if bound_hits:
            metadata["mode"] = "exact"
            return bound_hits, metadata
        metadata["mode"] = "fallback_no_bound_hits"
    else:
        metadata["mode"] = "fallback_no_bound_documents"

    fallback_hits = _retrieve_candidates(queries, retriever=retriever, limit=candidate_limit)
    metadata["fallback"] = True
    metadata["fallback_candidate_hit_count"] = len(fallback_hits)
    return fallback_hits, metadata


def _retrieve_candidates(
    queries: Sequence[RetrievalQuery],
    *,
    retriever,
    limit: int,
) -> tuple[RetrievalHit, ...]:
    return tuple(hit for query in queries for hit in retriever.retrieve(query, limit=limit))


def _source_bound_documents(
    documents: Sequence[RetrievalHit | Mapping[str, Any] | str],
    binding_keys: Sequence[str],
) -> tuple[RetrievalHit | Mapping[str, Any] | str, ...]:
    keys = set(binding_keys)
    return tuple(
        document
        for document in documents
        if keys & _document_source_binding_keys(document)
    )


def _document_source_binding_keys(document: RetrievalHit | Mapping[str, Any] | str) -> set[str]:
    if isinstance(document, RetrievalHit):
        keys = _source_binding_keys_from_mapping(document.metadata)
        return keys
    if isinstance(document, Mapping):
        keys = _source_binding_keys_from_mapping(document)
        metadata = document.get("metadata")
        if isinstance(metadata, Mapping):
            keys.update(_source_binding_keys_from_mapping(metadata))
        return keys
    return set()


def _source_binding_keys_from_mapping(payload: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field_name in (
        "source_queue_request_sha256",
        "source_request_sha256",
        "collection_request_sha256",
    ):
        keys.update(_string_tuple(payload.get(field_name)))
    return {key for key in keys if key}


def _source_binding_retriever_summary(index: Mapping[int, Sequence[str]]) -> dict[str, Any]:
    key_count = sum(len(tuple(keys)) for keys in index.values())
    return {
        "enabled": bool(index),
        "bound_record_count": len(index),
        "binding_key_count": key_count,
    }


def _clean_string_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    cleaned = tuple(str(value).strip() for value in values)
    if any(not value for value in cleaned):
        raise ValueError(f"{name} must contain non-empty strings.")
    return cleaned


def _query_text(statement: Mapping[str, Any], *, query_field: str) -> str:
    if query_field == "text":
        return _statement_text(statement)
    if query_field == "answer":
        text = str(statement.get("answer", "")).strip()
    elif query_field == "question":
        text = str(statement.get("question", "")).strip()
    elif query_field == "question_answer":
        text = f"{statement.get('question', '')} {statement.get('answer', '')}".strip()
    elif query_field in CITATION_QUERY_FIELDS:
        text = _citation_query_text(statement, query_field=query_field)
    elif query_field == "triple_slot":
        text = _triple_slot_query_text(statement, claim_id=None)
    else:
        raise ValueError(f"query_field must be one of: {', '.join(QUERY_FIELDS)}.")
    if not text:
        raise ValueError(f"statement record is missing query field {query_field!r}.")
    return text


def _retrieval_query_bundle(
    statement: Mapping[str, Any],
    *,
    query_field: str,
    claim_id: str,
) -> tuple[tuple[RetrievalQuery, ...], dict[str, Any]]:
    if query_field != "triple_slot":
        query = RetrievalQuery(query=_query_text(statement, query_field=query_field), claim_id=claim_id)
        return (query,), {
            "query_type": query_field,
            "fallback_query_field": None,
            "triple_slot_plan": None,
        }

    claim = _triple_slot_claim(statement, claim_id=claim_id)
    plan = plan_triple_slot_retrieval(claim)
    if plan.queries:
        return tuple(plan.queries), {
            "query_type": "triple_slot",
            "fallback_query_field": None,
            "triple_slot_plan": plan.to_dict(),
        }

    fallback_field = _triple_slot_fallback_query_field(statement)
    fallback_query = RetrievalQuery(
        query=_query_text(statement, query_field=fallback_field),
        claim_id=claim_id,
        metadata={
            "query_type": "triple_slot",
            "fallback_query_field": fallback_field,
            "triple_slot_plan": plan.to_dict(),
        },
    )
    return (fallback_query,), {
        "query_type": "triple_slot",
        "fallback_query_field": fallback_field,
        "triple_slot_plan": plan.to_dict(),
    }


def _triple_slot_query_text(statement: Mapping[str, Any], *, claim_id: str | None) -> str:
    claim = _triple_slot_claim(statement, claim_id=claim_id)
    plan = plan_triple_slot_retrieval(claim)
    if plan.queries:
        return " | ".join(query.query for query in plan.queries)
    return _query_text(statement, query_field=_triple_slot_fallback_query_field(statement))


def _triple_slot_claim(statement: Mapping[str, Any], *, claim_id: str | None) -> Claim:
    question_answer = f"{statement.get('question', '')} {statement.get('answer', '')}".strip()
    text = question_answer or _statement_text(statement)
    metadata = dict(statement.get("metadata", {}))
    return Claim(text=text, claim_id=claim_id, metadata=metadata)


def _triple_slot_fallback_query_field(statement: Mapping[str, Any]) -> str:
    if str(statement.get("question", "")).strip() and str(statement.get("answer", "")).strip():
        return "question_answer"
    return "text"


def _citation_query_text(statement: Mapping[str, Any], *, query_field: str) -> str:
    plan = _citation_query_plan(statement, query_field=query_field)
    source_plan = plan.source_family_plan
    source_hints = () if source_plan is None else tuple(source_plan.query_hints)
    return " ".join(
        part
        for part in (
            plan.query,
            *tuple(plan.alternate_queries),
            *source_hints,
        )
        if str(part).strip()
    ).strip()


def _citation_query_plan(statement: Mapping[str, Any], *, query_field: str):
    question = str(statement.get("question", "")).strip()
    if not question:
        raise ValueError(f"statement record is missing query field {query_field!r}.")
    strategy = "claim_entity" if query_field == "citation_entity" else "question_and_query"
    answer_hints = _answer_slot_hints(statement, question_text=question)
    return plan_citation_search_query(
        question=question,
        candidate_query=" ".join(answer_hints),
        question_type=_question_type(statement),
        strategy=strategy,
    )


def _question_type(statement: Mapping[str, Any]) -> str:
    metadata = statement.get("metadata")
    metadata_question_type = ""
    if isinstance(metadata, Mapping):
        metadata_question_type = str(metadata.get("question_type") or "").strip()
    return str(statement.get("question_type") or metadata_question_type).strip()


def _source_family_filter(value: str) -> str:
    mode = str(value).strip().casefold() or "off"
    if mode not in SOURCE_FAMILY_FILTERS:
        raise ValueError(f"source_family_filter must be one of: {', '.join(SOURCE_FAMILY_FILTERS)}.")
    return mode


def _retrieval_candidate_limit(limit: int, *, source_family_filter: str) -> int:
    if source_family_filter == "off":
        return int(limit)
    return int(limit) * _SOURCE_FAMILY_FILTER_FETCH_MULTIPLIER


def _apply_source_family_filter(
    hits: Sequence[RetrievalHit],
    *,
    statement: Mapping[str, Any],
    query_field: str,
    source_family_filter: str,
    limit: int,
) -> tuple[tuple[RetrievalHit, ...], dict[str, Any]]:
    if source_family_filter == "off":
        kept = tuple(hits[:limit])
        return kept, {
            "mode": "off",
            "candidate_hit_count": len(hits),
            "kept_hit_count": len(kept),
            "dropped_hit_count": 0,
        }

    plan = _source_family_plan(statement, query_field=query_field)
    slot_binding = _statement_slot_binding_metadata(statement)
    unique_hits, duplicate_hit_count = _deduplicate_retrieval_hits(hits)
    if plan is None:
        kept = tuple(unique_hits[:limit])
        return kept, {
            "mode": source_family_filter,
            "status": "skipped",
            "reason": "missing_question",
            "candidate_hit_count": len(hits),
            "unique_candidate_hit_count": len(unique_hits),
            "duplicate_hit_count": duplicate_hit_count,
            "kept_hit_count": len(kept),
            "dropped_hit_count": 0,
            "slot_binding": slot_binding,
        }

    accepted_families = _compatible_source_families(plan)
    compatible: list[RetrievalHit] = []
    incompatible: list[RetrievalHit] = []
    incompatible_examples: list[dict[str, Any]] = []
    for hit in unique_hits:
        hit_families = _hit_source_families(hit)
        structured_slot_score = _structured_slot_match_score(statement, hit)
        if hit_families and any(family in accepted_families for family in hit_families):
            compatible.append(_annotate_source_family_hit(
                hit,
                mode=source_family_filter,
                source_families=hit_families,
                accepted_families=accepted_families,
                structured_slot_score=structured_slot_score,
            ))
            continue
        incompatible.append(_annotate_source_family_hit(
            hit,
            mode=source_family_filter,
            source_families=hit_families,
            accepted_families=accepted_families,
            structured_slot_score=structured_slot_score,
        ))
        if len(incompatible_examples) < 5:
            incompatible_examples.append({
                "source": hit.source,
                "source_families": hit_families,
                "score": hit.score,
                "structured_slot_score": structured_slot_score,
            })

    if source_family_filter == "planned_rerank":
        compatible = _source_family_ranked_hits(compatible, statement=statement, plan=plan)
        incompatible = _source_family_ranked_hits(incompatible, statement=statement, plan=plan)
        kept = tuple((compatible + incompatible)[:limit])
        dropped_count = 0
        status = "reranked"
    else:
        compatible = _source_family_ranked_hits(compatible, statement=statement, plan=plan)
        kept = tuple(compatible[:limit])
        dropped_count = len(incompatible)
        status = "applied"

    return tuple(kept), {
        "mode": source_family_filter,
        "status": status,
        "candidate_hit_count": len(hits),
        "unique_candidate_hit_count": len(unique_hits),
        "duplicate_hit_count": duplicate_hit_count,
        "kept_hit_count": len(kept),
        "dropped_hit_count": dropped_count,
        "compatible_hit_count": len(compatible),
        "incompatible_hit_count": len(incompatible),
        "planned_families": tuple(plan.families),
        "accepted_families": accepted_families,
        "official_source_preferred": bool(plan.official_source_preferred),
        "rationale": tuple(plan.rationale),
        "slot_binding": slot_binding,
        "incompatible_hit_examples": tuple(incompatible_examples),
        "dropped_hit_examples": tuple(incompatible_examples),
    }


def _deduplicate_retrieval_hits(hits: Sequence[RetrievalHit]) -> tuple[tuple[RetrievalHit, ...], int]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[RetrievalHit] = []
    for hit in hits:
        key = _retrieval_hit_dedup_key(hit)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return tuple(unique), len(hits) - len(unique)


def _retrieval_hit_dedup_key(hit: RetrievalHit) -> tuple[str, str, str]:
    metadata = dict(hit.metadata)
    source = "" if hit.source is None else str(hit.source).strip()
    if source:
        return (source, "", "")
    result_id = str(metadata.get("result_sha256") or metadata.get("url") or "").strip()
    if result_id:
        return ("", result_id, "")
    return ("", "", hit.text.strip())


def _source_family_ranked_hits(
    hits: Sequence[RetrievalHit],
    *,
    statement: Mapping[str, Any],
    plan,
) -> list[RetrievalHit]:
    family_order = _source_family_preference_order(plan)
    return sorted(
        hits,
        key=lambda hit: (
            _source_family_rank(hit, family_order),
            -_structured_slot_match_score(statement, hit),
            -_evidence_text_overlap(statement, hit),
            -float(hit.score),
            "" if hit.source is None else str(hit.source),
        ),
    )


def _source_family_preference_order(plan) -> tuple[str, ...]:
    ordered: list[str] = []
    for raw_family in tuple(plan.families):
        family = _normalize_source_family(raw_family)
        for candidate in (family, *tuple(_SOURCE_FAMILY_COMPATIBILITY.get(family, ()))):
            if candidate and candidate not in ordered:
                ordered.append(candidate)
    return tuple(ordered)


def _source_family_rank(hit: RetrievalHit, family_order: Sequence[str]) -> int:
    hit_families = _hit_source_families(hit)
    ranks = [family_order.index(family) for family in hit_families if family in family_order]
    return min(ranks) if ranks else len(family_order) + 1


def _evidence_text_overlap(statement: Mapping[str, Any], hit: RetrievalHit) -> float:
    claim_tokens = _rerank_tokens(_statement_text(statement))
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(_rerank_tokens(hit.text))
    if not evidence_tokens:
        return 0.0
    return len(set(claim_tokens) & evidence_tokens) / len(set(claim_tokens))


def _structured_slot_match_score(statement: Mapping[str, Any], hit: RetrievalHit) -> float:
    """Score structured source metadata against the question-side fact slot."""
    slot_text = _statement_slot_text(statement)
    slot_tokens = set(_rerank_tokens(slot_text))
    slot_normalized = _normalize_slot_text(slot_text)
    metadata = dict(hit.metadata)

    entity_score = max(
        (
            _slot_value_match_score(slot_normalized, slot_tokens, value)
            for value in _metadata_slot_values(metadata, _STRUCTURED_ENTITY_METADATA_KEYS)
        ),
        default=0.0,
    )
    property_score = max(
        (
            _property_slot_match_score(slot_tokens, value)
            for value in _metadata_slot_values(metadata, _STRUCTURED_PROPERTY_METADATA_KEYS)
        ),
        default=0.0,
    )
    code_score = max(
        (
            _code_slot_match_score(slot_normalized, value)
            for value in _metadata_slot_values(metadata, _STRUCTURED_CODE_METADATA_KEYS)
        ),
        default=0.0,
    )
    return round(entity_score + property_score + code_score, 6)


def _statement_slot_text(statement: Mapping[str, Any]) -> str:
    question_text = _statement_question_text(statement) or _statement_text(statement)
    answer_hints = _answer_slot_hints(statement, question_text=question_text)
    return " ".join(part for part in (question_text, *answer_hints) if part)


def _statement_slot_binding_metadata(statement: Mapping[str, Any]) -> dict[str, Any]:
    question_text = _statement_question_text(statement) or _statement_text(statement)
    answer_hints = _answer_slot_hints(statement, question_text=question_text)
    return {
        "answer_slot_hints": answer_hints,
        "answer_slot_hint_count": len(answer_hints),
        "mode": "question_plus_answer_entity_hints" if answer_hints else "question_only",
    }


def _statement_question_text(statement: Mapping[str, Any]) -> str:
    question = str(statement.get("question", "")).strip()
    if question:
        return question
    metadata = statement.get("metadata")
    if isinstance(metadata, Mapping):
        return str(metadata.get("question", "")).strip()
    return ""


def _answer_slot_hints(
    statement: Mapping[str, Any],
    *,
    question_text: str,
) -> tuple[str, ...]:
    answer = str(statement.get("answer", "")).strip()
    if not answer:
        return ()
    question_normalized = _normalize_slot_text(question_text)
    question_tokens = set(_rerank_tokens(question_text))
    hints: list[str] = []
    for candidate in _answer_entity_candidates(answer, max_items=8):
        normalized = _normalize_slot_text(candidate)
        if not _valid_answer_slot_hint(
            candidate,
            normalized=normalized,
            question_tokens=question_tokens,
        ):
            continue
        if normalized and normalized in question_normalized:
            continue
        hints.append(candidate)
    return _unique(hints)


def _answer_entity_candidates(answer: str, *, max_items: int) -> tuple[str, ...]:
    candidates: list[str] = []
    for match in _ANSWER_QUOTED_SPAN_RE.finditer(answer):
        candidates.append(_clean_answer_slot_candidate(match.group("span")))
    for match in _ANSWER_CAPITALIZED_SPAN_RE.finditer(answer):
        candidates.append(_strip_answer_leading_entity_words(_clean_answer_slot_candidate(match.group(0))))
    return _unique(candidates)[: int(max_items)]


def _clean_answer_slot_candidate(value: str) -> str:
    return str(value).strip(" \t\r\n?.!,;:\"'()[]{}")


def _strip_answer_leading_entity_words(value: str) -> str:
    parts = value.split()
    while len(parts) > 1 and parts[0].casefold() in {"a", "an", "the", "this", "that", "these", "those", "all"}:
        parts.pop(0)
    return " ".join(parts)


def _valid_answer_slot_hint(
    candidate: str,
    *,
    normalized: str,
    question_tokens: set[str],
) -> bool:
    if not normalized or normalized in _ANSWER_SLOT_HINT_BLOCKLIST:
        return False
    if any(character.isdigit() for character in normalized):
        return False
    tokens = tuple(token for token in normalized.split() if token)
    if not tokens:
        return False
    if all(token in _RERANK_STOPWORDS or token in _ANSWER_SLOT_HINT_BLOCKLIST for token in tokens):
        return False
    return _has_new_proper_entity_token(candidate, question_tokens=question_tokens)


def _has_new_proper_entity_token(candidate: str, *, question_tokens: set[str]) -> bool:
    for match in re.finditer(r"[A-Za-z][A-Za-z&.'-]*", str(candidate)):
        raw_token = match.group(0).strip(".'-")
        token = raw_token.casefold()
        if not raw_token or len(token) <= 1:
            continue
        if token in question_tokens or token in _RERANK_STOPWORDS or token in _ANSWER_SLOT_HINT_BLOCKLIST:
            continue
        if raw_token[0].isupper():
            return True
    return False


def _metadata_slot_values(metadata: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        raw = metadata.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
            values.extend(str(item) for item in raw if str(item).strip())
        elif isinstance(raw, Mapping):
            values.extend(str(item) for item in raw.values() if str(item).strip())
        else:
            values.append(str(raw))
    return tuple(value.strip() for value in values if value.strip())


def _slot_value_match_score(question_normalized: str, question_tokens: set[str], value: str) -> float:
    aliases = _slot_aliases(value)
    scores = []
    for alias in aliases:
        normalized = _normalize_slot_text(alias)
        if not normalized:
            continue
        if normalized in question_normalized:
            scores.append(2.0)
            continue
        alias_tokens = set(_rerank_tokens(alias))
        if alias_tokens and alias_tokens <= question_tokens:
            scores.append(1.5)
        elif len(alias_tokens) > 1 and alias_tokens & question_tokens:
            scores.append(len(alias_tokens & question_tokens) / len(alias_tokens))
    return max(scores, default=0.0)


def _property_slot_match_score(question_tokens: set[str], value: str) -> float:
    property_tokens = set(_rerank_tokens(value))
    if not property_tokens:
        return 0.0
    overlap = len(property_tokens & question_tokens) / len(property_tokens)
    if overlap <= 0.0:
        return 0.0
    return min(1.0, max(0.25, overlap))


def _code_slot_match_score(question_normalized: str, value: str) -> float:
    normalized = _normalize_slot_text(value)
    if len(normalized) < 2:
        return 0.0
    if len(normalized) == 2 and normalized not in {"uk", "us"}:
        return 0.0
    padded_question = f" {question_normalized} "
    if f" {normalized} " in padded_question:
        return 0.5
    if normalized == "us" and " u s " in padded_question:
        return 0.5
    if normalized == "uk" and " u k " in padded_question:
        return 0.5
    return 0.0


def _slot_aliases(value: str) -> tuple[str, ...]:
    cleaned = str(value).strip()
    if not cleaned:
        return ()
    aliases = [cleaned]
    normalized = _normalize_slot_text(cleaned)
    if normalized:
        aliases.extend(_SLOT_ALIASES.get(normalized, ()))
    if ", the" in cleaned.casefold():
        aliases.append(cleaned.replace(", The", "").replace(", the", ""))
        aliases.append(f"The {cleaned.split(',', 1)[0].strip()}")
    return tuple(item for item in _unique(aliases) if str(item).strip())


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = str(value).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(str(value))
    return tuple(unique)


def _normalize_slot_text(value: str) -> str:
    normalized = _SLOT_NORMALIZE_RE.sub(" ", str(value).casefold())
    return " ".join(normalized.split())


def _rerank_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text)))
        if token and token not in _RERANK_STOPWORDS
    )


def _source_family_plan(statement: Mapping[str, Any], *, query_field: str):
    question = str(statement.get("question", "")).strip()
    if not question:
        return None
    strategy = "claim_entity" if query_field == "citation_entity" else "question"
    return plan_citation_search_query(
        question=question,
        candidate_query="",
        question_type=_question_type(statement),
        strategy=strategy,
    ).source_family_plan


def _compatible_source_families(plan) -> tuple[str, ...]:
    families = {_normalize_source_family(family) for family in plan.families}
    for family in tuple(families):
        families.update(_SOURCE_FAMILY_COMPATIBILITY.get(family, ()))
    return tuple(sorted(family for family in families if family))


def _hit_source_families(hit: RetrievalHit) -> tuple[str, ...]:
    metadata = dict(hit.metadata)
    raw_family = metadata.get("source_family", metadata.get("source_type", metadata.get("family")))
    families = tuple(
        family
        for family in _normalize_source_family_values(raw_family)
        if family in SOURCE_FAMILY_NAMES
    )
    if families:
        return families
    source = "" if hit.source is None else str(hit.source).casefold()
    for prefix, family in _SOURCE_PREFIX_FAMILY_HINTS:
        if source.startswith(prefix):
            return (family,)
    return ()


def _normalize_source_family_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_normalize_source_family(value),)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(_normalize_source_family(item) for item in value)
    return (_normalize_source_family(value),)


def _normalize_source_family(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _annotate_source_family_hit(
    hit: RetrievalHit,
    *,
    mode: str,
    source_families: Sequence[str],
    accepted_families: Sequence[str],
    structured_slot_score: float = 0.0,
) -> RetrievalHit:
    metadata = {
        **dict(hit.metadata),
        "source_family_filter": {
            "mode": mode,
            "source_families": tuple(source_families),
            "accepted_families": tuple(accepted_families),
            "structured_slot_score": float(structured_slot_score),
        },
    }
    return RetrievalHit(hit.text, hit.source, hit.score, metadata)


def _documents_from_json(path: Path) -> list[RetrievalHit]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, Mapping):
        raw_documents = payload.get("documents", payload.get("records", ()))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        raw_documents = payload
    else:
        raise ValueError(f"{path} must contain a JSON object or list.")
    return [_coerce_document(item, source_default=str(path)) for item in raw_documents]


def _documents_from_jsonl(path: Path) -> list[RetrievalHit]:
    documents = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = {"text": line, "source": f"{path}:{line_no}"}
            documents.append(_coerce_document(item, source_default=f"{path}:{line_no}"))
    return documents


def _documents_from_text(path: Path) -> list[RetrievalHit]:
    text = path.read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if len(chunks) == 1:
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        RetrievalHit(
            text=chunk,
            source=f"{path}#{idx}",
            metadata={"loader": "text", "path": str(path), "chunk_index": idx},
        )
        for idx, chunk in enumerate(chunks, start=1)
    ]


def _coerce_document(value: Any, *, source_default: str) -> RetrievalHit:
    if isinstance(value, str):
        return RetrievalHit(value, source=source_default, metadata={"loader": "json"})
    if not isinstance(value, Mapping):
        raise ValueError("corpus documents must be strings or mappings.")
    raw_source = value.get("source")
    source = source_default if raw_source is None else str(raw_source)
    metadata = {"loader": "json", **dict(value.get("metadata", {}))}
    for key in ("question", "answer"):
        if key in value and value[key] is not None:
            metadata[key] = str(value[key])
    for key, item in value.items():
        metadata_key = str(key)
        if metadata_key not in {"text", "content", "source", "score", "metadata"} and metadata_key not in metadata:
            metadata[metadata_key] = item
    return RetrievalHit(
        text=str(value.get("text", value.get("content", ""))),
        source=source,
        score=float(value.get("score", 1.0)),
        metadata=metadata,
    )


def _parse_csv(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    parts: list[str] = []
    for value in values:
        parts.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tuple(parts)


def _parse_key_values(values: Sequence[str] | None) -> dict[str, str]:
    metadata = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"metadata entry must be key=value: {value!r}")
        key, text = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key must not be empty: {value!r}")
        metadata[key] = text
    return metadata


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local evidence fixture for verifier ensemble benchmarks")
    parser.add_argument("--scores", required=True, help="statement-bearing score dump")
    parser.add_argument("--corpus", action="append", required=True,
                        help="local evidence corpus path; supports JSON, JSONL, and text; repeatable")
    parser.add_argument("--output", required=True, help="path to write claim/evidence fixture JSON")
    parser.add_argument("--retriever-min-overlap", type=float, default=0.2)
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--retriever-backend", choices=RETRIEVER_BACKENDS, default="memory")
    parser.add_argument("--retriever-index-path", default=None,
                        help="optional persistent SQLite FTS index path for sqlite_fts/auto backends")
    parser.add_argument("--query-field", choices=QUERY_FIELDS, default="text",
                        help="statement field used for retrieval query; claim text remains unchanged")
    parser.add_argument("--omit-label-metadata", action="store_true",
                        help="do not copy score labels into fixture record metadata")
    parser.add_argument("--require-retrieval-source", action="store_true",
                        help="drop retrieved hits that do not carry a source")
    parser.add_argument("--allowed-retrieval-source-prefix", action="append", default=None,
                        help="allowed source prefix for retrieved hits; comma-separated or repeatable")
    parser.add_argument("--denied-retrieval-source-prefix", action="append", default=None,
                        help="denied source prefix for retrieved hits; comma-separated or repeatable")
    parser.add_argument("--min-retrieval-score", type=float, default=0.0,
                        help="minimum retriever score required before a hit becomes evidence")
    parser.add_argument("--required-retrieval-metadata", action="append", default=None,
                        help="required hit metadata key=value; repeatable")
    parser.add_argument("--max-retrieval-hits-per-source", type=int, default=None,
                        help="maximum accepted hits per source before verifier evidence handoff")
    parser.add_argument("--source-family-filter", choices=SOURCE_FAMILY_FILTERS, default="off",
                        help="optionally filter or rerank retrieved evidence by planned source-family compatibility")
    parser.add_argument("--source-binding-queue", default=None,
                        help="optional evidence queue JSON used to bind retrieval to matching source requests")
    parser.add_argument("--use-precomputed-retrieval-hits", action="store_true",
                        help="mark fixture hits as already retrieved so verifier ensemble skips second-pass retrieval")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
