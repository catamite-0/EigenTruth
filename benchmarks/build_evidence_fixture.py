"""Build local retrieval evidence fixtures for verifier-ensemble benchmarks.

This is a no-network bridge from statement-bearing score dumps to
``eval_verifier_ensemble.py`` claim fixtures. It retrieves evidence from local
JSON/JSONL/text corpora using dependency-free local retrievers and writes one
fixture record per score row.
"""

from __future__ import annotations

import argparse
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
)
from eigentruth.eval.score_dump import (
    ScoreDump,
    score_dump_file_metadata,
)
from eigentruth.eval.score_dump import (
    load_score_dump as _load_validated_score_dump,
)
from eigentruth.registry import fingerprint_path
from eigentruth.verify.search_planning import plan_citation_search_query

RETRIEVER_BACKENDS = ("memory", "sqlite_fts", "auto")
QUERY_FIELDS = (
    "text",
    "answer",
    "question",
    "question_answer",
    "citation_question",
    "citation_entity",
)
CITATION_QUERY_FIELDS = ("citation_question", "citation_entity")


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
    labels = tuple(int(label) for label in dump.get("labels", ()))
    statements = tuple(dict(statement) for statement in dump.get("statements", ()))
    if len(labels) != len(statements):
        raise ValueError("labels and statements must have the same length.")

    documents = tuple(corpus_documents)
    index_path = None if retriever_index_path is None else Path(retriever_index_path)
    retriever, retriever_info = _build_retriever(
        documents,
        min_overlap=retriever_min_overlap,
        backend=retriever_backend,
        index_path=index_path,
        provenance_filter=_retrieval_provenance_filter_config(
            require_source=require_retrieval_source,
            allowed_source_prefixes=allowed_retrieval_source_prefixes,
            denied_source_prefixes=denied_retrieval_source_prefixes,
            min_score=min_retrieval_score,
            required_metadata=required_retrieval_metadata,
            max_hits_per_source=max_retrieval_hits_per_source,
        ),
    )
    records = []
    total_hits = 0
    for idx, (label, statement) in enumerate(zip(labels, statements), start=1):
        claim_text = _statement_text(statement)
        query_text = _query_text(statement, query_field=query_field)
        claim_id = str(statement.get("claim_id") or f"c{idx}")
        hits = tuple(retriever.retrieve(
            RetrievalQuery(query=query_text, claim_id=claim_id),
            limit=retrieval_limit,
        ))
        total_hits += len(hits)
        record_metadata: dict[str, Any] = {
            "index": idx - 1,
            "statement": statement,
            "retrieval": {
                "n_hits": len(hits),
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
                "provenance_filter": retriever_info.get("provenance_filter"),
            },
        }
        if include_label_metadata:
            record_metadata["score_label"] = label
        records.append({
            "claim": claim_text,
            "claim_id": claim_id,
            "claim_metadata": dict(statement.get("metadata", {})),
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
        },
        "summary": {
            "n_records": len(records),
            "records_with_hits": sum(1 for record in records if record["retrieval_documents"]),
            "total_hits": total_hits,
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
) -> dict[str, Any]:
    """Build input fingerprints and builder settings for fixture reproducibility."""
    score_dump_obj = _coerce_score_dump_for_metadata(score_dump)
    index_path = None if retriever_index_path is None else Path(retriever_index_path)
    return {
        "schema_version": 1,
        "builder": "build_evidence_fixture",
        "score_dump": score_dump_file_metadata(scores_path, score_dump_obj),
        "corpora": [fingerprint_path(path).to_dict() for path in corpus_paths],
        "retriever_index": None if index_path is None else fingerprint_path(index_path).to_dict(),
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
    else:
        raise ValueError(f"query_field must be one of: {', '.join(QUERY_FIELDS)}.")
    if not text:
        raise ValueError(f"statement record is missing query field {query_field!r}.")
    return text


def _citation_query_text(statement: Mapping[str, Any], *, query_field: str) -> str:
    question = str(statement.get("question", "")).strip()
    if not question:
        raise ValueError(f"statement record is missing query field {query_field!r}.")
    strategy = "claim_entity" if query_field == "citation_entity" else "question_and_query"
    metadata = statement.get("metadata")
    question_type = ""
    if isinstance(metadata, Mapping):
        question_type = str(metadata.get("question_type") or "").strip()
    question_type = str(statement.get("question_type") or question_type).strip()
    plan = plan_citation_search_query(
        question=question,
        candidate_query="",
        question_type=question_type,
        strategy=strategy,
    )
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
