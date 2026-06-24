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

from eigentruth.adapters import InMemoryRetriever, RetrievalHit, RetrievalQuery, SQLiteFTSRetriever
from eigentruth.eval.score_dump import (
    ScoreDump,
    score_dump_file_metadata,
)
from eigentruth.eval.score_dump import (
    load_score_dump as _load_validated_score_dump,
)
from eigentruth.registry import fingerprint_path

RETRIEVER_BACKENDS = ("memory", "sqlite_fts", "auto")


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
) -> dict[str, Any]:
    """Build a claim/evidence fixture using only local retrieval over claim text."""
    if retrieval_limit <= 0:
        raise ValueError("retrieval_limit must be positive.")
    if query_field not in {"text", "answer", "question", "question_answer"}:
        raise ValueError("query_field must be one of: text, answer, question, question_answer.")
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
):
    requested_index_path = None if index_path is None else str(index_path)
    if backend == "memory":
        retriever = InMemoryRetriever(documents, min_overlap=min_overlap)
        return retriever, {
            "type": type(retriever).__name__,
            "requested_backend": backend,
            "actual_backend": "memory",
            "fallback_reason": None,
            "requested_index_path": requested_index_path,
            "actual_index_path": None,
            "index_reused": False,
        }
    retriever = SQLiteFTSRetriever(documents, min_overlap=min_overlap, index_path=index_path)
    if retriever.available:
        return retriever, {
            "type": type(retriever).__name__,
            "requested_backend": backend,
            "actual_backend": "sqlite_fts",
            "fallback_reason": None,
            "requested_index_path": requested_index_path,
            "actual_index_path": None if retriever.index_path is None else str(retriever.index_path),
            "index_reused": retriever.index_reused,
            "document_fingerprint": retriever.document_fingerprint,
        }
    if backend == "sqlite_fts":
        return retriever, {
            "type": "InMemoryRetriever",
            "requested_backend": backend,
            "actual_backend": "memory",
            "fallback_reason": retriever.fallback_reason,
            "requested_index_path": requested_index_path,
            "actual_index_path": None,
            "index_reused": False,
            "document_fingerprint": retriever.document_fingerprint,
        }
    fallback = InMemoryRetriever(documents, min_overlap=min_overlap)
    return fallback, {
        "type": type(fallback).__name__,
        "requested_backend": backend,
        "actual_backend": "memory",
        "fallback_reason": retriever.fallback_reason,
        "requested_index_path": requested_index_path,
        "actual_index_path": None,
        "index_reused": False,
        "document_fingerprint": retriever.document_fingerprint,
    }


def _query_text(statement: Mapping[str, Any], *, query_field: str) -> str:
    if query_field == "text":
        return _statement_text(statement)
    if query_field == "answer":
        text = str(statement.get("answer", "")).strip()
    elif query_field == "question":
        text = str(statement.get("question", "")).strip()
    elif query_field == "question_answer":
        text = f"{statement.get('question', '')} {statement.get('answer', '')}".strip()
    else:
        raise ValueError("query_field must be one of: text, answer, question, question_answer.")
    if not text:
        raise ValueError(f"statement record is missing query field {query_field!r}.")
    return text


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
    return RetrievalHit(
        text=str(value.get("text", value.get("content", ""))),
        source=source,
        score=float(value.get("score", 1.0)),
        metadata=metadata,
    )


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
    parser.add_argument("--query-field", choices=("text", "answer", "question", "question_answer"), default="text",
                        help="statement field used for retrieval query; claim text remains unchanged")
    parser.add_argument("--omit-label-metadata", action="store_true",
                        help="do not copy score labels into fixture record metadata")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
