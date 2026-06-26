"""Build explicit external-candidate retrieval corpora from local source files.

This is an ingestion boundary for retrieval experiments. It converts supplied
JSON, JSONL, or text documents into the corpus schema accepted by
``build_evidence_fixture.py`` and ``audit_retrieval_corpus_provenance.py`` while
failing closed on score-dump label or row-link metadata. The command does not
fetch network content; callers should materialize any licensed external source
files first, then run this builder to fingerprint and normalize them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import build_artifact_manifest, fingerprint_path  # noqa: E402

EXTERNAL_CORPUS_TYPE = "external_evidence_candidate"
RESERVED_METADATA_KEYS = {
    "claim_id",
    "is_false",
    "label",
    "labels",
    "record_index",
    "row_index",
    "score_label",
    "source_index",
}
TEXT_FIELDS = ("text", "content", "document", "body")


def build_external_retrieval_corpus(
    source_paths: Sequence[str | Path],
    *,
    corpus_name: str = "external_candidate",
    source_kind: str = "local_external_source",
    min_chars: int = 1,
    require_source: bool = True,
    trusted_source: Sequence[str] = (),
    default_timestamp: str | None = None,
) -> dict[str, Any]:
    """Return a normalized external-candidate retrieval corpus."""
    if not source_paths:
        raise ValueError("source_paths must contain at least one path.")
    if min_chars < 1:
        raise ValueError("min_chars must be positive.")
    documents = []
    skipped_short = 0
    for source_path in tuple(Path(path) for path in source_paths):
        source_documents = _load_source_documents(
            source_path,
            corpus_name=corpus_name,
            source_kind=source_kind,
            require_source=require_source,
            default_timestamp=default_timestamp,
        )
        for document in source_documents:
            if len(document["text"]) < min_chars:
                skipped_short += 1
                continue
            documents.append(document)
    if not documents:
        raise ValueError("external retrieval corpus would be empty after filtering.")

    trusted = tuple(str(item) for item in trusted_source if str(item).strip())
    return {
        "schema_version": 1,
        "corpus_type": EXTERNAL_CORPUS_TYPE,
        "description": (
            "External-candidate retrieval corpus built from caller-supplied local source files. "
            "This builder rejects score labels, claim ids, and score-dump row links in document metadata."
        ),
        "label_usage": {
            "labels_used_for_documents": False,
            "labels_copied_to_document_metadata": False,
        },
        "config": {
            "builder": "build_external_retrieval_corpus",
            "corpus_name": corpus_name,
            "source_kind": source_kind,
            "min_chars": int(min_chars),
            "require_source": bool(require_source),
            "trusted_source": trusted,
            "default_timestamp": default_timestamp,
        },
        "input_provenance": {
            "builder": "build_external_retrieval_corpus",
            "sources": [fingerprint_path(path).to_dict() for path in source_paths],
        },
        "summary": {
            "n_documents": len(documents),
            "n_sources": len(source_paths),
            "n_skipped_short": skipped_short,
            "n_documents_with_source": sum(1 for document in documents if document.get("source")),
            "n_documents_with_timestamp": sum(
                1 for document in documents
                if document.get("metadata", {}).get("timestamp")
            ),
            "n_documents_with_url": sum(
                1 for document in documents
                if document.get("metadata", {}).get("url")
            ),
        },
        "documents": documents,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    source_paths = tuple(Path(path) for path in args.source)
    payload = build_external_retrieval_corpus(
        source_paths,
        corpus_name=args.corpus_name,
        source_kind=args.source_kind,
        min_chars=args.min_chars,
        require_source=not bool(args.allow_missing_source),
        trusted_source=tuple(args.trusted_source or ()),
        default_timestamp=args.default_timestamp,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.artifact_manifest:
        manifest_path = Path(args.artifact_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = {"corpus": output_path}
        for idx, path in enumerate(source_paths, start=1):
            artifacts[f"source.{idx}.{path.stem}"] = path
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": "build_external_retrieval_corpus",
                "corpus_type": payload["corpus_type"],
                "summary": payload["summary"],
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "external_retrieval_corpus_ok "
        f"documents={payload['summary']['n_documents']} output={output_path}"
    )
    return payload


def _load_source_documents(
    path: Path,
    *,
    corpus_name: str,
    source_kind: str,
    require_source: bool,
    default_timestamp: str | None,
) -> tuple[dict[str, Any], ...]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, Mapping):
            raw_documents = loaded.get("documents", loaded.get("records", ()))
        elif isinstance(loaded, Sequence) and not isinstance(loaded, (str, bytes, bytearray)):
            raw_documents = loaded
        else:
            raise ValueError(f"{path} must contain a JSON object or list.")
        return tuple(
            _coerce_source_document(
                item,
                source_default=f"{path}:{idx}",
                source_path=path,
                ingest_ordinal=idx,
                corpus_name=corpus_name,
                source_kind=source_kind,
                require_source=require_source,
                default_timestamp=default_timestamp,
            )
            for idx, item in enumerate(raw_documents, start=1)
        )
    if path.suffix.lower() == ".jsonl":
        documents = []
        with path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = {"text": line}
                documents.append(_coerce_source_document(
                    item,
                    source_default=f"{path}:{idx}",
                    source_path=path,
                    ingest_ordinal=idx,
                    corpus_name=corpus_name,
                    source_kind=source_kind,
                    require_source=require_source,
                    default_timestamp=default_timestamp,
                ))
        return tuple(documents)
    text = path.read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if len(chunks) == 1:
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
    return tuple(
        _coerce_source_document(
            {"text": chunk},
            source_default=f"{path}#{idx}",
            source_path=path,
            ingest_ordinal=idx,
            corpus_name=corpus_name,
            source_kind=source_kind,
            require_source=require_source,
            default_timestamp=default_timestamp,
        )
        for idx, chunk in enumerate(chunks, start=1)
    )


def _coerce_source_document(
    value: Any,
    *,
    source_default: str,
    source_path: Path,
    ingest_ordinal: int,
    corpus_name: str,
    source_kind: str,
    require_source: bool,
    default_timestamp: str | None,
) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"text": value}
    if not isinstance(value, Mapping):
        raise ValueError("source documents must be strings or mappings.")
    text = ""
    for field in TEXT_FIELDS:
        if value.get(field):
            text = str(value[field]).strip()
            break
    if not text:
        raise ValueError("source document is missing text/content/document/body.")
    metadata = dict(value.get("metadata", {}))
    for field in ("title", "url", "published_at", "timestamp"):
        if value.get(field) is not None and field not in metadata:
            metadata[field] = str(value[field])
    _reject_reserved_metadata(metadata, source=source_default)
    source = value.get("source")
    if source is None:
        if require_source:
            source = source_default
        else:
            source = None
    metadata = {
        **metadata,
        "external_source": True,
        "ingestion_builder": "build_external_retrieval_corpus",
        "corpus_name": corpus_name,
        "source_kind": source_kind,
        "source_file": str(source_path),
        "ingest_ordinal": int(ingest_ordinal),
    }
    if default_timestamp is not None and "timestamp" not in metadata:
        metadata["timestamp"] = str(default_timestamp)
    return {
        "text": text,
        "source": None if source is None else str(source),
        "metadata": metadata,
    }


def _reject_reserved_metadata(metadata: Mapping[str, Any], *, source: str) -> None:
    keys = {str(key) for key in metadata}
    reserved = sorted(keys & RESERVED_METADATA_KEYS)
    if reserved:
        raise ValueError(
            f"external source document {source} contains reserved score-dump metadata keys: "
            f"{', '.join(reserved)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicit external-candidate retrieval corpus")
    parser.add_argument("--source", action="append", required=True, help="source JSON/JSONL/text path; repeatable")
    parser.add_argument("--output", required=True, help="path to write corpus JSON")
    parser.add_argument("--corpus-name", default="external_candidate")
    parser.add_argument("--source-kind", default="local_external_source")
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--allow-missing-source", action="store_true")
    parser.add_argument("--trusted-source", action="append", default=())
    parser.add_argument("--default-timestamp", default=None)
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest for source/corpus files")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
