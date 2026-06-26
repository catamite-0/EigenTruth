"""Build structured QA corpora from Wikidata reference documents.

This is a dependency-free bridge from externally sourced Wikidata facts to the
existing ``QuestionAnswerVerifier`` / ``retrieval_structured_qa`` route. It does
not use score labels, claim ids, or score-dump row links.
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

DEFAULT_PROPERTY = "P36"
DEFAULT_PROPERTY_LABEL = "capital"
DEFAULT_QUESTION_TEMPLATE = "What is the capital of {country}?"
_QID_RE = re.compile(r"^Q[1-9][0-9]*$")


def build_wikidata_qa_corpus(
    source_documents: Sequence[Mapping[str, Any]],
    *,
    statement_property: str = DEFAULT_PROPERTY,
    statement_property_label: str = DEFAULT_PROPERTY_LABEL,
    question_template: str = DEFAULT_QUESTION_TEMPLATE,
    skip_qid_labels: bool = True,
) -> dict[str, Any]:
    """Return a structured QA corpus from Wikidata fact documents."""
    if not source_documents:
        raise ValueError("source_documents must not be empty.")
    statement_property = str(statement_property).strip()
    if not statement_property:
        raise ValueError("statement_property must be non-empty.")
    if "{country}" not in question_template:
        raise ValueError("question_template must contain {country}.")

    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped = {
        "missing_metadata": 0,
        "provider_mismatch": 0,
        "property_mismatch": 0,
        "missing_fields": 0,
        "qid_labels": 0,
        "duplicates": 0,
    }
    for source_index, item in enumerate(source_documents, start=1):
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            skipped["missing_metadata"] += 1
            continue
        if str(metadata.get("provider", "")).strip().casefold() != "wikidata":
            skipped["provider_mismatch"] += 1
            continue
        if str(metadata.get("statement_property", "")).strip() != statement_property:
            skipped["property_mismatch"] += 1
            continue
        country = _clean_text(metadata.get("country"))
        capital = _clean_text(metadata.get("capital"))
        if not country or not capital:
            skipped["missing_fields"] += 1
            continue
        if skip_qid_labels and (_is_qid(country) or _is_qid(capital)):
            skipped["qid_labels"] += 1
            continue
        question = question_template.format(country=country)
        key = (question.casefold(), capital.casefold())
        if key in seen:
            skipped["duplicates"] += 1
            continue
        seen.add(key)
        source = item.get("source")
        document_metadata = {
            "provider": "wikidata",
            "external_source": True,
            "structured_qa_builder": "build_wikidata_qa_corpus",
            "statement_property": statement_property,
            "statement_property_label": _clean_text(metadata.get("statement_property_label"))
            or statement_property_label,
            "country": country,
            "country_qid": _clean_text(metadata.get("country_qid")),
            "capital": capital,
            "capital_qid": _clean_text(metadata.get("capital_qid")),
            "source_document_index": source_index - 1,
            "source_document_source": None if source is None else str(source),
        }
        for key_name in (
            "license",
            "license_url",
            "endpoint",
            "query_preset",
            "retrieved_at",
            "timestamp",
            "url",
            "source_kind",
            "source_file",
            "corpus_name",
        ):
            if metadata.get(key_name) is not None:
                document_metadata[key_name] = metadata[key_name]
        documents.append({
            "question": question,
            "answer": capital,
            "text": f"{question} {capital}",
            "source": None if source is None else str(source),
            "metadata": document_metadata,
        })
    if not documents:
        raise ValueError("no Wikidata QA facts were produced from source documents.")
    return {
        "schema_version": 1,
        "corpus_type": "structured_qa_external_evidence",
        "description": (
            "Structured QA corpus derived from Wikidata reference facts. "
            "Use as a source corpus for QuestionAnswerVerifier or retrieval_structured_qa."
        ),
        "label_usage": {
            "labels_used_for_documents": False,
            "labels_copied_to_document_metadata": False,
            "claim_ids_copied_to_document_metadata": False,
            "score_dump_rows_copied_to_document_metadata": False,
        },
        "source": {
            "provider": "wikidata",
            "statement_property": statement_property,
            "statement_property_label": statement_property_label,
            "question_template": question_template,
            "skip_qid_labels": bool(skip_qid_labels),
        },
        "summary": {
            "n_source_documents": len(source_documents),
            "n_documents": len(documents),
            "skipped": skipped,
        },
        "documents": documents,
    }


def load_source_documents(paths: Sequence[str | Path]) -> tuple[dict[str, Any], ...]:
    """Load source documents from JSON, JSONL, or raw list files."""
    documents: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() == ".jsonl":
            documents.extend(_load_jsonl(path))
        else:
            documents.extend(_load_json(path))
    if not documents:
        raise ValueError("source paths did not contain any source documents.")
    return tuple(documents)


def build_input_provenance(
    *,
    source_paths: Sequence[str | Path],
    statement_property: str,
    statement_property_label: str,
    question_template: str,
    skip_qid_labels: bool,
) -> dict[str, Any]:
    """Return source fingerprints and builder config."""
    return {
        "schema_version": 1,
        "builder": "build_wikidata_qa_corpus",
        "sources": [fingerprint_path(path).to_dict() for path in source_paths],
        "config": {
            "statement_property": statement_property,
            "statement_property_label": statement_property_label,
            "question_template": question_template,
            "skip_qid_labels": bool(skip_qid_labels),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    source_paths = tuple(Path(path) for path in args.source)
    source_documents = load_source_documents(source_paths)
    corpus = build_wikidata_qa_corpus(
        source_documents,
        statement_property=args.statement_property,
        statement_property_label=args.statement_property_label,
        question_template=args.question_template,
        skip_qid_labels=not bool(args.keep_qid_labels),
    )
    corpus["input_provenance"] = build_input_provenance(
        source_paths=source_paths,
        statement_property=args.statement_property,
        statement_property_label=args.statement_property_label,
        question_template=args.question_template,
        skip_qid_labels=not bool(args.keep_qid_labels),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.artifact_manifest:
        manifest_path = Path(args.artifact_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = {"qa_corpus": output_path}
        for idx, path in enumerate(source_paths, start=1):
            artifacts[f"source.{idx}.{path.stem}"] = path
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": "build_wikidata_qa_corpus",
                "provider": "wikidata",
                "statement_property": args.statement_property,
                "statement_property_label": args.statement_property_label,
                "n_documents": corpus["summary"]["n_documents"],
                "skip_qid_labels": not bool(args.keep_qid_labels),
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wikidata_qa_corpus_ok documents={corpus['summary']['n_documents']} output={output_path}")
    return corpus


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_documents = payload.get("documents", payload.get("records", ()))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        raw_documents = payload
    else:
        raise ValueError(f"{path} must contain a JSON object or list.")
    return [_coerce_mapping(item, path=path) for item in raw_documents]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            documents.append(_coerce_mapping(json.loads(line), path=path, line_no=line_no))
    return documents


def _coerce_mapping(value: Any, *, path: Path, line_no: int | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        location = str(path) if line_no is None else f"{path}:{line_no}"
        raise ValueError(f"{location} contained a non-object document.")
    return dict(value)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_qid(value: str) -> bool:
    return bool(_QID_RE.fullmatch(value.strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured QA corpus from Wikidata reference documents")
    parser.add_argument("--source", action="append", required=True,
                        help="Wikidata source corpus JSON or JSONL; repeatable")
    parser.add_argument("--output", required=True, help="structured QA corpus output path")
    parser.add_argument("--statement-property", default=DEFAULT_PROPERTY)
    parser.add_argument("--statement-property-label", default=DEFAULT_PROPERTY_LABEL)
    parser.add_argument("--question-template", default=DEFAULT_QUESTION_TEMPLATE)
    parser.add_argument("--keep-qid-labels", action="store_true",
                        help="keep rows whose country/capital label is only a Wikidata QID")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
