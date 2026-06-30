"""Build structured QA facts from source-family catalog or adapter results.

This is a conservative bridge from source-family evidence acquisition to the
existing ``QuestionAnswerVerifier`` route. It only materializes QA facts from
documents that already carry structured metadata, such as Wikidata
``subject/property/value`` rows or World Bank ``country/indicator/year/value``
rows. Free-form news, scholarly abstracts, and generic web pages remain source
documents; they are not promoted into structured facts by this builder.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from eigentruth.verify import normalize_claim_text  # noqa: E402

WORKFLOW = "source_family_structured_qa_corpus_builder"
CORPUS_TYPE = "source_family_structured_qa_external_evidence"

RESERVED_METADATA_KEYS = {
    "claim_id",
    "is_false",
    "label",
    "model_answer",
    "queue_id",
    "record_index",
    "request_id",
    "row_index",
    "score_dump_row",
    "score_label",
    "source_request_id",
    "target_id",
}

PROVENANCE_METADATA_KEYS = (
    "alignment_candidate_id",
    "alignment_source_document_id",
    "provider",
    "source_family",
    "source_family_confidence",
    "source",
    "url",
    "title",
    "published_at",
    "retrieved_at",
    "license",
    "license_url",
    "endpoint",
    "reference_year",
    "statement_property",
    "statement_property_label",
    "indicator",
    "indicator_name",
    "country_name",
    "country_code_iso3",
    "review_id",
    "review_status",
    "reviewed_at",
    "reviewer",
)


def build_source_family_qa_corpus(
    source_documents: Sequence[Mapping[str, Any]],
    *,
    skip_qid_values: bool = True,
) -> dict[str, Any]:
    """Return a structured QA corpus from safe source-family metadata."""
    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped = Counter({
        "unsupported_provider": 0,
        "reserved_metadata": 0,
        "missing_fields": 0,
        "qid_values": 0,
        "duplicates": 0,
    })
    candidates = 0
    for source_index, item in enumerate(source_documents):
        metadata = _metadata(item)
        provider = _provider(item, metadata)
        if _has_reserved_metadata(item, metadata):
            skipped["reserved_metadata"] += 1
            continue
        builder = _builder_for_provider(provider)
        if builder is None:
            skipped["unsupported_provider"] += 1
            continue
        candidates += 1
        document = builder(item, metadata, source_index=source_index, skip_qid_values=skip_qid_values)
        if document is None:
            reason = _last_skip_reason(item, metadata, provider, skip_qid_values=skip_qid_values)
            skipped[reason] += 1
            continue
        key = (normalize_claim_text(document["question"]), normalize_claim_text(document["answer"]))
        if key in seen:
            skipped["duplicates"] += 1
            continue
        seen.add(key)
        documents.append(document)

    by_provider = Counter(str(doc["metadata"].get("provider", "unknown")) for doc in documents)
    by_source_family = Counter(str(doc["metadata"].get("source_family", "unknown")) for doc in documents)
    return {
        "schema_version": 1,
        "corpus_type": CORPUS_TYPE,
        "description": (
            "Structured QA corpus derived only from source-family rows with "
            "explicit structured metadata. Use as a covered-fact input for "
            "QuestionAnswerVerifier; do not treat it as broad open-domain "
            "TruthfulQA coverage."
        ),
        "label_usage": {
            "labels_used_for_documents": False,
            "labels_copied_to_document_metadata": False,
            "claim_ids_copied_to_document_metadata": False,
            "score_dump_rows_copied_to_document_metadata": False,
            "model_answers_copied_to_document_metadata": False,
        },
        "source": {
            "builder": WORKFLOW,
            "accepted_providers": ("wikidata", "worldbank"),
            "skip_qid_values": bool(skip_qid_values),
        },
        "summary": {
            "n_source_documents": len(source_documents),
            "n_candidate_documents": candidates,
            "n_documents": len(documents),
            "by_provider": dict(sorted(by_provider.items())),
            "by_source_family": dict(sorted(by_source_family.items())),
            "skipped": dict(sorted(skipped.items())),
        },
        "documents": documents,
    }


def run(
    *,
    source_paths: Sequence[str | Path],
    output_path: str | Path,
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    skip_qid_values: bool = True,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a structured QA corpus."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if not source_paths:
        raise ValueError("source_paths must contain at least one path.")

    output = Path(output_path)
    report_path = Path(report_json_path) if report_json_path is not None else output.with_suffix(".report.json")
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output.parent / "artifact-manifest.json"
    )
    source_documents = load_source_documents(source_paths)
    corpus = build_source_family_qa_corpus(source_documents, skip_qid_values=skip_qid_values)
    status = "ready" if corpus["summary"]["n_documents"] else "blocked"
    report = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "summary": dict(corpus["summary"]),
        "source": {
            "source_paths": tuple(str(path) for path in source_paths),
        },
        "config": {
            "skip_qid_values": bool(skip_qid_values),
        },
        "paths": {
            "qa_corpus": str(output),
            "report": str(report_path),
            "artifact_manifest": str(manifest_path),
        },
        "metadata": dict(metadata or {}),
    }
    _write_json(output, corpus, compact=compact_json)
    _write_json(report_path, report, compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "source_family_structured_qa_corpus": output,
            "source_family_structured_qa_report": report_path,
            **{f"source_{index}": Path(path) for index, path in enumerate(source_paths, start=1)},
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": status,
            "corpus_type": CORPUS_TYPE,
            "document_count": corpus["summary"]["n_documents"],
            "candidate_document_count": corpus["summary"]["n_candidate_documents"],
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "corpus_type": CORPUS_TYPE,
                "document_count": corpus["summary"]["n_documents"],
                "candidate_document_count": corpus["summary"]["n_candidate_documents"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return {"corpus": corpus, "report": report}


def load_source_documents(paths: Sequence[str | Path]) -> tuple[dict[str, Any], ...]:
    """Load catalog docs, corpus docs, or adapter-result rows."""
    documents: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() == ".jsonl":
            for item in _load_jsonl(path):
                documents.extend(_documents_from_payload(item))
        else:
            documents.extend(_documents_from_payload(_load_json(path)))
    return tuple(documents)


def _documents_from_payload(payload: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        docs: list[dict[str, Any]] = []
        for item in payload:
            docs.extend(_documents_from_payload(item))
        return tuple(docs)
    if not isinstance(payload, Mapping):
        return ()
    for key in ("results", "documents", "records"):
        values = _non_string_sequence(payload.get(key))
        if values is not None:
            return tuple(dict(item) for item in values if isinstance(item, Mapping))
    return (dict(payload),)


def _non_string_sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


def _wikidata_document(
    item: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    source_index: int,
    skip_qid_values: bool,
) -> dict[str, Any] | None:
    subject = _clean_text(metadata.get("subject"))
    answer = _clean_text(metadata.get("value"))
    statement_property = _clean_text(metadata.get("statement_property"))
    property_label = _clean_text(metadata.get("statement_property_label")) or statement_property
    if not subject or not answer or not statement_property or not property_label:
        return None
    if skip_qid_values and _is_qid(answer):
        return None
    question = f"What does Wikidata list as the {property_label} for {subject}?"
    return _document(
        question=question,
        answer=answer,
        text=f"{question} {answer}",
        source=_clean_text(item.get("source")),
        item=item,
        metadata=metadata,
        source_index=source_index,
        provider="wikidata",
        extraction_rule="wikidata_subject_property_value",
        extra_metadata={
            "statement_property": statement_property,
            "statement_property_label": property_label,
            "subject": subject,
            "subject_qid": _clean_text(metadata.get("subject_qid")),
            "value_datatype": _clean_text(metadata.get("value_datatype")),
        },
    )


def _worldbank_document(
    item: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    source_index: int,
    skip_qid_values: bool,
) -> dict[str, Any] | None:
    del skip_qid_values
    country = _clean_text(metadata.get("country_name"))
    indicator_name = _clean_text(metadata.get("indicator_name"))
    year = _clean_text(metadata.get("reference_year"))
    value = _clean_text(metadata.get("value"))
    if not country or not indicator_name or not year or not value:
        return None
    answer = _format_answer_value(value)
    question = f"What does the World Bank list as {indicator_name} for {country} in {year}?"
    return _document(
        question=question,
        answer=answer,
        text=f"{question} {answer}",
        source=_clean_text(item.get("source")),
        item=item,
        metadata=metadata,
        source_index=source_index,
        provider="worldbank",
        extraction_rule="worldbank_country_indicator_year_value",
        extra_metadata={
            "country_name": country,
            "country_code_iso3": _clean_text(metadata.get("country_code_iso3")),
            "indicator": _clean_text(metadata.get("indicator")),
            "indicator_name": indicator_name,
            "reference_year": year,
            "raw_value": value,
        },
    )


def _document(
    *,
    question: str,
    answer: str,
    text: str,
    source: str | None,
    item: Mapping[str, Any],
    metadata: Mapping[str, Any],
    source_index: int,
    provider: str,
    extraction_rule: str,
    extra_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    document_metadata = {
        "provider": provider,
        "source_index": source_index,
        "extraction_rule": extraction_rule,
        "source_family": _clean_text(item.get("source_family")) or _clean_text(metadata.get("source_family")),
    }
    for key in PROVENANCE_METADATA_KEYS:
        value = item.get(key, metadata.get(key))
        cleaned = _json_safe(value)
        if cleaned is not None and key not in RESERVED_METADATA_KEYS:
            document_metadata[key] = cleaned
    for key, value in extra_metadata.items():
        cleaned = _json_safe(value)
        if cleaned is not None and key not in RESERVED_METADATA_KEYS:
            document_metadata[key] = cleaned
    return {
        "question": question,
        "answer": answer,
        "text": text,
        "source": source,
        "metadata": document_metadata,
    }


def _builder_for_provider(provider: str):
    if provider == "wikidata":
        return _wikidata_document
    if provider == "worldbank":
        return _worldbank_document
    return None


def _last_skip_reason(
    item: Mapping[str, Any],
    metadata: Mapping[str, Any],
    provider: str,
    *,
    skip_qid_values: bool,
) -> str:
    if provider == "wikidata":
        answer = _clean_text(metadata.get("value"))
        if skip_qid_values and answer and _is_qid(answer):
            return "qid_values"
    del item
    return "missing_fields"


def _metadata(item: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = item.get("metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _provider(item: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    provider = item.get("provider", metadata.get("provider", ""))
    return str(provider).strip().casefold()


def _has_reserved_metadata(item: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    return any(key in item or key in metadata for key in RESERVED_METADATA_KEYS)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_answer_value(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,}"


def _is_qid(value: str) -> bool:
    if len(value) < 2 or value[0] != "Q":
        return False
    return value[1:].isdigit()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        cleaned_items = ((str(key), _json_safe(item)) for key, item in value.items())
        return {key: item for key, item in cleaned_items if item is not None}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in (_json_safe(item) for item in value) if item is not None)
    return str(value)


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> tuple[Any, ...]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


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
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--keep-qid-values", action="store_true")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        source_paths=tuple(args.source or ()),
        output_path=args.output,
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        skip_qid_values=not bool(args.keep_qid_values),
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        "source_family_structured_qa_corpus_ok "
        f"status={payload['report']['status']} "
        f"documents={payload['corpus']['summary']['n_documents']} "
        f"candidates={payload['corpus']['summary']['n_candidate_documents']}"
    )


if __name__ == "__main__":
    main()
