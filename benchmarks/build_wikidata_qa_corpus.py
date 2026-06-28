"""Build structured QA corpora from Wikidata reference documents.

This is a dependency-free bridge from externally sourced Wikidata facts to the
existing ``QuestionAnswerVerifier`` / ``retrieval_structured_qa`` route. It does
not use score labels, claim ids, or score-dump row links.
"""

from __future__ import annotations

import argparse
import json
import re
import string
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
DEFAULT_ANSWER_FIELD = "capital"
DEFAULT_AUTO_QUESTION_TEMPLATE = "What does Wikidata list as the {statement_property_label} for {subject}?"
DEFAULT_AUTO_ANSWER_FIELD = "value"
_QID_RE = re.compile(r"^Q[1-9][0-9]*$")
_RESERVED_METADATA_KEYS = {
    "claim_id",
    "is_false",
    "label",
    "row_index",
    "score_dump_row",
    "score_label",
}
_PROVENANCE_METADATA_FIELDS = (
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
)


def build_wikidata_qa_corpus(
    source_documents: Sequence[Mapping[str, Any]],
    *,
    statement_property: str = DEFAULT_PROPERTY,
    statement_property_label: str = DEFAULT_PROPERTY_LABEL,
    question_template: str = DEFAULT_QUESTION_TEMPLATE,
    answer_field: str = DEFAULT_ANSWER_FIELD,
    qid_label_fields: Sequence[str] | None = None,
    templates: Sequence[Mapping[str, Any]] | None = None,
    skip_qid_labels: bool = True,
) -> dict[str, Any]:
    """Return a structured QA corpus from Wikidata fact documents."""
    if not source_documents:
        raise ValueError("source_documents must not be empty.")
    normalized_templates = _normalize_templates(
        templates,
        statement_property=statement_property,
        statement_property_label=statement_property_label,
        question_template=question_template,
        answer_field=answer_field,
        qid_label_fields=qid_label_fields,
    )
    templates_by_property: dict[str, list[dict[str, Any]]] = {}
    for template in normalized_templates:
        templates_by_property.setdefault(str(template["statement_property"]), []).append(template)

    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped = {
        "missing_metadata": 0,
        "provider_mismatch": 0,
        "property_mismatch": 0,
        "missing_fields": 0,
        "qid_labels": 0,
        "duplicates": 0,
        "reserved_metadata": 0,
    }
    by_property: dict[str, dict[str, Any]] = {
        str(template["statement_property"]): {
            "statement_property_label": template["statement_property_label"],
            "n_documents": 0,
            "skipped": {key: 0 for key in skipped},
        }
        for template in normalized_templates
    }
    for source_index, item in enumerate(source_documents, start=1):
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            skipped["missing_metadata"] += 1
            continue
        if str(metadata.get("provider", "")).strip().casefold() != "wikidata":
            skipped["provider_mismatch"] += 1
            continue
        reserved = sorted(key for key in _RESERVED_METADATA_KEYS if key in metadata)
        if reserved:
            skipped["reserved_metadata"] += 1
            continue
        item_property = str(metadata.get("statement_property", "")).strip()
        matching_templates = templates_by_property.get(item_property, ())
        if not matching_templates:
            skipped["property_mismatch"] += 1
            continue
        for template in matching_templates:
            property_key = str(template["statement_property"])
            property_summary = by_property[property_key]
            template_fields = tuple(template["question_fields"])
            template_answer_field = str(template["answer_field"])
            missing_fields = [
                field for field in (*template_fields, template_answer_field)
                if not _clean_text(metadata.get(field))
            ]
            if missing_fields:
                _increment_skip(skipped, property_summary, "missing_fields")
                continue
            qid_fields = tuple(template["qid_label_fields"])
            if skip_qid_labels and any(_is_qid(str(metadata.get(field, "")).strip()) for field in qid_fields):
                _increment_skip(skipped, property_summary, "qid_labels")
                continue
            question_values = {field: _clean_text(metadata.get(field)) for field in template_fields}
            question = str(template["question_template"]).format(**question_values)
            answer = _clean_text(metadata.get(template_answer_field))
            assert answer is not None
            key = (question.casefold(), answer.casefold())
            if key in seen:
                _increment_skip(skipped, property_summary, "duplicates")
                continue
            seen.add(key)
            source = item.get("source")
            document_metadata = _qa_document_metadata(
                metadata,
                template=template,
                source_index=source_index - 1,
                source=None if source is None else str(source),
            )
            documents.append({
                "question": question,
                "answer": answer,
                "text": f"{question} {answer}",
                "source": None if source is None else str(source),
                "metadata": document_metadata,
            })
            property_summary["n_documents"] += 1
    if not documents:
        raise ValueError("no Wikidata QA facts were produced from source documents.")
    source_payload = {
        "provider": "wikidata",
        "templates": tuple(_public_template(template) for template in normalized_templates),
        "skip_qid_labels": bool(skip_qid_labels),
    }
    if len(normalized_templates) == 1:
        only = normalized_templates[0]
        source_payload.update({
            "statement_property": only["statement_property"],
            "statement_property_label": only["statement_property_label"],
            "question_template": only["question_template"],
            "answer_field": only["answer_field"],
        })
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
        "source": source_payload,
        "summary": {
            "n_source_documents": len(source_documents),
            "n_documents": len(documents),
            "by_property": by_property,
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
    templates: Sequence[Mapping[str, Any]],
    skip_qid_labels: bool,
) -> dict[str, Any]:
    """Return source fingerprints and builder config."""
    return {
        "schema_version": 1,
        "builder": "build_wikidata_qa_corpus",
        "sources": [fingerprint_path(path).to_dict() for path in source_paths],
        "config": {
            "templates": tuple(_public_template(template) for template in templates),
            "skip_qid_labels": bool(skip_qid_labels),
        },
    }


def infer_wikidata_qa_templates(
    source_documents: Sequence[Mapping[str, Any]],
    *,
    question_template: str = DEFAULT_AUTO_QUESTION_TEMPLATE,
    answer_field: str = DEFAULT_AUTO_ANSWER_FIELD,
    qid_label_fields: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Infer one structured-QA template per Wikidata statement property."""
    if not source_documents:
        raise ValueError("source_documents must not be empty.")
    templates_by_property: dict[str, dict[str, Any]] = {}
    for item in source_documents:
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        if str(metadata.get("provider", "")).strip().casefold() != "wikidata":
            continue
        statement_property = _clean_text(metadata.get("statement_property"))
        if statement_property is None:
            continue
        if _clean_text(metadata.get(answer_field)) is None:
            continue
        for field in _template_fields(question_template):
            if _clean_text(metadata.get(field)) is None:
                break
        else:
            templates_by_property.setdefault(
                statement_property,
                {
                    "statement_property": statement_property,
                    "statement_property_label": (
                        _clean_text(metadata.get("statement_property_label"))
                        or statement_property
                    ),
                    "question_template": question_template,
                    "answer_field": answer_field,
                    "qid_label_fields": qid_label_fields,
                },
            )
    if not templates_by_property:
        raise ValueError("no Wikidata QA templates could be inferred from source documents.")
    return _normalize_templates(
        tuple(templates_by_property[key] for key in sorted(templates_by_property)),
        statement_property=DEFAULT_PROPERTY,
        statement_property_label=DEFAULT_PROPERTY_LABEL,
        question_template=question_template,
        answer_field=answer_field,
        qid_label_fields=qid_label_fields,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    source_paths = tuple(Path(path) for path in args.source)
    source_documents = load_source_documents(source_paths)
    auto_template_from_source = bool(getattr(args, "auto_template_from_source", False))
    templates = (
        infer_wikidata_qa_templates(
            source_documents,
            question_template=getattr(args, "auto_question_template", DEFAULT_AUTO_QUESTION_TEMPLATE),
            answer_field=getattr(args, "auto_answer_field", DEFAULT_AUTO_ANSWER_FIELD),
            qid_label_fields=getattr(args, "qid_label_field", None),
        )
        if auto_template_from_source
        else _templates_from_args(args)
    )
    corpus = build_wikidata_qa_corpus(
        source_documents,
        templates=templates,
        skip_qid_labels=not bool(args.keep_qid_labels),
    )
    corpus["input_provenance"] = build_input_provenance(
        source_paths=source_paths,
        templates=templates,
        skip_qid_labels=not bool(args.keep_qid_labels),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    template_output = getattr(args, "template_json_output", None)
    if template_output:
        template_output_path = Path(template_output)
        template_output_path.parent.mkdir(parents=True, exist_ok=True)
        template_output_path.write_text(
            json.dumps({"templates": tuple(_public_template(template) for template in templates)},
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.artifact_manifest:
        manifest_path = Path(args.artifact_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = {"qa_corpus": output_path}
        if template_output:
            artifacts["qa_templates"] = Path(template_output)
        for idx, path in enumerate(source_paths, start=1):
            artifacts[f"source.{idx}.{path.stem}"] = path
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": "build_wikidata_qa_corpus",
                "provider": "wikidata",
                "statement_properties": tuple(template["statement_property"] for template in templates),
                "templates": tuple(_public_template(template) for template in templates),
                "n_documents": corpus["summary"]["n_documents"],
                "skip_qid_labels": not bool(args.keep_qid_labels),
                "auto_template_from_source": auto_template_from_source,
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


def _normalize_templates(
    templates: Sequence[Mapping[str, Any]] | None,
    *,
    statement_property: str,
    statement_property_label: str,
    question_template: str,
    answer_field: str,
    qid_label_fields: Sequence[str] | None,
) -> tuple[dict[str, Any], ...]:
    raw_templates: Sequence[Mapping[str, Any]]
    if templates is None:
        raw_templates = ({
            "statement_property": statement_property,
            "statement_property_label": statement_property_label,
            "question_template": question_template,
            "answer_field": answer_field,
            "qid_label_fields": qid_label_fields,
        },)
    else:
        raw_templates = templates
    if not raw_templates:
        raise ValueError("at least one Wikidata QA template is required.")
    normalized = []
    for idx, raw in enumerate(raw_templates, start=1):
        normalized.append(_normalize_template(raw, index=idx))
    return tuple(normalized)


def _normalize_template(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    statement_property = _clean_text(raw.get("statement_property"))
    if statement_property is None:
        raise ValueError(f"template {index} must define statement_property.")
    question_template = _clean_text(raw.get("question_template"))
    if question_template is None:
        raise ValueError(f"template {index} must define question_template.")
    question_fields = _template_fields(question_template)
    if not question_fields:
        raise ValueError(f"template {index} question_template must contain at least one field.")
    answer_field = _clean_text(raw.get("answer_field", DEFAULT_ANSWER_FIELD))
    if answer_field is None:
        raise ValueError(f"template {index} answer_field must be non-empty.")
    qid_label_fields = raw.get("qid_label_fields")
    if qid_label_fields is None:
        qid_fields = tuple(dict.fromkeys((*question_fields, answer_field)))
    elif isinstance(qid_label_fields, str):
        qid_fields = tuple(field.strip() for field in qid_label_fields.split(",") if field.strip())
    elif isinstance(qid_label_fields, Sequence):
        qid_fields = tuple(str(field).strip() for field in qid_label_fields if str(field).strip())
    else:
        raise ValueError(f"template {index} qid_label_fields must be a sequence or comma-list.")
    statement_property_label = _clean_text(raw.get("statement_property_label")) or statement_property
    return {
        "statement_property": statement_property,
        "statement_property_label": statement_property_label,
        "question_template": question_template,
        "question_fields": question_fields,
        "answer_field": answer_field,
        "qid_label_fields": qid_fields,
    }


def _template_fields(template: str) -> tuple[str, ...]:
    fields = []
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name.isidentifier():
            raise ValueError(f"unsupported template field {field_name!r}; use simple metadata keys only.")
        fields.append(field_name)
    return tuple(dict.fromkeys(fields))


def _templates_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    if args.template_json:
        payload = json.loads(Path(args.template_json).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            raw_templates = payload.get("templates", ())
        else:
            raw_templates = payload
        if not isinstance(raw_templates, Sequence) or isinstance(raw_templates, (str, bytes, bytearray)):
            raise ValueError("template JSON must contain a list or an object with a templates list.")
        return _normalize_templates(tuple(_coerce_template(item) for item in raw_templates),
                                    statement_property=args.statement_property,
                                    statement_property_label=args.statement_property_label,
                                    question_template=args.question_template,
                                    answer_field=args.answer_field,
                                    qid_label_fields=args.qid_label_field)
    return _normalize_templates(
        None,
        statement_property=args.statement_property,
        statement_property_label=args.statement_property_label,
        question_template=args.question_template,
        answer_field=args.answer_field,
        qid_label_fields=args.qid_label_field,
    )


def _coerce_template(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("each template must be a JSON object.")
    return dict(value)


def _qa_document_metadata(
    metadata: Mapping[str, Any],
    *,
    template: Mapping[str, Any],
    source_index: int,
    source: str | None,
) -> dict[str, Any]:
    document_metadata: dict[str, Any] = {
        "provider": "wikidata",
        "external_source": True,
        "structured_qa_builder": "build_wikidata_qa_corpus",
        "statement_property": template["statement_property"],
        "statement_property_label": _clean_text(metadata.get("statement_property_label"))
        or str(template["statement_property_label"]),
        "question_template": template["question_template"],
        "answer_field": template["answer_field"],
        "source_document_index": source_index,
        "source_document_source": source,
    }
    for field in (*template["question_fields"], template["answer_field"], *template["qid_label_fields"]):
        value = _clean_text(metadata.get(str(field)))
        if value is not None:
            document_metadata[str(field)] = value
        qid_value = _clean_text(metadata.get(f"{field}_qid"))
        if qid_value is not None:
            document_metadata[f"{field}_qid"] = qid_value
    for key_name in _PROVENANCE_METADATA_FIELDS:
        if metadata.get(key_name) is not None:
            document_metadata[key_name] = metadata[key_name]
    return document_metadata


def _public_template(template: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "statement_property": template["statement_property"],
        "statement_property_label": template["statement_property_label"],
        "question_template": template["question_template"],
        "answer_field": template["answer_field"],
        "qid_label_fields": tuple(template["qid_label_fields"]),
    }


def _increment_skip(
    skipped: dict[str, int],
    property_summary: dict[str, Any],
    key: str,
) -> None:
    skipped[key] += 1
    property_summary["skipped"][key] += 1


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
    parser.add_argument("--template-json", default=None,
                        help="optional JSON file with a templates list for multiple Wikidata properties")
    parser.add_argument("--statement-property", default=DEFAULT_PROPERTY)
    parser.add_argument("--statement-property-label", default=DEFAULT_PROPERTY_LABEL)
    parser.add_argument("--question-template", default=DEFAULT_QUESTION_TEMPLATE)
    parser.add_argument("--answer-field", default=DEFAULT_ANSWER_FIELD)
    parser.add_argument("--qid-label-field", action="append", default=None,
                        help="metadata field to reject when its value is a bare Wikidata QID; repeatable")
    parser.add_argument("--keep-qid-labels", action="store_true",
                        help="keep rows whose configured label fields are only Wikidata QIDs")
    parser.add_argument("--auto-template-from-source", action="store_true",
                        help="infer one generic QA template per Wikidata statement_property in the source docs")
    parser.add_argument("--auto-question-template", default=DEFAULT_AUTO_QUESTION_TEMPLATE,
                        help="question template used with --auto-template-from-source")
    parser.add_argument("--auto-answer-field", default=DEFAULT_AUTO_ANSWER_FIELD,
                        help="metadata answer field used with --auto-template-from-source")
    parser.add_argument("--template-json-output", default=None,
                        help="optional path to save inferred or resolved templates")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
