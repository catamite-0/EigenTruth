"""Build target-specific QA diagnostics from covered-fact mapping audits.

This bridge turns conservative ``blind_spot_covered_fact_mapping_audit`` rows
into a retrieval-visible structured-QA corpus for diagnostics. It is explicitly
target-specific: generated documents reuse the original blind-spot question but
substitute candidate covered-fact answers from source metadata. The corpus is
therefore useful for measuring how much a covered-fact QA route could recover,
but it is not an open-domain retrieval corpus or release evidence by itself.
"""

from __future__ import annotations

import argparse
import json
import re
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

WORKFLOW = "covered_fact_retrieval_qa_corpus_builder"
CORPUS_TYPE = "target_specific_covered_fact_retrieval_qa_diagnostic"
DEFAULT_ROUTE_NAME = "covered_fact_retrieval_structured_qa"
DEFAULT_INCLUDE_STATUSES = ("candidate_fact_coverage",)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand|%|percent))?", re.I)
NUMERIC_QUESTION_TERMS = {
    "amount",
    "count",
    "fewer",
    "greater",
    "how many",
    "less",
    "more",
    "number",
    "percent",
    "percentage",
    "population",
    "rate",
    "share",
    "total",
}
NUMERIC_PROPERTY_TERMS = {
    "amount",
    "count",
    "headcount",
    "number",
    "percent",
    "percentage",
    "population",
    "rate",
    "share",
    "total",
    "value",
}
PERSON_RELATION_TERMS = {
    "author",
    "cofounder",
    "created",
    "creator",
    "founded",
    "founder",
    "invented",
    "inventor",
    "started",
}
PERSON_PROPERTY_TERMS = {
    "author",
    "creator",
    "developer",
    "founded by",
    "founder",
    "inventor",
}
LOCATION_TERMS = {"country", "location", "nation", "place", "where"}
LOCATION_PROPERTY_TERMS = {"capital", "country", "country of origin", "located in", "location"}
TEMPORAL_TERMS = {"date", "time", "when", "year"}
TEMPORAL_PROPERTY_TERMS = {"date", "inception", "point in time", "publication date", "time", "year"}
QUESTION_CONTENT_STOPWORDS = {
    "a",
    "an",
    "are",
    "is",
    "of",
    "the",
    "was",
    "were",
    "what",
}


def build_covered_fact_retrieval_qa_corpus(
    mapping_audit: Mapping[str, Any],
    *,
    include_statuses: Sequence[str] = DEFAULT_INCLUDE_STATUSES,
    max_facts_per_record: int = 3,
    route_name: str = DEFAULT_ROUTE_NAME,
    require_question_intent: bool = False,
) -> dict[str, Any]:
    """Return a target-specific QA corpus from covered-fact mapping rows."""
    statuses = _include_statuses(include_statuses)
    max_facts = int(max_facts_per_record)
    if max_facts <= 0:
        raise ValueError("max_facts_per_record must be positive.")
    route = str(route_name).strip()
    if not route:
        raise ValueError("route_name must be non-empty.")

    records = _records(mapping_audit)
    documents: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter({
        "answer_entity_collision": 0,
        "answer_value_supported": 0,
        "duplicate_fact": 0,
        "missing_fact_answer": 0,
        "missing_question": 0,
        "status_not_included": 0,
        "without_facts": 0,
        "question_intent_mismatch": 0,
    })
    included_statuses: Counter[str] = Counter()
    seen: set[tuple[int, str]] = set()

    for record in records:
        mapping_status = str(record.get("mapping_status") or "")
        if mapping_status not in statuses:
            skipped["status_not_included"] += 1
            continue
        if bool(record.get("answer_value_supported")):
            skipped["answer_value_supported"] += 1
            continue
        if bool(record.get("answer_entity_collision")):
            skipped["answer_entity_collision"] += 1
            continue
        question = str(record.get("question") or "").strip()
        if not question:
            skipped["missing_question"] += 1
            continue
        facts = _ranked_facts(record.get("facts", ()))
        if not facts:
            skipped["without_facts"] += 1
            continue

        record_index = int(record.get("record_index", -1))
        kept_for_record = 0
        for fact in facts:
            if kept_for_record >= max_facts:
                break
            answer = str(fact.get("answer") or "").strip()
            if not answer:
                skipped["missing_fact_answer"] += 1
                continue
            intent = _question_intent_match(record, fact)
            if require_question_intent and not bool(intent["match"]):
                skipped["question_intent_mismatch"] += 1
                continue
            key = (record_index, normalize_claim_text(answer))
            if key in seen:
                skipped["duplicate_fact"] += 1
                continue
            seen.add(key)
            kept_for_record += 1
            included_statuses[mapping_status] += 1
            documents.append(_document_for_fact(
                question=question,
                fact=fact,
                record=record,
                record_index=record_index,
                mapping_status=mapping_status,
                route_name=route,
                question_intent=intent,
            ))

    by_property = Counter(
        str(_mapping(document.get("metadata")).get("statement_property") or "unknown")
        for document in documents
    )
    return {
        "schema_version": 1,
        "corpus_type": CORPUS_TYPE,
        "description": (
            "Target-specific diagnostic QA corpus derived from covered-fact "
            "mapping audit candidates. It measures potential retrieval_structured_qa "
            "coverage and must not be treated as broad open-domain citation evidence."
        ),
        "label_usage": {
            "uses_blind_spot_target_selection": True,
            "uses_model_answer_for_exclusion_guard": True,
            "labels_copied_to_document_metadata": False,
            "model_answers_copied_to_document_metadata": False,
            "score_dump_rows_copied_to_document_metadata": False,
            "not_general_retrieval_corpus": True,
        },
        "source": {
            "builder": WORKFLOW,
            "mapping_audit_workflow": mapping_audit.get("workflow"),
            "mapping_audit_status": mapping_audit.get("status"),
            "route_name": route,
            "include_statuses": statuses,
            "max_facts_per_record": max_facts,
            "require_question_intent": bool(require_question_intent),
        },
        "summary": {
            "mapping_record_count": len(records),
            "n_documents": len(documents),
            "n_questions": len({normalize_claim_text(str(item.get("question") or "")) for item in documents}),
            "included_status_counts": dict(sorted(included_statuses.items())),
            "by_property": dict(sorted(by_property.items())),
            "skipped": dict(sorted(skipped.items())),
        },
        "documents": tuple(documents),
    }


def run(
    *,
    mapping_audit_path: str | Path,
    output_path: str | Path,
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    include_statuses: Sequence[str] = DEFAULT_INCLUDE_STATUSES,
    max_facts_per_record: int = 3,
    route_name: str = DEFAULT_ROUTE_NAME,
    require_question_intent: bool = False,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register the diagnostic corpus."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_path)
    report_path = Path(report_json_path) if report_json_path is not None else output.with_suffix(".report.json")
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output.parent / "artifact-manifest.json"
    )
    mapping_audit = _load_json_mapping(mapping_audit_path)
    corpus = build_covered_fact_retrieval_qa_corpus(
        mapping_audit,
        include_statuses=include_statuses,
        max_facts_per_record=max_facts_per_record,
        route_name=route_name,
        require_question_intent=require_question_intent,
    )
    status = "ready" if corpus["summary"]["n_documents"] else "blocked"
    report = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Diagnostic target-specific covered-fact QA bridge. Use for "
            "measuring route potential; do not promote as independent citation evidence."
        ),
        "source": {
            "mapping_audit": str(mapping_audit_path),
            "mapping_audit_workflow": mapping_audit.get("workflow"),
            "mapping_audit_status": mapping_audit.get("status"),
            "mapping_target_count": _nested_int(mapping_audit, "summary", "target_count"),
        },
        "config": {
            "include_statuses": tuple(str(item) for item in include_statuses),
            "max_facts_per_record": int(max_facts_per_record),
            "route_name": route_name,
            "require_question_intent": bool(require_question_intent),
        },
        "summary": dict(corpus["summary"]),
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
            "covered_fact_retrieval_qa_corpus": output,
            "covered_fact_retrieval_qa_report": report_path,
            "covered_fact_mapping_audit": Path(mapping_audit_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": status,
            "corpus_type": CORPUS_TYPE,
            "document_count": corpus["summary"]["n_documents"],
            "question_count": corpus["summary"]["n_questions"],
            "require_question_intent": bool(require_question_intent),
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
                "question_count": corpus["summary"]["n_questions"],
                "require_question_intent": bool(require_question_intent),
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return {"corpus": corpus, "report": report}


def _document_for_fact(
    *,
    question: str,
    fact: Mapping[str, Any],
    record: Mapping[str, Any],
    record_index: int,
    mapping_status: str,
    route_name: str,
    question_intent: Mapping[str, Any],
) -> dict[str, Any]:
    answer = str(fact.get("answer") or "").strip()
    subject = _optional_str(fact.get("subject"))
    property_label = _optional_str(fact.get("statement_property_label"))
    fact_question = _optional_str(fact.get("question"))
    source = fact.get("source")
    metadata = {
        "provider": "wikidata",
        "correction_scope": "target_specific_covered_fact_candidate",
        "route_name": route_name,
        "source_record_index": record_index,
        "source_question_type": record.get("question_type"),
        "source_mapping_status": mapping_status,
        "source_fact_question": fact_question,
        "source": source,
        "statement_property": fact.get("statement_property"),
        "statement_property_label": property_label,
        "subject": subject,
        "subject_qid": fact.get("subject_qid"),
        "value_qid": fact.get("value_qid"),
        "question_overlap": fact.get("question_overlap"),
        "answer_value_overlap": fact.get("answer_value_overlap"),
        "answer_subject_overlap": fact.get("answer_subject_overlap"),
        "question_intent_match": bool(question_intent.get("match")),
        "question_intent_reason": question_intent.get("reason"),
        "question_intent_terms": tuple(str(item) for item in _sequence(question_intent.get("terms"))),
        "retrieval_index_text": tuple(
            part
            for part in (
                question,
                fact_question,
                subject,
                property_label,
                answer,
            )
            if part
        ),
    }
    return {
        "question": question,
        "answer": answer,
        "text": f"{question} {answer}",
        "source": source,
        "metadata": metadata,
    }


def _question_intent_match(record: Mapping[str, Any], fact: Mapping[str, Any]) -> dict[str, Any]:
    question = str(record.get("question") or "")
    question_type = str(record.get("question_type") or "").strip().casefold()
    answer = str(fact.get("answer") or "")
    subject = str(fact.get("subject") or "")
    property_id = str(fact.get("statement_property") or "").strip()
    property_label = str(fact.get("statement_property_label") or "").strip()
    question_key = normalize_claim_text(question)
    property_key = normalize_claim_text(f"{property_id} {property_label}")
    question_tokens = set(_tokens(question))
    property_tokens = set(_tokens(property_label))
    subject_tokens = set(_tokens(subject))

    if _has_numeric_intent(question_key):
        if _answer_or_property_is_numeric(answer=answer, property_key=property_key):
            return _intent_result(True, "numeric_question_numeric_fact", ("numeric",))
        return _intent_result(False, "numeric_question_requires_numeric_fact", ("numeric",))

    if "why" in question_tokens or question_type == "causal":
        if _contains_any(property_key, {"cause", "reason", "because", "etiology"}):
            return _intent_result(True, "causal_property_match", ("cause",))
        return _intent_result(False, "causal_question_requires_causal_property", ("cause",))

    if "who" in question_tokens or question_type == "person":
        if _contains_any(question_key, PERSON_RELATION_TERMS) and _contains_any(property_key, PERSON_PROPERTY_TERMS):
            return _intent_result(True, "person_relation_property_match", ("person_relation",))
        if _contains_any(property_key, PERSON_PROPERTY_TERMS) and property_tokens & question_tokens:
            return _intent_result(True, "person_property_token_match", sorted(property_tokens & question_tokens))
        return _intent_result(False, "person_question_requires_person_relation_property", ("person_relation",))

    if _contains_any(question_key, LOCATION_TERMS) or question_type == "location":
        if _contains_any(property_key, LOCATION_PROPERTY_TERMS):
            return _intent_result(True, "location_property_match", ("location",))
        return _intent_result(False, "location_question_requires_location_property", ("location",))

    if _contains_any(question_key, TEMPORAL_TERMS) or question_type == "temporal":
        if _contains_any(property_key, TEMPORAL_PROPERTY_TERMS):
            return _intent_result(True, "temporal_property_match", ("temporal",))
        return _intent_result(False, "temporal_question_requires_temporal_property", ("temporal",))

    if question_type == "definition" and question_key.startswith(("what is ", "what are ", "what was ", "what were ")):
        if property_id in {"description", "P31", "P279"} or _contains_any(
            property_key,
            {"description", "instance of", "subclass of"},
        ):
            question_content = question_tokens - QUESTION_CONTENT_STOPWORDS
            off_subject_terms = question_content - subject_tokens
            if subject_tokens and len(off_subject_terms) <= 1:
                return _intent_result(True, "definition_property_match", ("definition",))
            return _intent_result(False, "definition_question_subject_mismatch", tuple(sorted(off_subject_terms)))
        return _intent_result(False, "definition_question_requires_definition_property", ("definition",))

    overlap = tuple(sorted(property_tokens & question_tokens))
    if overlap:
        return _intent_result(True, "property_token_overlap", overlap)
    return _intent_result(False, "no_question_property_intent_match", ())


def _intent_result(match: bool, reason: str, terms: Sequence[str]) -> dict[str, Any]:
    return {"match": bool(match), "reason": reason, "terms": tuple(terms)}


def _ranked_facts(value: Any) -> tuple[dict[str, Any], ...]:
    facts = [dict(item) for item in _sequence(value) if isinstance(item, Mapping)]
    return tuple(
        sorted(
            facts,
            key=lambda item: (
                -_float(item.get("question_overlap")),
                -_float(item.get("answer_subject_overlap")),
                str(item.get("statement_property") or ""),
                str(item.get("source") or ""),
            ),
        )
    )


def _include_statuses(values: Sequence[str]) -> tuple[str, ...]:
    statuses = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not statuses:
        raise ValueError("include_statuses must not be empty.")
    return statuses


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in TOKEN_RE.finditer(value))


def _contains_any(text: str, terms: set[str]) -> bool:
    padded = f" {text} "
    words = set(text.split())
    for term in terms:
        normalized = term.casefold()
        if " " in normalized:
            if f" {normalized} " in padded:
                return True
        elif normalized in words:
            return True
    return False


def _has_numeric_intent(question_key: str) -> bool:
    if NUMBER_RE.search(question_key) or "%" in question_key:
        return True
    return _contains_any(question_key, NUMERIC_QUESTION_TERMS)


def _answer_or_property_is_numeric(*, answer: str, property_key: str) -> bool:
    if NUMBER_RE.search(answer):
        return True
    return _contains_any(property_key, NUMERIC_PROPERTY_TERMS)


def _records(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_records = payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("mapping audit must contain a records list.")
    records = tuple(dict(item) for item in raw_records if isinstance(item, Mapping))
    if not records:
        raise ValueError("mapping audit did not contain usable records.")
    return records


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_metadata(values: Sequence[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in values or ():
        if "=" not in raw:
            raise ValueError("--metadata entries must use key=value.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        metadata[key] = value.strip()
    return metadata


def _nested_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    try:
        return int(current)
    except (TypeError, ValueError):
        return None


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-audit", required=True)
    parser.add_argument("--json", required=True, help="Path to write the QA corpus JSON.")
    parser.add_argument("--report-json")
    parser.add_argument("--artifact-manifest")
    parser.add_argument("--registry")
    parser.add_argument("--name")
    parser.add_argument("--version")
    parser.add_argument("--include-statuses", default=",".join(DEFAULT_INCLUDE_STATUSES))
    parser.add_argument("--max-facts-per-record", type=int, default=3)
    parser.add_argument("--route-name", default=DEFAULT_ROUTE_NAME)
    parser.add_argument(
        "--require-question-intent",
        action="store_true",
        help="Keep only target-specific facts whose property/value type matches the source question intent.",
    )
    parser.add_argument("--metadata", action="append")
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args()

    payload = run(
        mapping_audit_path=args.mapping_audit,
        output_path=args.json,
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        include_statuses=_parse_csv(args.include_statuses),
        max_facts_per_record=args.max_facts_per_record,
        route_name=args.route_name,
        require_question_intent=bool(args.require_question_intent),
        metadata=_parse_metadata(args.metadata),
        compact_json=args.compact_json,
    )
    report = payload["report"]
    summary = report["summary"]
    print(
        "covered_fact_retrieval_qa_corpus_ok "
        f"status={report['status']} documents={summary['n_documents']} "
        f"questions={summary['n_questions']}"
    )


if __name__ == "__main__":
    main()
