"""Audit whether blind-spot records map to collected covered facts.

This workflow joins four artifacts without changing the evidence corpora:

* blind-spot records from ``analyze_detectability_blind_spots.py``
* Wikidata source documents from ``fetch_blind_spot_wikidata_evidence.py``
* that fetch report's request/target trace
* a structured QA corpus built from the same source documents

The source docs and QA corpus intentionally avoid score-row labels or target ids.
This audit reconstructs target links from request fingerprints in the fetch
report, then reports conservative mapping statuses. A joined fact is treated as
a candidate correction input, not as proof that the original TruthfulQA answer
is refuted.
"""

from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_QUESTION_OVERLAP_THRESHOLD = 0.20
DEFAULT_ANSWER_OVERLAP_THRESHOLD = 0.80
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
}


def audit_blind_spot_covered_fact_mapping(
    *,
    blind_spots: Mapping[str, Any],
    qa_corpus: Mapping[str, Any],
    source_documents: Sequence[Mapping[str, Any]],
    wikidata_fetch_report: Mapping[str, Any],
    question_overlap_threshold: float = DEFAULT_QUESTION_OVERLAP_THRESHOLD,
    answer_overlap_threshold: float = DEFAULT_ANSWER_OVERLAP_THRESHOLD,
) -> dict[str, Any]:
    """Return a conservative mapping audit from blind spots to covered facts."""
    records = _blind_records(blind_spots)
    qa_documents = _qa_documents(qa_corpus)
    source_by_source = {
        str(item.get("source")): dict(item)
        for item in source_documents
        if item.get("source") is not None
    }
    requests_by_fingerprint = _requests_by_fingerprint(wikidata_fetch_report)
    qa_by_target = _qa_documents_by_target(
        qa_documents=qa_documents,
        source_by_source=source_by_source,
        requests_by_fingerprint=requests_by_fingerprint,
    )
    audited_records = tuple(
        _audit_record(
            record,
            qa_documents=qa_by_target.get(f"record-{int(record['record_index'])}", ()),
            question_overlap_threshold=question_overlap_threshold,
            answer_overlap_threshold=answer_overlap_threshold,
        )
        for record in records
    )
    summary = _summary(audited_records)
    status = (
        "observed"
        if summary["candidate_fact_coverage_count"] > 0
        else "blocked"
    )
    return {
        "schema_version": 1,
        "workflow": "blind_spot_covered_fact_mapping_audit",
        "status": status,
        "scope": (
            "Audits target-to-covered-fact mapping candidates. It does not prove "
            "that the original open-domain blind-spot answers are refuted."
        ),
        "config": {
            "question_overlap_threshold": float(question_overlap_threshold),
            "answer_overlap_threshold": float(answer_overlap_threshold),
        },
        "summary": summary,
        "records": audited_records,
        "next_step": (
            "Promote only records with explicit question/property claim mapping; "
            "send answer-entity collisions, low-relevance joins, and no-fact "
            "targets to citation retrieval or world-model evidence collection."
        ),
    }


def run(
    *,
    blind_spots_path: str | Path,
    qa_corpus_path: str | Path,
    source_jsonl_path: str | Path,
    wikidata_fetch_report_path: str | Path,
    output_path: str | Path,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    question_overlap_threshold: float = DEFAULT_QUESTION_OVERLAP_THRESHOLD,
    answer_overlap_threshold: float = DEFAULT_ANSWER_OVERLAP_THRESHOLD,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Load inputs, write the audit, and optionally manifest/register it."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    blind_spots = _load_json_mapping(blind_spots_path)
    qa_corpus = _load_json_mapping(qa_corpus_path)
    source_documents = _load_jsonl(source_jsonl_path)
    wikidata_fetch_report = _load_json_mapping(wikidata_fetch_report_path)
    payload = audit_blind_spot_covered_fact_mapping(
        blind_spots=blind_spots,
        qa_corpus=qa_corpus,
        source_documents=source_documents,
        wikidata_fetch_report=wikidata_fetch_report,
        question_overlap_threshold=question_overlap_threshold,
        answer_overlap_threshold=answer_overlap_threshold,
    )
    payload["paths"] = {
        "blind_spots": str(blind_spots_path),
        "qa_corpus": str(qa_corpus_path),
        "source_jsonl": str(source_jsonl_path),
        "wikidata_fetch_report": str(wikidata_fetch_report_path),
    }
    payload["metadata"] = dict(metadata or {})
    output = Path(output_path)
    _write_json(output, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "covered_fact_mapping_audit": output,
                "blind_spots": Path(blind_spots_path),
                "qa_corpus": Path(qa_corpus_path),
                "wikidata_source_docs": Path(source_jsonl_path),
                "wikidata_fetch_report": Path(wikidata_fetch_report_path),
            },
            root=manifest_path.parent,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "candidate_fact_coverage_count": payload["summary"]["candidate_fact_coverage_count"],
                "answer_entity_collision_count": payload["summary"]["answer_entity_collision_count"],
                "no_joined_fact_count": payload["summary"]["no_joined_fact_count"],
                **dict(metadata or {}),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "candidate_fact_coverage_count": payload["summary"]["candidate_fact_coverage_count"],
                "answer_entity_collision_count": payload["summary"]["answer_entity_collision_count"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _audit_record(
    record: Mapping[str, Any],
    *,
    qa_documents: Sequence[Mapping[str, Any]],
    question_overlap_threshold: float,
    answer_overlap_threshold: float,
) -> dict[str, Any]:
    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))
    answer_tokens = _tokens(answer)
    fact_summaries = tuple(
        _fact_summary(document, question=question, answer=answer)
        for document in qa_documents
    )
    best_question_overlap = max((item["question_overlap"] for item in fact_summaries), default=0.0)
    best_answer_value_overlap = max((item["answer_value_overlap"] for item in fact_summaries), default=0.0)
    best_answer_subject_overlap = max((item["answer_subject_overlap"] for item in fact_summaries), default=0.0)
    answer_value_supported = best_answer_value_overlap >= float(answer_overlap_threshold)
    answer_entity_collision = (
        best_answer_subject_overlap >= float(answer_overlap_threshold)
        and best_question_overlap < float(question_overlap_threshold)
    )
    candidate_fact_coverage = (
        bool(fact_summaries)
        and best_question_overlap >= float(question_overlap_threshold)
        and not answer_entity_collision
        and not answer_value_supported
    )
    if not fact_summaries:
        mapping_status = "no_joined_facts"
    elif answer_value_supported:
        mapping_status = "answer_value_supported"
    elif answer_entity_collision:
        mapping_status = "answer_entity_collision"
    elif candidate_fact_coverage:
        mapping_status = "candidate_fact_coverage"
    else:
        mapping_status = "joined_low_relevance"
    return {
        "record_index": int(record["record_index"]),
        "question": question,
        "answer": answer,
        "question_type": record.get("question_type"),
        "priority": record.get("priority"),
        "mapping_status": mapping_status,
        "candidate_fact_coverage": candidate_fact_coverage,
        "answer_value_supported": answer_value_supported,
        "answer_entity_collision": answer_entity_collision,
        "joined_fact_count": len(fact_summaries),
        "joined_property_counts": _sorted_counter(Counter(item["statement_property"] for item in fact_summaries)),
        "best_question_overlap": best_question_overlap,
        "best_answer_value_overlap": best_answer_value_overlap,
        "best_answer_subject_overlap": best_answer_subject_overlap,
        "answer_tokens": sorted(answer_tokens),
        "facts": fact_summaries[:10],
    }


def _fact_summary(
    document: Mapping[str, Any],
    *,
    question: str,
    answer: str,
) -> dict[str, Any]:
    metadata = _mapping(document.get("metadata"))
    subject = str(metadata.get("subject") or "")
    value = str(document.get("answer") or metadata.get("value") or "")
    property_label = str(metadata.get("statement_property_label") or "")
    property_id = str(metadata.get("statement_property") or "unknown")
    question_tokens = _tokens(question)
    fact_question_tokens = _tokens(f"{subject} {property_label}")
    answer_tokens = _tokens(answer)
    value_tokens = _tokens(value)
    subject_tokens = _tokens(subject)
    return {
        "question": document.get("question"),
        "answer": value,
        "source": document.get("source"),
        "statement_property": property_id,
        "statement_property_label": property_label,
        "subject": subject,
        "subject_qid": metadata.get("subject_qid"),
        "value_qid": metadata.get("value_qid"),
        "question_overlap": _overlap(question_tokens, fact_question_tokens),
        "answer_value_overlap": _overlap(answer_tokens, value_tokens),
        "answer_subject_overlap": _overlap(answer_tokens, subject_tokens),
    }


def _qa_documents_by_target(
    *,
    qa_documents: Sequence[Mapping[str, Any]],
    source_by_source: Mapping[str, Mapping[str, Any]],
    requests_by_fingerprint: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for document in qa_documents:
        source = str(document.get("source"))
        source_document = source_by_source.get(source)
        if source_document is None:
            continue
        fingerprint = _mapping(source_document.get("metadata")).get("collection_request_sha256")
        if not fingerprint:
            continue
        requests = requests_by_fingerprint.get(str(fingerprint), ())
        for request in requests:
            target_id = request.get("target_id")
            if target_id is None:
                continue
            by_target.setdefault(str(target_id), []).append(dict(document))
    return {key: tuple(value) for key, value in by_target.items()}


def _requests_by_fingerprint(report: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    raw_results = report.get("request_results", ())
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes, bytearray)):
        raise ValueError("Wikidata fetch report is missing request_results.")
    by_fingerprint: dict[str, list[dict[str, Any]]] = {}
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        payload = dict(item)
        fingerprint = _request_fingerprint(payload)
        by_fingerprint.setdefault(fingerprint, []).append(payload)
    return {key: tuple(value) for key, value in by_fingerprint.items()}


def _request_fingerprint(request: Mapping[str, Any]) -> str:
    stable = {
        "entity": request.get("entity"),
        "property_id": request.get("property_id"),
        "property_hint": request.get("property_hint"),
        "request_type": request.get("request_type", "wikidata_entity_property"),
    }
    encoded = strict_json_dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(record.get("mapping_status")) for record in records)
    property_counts = Counter()
    question_type_counts = Counter()
    for record in records:
        question_type_counts[str(record.get("question_type") or "unknown")] += 1
        property_counts.update(_mapping(record.get("joined_property_counts")))
    return {
        "target_count": len(records),
        "records_with_joined_facts": sum(1 for record in records if int(record.get("joined_fact_count", 0)) > 0),
        "candidate_fact_coverage_count": sum(1 for record in records if record.get("candidate_fact_coverage")),
        "answer_value_supported_count": sum(1 for record in records if record.get("answer_value_supported")),
        "answer_entity_collision_count": sum(1 for record in records if record.get("answer_entity_collision")),
        "no_joined_fact_count": status_counts.get("no_joined_facts", 0),
        "mapping_status_counts": _sorted_counter(status_counts),
        "joined_property_counts": _sorted_counter(property_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
    }


def _blind_records(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_records = payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("blind-spot report must contain a records list.")
    records = []
    for item in raw_records:
        if not isinstance(item, Mapping):
            continue
        if item.get("record_index") is None:
            continue
        records.append(dict(item))
    if not records:
        raise ValueError("blind-spot report did not contain usable records.")
    return tuple(records)


def _qa_documents(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_documents = payload.get("documents", payload.get("records", ()))
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes, bytearray)):
        raise ValueError("QA corpus must contain documents or records.")
    documents = [dict(item) for item in raw_documents if isinstance(item, Mapping)]
    if not documents:
        raise ValueError("QA corpus did not contain usable documents.")
    return tuple(documents)


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            rows.append(dict(payload))
    if not rows:
        raise ValueError(f"{path} did not contain any JSONL rows.")
    return tuple(rows)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _tokens(value: Any) -> set[str]:
    text = normalize_claim_text(str(value))
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) > 1 and token not in STOPWORDS
    }


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sorted_counter(counter: Counter[str] | Mapping[str, Any]) -> dict[str, int]:
    items = Counter({str(key): int(value) for key, value in dict(counter).items()})
    return dict(sorted(items.items(), key=lambda item: (-item[1], item[0])))


def _parse_metadata(values: Sequence[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not values:
        return metadata
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"metadata item {item!r} must use key=value format.")
            key, raw = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("metadata key must be non-empty.")
            metadata[key] = raw.strip()
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-spots", required=True)
    parser.add_argument("--qa-corpus", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--wikidata-fetch-report", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--question-overlap-threshold", type=float, default=DEFAULT_QUESTION_OVERLAP_THRESHOLD)
    parser.add_argument("--answer-overlap-threshold", type=float, default=DEFAULT_ANSWER_OVERLAP_THRESHOLD)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        blind_spots_path=args.blind_spots,
        qa_corpus_path=args.qa_corpus,
        source_jsonl_path=args.source_jsonl,
        wikidata_fetch_report_path=args.wikidata_fetch_report,
        output_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        question_overlap_threshold=args.question_overlap_threshold,
        answer_overlap_threshold=args.answer_overlap_threshold,
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "blind_spot_covered_fact_mapping_audit_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"joined={summary['records_with_joined_facts']} "
        f"candidates={summary['candidate_fact_coverage_count']} "
        f"collisions={summary['answer_entity_collision_count']}"
    )


if __name__ == "__main__":
    main()
