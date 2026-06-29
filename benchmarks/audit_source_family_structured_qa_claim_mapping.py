"""Audit claim coverage against source-family structured QA facts.

This workflow is the conservative bridge after
``run_source_family_structured_qa_route_workflow.py``. The route audit proves
that known covered QA facts are handled correctly; this script checks whether
real blind-spot or product claims can be mapped into those exact covered facts
before any correction handoff is created.

The audit is intentionally text-and-metadata based. It does not use labels for
retrieval, does not mutate the QA corpus, and does not treat weak topical
overlap as evidence.
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

WORKFLOW = "source_family_structured_qa_claim_mapping_audit"
DEFAULT_SUBJECT_COVERAGE_THRESHOLD = 0.60
DEFAULT_MAPPING_SCORE_THRESHOLD = 0.70
DEFAULT_ANSWER_OVERLAP_THRESHOLD = 0.80
DEFAULT_WEAK_OVERLAP_THRESHOLD = 0.20

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
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
    "list",
    "listed",
    "lists",
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

FACT_TYPE_ALIASES = {
    "description": (
        "definition",
        "description",
        "described",
        "is",
        "kind",
        "type",
    ),
    "p31": (
        "class",
        "instance",
        "kind",
        "type",
        "what",
    ),
    "p50": (
        "author",
        "authored",
        "wrote",
        "written",
    ),
    "p112": (
        "cofounder",
        "cofounders",
        "established",
        "founded",
        "founder",
        "founders",
        "launched",
        "started",
    ),
    "p170": (
        "created",
        "creator",
        "invented",
        "made",
    ),
    "p17": (
        "country",
        "from",
        "located",
        "nationality",
    ),
    "p27": (
        "country",
        "from",
        "nationality",
    ),
    "p856": (
        "official",
        "url",
        "website",
    ),
    "sp_pop_totl": (
        "people",
        "population",
        "residents",
        "total",
    ),
    "sp.pop.totl": (
        "people",
        "population",
        "residents",
        "total",
    ),
}


def audit_source_family_structured_qa_claim_mapping(
    *,
    claims_payload: Any,
    qa_corpus: Mapping[str, Any],
    route_summary: Mapping[str, Any] | None = None,
    subject_coverage_threshold: float = DEFAULT_SUBJECT_COVERAGE_THRESHOLD,
    mapping_score_threshold: float = DEFAULT_MAPPING_SCORE_THRESHOLD,
    answer_overlap_threshold: float = DEFAULT_ANSWER_OVERLAP_THRESHOLD,
    weak_overlap_threshold: float = DEFAULT_WEAK_OVERLAP_THRESHOLD,
) -> dict[str, Any]:
    """Return a conservative claim-to-QA covered-fact mapping audit."""
    records = _claim_records(claims_payload)
    qa_documents = _qa_documents(qa_corpus)
    audited_records = tuple(
        _audit_record(
            record,
            qa_documents=qa_documents,
            subject_coverage_threshold=float(subject_coverage_threshold),
            mapping_score_threshold=float(mapping_score_threshold),
            answer_overlap_threshold=float(answer_overlap_threshold),
            weak_overlap_threshold=float(weak_overlap_threshold),
        )
        for record in records
    )
    summary = _summary(audited_records)
    route_status = None if route_summary is None else str(route_summary.get("status") or "unknown")
    route_promoted = None if route_status is None else route_status == "promote"
    status = (
        "observed"
        if summary["covered_fact_match_count"] > 0 and route_promoted is not False
        else "blocked"
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Maps claims or blind spots into exact source-family structured QA "
            "covered facts. It does not prove open-domain coverage or use weak "
            "topical overlap as correction evidence."
        ),
        "source": {
            "claim_record_count": len(records),
            "qa_corpus_type": qa_corpus.get("corpus_type"),
            "qa_document_count": len(qa_documents),
            "route_summary_workflow": None if route_summary is None else route_summary.get("workflow"),
            "route_summary_status": route_status,
            "route_summary_promoted": route_promoted,
        },
        "config": {
            "subject_coverage_threshold": float(subject_coverage_threshold),
            "mapping_score_threshold": float(mapping_score_threshold),
            "answer_overlap_threshold": float(answer_overlap_threshold),
            "weak_overlap_threshold": float(weak_overlap_threshold),
        },
        "summary": summary,
        "records": audited_records,
        "next_step": (
            "Use mapped_qa_fact_candidate rows only when the route summary is "
            "promoted; route supported answers, weak matches, and no-fact gaps "
            "to citation retrieval, richer source-family collection, or "
            "world-model/calculator rule authoring."
        ),
    }


def run(
    *,
    claims_path: str | Path,
    qa_corpus_path: str | Path,
    output_path: str | Path,
    route_summary_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    subject_coverage_threshold: float = DEFAULT_SUBJECT_COVERAGE_THRESHOLD,
    mapping_score_threshold: float = DEFAULT_MAPPING_SCORE_THRESHOLD,
    answer_overlap_threshold: float = DEFAULT_ANSWER_OVERLAP_THRESHOLD,
    weak_overlap_threshold: float = DEFAULT_WEAK_OVERLAP_THRESHOLD,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Load inputs, write the audit, and optionally manifest/register it."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    claims = _load_claim_payload(claims_path)
    qa_corpus = _load_json_mapping(qa_corpus_path)
    route_summary = (
        None if route_summary_path is None else _load_json_mapping(route_summary_path)
    )
    payload = audit_source_family_structured_qa_claim_mapping(
        claims_payload=claims,
        qa_corpus=qa_corpus,
        route_summary=route_summary,
        subject_coverage_threshold=subject_coverage_threshold,
        mapping_score_threshold=mapping_score_threshold,
        answer_overlap_threshold=answer_overlap_threshold,
        weak_overlap_threshold=weak_overlap_threshold,
    )
    payload["paths"] = {
        "claims": str(claims_path),
        "qa_corpus": str(qa_corpus_path),
        "route_summary": None if route_summary_path is None else str(route_summary_path),
    }
    payload["metadata"] = dict(metadata or {})
    output = Path(output_path)
    _write_json(output, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        artifacts = {
            "source_family_structured_qa_claim_mapping": output,
            "claims": Path(claims_path),
            "qa_corpus": Path(qa_corpus_path),
        }
        if route_summary_path is not None:
            artifacts["route_summary"] = Path(route_summary_path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "covered_fact_match_count": payload["summary"]["covered_fact_match_count"],
                "mapped_qa_fact_candidate_count": payload["summary"][
                    "mapped_qa_fact_candidate_count"
                ],
                "answer_value_supported_count": payload["summary"][
                    "answer_value_supported_count"
                ],
                "no_candidate_fact_count": payload["summary"]["no_candidate_fact_count"],
                "route_summary_status": payload["source"]["route_summary_status"],
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
                "covered_fact_match_count": payload["summary"]["covered_fact_match_count"],
                "mapped_qa_fact_candidate_count": payload["summary"][
                    "mapped_qa_fact_candidate_count"
                ],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _audit_record(
    record: Mapping[str, Any],
    *,
    qa_documents: Sequence[Mapping[str, Any]],
    subject_coverage_threshold: float,
    mapping_score_threshold: float,
    answer_overlap_threshold: float,
    weak_overlap_threshold: float,
) -> dict[str, Any]:
    question = str(record.get("question") or "")
    answer = str(record.get("answer") or "")
    claim_text = str(record.get("text") or record.get("claim") or "")
    scored_facts = tuple(
        sorted(
            (
                _score_fact(
                    document,
                    question=question,
                    answer=answer,
                    claim_text=claim_text,
                    subject_coverage_threshold=subject_coverage_threshold,
                    mapping_score_threshold=mapping_score_threshold,
                    answer_overlap_threshold=answer_overlap_threshold,
                    weak_overlap_threshold=weak_overlap_threshold,
                )
                for document in qa_documents
            ),
            key=lambda item: (
                -float(item["mapping_score"]),
                -float(item["subject_coverage"]),
                -float(item["intent_score"]),
                str(item["source"]),
            ),
        )
    )
    covered = tuple(item for item in scored_facts if item["covered_fact_match"])
    mapped_candidates = tuple(
        item
        for item in covered
        if not item["answer_value_supported"] and not item["answer_entity_collision"]
    )
    supported = tuple(item for item in covered if item["answer_value_supported"])
    collisions = tuple(item for item in scored_facts if item["answer_entity_collision"])
    decision = _mapping_decision(
        scored_facts=scored_facts,
        mapped_candidates=mapped_candidates,
        supported=supported,
        collisions=collisions,
        subject_coverage_threshold=subject_coverage_threshold,
        weak_overlap_threshold=weak_overlap_threshold,
    )
    return {
        "record_id": record["record_id"],
        "record_index": record.get("record_index"),
        "claim_id": record.get("claim_id"),
        "question": question,
        "answer": answer,
        "text": claim_text,
        "label": record.get("label"),
        "question_type": record.get("question_type"),
        "mapping_decision": decision,
        "covered_fact_match": bool(covered),
        "mapped_qa_fact_candidate": bool(mapped_candidates),
        "answer_value_supported": bool(supported),
        "answer_entity_collision": bool(collisions),
        "best_mapping_score": max((float(item["mapping_score"]) for item in scored_facts), default=0.0),
        "best_subject_coverage": max(
            (float(item["subject_coverage"]) for item in scored_facts),
            default=0.0,
        ),
        "best_intent_score": max((float(item["intent_score"]) for item in scored_facts), default=0.0),
        "matched_provider_counts": _sorted_counter(
            Counter(str(item["provider"]) for item in covered)
        ),
        "matched_source_family_counts": _sorted_counter(
            Counter(str(item["source_family"]) for item in covered)
        ),
        "mapped_facts": mapped_candidates[:10],
        "supported_facts": supported[:10],
        "collision_facts": collisions[:10],
        "top_fact_candidates": scored_facts[:10],
        "gate_recommendation": _gate_recommendation(decision),
    }


def _score_fact(
    document: Mapping[str, Any],
    *,
    question: str,
    answer: str,
    claim_text: str,
    subject_coverage_threshold: float,
    mapping_score_threshold: float,
    answer_overlap_threshold: float,
    weak_overlap_threshold: float,
) -> dict[str, Any]:
    metadata = _mapping(document.get("metadata"))
    provider = _provider(metadata)
    source_family = _source_family(metadata)
    fact_type = _fact_type(metadata)
    subject = _subject(metadata)
    fact_answer = str(document.get("answer") or "")
    fact_question = str(document.get("question") or "")
    record_text = " ".join(part for part in (question, claim_text, answer) if part)
    record_tokens = _tokens(record_text)
    question_tokens = _tokens(question or claim_text)
    fact_question_tokens = _tokens(fact_question)
    subject_tokens = _tokens(subject)
    intent_tokens = _intent_tokens(metadata)
    answer_tokens = _tokens(answer)
    fact_answer_tokens = _tokens(fact_answer)

    subject_coverage = _coverage(subject_tokens, record_tokens)
    subject_question_coverage = _coverage(subject_tokens, question_tokens)
    intent_score = _intent_score(intent_tokens, record_tokens)
    question_overlap = _overlap(question_tokens, fact_question_tokens)
    weak_textual_overlap = _overlap(record_tokens, fact_question_tokens | fact_answer_tokens)
    answer_value_overlap = _overlap(answer_tokens, fact_answer_tokens)
    answer_subject_overlap = _overlap(answer_tokens, subject_tokens)
    exact_question_match = bool(question) and normalize_claim_text(question) == normalize_claim_text(fact_question)
    answer_value_supported = answer_value_overlap >= answer_overlap_threshold
    answer_entity_collision = answer_subject_overlap >= answer_overlap_threshold
    mapping_score = (
        0.55 * max(subject_coverage, subject_question_coverage)
        + 0.30 * intent_score
        + 0.10 * question_overlap
        + 0.05 * weak_textual_overlap
    )
    covered_fact_match = (
        exact_question_match
        or (
            max(subject_coverage, subject_question_coverage) >= subject_coverage_threshold
            and intent_score > 0.0
            and mapping_score >= mapping_score_threshold
        )
    )
    weak_candidate = (
        not covered_fact_match
        and weak_textual_overlap >= weak_overlap_threshold
        and (subject_coverage > 0.0 or intent_score > 0.0)
    )
    return {
        "question": fact_question,
        "answer": fact_answer,
        "source": document.get("source"),
        "provider": provider,
        "source_family": source_family,
        "fact_type": fact_type,
        "subject": subject,
        "intent_terms": tuple(sorted(intent_tokens)),
        "covered_fact_match": covered_fact_match,
        "weak_candidate": weak_candidate,
        "exact_question_match": exact_question_match,
        "subject_coverage": subject_coverage,
        "subject_question_coverage": subject_question_coverage,
        "intent_score": intent_score,
        "question_overlap": question_overlap,
        "weak_textual_overlap": weak_textual_overlap,
        "answer_value_overlap": answer_value_overlap,
        "answer_subject_overlap": answer_subject_overlap,
        "answer_value_supported": answer_value_supported,
        "answer_entity_collision": answer_entity_collision,
        "mapping_score": mapping_score,
        "metadata": _fact_metadata_summary(metadata),
    }


def _mapping_decision(
    *,
    scored_facts: Sequence[Mapping[str, Any]],
    mapped_candidates: Sequence[Mapping[str, Any]],
    supported: Sequence[Mapping[str, Any]],
    collisions: Sequence[Mapping[str, Any]],
    subject_coverage_threshold: float,
    weak_overlap_threshold: float,
) -> str:
    if mapped_candidates:
        return "mapped_qa_fact_candidate"
    if supported:
        return "answer_value_supported_by_covered_fact"
    if collisions:
        return "answer_entity_collision"
    if any(bool(item.get("covered_fact_match")) for item in scored_facts):
        return "covered_fact_match_without_correction"
    if any(float(item.get("subject_coverage") or 0.0) >= subject_coverage_threshold for item in scored_facts):
        return "subject_only_or_missing_intent"
    if any(float(item.get("intent_score") or 0.0) > 0.0 for item in scored_facts):
        return "intent_only_or_missing_subject"
    if any(float(item.get("weak_textual_overlap") or 0.0) >= weak_overlap_threshold for item in scored_facts):
        return "weak_textual_overlap"
    return "no_candidate_fact"


def _gate_recommendation(decision: str) -> str:
    if decision == "mapped_qa_fact_candidate":
        return "structured_qa_correction_handoff"
    if decision == "answer_value_supported_by_covered_fact":
        return "answer_support_audit"
    if decision == "answer_entity_collision":
        return "answer_collision_audit"
    if decision in {"subject_only_or_missing_intent", "intent_only_or_missing_subject"}:
        return "richer_property_or_indicator_collection"
    if decision == "weak_textual_overlap":
        return "citation_retrieval_before_handoff"
    return "source_family_coverage_expansion"


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = Counter(str(record.get("mapping_decision")) for record in records)
    provider_counts = Counter()
    family_counts = Counter()
    question_type_counts = Counter()
    for record in records:
        question_type_counts[str(record.get("question_type") or "unknown")] += 1
        provider_counts.update(_mapping(record.get("matched_provider_counts")))
        family_counts.update(_mapping(record.get("matched_source_family_counts")))
    return {
        "target_count": len(records),
        "covered_fact_match_count": sum(1 for record in records if record.get("covered_fact_match")),
        "mapped_qa_fact_candidate_count": decisions.get("mapped_qa_fact_candidate", 0),
        "answer_value_supported_count": decisions.get("answer_value_supported_by_covered_fact", 0),
        "answer_entity_collision_count": decisions.get("answer_entity_collision", 0),
        "subject_only_or_missing_intent_count": decisions.get("subject_only_or_missing_intent", 0),
        "intent_only_or_missing_subject_count": decisions.get("intent_only_or_missing_subject", 0),
        "weak_textual_overlap_count": decisions.get("weak_textual_overlap", 0),
        "no_candidate_fact_count": decisions.get("no_candidate_fact", 0),
        "mapping_decision_counts": _sorted_counter(decisions),
        "matched_provider_counts": _sorted_counter(provider_counts),
        "matched_source_family_counts": _sorted_counter(family_counts),
        "question_type_counts": _sorted_counter(question_type_counts),
    }


def _claim_records(payload: Any) -> tuple[dict[str, Any], ...]:
    raw_records = _raw_record_sequence(payload)
    records = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            continue
        record = _claim_record(item, fallback_index=index)
        if record is not None:
            records.append(record)
    if not records:
        raise ValueError("claims payload did not contain usable claim records.")
    return tuple(records)


def _raw_record_sequence(payload: Any) -> tuple[Any, ...]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return tuple(payload)
    if isinstance(payload, Mapping):
        for key in ("records", "claims", "statements", "documents", "rows", "data"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return tuple(value)
        return (payload,)
    return ()


def _claim_record(item: Mapping[str, Any], *, fallback_index: int) -> dict[str, Any] | None:
    question = _clean_text(item.get("question"))
    answer = _clean_text(item.get("answer"))
    text = _clean_text(
        item.get("claim")
        or item.get("statement")
        or item.get("text")
        or item.get("content")
    )
    if text is None and question is not None and answer is not None:
        text = f"{question} {answer}"
    if text is None and question is None:
        return None
    claim_id = _clean_text(item.get("claim_id") or item.get("id") or item.get("key"))
    record_index = item.get("record_index", item.get("index"))
    record_id = claim_id or (
        f"record-{record_index}" if record_index is not None else f"row-{fallback_index}"
    )
    return {
        "record_id": str(record_id),
        "record_index": _int_or_none(record_index),
        "claim_id": claim_id,
        "question": question or "",
        "answer": answer or "",
        "text": text or question or "",
        "label": _label(item),
        "question_type": item.get("question_type"),
    }


def _qa_documents(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_documents = payload.get("documents", payload.get("records", ()))
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes, bytearray)):
        raise ValueError("QA corpus must contain a documents or records list.")
    documents = []
    seen: set[tuple[str, str]] = set()
    for item in raw_documents:
        if not isinstance(item, Mapping):
            continue
        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("answer"))
        if not question or not answer:
            continue
        key = (normalize_claim_text(question), normalize_claim_text(answer))
        if key in seen:
            continue
        seen.add(key)
        documents.append({
            "question": question,
            "answer": answer,
            "source": item.get("source"),
            "metadata": dict(_mapping(item.get("metadata"))),
        })
    if not documents:
        raise ValueError("QA corpus did not contain usable documents.")
    return tuple(documents)


def _intent_tokens(metadata: Mapping[str, Any]) -> set[str]:
    fact_type = _fact_type(metadata)
    tokens = set()
    tokens.update(_tokens(metadata.get("statement_property_label")))
    tokens.update(_tokens(metadata.get("indicator_name")))
    tokens.update(_tokens(fact_type))
    normalized_type = _metadata_key_component(fact_type)
    tokens.update(_tokens(" ".join(FACT_TYPE_ALIASES.get(normalized_type, ()))))
    if _provider(metadata) == "worldbank":
        tokens.update(_tokens(metadata.get("reference_year")))
    return tokens


def _intent_score(intent_tokens: set[str], record_tokens: set[str]) -> float:
    if not intent_tokens or not record_tokens:
        return 0.0
    overlap_count = len(intent_tokens & record_tokens)
    if overlap_count == 0:
        return 0.0
    return min(1.0, overlap_count / min(3, len(intent_tokens)))


def _fact_metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "provider",
        "source_family",
        "statement_property",
        "statement_property_label",
        "subject",
        "subject_qid",
        "indicator",
        "indicator_name",
        "country_name",
        "country_code_iso3",
        "reference_year",
        "url",
    )
    return {key: metadata[key] for key in keys if metadata.get(key) is not None}


def _provider(metadata: Mapping[str, Any]) -> str:
    return _clean_text(metadata.get("provider")) or "unknown_provider"


def _source_family(metadata: Mapping[str, Any]) -> str:
    return _clean_text(metadata.get("source_family")) or "unknown_family"


def _fact_type(metadata: Mapping[str, Any]) -> str:
    return (
        _clean_text(metadata.get("statement_property"))
        or _clean_text(metadata.get("indicator"))
        or _clean_text(metadata.get("extraction_rule"))
        or "unknown_fact"
    )


def _subject(metadata: Mapping[str, Any]) -> str:
    return (
        _clean_text(metadata.get("subject"))
        or _clean_text(metadata.get("country_name"))
        or _clean_text(metadata.get("title"))
        or ""
    )


def _label(item: Mapping[str, Any]) -> int | None:
    raw = item.get("label")
    if raw is None and item.get("is_false") is not None:
        raw = 1 if item.get("is_false") else 0
    try:
        return None if raw is None else int(raw)
    except (TypeError, ValueError):
        return None


def _tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    text = normalize_claim_text(str(value))
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) > 1 and token not in STOPWORDS
    }


def _coverage(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _metadata_key_component(value: Any) -> str:
    text = str(value).strip().lower()
    return "_".join(part for part in re.split(r"[^a-z0-9]+", text) if part) or "unknown"


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sorted_counter(counter: Counter[str] | Mapping[str, Any]) -> dict[str, int]:
    items = Counter({str(key): int(value) for key, value in dict(counter).items()})
    return dict(sorted(items.items(), key=lambda item: (-item[1], item[0])))


def _load_claim_payload(path: str | Path) -> Any:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            rows.append(dict(payload))
    return tuple(rows)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


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
    parser.add_argument("--claims", required=True)
    parser.add_argument("--qa-corpus", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--route-summary", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--subject-coverage-threshold", type=float, default=DEFAULT_SUBJECT_COVERAGE_THRESHOLD)
    parser.add_argument("--mapping-score-threshold", type=float, default=DEFAULT_MAPPING_SCORE_THRESHOLD)
    parser.add_argument("--answer-overlap-threshold", type=float, default=DEFAULT_ANSWER_OVERLAP_THRESHOLD)
    parser.add_argument("--weak-overlap-threshold", type=float, default=DEFAULT_WEAK_OVERLAP_THRESHOLD)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        claims_path=args.claims,
        qa_corpus_path=args.qa_corpus,
        output_path=args.json,
        route_summary_path=args.route_summary,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        subject_coverage_threshold=args.subject_coverage_threshold,
        mapping_score_threshold=args.mapping_score_threshold,
        answer_overlap_threshold=args.answer_overlap_threshold,
        weak_overlap_threshold=args.weak_overlap_threshold,
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "source_family_structured_qa_claim_mapping_audit_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"matched={summary['covered_fact_match_count']} "
        f"mapped={summary['mapped_qa_fact_candidate_count']} "
        f"supported={summary['answer_value_supported_count']}"
    )


if __name__ == "__main__":
    main()
