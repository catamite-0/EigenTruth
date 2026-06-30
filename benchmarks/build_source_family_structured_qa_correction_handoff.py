"""Build a ProductTrace handoff from source-family structured-QA claim maps.

This workflow starts after
``audit_source_family_structured_qa_claim_mapping.py``. It only promotes rows
whose mapping decision is ``mapped_qa_fact_candidate`` and only when the
upstream structured-QA covered-fact route was promoted. The output is a
target-specific correction corpus plus ProductTrace/action-result JSONL rows;
it is not a broad retrieval corpus and does not treat weak matches as evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.adapters import QuestionAnswerVerifier  # noqa: E402
from eigentruth.control import (  # noqa: E402
    ActionExecutorRegistry,
    ControlAction,
    DefaultCorrectionPolicy,
    DryRunActionExecutor,
    ProductTrace,
    RiskDecision,
    RiskLevel,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify import Claim, VerificationResult, VerificationStatus, normalize_claim_text  # noqa: E402

WORKFLOW = "source_family_structured_qa_correction_handoff"
DEFAULT_ROUTE_NAME = "source_family_structured_qa_correction"
DEFAULT_VERIFIER_NAME = "QuestionAnswerVerifier"


def build_source_family_structured_qa_correction_handoff(
    claim_mapping: Mapping[str, Any],
    *,
    route_name: str = DEFAULT_ROUTE_NAME,
    verifier_name: str = DEFAULT_VERIFIER_NAME,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return report, correction corpus, ProductTrace rows, and action results."""
    route_promoted = _route_promoted(claim_mapping)
    input_records = _records(claim_mapping)
    candidate_records = tuple(
        record for record in input_records if route_promoted and _is_correction_candidate(record)
    )
    documents = _correction_documents(candidate_records, route_name=route_name)
    corpus = _qa_corpus(
        documents=documents,
        claim_mapping=claim_mapping,
        route_name=route_name,
        verifier_name=verifier_name,
    )
    traces = tuple(
        _trace_for_record(
            record,
            corpus=corpus,
            route_name=route_name,
            verifier_name=verifier_name,
        )
        for record in candidate_records
    )
    action_results = tuple(
        result
        for trace in traces
        for result in _sequence(trace.get("action_results"))
        if isinstance(result, Mapping)
    )
    report = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(route_promoted=route_promoted, traces=traces),
        "scope": (
            "Target-specific correction handoff for source-family structured "
            "QA covered facts. It only uses mapped_qa_fact_candidate rows from "
            "a promoted route audit and does not claim open-domain coverage."
        ),
        "source": {
            "claim_mapping_workflow": claim_mapping.get("workflow"),
            "claim_mapping_status": claim_mapping.get("status"),
            "route_summary_promoted": route_promoted,
            "route_summary_status": _mapping(claim_mapping.get("source")).get("route_summary_status"),
            "target_count": _nested_int(claim_mapping, "summary", "target_count"),
            "mapped_qa_fact_candidate_count": _nested_int(
                claim_mapping,
                "summary",
                "mapped_qa_fact_candidate_count",
            ),
        },
        "config": {
            "route_name": route_name,
            "verifier_name": verifier_name,
        },
        "summary": _summary(
            input_records=input_records,
            candidate_records=candidate_records,
            documents=documents,
            traces=traces,
        ),
        "trace_summaries": tuple(_trace_summary(trace) for trace in traces),
        "metadata": dict(metadata or {}),
    }
    return {
        "report": report,
        "qa_corpus": corpus,
        "product_traces": traces,
        "action_results": action_results,
    }


def run(
    *,
    claim_mapping_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    qa_corpus_json_path: str | Path | None = None,
    trace_jsonl_path: str | Path | None = None,
    action_results_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    route_name: str = DEFAULT_ROUTE_NAME,
    verifier_name: str = DEFAULT_VERIFIER_NAME,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register the correction handoff."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    report_path = (
        Path(report_json_path)
        if report_json_path is not None
        else output / "source-family-structured-qa-correction-handoff.json"
    )
    qa_path = (
        Path(qa_corpus_json_path)
        if qa_corpus_json_path is not None
        else output / "source-family-structured-qa-correction-corpus.json"
    )
    trace_path = Path(trace_jsonl_path) if trace_jsonl_path is not None else output / "product-traces.jsonl"
    action_results_path = (
        Path(action_results_jsonl_path)
        if action_results_jsonl_path is not None
        else output / "action-results.jsonl"
    )
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output / "artifact-manifest.json"
    )
    claim_mapping = _load_json_mapping(claim_mapping_path)
    payload = build_source_family_structured_qa_correction_handoff(
        claim_mapping,
        route_name=route_name,
        verifier_name=verifier_name,
        metadata=metadata,
    )
    report = dict(payload["report"])
    report["paths"] = {
        "claim_mapping": str(claim_mapping_path),
        "qa_corpus": str(qa_path),
        "product_traces": str(trace_path),
        "action_results": str(action_results_path),
        "artifact_manifest": str(manifest_path),
    }
    payload["report"] = report

    _write_json(report_path, report, compact=compact_json)
    _write_json(qa_path, payload["qa_corpus"], compact=compact_json)
    _write_jsonl(trace_path, payload["product_traces"])
    _write_jsonl(action_results_path, payload["action_results"])

    manifest = build_artifact_manifest(
        {
            "source_family_structured_qa_correction_handoff": report_path,
            "source_family_structured_qa_correction_corpus": qa_path,
            "product_traces": trace_path,
            "action_results": action_results_path,
            "source_family_structured_qa_claim_mapping": Path(claim_mapping_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": report["workflow"],
            "status": report["status"],
            "route_summary_promoted": report["source"]["route_summary_promoted"],
            "correction_candidate_count": report["summary"]["correction_candidate_count"],
            "trace_count": report["summary"]["trace_count"],
            "refuted_count": report["summary"]["verification_status_counts"].get(
                VerificationStatus.REFUTED.value,
                0,
            ),
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": report["workflow"],
                "status": report["status"],
                "correction_candidate_count": report["summary"]["correction_candidate_count"],
                "corpus_document_count": report["summary"]["corpus_document_count"],
                "trace_count": report["summary"]["trace_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        )
        registry.record_trace(
            name=name,
            version=version,
            path=trace_path,
            metadata={
                "workflow": report["workflow"],
                "status": report["status"],
                "trace_count": report["summary"]["trace_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        )
        registry.record_action_result(
            name=name,
            version=version,
            path=action_results_path,
            metadata={
                "workflow": report["workflow"],
                "status": report["status"],
                "action_result_count": len(payload["action_results"]),
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        )
        registry.save_json()
    return payload


def _correction_documents(
    records: Sequence[Mapping[str, Any]],
    *,
    route_name: str,
) -> tuple[dict[str, Any], ...]:
    documents: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str, str]] = set()
    for record in records:
        question = str(record.get("question") or "").strip()
        record_index = int(record.get("record_index", -1))
        for fact in _sequence(record.get("mapped_facts")):
            if not isinstance(fact, Mapping):
                continue
            answer = str(fact.get("answer") or "").strip()
            if not question or not answer:
                continue
            fact_metadata = _fact_metadata(fact)
            source = fact.get("source")
            key = (
                record_index,
                normalize_claim_text(question),
                normalize_claim_text(answer),
                str(source or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            metadata = {
                "correction_scope": "target_specific_source_family_structured_qa",
                "route_name": route_name,
                "source_record_index": record_index,
                "source_record_id": record.get("record_id"),
                "source_mapping_decision": record.get("mapping_decision"),
                "source_gate_recommendation": record.get("gate_recommendation"),
                "source_fact_question": fact.get("question"),
                "source": source,
                "provider": fact.get("provider") or fact_metadata.get("provider"),
                "source_family": fact.get("source_family") or fact_metadata.get("source_family"),
                "fact_type": fact.get("fact_type") or fact_metadata.get("statement_property"),
                "subject": fact.get("subject") or fact_metadata.get("subject"),
                "subject_qid": fact_metadata.get("subject_qid"),
                "statement_property": fact_metadata.get("statement_property") or fact.get("fact_type"),
                "statement_property_label": fact_metadata.get("statement_property_label"),
                "indicator": fact_metadata.get("indicator"),
                "indicator_name": fact_metadata.get("indicator_name"),
                "country_name": fact_metadata.get("country_name"),
                "country_code_iso3": fact_metadata.get("country_code_iso3"),
                "reference_year": fact_metadata.get("reference_year"),
                "mapping_score": fact.get("mapping_score"),
                "subject_coverage": fact.get("subject_coverage"),
                "intent_score": fact.get("intent_score"),
                "url": fact_metadata.get("url"),
            }
            documents.append({
                "question": question,
                "answer": answer,
                "text": f"{question} {answer}",
                "source": source,
                "metadata": {key: value for key, value in metadata.items() if value is not None},
            })
    return tuple(documents)


def _qa_corpus(
    *,
    documents: Sequence[Mapping[str, Any]],
    claim_mapping: Mapping[str, Any],
    route_name: str,
    verifier_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus_type": "target_specific_source_family_structured_qa_correction",
        "description": (
            "Structured QA correction corpus generated only from exact "
            "source-family claim-mapping candidates."
        ),
        "label_usage": {
            "uses_blind_spot_target_selection": True,
            "labels_copied_to_document_metadata": False,
            "score_dump_rows_copied_to_document_metadata": False,
            "not_general_retrieval_corpus": True,
        },
        "source": {
            "workflow": claim_mapping.get("workflow"),
            "status": claim_mapping.get("status"),
            "route_summary_promoted": _route_promoted(claim_mapping),
            "route_name": route_name,
            "verifier_name": verifier_name,
        },
        "summary": {
            "n_documents": len(documents),
            "n_questions": len({normalize_claim_text(str(item.get("question", ""))) for item in documents}),
            "by_provider": _sorted_counter(
                Counter(str(_mapping(item.get("metadata")).get("provider") or "unknown") for item in documents)
            ),
            "by_source_family": _sorted_counter(
                Counter(
                    str(_mapping(item.get("metadata")).get("source_family") or "unknown")
                    for item in documents
                )
            ),
            "by_fact_type": _sorted_counter(
                Counter(str(_mapping(item.get("metadata")).get("fact_type") or "unknown") for item in documents)
            ),
        },
        "documents": tuple(dict(item) for item in documents),
    }


def _trace_for_record(
    record: Mapping[str, Any],
    *,
    corpus: Mapping[str, Any],
    route_name: str,
    verifier_name: str,
) -> dict[str, Any]:
    record_index = int(record.get("record_index", -1))
    request_id = f"source-family-structured-qa-record-{record_index}"
    claim_id = f"{request_id}:model-answer"
    claim_triples = _claim_triples_for_record(record, claim_id=claim_id)
    claim_metadata = {
        "question": record.get("question"),
        "answer": record.get("answer"),
        "source_record_index": record_index,
        "route_hints": ("structured_qa", route_name),
        "requires_verification": True,
        "source_family_structured_qa_gate": True,
    }
    if claim_triples:
        claim_metadata["claim_triples"] = claim_triples
        claim_metadata["requires_triple_audit"] = True
    claim = Claim(
        text=f"{record.get('question', '')} {record.get('answer', '')}".strip(),
        claim_id=claim_id,
        metadata=claim_metadata,
    )
    verifier = QuestionAnswerVerifier.from_corpus(corpus)
    raw_result = verifier.verify(claim)
    result = _annotated_result(
        raw_result,
        route_name=route_name,
        verifier_name=verifier_name,
        record=record,
    )
    decision = _risk_decision(result, route_name=route_name)
    actions = tuple(
        replace(action, request_id=f"{request_id}:action:{idx}")
        for idx, action in enumerate(
            DefaultCorrectionPolicy().plan(
                decision,
                claims=(claim,),
                verification_results=(result,),
                context={"route_name": route_name, "workflow": WORKFLOW},
            ),
            start=1,
        )
    )
    executor = ActionExecutorRegistry(
        fallback_executor=DryRunActionExecutor(executor_name=WORKFLOW)
    )
    action_results = executor.execute_many(
        actions,
        context={"request_id": request_id, "route_name": route_name, "workflow": WORKFLOW},
    )
    trace = ProductTrace(
        request_id=request_id,
        diagnostics={
            "source_family_structured_qa_gate": 1.0,
            "mapped_fact_count": len(_sequence(record.get("mapped_facts"))),
            "best_mapping_score": record.get("best_mapping_score"),
            "best_subject_coverage": record.get("best_subject_coverage"),
            "best_intent_score": record.get("best_intent_score"),
        },
        claims=(claim,),
        verification_results=(result,),
        risk_decision=decision,
        actions=actions,
        action_results=action_results,
        metadata={
            "workflow": WORKFLOW,
            "route_name": route_name,
            "verifier_name": verifier_name,
            "source_mapping_decision": record.get("mapping_decision"),
            "source_gate_recommendation": record.get("gate_recommendation"),
            "source_record_index": record_index,
        },
    )
    payload = trace.to_dict()
    payload["bounded"] = trace.to_bounded_dict(
        metadata_keys=(
            "workflow",
            "route_name",
            "verifier_name",
            "source_mapping_decision",
            "source_gate_recommendation",
            "source_record_index",
        )
    )
    return payload


def _annotated_result(
    result: VerificationResult,
    *,
    route_name: str,
    verifier_name: str,
    record: Mapping[str, Any],
) -> VerificationResult:
    provider_counts = Counter()
    family_counts = Counter()
    fact_type_counts = Counter()
    for fact in _sequence(record.get("mapped_facts")):
        if not isinstance(fact, Mapping):
            continue
        metadata = _fact_metadata(fact)
        provider_counts[str(fact.get("provider") or metadata.get("provider") or "unknown")] += 1
        family_counts[str(fact.get("source_family") or metadata.get("source_family") or "unknown")] += 1
        fact_type_counts[str(fact.get("fact_type") or metadata.get("statement_property") or "unknown")] += 1
    return VerificationResult(
        status=result.status,
        confidence=result.confidence,
        evidence=result.evidence,
        explanation=result.explanation,
        metadata={
            **dict(result.metadata),
            "selected_route": route_name,
            "selected_verifier": verifier_name,
            "matched_routes": ("structured_qa", route_name),
            "selected_route_duration_seconds": 0.0,
            "attempted_routes": (route_name,),
            "route_family": "source_family_structured_qa_correction",
            "source_record_index": record.get("record_index"),
            "source_mapping_decision": record.get("mapping_decision"),
            "source_gate_recommendation": record.get("gate_recommendation"),
            "mapped_provider_counts": _sorted_counter(provider_counts),
            "mapped_source_family_counts": _sorted_counter(family_counts),
            "mapped_fact_type_counts": _sorted_counter(fact_type_counts),
            "evidence_documents": _refutation_evidence_documents(record),
        },
    )


def _claim_triples_for_record(
    record: Mapping[str, Any],
    *,
    claim_id: str,
) -> tuple[dict[str, Any], ...]:
    """Build conservative claim triples from mapped facts and the model answer."""
    model_answer = str(record.get("answer") or "").strip()
    if not model_answer:
        return ()
    triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in _sequence(record.get("mapped_facts")):
        if not isinstance(fact, Mapping):
            continue
        subject = _fact_subject(fact)
        predicate = _fact_predicate(fact)
        if not subject or not predicate:
            continue
        key = (subject, predicate, model_answer)
        if key in seen:
            continue
        seen.add(key)
        triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": model_answer,
            "claim_id": claim_id,
            "source_text": f"{record.get('question', '')} {model_answer}".strip(),
            "confidence": 0.9,
            "metadata": {
                "source": "source_family_structured_qa_claim_mapping",
                "role": "model_answer_claim",
                "mapping_decision": record.get("mapping_decision"),
            },
        })
    return tuple(triples)


def _refutation_evidence_documents(record: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return structured evidence snippets suitable for trace-level triple audit."""
    model_answer = str(record.get("answer") or "").strip()
    if not model_answer:
        return ()
    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for fact in _sequence(record.get("mapped_facts")):
        if not isinstance(fact, Mapping):
            continue
        subject = _fact_subject(fact)
        predicate = _fact_predicate(fact)
        correct_answer = str(fact.get("answer") or "").strip()
        if not subject or not predicate or not correct_answer:
            continue
        text = f"{subject} {predicate} is {correct_answer}, not {model_answer}."
        source = str(fact.get("source") or "source_family_structured_qa")
        key = (source, text)
        if key in seen:
            continue
        seen.add(key)
        metadata = _fact_metadata(fact)
        documents.append({
            "text": text,
            "source": source,
            "metadata": {
                "workflow": WORKFLOW,
                "evidence_relation": "refutes_model_answer",
                "source_fact_question": fact.get("question"),
                "correct_answer": correct_answer,
                "model_answer": model_answer,
                "provider": fact.get("provider") or metadata.get("provider"),
                "source_family": fact.get("source_family") or metadata.get("source_family"),
                "statement_property": metadata.get("statement_property") or fact.get("fact_type"),
            },
        })
    return tuple(documents)


def _fact_subject(fact: Mapping[str, Any]) -> str:
    metadata = _fact_metadata(fact)
    for value in (
        fact.get("subject"),
        metadata.get("subject"),
        metadata.get("country_name"),
        metadata.get("indicator_name"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _fact_predicate(fact: Mapping[str, Any]) -> str:
    metadata = _fact_metadata(fact)
    for value in (
        metadata.get("statement_property_label"),
        fact.get("fact_type"),
        metadata.get("statement_property"),
        metadata.get("indicator_name"),
        metadata.get("indicator"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _risk_decision(result: VerificationResult, *, route_name: str) -> RiskDecision:
    if result.status is VerificationStatus.REFUTED:
        return RiskDecision(
            action=ControlAction.ABSTAIN,
            risk_level=RiskLevel.HIGH,
            confidence=max(result.confidence, 0.9),
            reason="source-family structured-QA correction refuted the model answer",
            diagnostics={
                "verification": {
                    "total": 1,
                    "counts": {VerificationStatus.REFUTED.value: 1},
                    "route": route_name,
                }
            },
        )
    return RiskDecision(
        action=ControlAction.CLARIFY,
        risk_level=RiskLevel.UNKNOWN,
        confidence=max(result.confidence, 0.5),
        reason=f"source-family structured-QA correction returned {result.status.value}",
        diagnostics={
            "verification": {
                "total": 1,
                "counts": {result.status.value: 1},
                "route": route_name,
            }
        },
    )


def _status(*, route_promoted: bool, traces: Sequence[Mapping[str, Any]]) -> str:
    if not route_promoted or not traces:
        return "blocked"
    return "promote" if all(_trace_status(trace) == VerificationStatus.REFUTED.value for trace in traces) else "blocked"


def _summary(
    *,
    input_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    verification_status_counts = Counter(_trace_status(trace) for trace in traces)
    action_counts = Counter(
        str(action.get("action") or "unknown")
        for trace in traces
        for action in _sequence(trace.get("actions"))
        if isinstance(action, Mapping)
    )
    provider_counts = Counter(
        str(_mapping(item.get("metadata")).get("provider") or "unknown")
        for item in documents
    )
    family_counts = Counter(
        str(_mapping(item.get("metadata")).get("source_family") or "unknown")
        for item in documents
    )
    fact_type_counts = Counter(
        str(_mapping(item.get("metadata")).get("fact_type") or "unknown")
        for item in documents
    )
    return {
        "input_record_count": len(input_records),
        "correction_candidate_count": len(candidate_records),
        "corpus_document_count": len(documents),
        "trace_count": len(traces),
        "action_result_count": sum(
            1
            for trace in traces
            for result in _sequence(trace.get("action_results"))
            if isinstance(result, Mapping)
        ),
        "verification_status_counts": _sorted_counter(verification_status_counts),
        "action_counts": _sorted_counter(action_counts),
        "provider_counts": _sorted_counter(provider_counts),
        "source_family_counts": _sorted_counter(family_counts),
        "fact_type_counts": _sorted_counter(fact_type_counts),
    }


def _trace_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    results = _sequence(trace.get("verification_results"))
    actions = _sequence(trace.get("actions"))
    result = _mapping(results[0]) if results else {}
    metadata = _mapping(result.get("metadata"))
    return {
        "request_id": trace.get("request_id"),
        "status": result.get("status"),
        "selected_route": metadata.get("selected_route"),
        "risk_action": _mapping(trace.get("risk_decision")).get("action"),
        "action_count": len(actions),
        "evidence_count": len(_sequence(result.get("evidence"))),
    }


def _trace_status(trace: Mapping[str, Any]) -> str:
    results = _sequence(trace.get("verification_results"))
    if not results:
        return "missing"
    return str(_mapping(results[0]).get("status") or "unknown")


def _is_correction_candidate(record: Mapping[str, Any]) -> bool:
    return (
        str(record.get("mapping_decision") or "") == "mapped_qa_fact_candidate"
        and bool(record.get("mapped_qa_fact_candidate"))
        and str(record.get("gate_recommendation") or "") == "structured_qa_correction_handoff"
        and bool(_sequence(record.get("mapped_facts")))
    )


def _route_promoted(claim_mapping: Mapping[str, Any]) -> bool:
    source = _mapping(claim_mapping.get("source"))
    return source.get("route_summary_promoted") is True


def _records(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_records = payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("source-family claim mapping report must contain a records list.")
    records = [dict(item) for item in raw_records if isinstance(item, Mapping)]
    if not records:
        raise ValueError("source-family claim mapping report did not contain usable records.")
    return tuple(records)


def _fact_metadata(fact: Mapping[str, Any]) -> dict[str, Any]:
    return dict(fact.get("metadata", {})) if isinstance(fact.get("metadata"), Mapping) else {}


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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
    parser.add_argument("--claim-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--qa-corpus-json", default=None)
    parser.add_argument("--trace-jsonl", default=None)
    parser.add_argument("--action-results-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--route-name", default=DEFAULT_ROUTE_NAME)
    parser.add_argument("--verifier-name", default=DEFAULT_VERIFIER_NAME)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        claim_mapping_path=args.claim_mapping,
        output_dir=args.output_dir,
        report_json_path=args.report_json,
        qa_corpus_json_path=args.qa_corpus_json,
        trace_jsonl_path=args.trace_jsonl,
        action_results_jsonl_path=args.action_results_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        route_name=args.route_name,
        verifier_name=args.verifier_name,
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    summary = payload["report"]["summary"]
    print(
        "source_family_structured_qa_correction_handoff_ok "
        f"status={payload['report']['status']} "
        f"candidates={summary['correction_candidate_count']} "
        f"docs={summary['corpus_document_count']} "
        f"traces={summary['trace_count']}"
    )


if __name__ == "__main__":
    main()
