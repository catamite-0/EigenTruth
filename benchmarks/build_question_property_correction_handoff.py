"""Build a ProductTrace-visible correction handoff from question/property maps.

The input is the stricter output of ``map_blind_spot_question_properties.py``.
Only records with ``correction_candidate=true`` become route inputs. The
workflow creates a target-specific structured-QA correction corpus for the
original question, verifies the model answer with the existing
``QuestionAnswerVerifier``, and writes ProductTrace rows showing the refutation,
risk decision, and dry-run action.
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

DEFAULT_ROUTE_NAME = "question_property_structured_qa"
DEFAULT_VERIFIER_NAME = "QuestionAnswerVerifier"


def build_question_property_correction_handoff(
    mapping_report: Mapping[str, Any],
    *,
    route_name: str = DEFAULT_ROUTE_NAME,
    verifier_name: str = DEFAULT_VERIFIER_NAME,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return report, QA corpus, and ProductTrace rows for mapped corrections."""
    records = tuple(record for record in _records(mapping_report) if bool(record.get("correction_candidate")))
    documents = _correction_documents(records, route_name=route_name)
    corpus = _qa_corpus(
        documents=documents,
        mapping_report=mapping_report,
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
        for record in records
    )
    report = {
        "schema_version": 1,
        "workflow": "question_property_correction_handoff",
        "status": _status(traces),
        "scope": (
            "Target-specific handoff for explicitly mapped blind-spot "
            "question/property corrections. It is not a broad retrieval corpus "
            "or a claim that unmapped blind spots are covered."
        ),
        "source": {
            "question_property_mapping_workflow": mapping_report.get("workflow"),
            "question_property_mapping_status": mapping_report.get("status"),
            "question_property_mapping_target_count": _nested_int(mapping_report, "summary", "target_count"),
            "mapped_correction_candidate_count": _nested_int(
                mapping_report,
                "summary",
                "mapped_correction_candidate_count",
            ),
        },
        "config": {
            "route_name": route_name,
            "verifier_name": verifier_name,
        },
        "summary": _summary(records=records, documents=documents, traces=traces),
        "trace_summaries": tuple(_trace_summary(trace) for trace in traces),
        "metadata": dict(metadata or {}),
    }
    return {
        "report": report,
        "qa_corpus": corpus,
        "product_traces": traces,
    }


def run(
    *,
    mapping_report_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    qa_corpus_json_path: str | Path | None = None,
    trace_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    route_name: str = DEFAULT_ROUTE_NAME,
    verifier_name: str = DEFAULT_VERIFIER_NAME,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register the handoff."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    report_path = (
        Path(report_json_path)
        if report_json_path is not None
        else output / "question-property-correction-handoff.json"
    )
    qa_path = (
        Path(qa_corpus_json_path)
        if qa_corpus_json_path is not None
        else output / "question-property-correction-corpus.json"
    )
    trace_path = Path(trace_jsonl_path) if trace_jsonl_path is not None else output / "product-traces.jsonl"
    manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output / "artifact-manifest.json"
    )
    mapping_report = _load_json_mapping(mapping_report_path)
    payload = build_question_property_correction_handoff(
        mapping_report,
        route_name=route_name,
        verifier_name=verifier_name,
        metadata=metadata,
    )
    report = dict(payload["report"])
    report["paths"] = {
        "question_property_mapping": str(mapping_report_path),
        "qa_corpus": str(qa_path),
        "product_traces": str(trace_path),
        "artifact_manifest": str(manifest_path),
    }
    payload["report"] = report

    _write_json(report_path, report, compact=compact_json)
    _write_json(qa_path, payload["qa_corpus"], compact=compact_json)
    _write_jsonl(trace_path, payload["product_traces"])

    manifest = build_artifact_manifest(
        {
            "question_property_correction_handoff": report_path,
            "question_property_correction_corpus": qa_path,
            "product_traces": trace_path,
            "question_property_mapping": Path(mapping_report_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": report["workflow"],
            "status": report["status"],
            "trace_count": report["summary"]["trace_count"],
            "correction_candidate_count": report["summary"]["correction_candidate_count"],
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
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": report["workflow"],
                "status": report["status"],
                "trace_count": report["summary"]["trace_count"],
                "correction_candidate_count": report["summary"]["correction_candidate_count"],
                "corpus_document_count": report["summary"]["corpus_document_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _correction_documents(
    records: Sequence[Mapping[str, Any]],
    *,
    route_name: str,
) -> tuple[dict[str, Any], ...]:
    documents: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for record in records:
        question = str(record.get("question", "")).strip()
        record_index = int(record.get("record_index", -1))
        for fact in _sequence(record.get("mapped_facts")):
            if not isinstance(fact, Mapping):
                continue
            answer = str(fact.get("answer", "")).strip()
            if not question or not answer:
                continue
            key = (record_index, normalize_claim_text(answer))
            if key in seen:
                continue
            seen.add(key)
            metadata = {
                "provider": "wikidata",
                "correction_scope": "target_specific_question_property",
                "route_name": route_name,
                "source_record_index": record_index,
                "source_mapping_decision": record.get("mapping_decision"),
                "statement_property": fact.get("statement_property"),
                "statement_property_label": fact.get("statement_property_label"),
                "subject": fact.get("subject"),
                "subject_qid": fact.get("subject_qid"),
                "value_qid": fact.get("value_qid"),
                "matched_intents": tuple(str(item) for item in _sequence(fact.get("matched_intents"))),
                "mapping_score": fact.get("mapping_score"),
                "source": fact.get("source"),
            }
            documents.append({
                "question": question,
                "answer": answer,
                "text": f"{question} {answer}",
                "source": fact.get("source"),
                "metadata": metadata,
            })
    return tuple(documents)


def _qa_corpus(
    *,
    documents: Sequence[Mapping[str, Any]],
    mapping_report: Mapping[str, Any],
    route_name: str,
    verifier_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus_type": "target_specific_question_property_correction_qa",
        "description": (
            "Structured QA correction corpus generated only from explicit "
            "question/property mapping candidates."
        ),
        "label_usage": {
            "uses_blind_spot_target_selection": True,
            "labels_copied_to_document_metadata": False,
            "score_dump_rows_copied_to_document_metadata": False,
            "not_general_retrieval_corpus": True,
        },
        "source": {
            "workflow": mapping_report.get("workflow"),
            "status": mapping_report.get("status"),
            "route_name": route_name,
            "verifier_name": verifier_name,
        },
        "summary": {
            "n_documents": len(documents),
            "n_questions": len({normalize_claim_text(str(item.get("question", ""))) for item in documents}),
            "by_property": _sorted_counter(
                Counter(
                    str(_mapping(item.get("metadata")).get("statement_property") or "unknown")
                    for item in documents
                )
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
    request_id = f"blind-spot-record-{record_index}"
    claim = Claim(
        text=f"{record.get('question', '')} {record.get('answer', '')}".strip(),
        claim_id=f"{request_id}:model-answer",
        metadata={
            "question": record.get("question"),
            "answer": record.get("answer"),
            "source_record_index": record_index,
            "route_hints": ("structured_qa", route_name),
            "requires_verification": True,
            "question_property_gate": True,
        },
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
                context={"route_name": route_name},
            ),
            start=1,
        )
    )
    action_results = DryRunActionExecutor(executor_name="question_property_correction_handoff").execute_many(
        actions,
        context={"request_id": request_id, "route_name": route_name},
    )
    trace = ProductTrace(
        request_id=request_id,
        diagnostics={
            "question_property_gate": 1.0,
            "mapped_fact_count": len(_sequence(record.get("mapped_facts"))),
            "best_mapping_score": record.get("best_mapping_score"),
        },
        claims=(claim,),
        verification_results=(result,),
        risk_decision=decision,
        actions=actions,
        action_results=action_results,
        metadata={
            "workflow": "question_property_correction_handoff",
            "route_name": route_name,
            "verifier_name": verifier_name,
            "source_mapping_decision": record.get("mapping_decision"),
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
            "route_family": "question_property_correction",
            "source_record_index": record.get("record_index"),
            "source_mapping_decision": record.get("mapping_decision"),
            "mapped_properties": tuple(
                str(_mapping(fact).get("statement_property") or "unknown")
                for fact in _sequence(record.get("mapped_facts"))
                if isinstance(fact, Mapping)
            ),
        },
    )


def _risk_decision(result: VerificationResult, *, route_name: str) -> RiskDecision:
    if result.status is VerificationStatus.REFUTED:
        return RiskDecision(
            action=ControlAction.ABSTAIN,
            risk_level=RiskLevel.HIGH,
            confidence=max(result.confidence, 0.9),
            reason="question/property correction refuted the model answer",
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
        reason=f"question/property correction returned {result.status.value}",
        diagnostics={
            "verification": {
                "total": 1,
                "counts": {result.status.value: 1},
                "route": route_name,
            }
        },
    )


def _status(traces: Sequence[Mapping[str, Any]]) -> str:
    if not traces:
        return "blocked"
    return "promote" if all(_trace_status(trace) == VerificationStatus.REFUTED.value for trace in traces) else "blocked"


def _summary(
    *,
    records: Sequence[Mapping[str, Any]],
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
    property_counts = Counter(
        str(_mapping(item.get("metadata")).get("statement_property") or "unknown")
        for item in documents
    )
    return {
        "correction_candidate_count": len(records),
        "corpus_document_count": len(documents),
        "trace_count": len(traces),
        "verification_status_counts": _sorted_counter(verification_status_counts),
        "action_counts": _sorted_counter(action_counts),
        "property_counts": _sorted_counter(property_counts),
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


def _records(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_records = payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("question/property mapping report must contain a records list.")
    records = [dict(item) for item in raw_records if isinstance(item, Mapping)]
    if not records:
        raise ValueError("question/property mapping report did not contain usable records.")
    return tuple(records)


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
    parser.add_argument("--question-property-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--qa-corpus-json", default=None)
    parser.add_argument("--trace-jsonl", default=None)
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
        mapping_report_path=args.question_property_mapping,
        output_dir=args.output_dir,
        report_json_path=args.report_json,
        qa_corpus_json_path=args.qa_corpus_json,
        trace_jsonl_path=args.trace_jsonl,
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
        "question_property_correction_handoff_ok "
        f"status={payload['report']['status']} "
        f"candidates={summary['correction_candidate_count']} "
        f"docs={summary['corpus_document_count']} "
        f"traces={summary['trace_count']}"
    )


if __name__ == "__main__":
    main()
