"""Enrich ProductTrace JSON with claim triples and slot-level evidence audits.

This workflow is intentionally offline and dependency-free. It takes existing
full ProductTrace payloads, extracts conservative claim triples, searches local
evidence snippets when a trace result did not already carry evidence, and writes
new trace files with auditable ``claim_triples`` and ``audit_report`` metadata.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import (  # noqa: E402
    planned_artifact_manifest_summary,
    reject_bounded_product_trace,
    strict_bool,
)
from eigentruth.adapters.retrieval import InMemoryRetriever, RetrievalHit, RetrievalQuery  # noqa: E402
from eigentruth.control import ProductTrace, product_runtime_metrics  # noqa: E402
from eigentruth.json_utils import strict_json_dumps, to_jsonable  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify import (  # noqa: E402
    Claim,
    EvidenceDocument,
    audit_claim_triples,
    extract_claim_triples,
)

WORKFLOW = "product_trace_triple_audit_enrichment"
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ProductTraceTripleAuditEnrichmentConfig:
    """Configuration for trace-level triple-audit enrichment."""

    trace_paths: Sequence[str | Path]
    output_dir: str | Path
    evidence_corpus_paths: Sequence[str | Path] = ()
    report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    retrieval_limit: int = 5
    min_retrieval_overlap: float = 0.55
    min_slot_coverage: float = 1.0
    min_audit_claim_coverage: float = 1.0
    min_audit_pass_rate: float = 1.0
    min_slot_coverage_rate: float = 1.0
    compact_json: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        if not trace_paths:
            raise ValueError("at least one ProductTrace path is required.")
        evidence_corpus_paths = tuple(Path(path) for path in self.evidence_corpus_paths)
        output_dir = Path(self.output_dir)
        report_path = (
            output_dir / "product-trace-triple-audit-enrichment.json"
            if self.report_path is None
            else Path(self.report_path)
        )
        artifact_manifest_path = (
            output_dir / "product-trace-triple-audit-artifact-manifest.json"
            if self.artifact_manifest_path is None
            else Path(self.artifact_manifest_path)
        )
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "evidence_corpus_paths", evidence_corpus_paths)
        object.__setattr__(self, "report_path", report_path)
        object.__setattr__(self, "artifact_manifest_path", artifact_manifest_path)
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(
            self,
            "retrieval_limit",
            _positive_int(self.retrieval_limit, name="retrieval_limit"),
        )
        object.__setattr__(
            self,
            "min_retrieval_overlap",
            _rate(self.min_retrieval_overlap, name="min_retrieval_overlap"),
        )
        object.__setattr__(
            self,
            "min_slot_coverage",
            _rate(self.min_slot_coverage, name="min_slot_coverage"),
        )
        object.__setattr__(
            self,
            "min_audit_claim_coverage",
            _rate(self.min_audit_claim_coverage, name="min_audit_claim_coverage"),
        )
        object.__setattr__(
            self,
            "min_audit_pass_rate",
            _rate(self.min_audit_pass_rate, name="min_audit_pass_rate"),
        )
        object.__setattr__(
            self,
            "min_slot_coverage_rate",
            _rate(self.min_slot_coverage_rate, name="min_slot_coverage_rate"),
        )
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_product_trace_triple_audit_enrichment(
    config: ProductTraceTripleAuditEnrichmentConfig,
) -> dict[str, Any]:
    """Write enriched ProductTrace files and return a JSON-ready report."""
    evidence_documents = _load_evidence_corpora(config.evidence_corpus_paths)
    retriever = (
        None
        if not evidence_documents
        else InMemoryRetriever(evidence_documents, min_overlap=config.min_retrieval_overlap)
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, trace_path in enumerate(config.trace_paths, start=1):
        trace = _load_trace(trace_path)
        reject_bounded_product_trace(trace, path=trace_path)
        enriched, record = _enrich_trace(
            trace,
            source_path=Path(trace_path),
            output_path=_trace_output_path(output_dir, Path(trace_path), index=index),
            retriever=retriever,
            retrieval_limit=config.retrieval_limit,
            min_slot_coverage=config.min_slot_coverage,
        )
        _write_json(record["output_path"], enriched, compact=config.compact_json)
        records.append(record)

    summary = _aggregate_records(records, evidence_document_count=len(evidence_documents))
    status, blocking_reasons = _status(summary, config=config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": blocking_reasons,
        },
        "summary": summary,
        "traces": records,
        "paths": {
            "report": str(config.report_path),
            "artifact_manifest": str(config.artifact_manifest_path),
            "output_dir": str(config.output_dir),
            "traces": tuple(str(path) for path in config.trace_paths),
            "evidence_corpora": tuple(str(path) for path in config.evidence_corpus_paths),
        },
        "config": {
            "retrieval_limit": config.retrieval_limit,
            "min_retrieval_overlap": config.min_retrieval_overlap,
            "min_slot_coverage": config.min_slot_coverage,
            "min_audit_claim_coverage": config.min_audit_claim_coverage,
            "min_audit_pass_rate": config.min_audit_pass_rate,
            "min_slot_coverage_rate": config.min_slot_coverage_rate,
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
    }
    report["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        _artifact_paths(config, records),
        assume_file_paths=(config.report_path,),
    )
    _write_json(config.report_path, report, compact=config.compact_json)
    manifest = _write_artifact_manifest(config, report, records)
    report["artifact_manifest_summary"] = manifest["summary"]
    _write_json(config.report_path, report, compact=config.compact_json)
    _record_registry(config, report)
    return report


def _enrich_trace(
    trace: Mapping[str, Any],
    *,
    source_path: Path,
    output_path: Path,
    retriever: InMemoryRetriever | None,
    retrieval_limit: int,
    min_slot_coverage: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(trace)
    claims = [dict(claim) for claim in _sequence(payload.get("claims")) if isinstance(claim, Mapping)]
    results = [
        dict(result)
        for result in _sequence(payload.get("verification_results"))
        if isinstance(result, Mapping)
    ]
    claim_records: list[dict[str, Any]] = []

    for index, claim_payload in enumerate(claims):
        claim = _claim_from_payload(claim_payload)
        triples = extract_claim_triples(claim)
        claim_metadata = dict(_mapping(claim_payload.get("metadata")))
        if triples:
            claim_metadata["claim_triples"] = tuple(triple.to_dict() for triple in triples)
            claim_metadata.setdefault("requires_triple_audit", True)
        claim_payload["metadata"] = claim_metadata

        result_payload = _result_for_claim(results, index=index, claim=claim)
        evidence, evidence_summary = _evidence_for_claim(
            claim,
            result_payload=result_payload,
            retriever=retriever,
            retrieval_limit=retrieval_limit,
        )
        audit_report = None
        if triples and evidence and result_payload is None:
            result_payload = _audit_only_result_payload(claim)
            results.append(result_payload)
        if triples and evidence and result_payload is not None:
            audit_report = audit_claim_triples(
                Claim(
                    text=claim.text,
                    claim_id=claim.claim_id,
                    span=claim.span,
                    metadata={**dict(claim.metadata), "claim_triples": tuple(triple.to_dict() for triple in triples)},
                ),
                evidence=evidence,
                min_slot_coverage=min_slot_coverage,
            ).to_dict()
            evidence_relation = _evidence_relation(result_payload.get("status"))
            audit_report["verification_status"] = result_payload.get("status")
            audit_report["evidence_relation"] = evidence_relation
            result_metadata = dict(_mapping(result_payload.get("metadata")))
            result_metadata["audit_report"] = audit_report
            result_metadata["triple_audit_enrichment"] = {
                "workflow": WORKFLOW,
                "status": "audited",
                "evidence_relation": evidence_relation,
                **evidence_summary,
            }
            result_payload["metadata"] = result_metadata
        elif result_payload is not None:
            result_metadata = dict(_mapping(result_payload.get("metadata")))
            result_metadata["triple_audit_enrichment"] = {
                "workflow": WORKFLOW,
                "status": "missing_triples" if not triples else "missing_evidence",
                **evidence_summary,
            }
            result_payload["metadata"] = result_metadata

        claim_records.append({
            "claim_id": claim.claim_id,
            "claim_index": index,
            "text": claim.text,
            "triple_count": len(triples),
            "evidence_count": len(evidence),
            "audit_report_attached": audit_report is not None,
            "audit_passed": None if audit_report is None else bool(audit_report.get("passed")),
            "evidence_summary": evidence_summary,
        })

    payload["claims"] = claims
    payload["verification_results"] = results
    summary = ProductTrace(
        claims=claims,
        verification_results=results,
    ).triple_coverage_summary()
    summaries = dict(_mapping(payload.get("summaries")))
    summaries["triple_coverage"] = summary
    payload["summaries"] = summaries
    metadata = dict(_mapping(payload.get("metadata")))
    trace_corpus = dict(_mapping(metadata.get("trace_corpus")))
    trace_corpus["triple_coverage_summary"] = summary
    metadata["trace_corpus"] = trace_corpus
    metadata["triple_audit_enrichment"] = {
        "workflow": WORKFLOW,
        "source_path": str(source_path),
        "claim_count": len(claims),
        "audit_report_count": summary.get("audit_report_count"),
        "audit_claim_coverage_rate": summary.get("audit_claim_coverage_rate"),
        "audit_pass_rate": summary.get("audit_pass_rate"),
        "slot_coverage_rate": summary.get("slot_coverage_rate"),
    }
    payload["metadata"] = metadata
    metrics = product_runtime_metrics(payload)
    return payload, {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "request_id": payload.get("request_id"),
        "claim_records": tuple(claim_records),
        "triple_coverage_summary": summary,
        "metrics": {
            "triple_claim_coverage_rate": metrics.get("triple_claim_coverage_rate"),
            "triple_audit_claim_coverage_rate": metrics.get("triple_audit_claim_coverage_rate"),
            "triple_audit_pass_rate": metrics.get("triple_audit_pass_rate"),
            "triple_slot_coverage_rate": metrics.get("triple_slot_coverage_rate"),
        },
    }


def _audit_only_result_payload(claim: Claim) -> dict[str, Any]:
    """Return a verifier-result shell for offline audit metadata."""
    return {
        "status": "not_applicable",
        "confidence": 0.0,
        "evidence": (),
        "explanation": (
            "offline triple-audit enrichment result; original trace did not "
            "include a verifier result for this claim"
        ),
        "metadata": {
            "claim_id": claim.claim_id,
            "selected_route": "triple_audit_enrichment",
            "selected_verifier": "TripleEvidenceVerifier",
            "audit_only": True,
        },
    }


def _result_for_claim(
    results: Sequence[dict[str, Any]],
    *,
    index: int,
    claim: Claim,
) -> dict[str, Any] | None:
    if index < len(results):
        indexed = results[index]
        indexed_claim_id = _result_claim_id(indexed)
        if indexed_claim_id is None or indexed_claim_id == claim.claim_id:
            return indexed
    if claim.claim_id is None:
        return None
    for result in results:
        if _result_claim_id(result) == claim.claim_id:
            return result
    return None


def _result_claim_id(result: Mapping[str, Any]) -> str | None:
    for value in (
        result.get("claim_id"),
        _mapping(result.get("metadata")).get("claim_id"),
        _mapping(result.get("metadata")).get("selected_claim_id"),
    ):
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _evidence_relation(status: Any) -> str:
    normalized = str(status or "").strip().casefold()
    if normalized == "refuted":
        return "refutes_claim"
    if normalized == "supported":
        return "supports_claim"
    if normalized:
        return f"{normalized}_claim"
    return "audits_claim"


def _evidence_for_claim(
    claim: Claim,
    *,
    result_payload: Mapping[str, Any] | None,
    retriever: InMemoryRetriever | None,
    retrieval_limit: int,
) -> tuple[tuple[EvidenceDocument, ...], dict[str, Any]]:
    documents: list[EvidenceDocument] = []
    trace_documents = _trace_evidence_documents(result_payload)
    documents.extend(trace_documents)
    retrieval_hits: tuple[RetrievalHit, ...] = ()
    if retriever is not None:
        retrieval_hits = retriever.retrieve(
            RetrievalQuery(query=claim.text, claim_id=claim.claim_id),
            limit=retrieval_limit,
        )
        documents.extend(
            EvidenceDocument(hit.text, source=hit.source, metadata=hit.metadata)
            for hit in retrieval_hits
        )
    deduped = _dedupe_evidence(documents)
    return deduped, {
        "trace_evidence_count": len(trace_documents),
        "retrieval_hit_count": len(retrieval_hits),
        "evidence_count": len(deduped),
        "retrieval_hits": tuple(hit.to_dict() for hit in retrieval_hits),
    }


def _trace_evidence_documents(result_payload: Mapping[str, Any] | None) -> tuple[EvidenceDocument, ...]:
    if result_payload is None:
        return ()
    documents: list[EvidenceDocument] = []
    for item in _sequence(result_payload.get("evidence")):
        document = _evidence_document_from_any(item, source_default="trace_verification_result")
        if document is not None:
            documents.append(document)
    metadata = _mapping(result_payload.get("metadata"))
    for key in ("retrieval_hits", "evidence_documents", "evidence"):
        for item in _sequence(metadata.get(key)):
            document = _evidence_document_from_any(item, source_default=f"trace_metadata.{key}")
            if document is not None:
                documents.append(document)
    return _dedupe_evidence(documents)


def _load_evidence_corpora(paths: Sequence[Path]) -> tuple[RetrievalHit, ...]:
    documents: list[RetrievalHit] = []
    for path in paths:
        documents.extend(_load_evidence_corpus(path))
    return _dedupe_hits(documents)


def _load_evidence_corpus(path: Path) -> list[RetrievalHit]:
    if path.suffix.lower() == ".jsonl":
        documents: list[RetrievalHit] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            documents.extend(_hits_from_payload(json.loads(line), source_default=str(path)))
        return documents
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _hits_from_payload(payload, source_default=str(path))


def _hits_from_payload(payload: Any, *, source_default: str) -> list[RetrievalHit]:
    documents: list[RetrievalHit] = []
    if isinstance(payload, str):
        text = payload.strip()
        return [] if not text else [RetrievalHit(text=text, source=source_default)]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            documents.extend(_hits_from_payload(item, source_default=source_default))
        return documents
    if not isinstance(payload, Mapping):
        return documents
    if payload.get("text") is not None or payload.get("content") is not None:
        try:
            documents.append(RetrievalHit.from_dict(payload))
        except ValueError:
            return documents
    for key in (
        "documents",
        "hits",
        "retrieval_hits",
        "evidence",
        "evidence_documents",
        "items",
        "records",
        "facts",
        "corpus",
    ):
        documents.extend(_hits_from_payload(payload.get(key), source_default=source_default))
    for key in ("record", "final", "initial", "metadata"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            documents.extend(_hits_from_payload(nested, source_default=source_default))
    return documents


def _evidence_document_from_any(
    value: Any,
    *,
    source_default: str,
) -> EvidenceDocument | None:
    if isinstance(value, EvidenceDocument):
        return value
    if isinstance(value, Mapping):
        try:
            return EvidenceDocument.from_dict(value)
        except ValueError:
            return None
    text = str(value).strip()
    if not text:
        return None
    source = source_default
    if ": " in text:
        candidate_source, candidate_text = text.split(": ", 1)
        if candidate_source.strip() and candidate_text.strip():
            source = candidate_source.strip()
            text = candidate_text.strip()
    return EvidenceDocument(text=text, source=source)


def _claim_from_payload(payload: Mapping[str, Any]) -> Claim:
    span = payload.get("span")
    parsed_span = None
    if isinstance(span, Sequence) and not isinstance(span, (str, bytes, bytearray)) and len(span) == 2:
        parsed_span = (int(span[0]), int(span[1]))
    claim_id = payload.get("claim_id")
    return Claim(
        text=str(payload.get("text", "")),
        claim_id=None if claim_id is None else str(claim_id),
        span=parsed_span,
        metadata=dict(_mapping(payload.get("metadata"))),
    )


def _aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    evidence_document_count: int,
) -> dict[str, Any]:
    summaries = tuple(_mapping(record.get("triple_coverage_summary")) for record in records)
    claim_count = _sum_int(summaries, "claim_count")
    claims_with_triples = _sum_int(summaries, "claims_with_triples")
    audit_claim_covered_count = _sum_int(summaries, "audit_claim_covered_count")
    audit_triple_count = _sum_int(summaries, "audit_triple_count")
    audit_passed_count = _sum_int(summaries, "audit_passed_count")
    covered_slot_count = _sum_int(summaries, "covered_slot_count")
    missing_slot_count = _sum_int(summaries, "missing_slot_count")
    total_slot_count = covered_slot_count + missing_slot_count
    return {
        "trace_count": len(records),
        "evidence_document_count": evidence_document_count,
        "claim_count": claim_count,
        "claims_with_triples": claims_with_triples,
        "claim_triple_count": _sum_int(summaries, "claim_triple_count"),
        "claim_triple_coverage_rate": _safe_div(claims_with_triples, claim_count),
        "audit_report_count": _sum_int(summaries, "audit_report_count"),
        "audit_claim_covered_count": audit_claim_covered_count,
        "audit_claim_coverage_rate": _safe_div(audit_claim_covered_count, claims_with_triples),
        "audit_triple_count": audit_triple_count,
        "audit_passed_count": audit_passed_count,
        "audit_failed_count": _sum_int(summaries, "audit_failed_count"),
        "audit_pass_rate": _safe_div(audit_passed_count, audit_triple_count),
        "covered_slot_count": covered_slot_count,
        "missing_slot_count": missing_slot_count,
        "slot_coverage_rate": _safe_div(covered_slot_count, total_slot_count),
        "per_trace": {
            "audit_claim_coverage_rate": tuple(
                summary.get("audit_claim_coverage_rate") for summary in summaries
            ),
            "audit_pass_rate": tuple(summary.get("audit_pass_rate") for summary in summaries),
            "slot_coverage_rate": tuple(summary.get("slot_coverage_rate") for summary in summaries),
        },
    }


def _status(
    summary: Mapping[str, Any],
    *,
    config: ProductTraceTripleAuditEnrichmentConfig,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if int(summary.get("claims_with_triples") or 0) <= 0:
        reasons.append("no_claim_triples_extracted")
    for key, minimum in (
        ("audit_claim_coverage_rate", config.min_audit_claim_coverage),
        ("audit_pass_rate", config.min_audit_pass_rate),
        ("slot_coverage_rate", config.min_slot_coverage_rate),
    ):
        value = summary.get(key)
        if value is None or float(value) < minimum:
            reasons.append(f"{key}_below_{minimum:g}")
    return ("blocked" if reasons else "promote", tuple(reasons))


def _write_artifact_manifest(
    config: ProductTraceTripleAuditEnrichmentConfig,
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(config, records),
        root=Path(config.artifact_manifest_path).parent,
        metadata={
            "runner": "enrich_product_trace_triple_audit",
            "workflow": WORKFLOW,
            "status": report.get("status"),
            "trace_count": _mapping(report.get("summary")).get("trace_count"),
            "audit_claim_coverage_rate": _mapping(report.get("summary")).get("audit_claim_coverage_rate"),
            "audit_pass_rate": _mapping(report.get("summary")).get("audit_pass_rate"),
            "slot_coverage_rate": _mapping(report.get("summary")).get("slot_coverage_rate"),
            **dict(config.metadata),
        },
    )
    _write_json(config.artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(
    config: ProductTraceTripleAuditEnrichmentConfig,
    report: Mapping[str, Any],
) -> None:
    if config.registry_path is None:
        return
    assert config.name is not None and config.version is not None
    summary = _mapping(report.get("summary"))
    ArtifactRegistry.load_json(config.registry_path).record_report(
        name=config.name,
        version=config.version,
        path=config.report_path,
        metadata={
            "workflow": WORKFLOW,
            "status": report.get("status"),
            "trace_count": summary.get("trace_count"),
            "audit_claim_coverage_rate": summary.get("audit_claim_coverage_rate"),
            "audit_pass_rate": summary.get("audit_pass_rate"),
            "slot_coverage_rate": summary.get("slot_coverage_rate"),
        },
    ).save_json(config.registry_path)


def _artifact_paths(
    config: ProductTraceTripleAuditEnrichmentConfig,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "triple_audit_enrichment_report": config.report_path,
        "enriched_trace_dir": Path(config.output_dir) / "traces",
    }
    for index, record in enumerate(records, start=1):
        artifacts[f"enriched_trace_{index}"] = str(record.get("output_path"))
    for index, path in enumerate(config.evidence_corpus_paths, start=1):
        artifacts[f"evidence_corpus_{index}"] = path
    return artifacts


def _load_trace(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"ProductTrace JSON must be an object: {path}")
    return dict(payload)


def _write_json(path: str | Path, payload: Any, *, compact: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")


def _trace_output_path(output_dir: Path, source_path: Path, *, index: int) -> Path:
    stem = _SAFE_STEM_RE.sub("-", source_path.stem).strip("-") or "trace"
    return output_dir / "traces" / f"trace-{index:04d}-{stem}.json"


def _dedupe_evidence(documents: Sequence[EvidenceDocument]) -> tuple[EvidenceDocument, ...]:
    deduped: list[EvidenceDocument] = []
    seen: set[tuple[str | None, str]] = set()
    for document in documents:
        key = (document.source, document.text)
        if key in seen:
            continue
        deduped.append(document)
        seen.add(key)
    return tuple(deduped)


def _dedupe_hits(documents: Sequence[RetrievalHit]) -> tuple[RetrievalHit, ...]:
    deduped: list[RetrievalHit] = []
    seen: set[tuple[str | None, str]] = set()
    for document in documents:
        key = (document.source, document.text)
        if key in seen:
            continue
        deduped.append(document)
        seen.add(key)
    return tuple(deduped)


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sum_int(items: Sequence[Mapping[str, Any]], key: str) -> int:
    total = 0
    for item in items:
        try:
            total += int(item.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool.")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _rate(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number in [0, 1], not bool.")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


def _config_from_args(args: argparse.Namespace) -> ProductTraceTripleAuditEnrichmentConfig:
    trace_paths = _trace_paths_from_args(args.trace or (), args.trace_glob or ())
    return ProductTraceTripleAuditEnrichmentConfig(
        trace_paths=trace_paths,
        output_dir=args.output_dir,
        evidence_corpus_paths=tuple(args.evidence_corpus or ()),
        report_path=args.report,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        retrieval_limit=args.retrieval_limit,
        min_retrieval_overlap=args.min_retrieval_overlap,
        min_slot_coverage=args.min_slot_coverage,
        min_audit_claim_coverage=args.min_audit_claim_coverage,
        min_audit_pass_rate=args.min_audit_pass_rate,
        min_slot_coverage_rate=args.min_slot_coverage_rate,
        compact_json=bool(args.compact_json),
        metadata={"cli": True},
    )


def _trace_paths_from_args(values: Sequence[str], globs: Sequence[str]) -> tuple[Path, ...]:
    paths = [Path(value) for value in values]
    for pattern in globs:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern, recursive=True)))
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise ValueError("at least one --trace or --trace-glob match is required.")
    return tuple(unique)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrich ProductTrace JSON with triple-audit metadata")
    parser.add_argument("--trace", action="append", default=[], help="full ProductTrace JSON path; repeatable")
    parser.add_argument("--trace-glob", action="append", default=[], help="glob for full ProductTrace JSON files")
    parser.add_argument("--output-dir", required=True, help="directory for enriched traces and default report paths")
    parser.add_argument(
        "--evidence-corpus",
        action="append",
        default=None,
        help="local evidence JSON/JSONL; repeatable",
    )
    parser.add_argument("--report", default=None, help="report JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--min-retrieval-overlap", type=float, default=0.55)
    parser.add_argument("--min-slot-coverage", type=float, default=1.0)
    parser.add_argument("--min-audit-claim-coverage", type=float, default=1.0)
    parser.add_argument("--min-audit-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-slot-coverage-rate", type=float, default=1.0)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    report = build_product_trace_triple_audit_enrichment(_config_from_args(args))
    print(strict_json_dumps(to_jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
