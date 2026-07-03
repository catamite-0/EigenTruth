"""Enrich verifier sidecars with claim/evidence alignment reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import to_jsonable
from eigentruth.verify import (
    Claim,
    EvidenceAlignmentPolicy,
    EvidenceAlignmentVerifier,
    VerificationResult,
)


def load_verified_record_sidecar(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load verifier verified-record JSONL sidecar records."""
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"verified-record line {line_number} must be a JSON object.")
            records.append(dict(payload))
    if not records:
        raise ValueError("verified-record sidecar must contain at least one record.")
    return tuple(records)


def load_evidence_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load evidence records from JSON or JSONL.

    Evidence can be grouped records with run/record_index or individual hit
    records carrying those join keys in top-level fields or metadata.
    """
    source = Path(path)
    if source.suffix == ".jsonl":
        return _load_jsonl_records(source, record_name="evidence")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        records = _records_from_payload(payload)
        if records is None:
            return (dict(payload),)
        return records
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return _records_from_sequence(payload, record_name="evidence")
    raise ValueError("evidence JSON must be an object, an array of objects, or a records object.")


def enrich_evidence_alignment_sidecar(
    verified_records: Sequence[Mapping[str, Any]],
    evidence_records: Sequence[Mapping[str, Any]],
    *,
    run_name: str | None = None,
    min_keyword_overlap: float = 0.2,
    min_number_recall: float = 1.0,
    min_entity_recall: float = 0.5,
    require_cited_evidence: bool = False,
    allow_missing: bool = False,
    overwrite: bool = False,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Return verified records enriched with evidence-alignment verification results."""
    policy = EvidenceAlignmentPolicy(
        min_keyword_overlap=min_keyword_overlap,
        min_number_recall=min_number_recall,
        min_entity_recall=min_entity_recall,
        require_cited_evidence=require_cited_evidence,
    )
    selected_run = None if run_name is None else str(run_name)
    default_run = _default_run(verified_records, run_name=selected_run)
    evidence_index = _index_evidence_records(evidence_records, default_run=default_run)
    enriched: list[dict[str, Any]] = []
    enriched_summaries: list[Mapping[str, Any]] = []
    missing_records: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, int]] = set()
    matched_claim_ids: set[str] = set()

    for record in verified_records:
        output_record = dict(record)
        if selected_run is not None and str(record.get("run", "")) != selected_run:
            enriched.append(output_record)
            continue

        key = _record_key(record, default_run=default_run, record_name="verified-record")
        claim = _claim_from_verified_record(record, key=key)
        matching_evidence_records = evidence_index.by_key.get(key, ())
        if not matching_evidence_records and claim.claim_id is not None:
            matching_evidence_records = evidence_index.by_claim_id.get(claim.claim_id, ())

        if not matching_evidence_records:
            missing_records.append({"run": key[0], "record_index": key[1], "claim_id": claim.claim_id})
            if not allow_missing:
                raise ValueError(
                    f"missing evidence record for run={key[0]!r} record_index={key[1]} "
                    f"claim_id={claim.claim_id!r}."
                )
            enriched.append(output_record)
            continue

        _mark_evidence_records_matched(
            matching_evidence_records,
            matched_keys=matched_keys,
            matched_claim_ids=matched_claim_ids,
            default_run=default_run,
        )
        if not overwrite and _has_existing_evidence_alignment(output_record):
            raise ValueError(
                f"verified record run={key[0]!r} record_index={key[1]} already has evidence_alignment."
            )

        evidence_items = tuple(
            item
            for evidence_record in matching_evidence_records
            for item in _evidence_items_from_record(evidence_record)
        )
        result = EvidenceAlignmentVerifier(evidence=evidence_items, policy=policy).verify(
            claim,
            context={
                "run": key[0],
                "record_index": key[1],
                "evidence_record_count": len(matching_evidence_records),
            },
        )
        _attach_evidence_alignment(output_record, result)
        summary = _evidence_alignment_summary(result)
        if summary:
            enriched_summaries.append(summary)
        enriched.append(output_record)

    unused_keys = sorted(key for key in evidence_index.by_key if key not in matched_keys)
    unused_claim_ids = sorted(claim_id for claim_id in evidence_index.by_claim_id if claim_id not in matched_claim_ids)
    report = {
        "schema_version": 1,
        "workflow": "evidence_alignment_sidecar_enrichment",
        "run_name": selected_run,
        "policy": policy.to_dict(),
        "input_record_count": len(verified_records),
        "evidence_record_count": len(evidence_records),
        "enriched_record_count": len(enriched_summaries),
        "missing_record_count": len(missing_records),
        "unused_evidence_key_count": len(unused_keys),
        "unused_evidence_claim_id_count": len(unused_claim_ids),
        "missing_records": missing_records,
        "unused_evidence_keys": [
            {"run": run, "record_index": record_index}
            for run, record_index in unused_keys
        ],
        "unused_evidence_claim_ids": unused_claim_ids,
        "summary": _aggregate_evidence_alignment_summaries(enriched_summaries),
    }
    return tuple(enriched), report


def write_verified_records_jsonl(records: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    """Write verified-record sidecar JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def build_report(
    *,
    verified_records_jsonl: str | Path,
    evidence: str | Path,
    output: str | Path,
    run_name: str | None = None,
    min_keyword_overlap: float = 0.2,
    min_number_recall: float = 1.0,
    min_entity_recall: float = 0.5,
    require_cited_evidence: bool = False,
    allow_missing: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Enrich a sidecar and return a compact enrichment report."""
    verified_records = load_verified_record_sidecar(verified_records_jsonl)
    evidence_records = load_evidence_records(evidence)
    enriched, report = enrich_evidence_alignment_sidecar(
        verified_records,
        evidence_records,
        run_name=run_name,
        min_keyword_overlap=min_keyword_overlap,
        min_number_recall=min_number_recall,
        min_entity_recall=min_entity_recall,
        require_cited_evidence=require_cited_evidence,
        allow_missing=allow_missing,
        overwrite=overwrite,
    )
    write_verified_records_jsonl(enriched, output)
    return {
        **report,
        "verified_records_jsonl": str(verified_records_jsonl),
        "evidence": str(evidence),
        "output": str(output),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entrypoint helper."""
    report = build_report(
        verified_records_jsonl=args.verified_records_jsonl,
        evidence=args.evidence,
        output=args.output,
        run_name=args.run_name,
        min_keyword_overlap=args.min_keyword_overlap,
        min_number_recall=args.min_number_recall,
        min_entity_recall=args.min_entity_recall,
        require_cited_evidence=bool(args.require_cited_evidence),
        allow_missing=bool(args.allow_missing),
        overwrite=bool(args.overwrite),
    )
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote evidence-alignment enriched sidecar to {args.output}")
    return report


def _verification_result_to_dict(result: VerificationResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "confidence": result.confidence,
        "evidence": tuple(result.evidence),
        "explanation": result.explanation,
        "metadata": to_jsonable(dict(result.metadata)),
    }


def _attach_evidence_alignment(record: dict[str, Any], result: VerificationResult) -> None:
    nested = dict(_mapping(record.get("record")))
    nested["evidence_alignment"] = _verification_result_to_dict(result)
    record["record"] = nested


def _has_existing_evidence_alignment(record: Mapping[str, Any]) -> bool:
    nested = _mapping(record.get("record"))
    final = _mapping(nested.get("final"))
    final_metadata = _mapping(final.get("metadata"))
    return any(
        isinstance(candidate, Mapping)
        for candidate in (
            nested.get("evidence_alignment"),
            nested.get("citation_evidence_alignment"),
            nested.get("claim_evidence_alignment"),
            final.get("evidence_alignment"),
            final_metadata.get("evidence_alignment"),
        )
    )


def _evidence_alignment_summary(result: VerificationResult) -> Mapping[str, Any]:
    metadata = _mapping(result.metadata)
    report = _mapping(metadata.get("evidence_alignment"))
    return _mapping(report.get("summary"))


def _claim_from_verified_record(record: Mapping[str, Any], *, key: tuple[str, int]) -> Claim:
    nested = _mapping(record.get("record"))
    claim_payload = _first_claim_payload(
        nested.get("claim"),
        record.get("claim"),
        nested.get("claims"),
        record.get("claims"),
    )
    claim_text = _text_from_claim_payload(claim_payload)
    if claim_text is None:
        claim_text = _first_text(
            nested.get("claim_text"),
            record.get("claim_text"),
            nested.get("statement"),
            record.get("statement"),
            nested.get("text"),
        )
    if claim_text is None:
        raise ValueError(f"verified record run={key[0]!r} record_index={key[1]} has no claim text.")

    claim_metadata = dict(_mapping(_mapping(claim_payload).get("metadata")))
    for source in (
        _mapping(nested.get("claim_metadata")),
        _mapping(record.get("claim_metadata")),
    ):
        claim_metadata.update(source)
    claim_metadata.update({"run": key[0], "record_index": key[1]})
    claim_id = _first_text(
        _mapping(claim_payload).get("claim_id"),
        _mapping(claim_payload).get("id"),
        nested.get("claim_id"),
        record.get("claim_id"),
    )
    if claim_id is None:
        claim_id = f"{key[0]}:{key[1]}"
    return Claim(text=claim_text, claim_id=claim_id, metadata=claim_metadata)


def _first_claim_payload(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
            for item in value:
                if item is not None:
                    return item
            continue
        return value
    return None


def _text_from_claim_payload(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping):
        return _first_text(
            value.get("text"),
            value.get("claim_text"),
            value.get("statement"),
            value.get("content"),
        )
    return None


def _evidence_items_from_record(record: Mapping[str, Any]) -> tuple[Any, ...]:
    items: list[Any] = []
    for container in _evidence_containers(record):
        if isinstance(container, Mapping):
            items.append(_evidence_mapping_with_metadata(container, record))
        elif isinstance(container, Sequence) and not isinstance(container, (str, bytes, bytearray)):
            for item in container:
                if isinstance(item, Mapping):
                    items.append(_evidence_mapping_with_metadata(item, record))
                else:
                    items.append(item)
        else:
            items.append(container)
    if not items and _has_evidence_text(record):
        items.append(_evidence_mapping_with_metadata(record, record))
    return tuple(items)


def _evidence_containers(record: Mapping[str, Any]) -> tuple[Any, ...]:
    containers = []
    nested = _mapping(record.get("record"))
    output = _mapping(record.get("output"))
    for source in (record, nested, output):
        for key in (
            "evidence",
            "evidence_items",
            "evidence_texts",
            "retrieval_hits",
            "hits",
            "citation_evidence",
            "citation_hits",
            "search_hits",
            "source_documents",
            "documents",
        ):
            if key in source:
                containers.append(source[key])
    for query_result in _sequence(output.get("hits_by_query", ())):
        hits = _mapping(query_result).get("hits")
        if hits is not None:
            containers.append(hits)
    return tuple(containers)


def _evidence_mapping_with_metadata(item: Mapping[str, Any], source_record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    metadata = dict(_mapping(payload.get("metadata")))
    source_metadata = _mapping(source_record.get("metadata"))
    for key in ("run", "record_index", "claim_id", "citation_id", "ref", "request_id"):
        value = payload.get(key, source_record.get(key, source_metadata.get(key)))
        if value is not None:
            metadata.setdefault(key, value)
    if metadata:
        payload["metadata"] = metadata
    return payload


def _has_evidence_text(record: Mapping[str, Any]) -> bool:
    return _first_text(
        record.get("text"),
        record.get("content"),
        record.get("snippet"),
        record.get("title"),
        record.get("source_text"),
    ) is not None


class _EvidenceIndex:
    def __init__(
        self,
        *,
        by_key: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
        by_claim_id: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> None:
        self.by_key = {key: tuple(value) for key, value in by_key.items()}
        self.by_claim_id = {key: tuple(value) for key, value in by_claim_id.items()}


def _index_evidence_records(
    records: Sequence[Mapping[str, Any]],
    *,
    default_run: str | None,
) -> _EvidenceIndex:
    by_key: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    by_claim_id: dict[str, list[Mapping[str, Any]]] = {}
    unindexed = 0
    for record in records:
        indexed = False
        try:
            key = _record_key(record, default_run=default_run, record_name="evidence")
        except ValueError:
            key = None
        if key is not None:
            by_key.setdefault(key, []).append(record)
            indexed = True

        claim_id = _claim_id_from_evidence_record(record)
        if claim_id is not None:
            by_claim_id.setdefault(claim_id, []).append(record)
            indexed = True

        if not indexed:
            unindexed += 1
    if unindexed:
        raise ValueError(
            f"{unindexed} evidence record(s) lack run/record_index metadata and claim_id; cannot join to sidecar."
        )
    if not by_key and not by_claim_id:
        raise ValueError("evidence records must include run/record_index metadata or claim_id.")
    return _EvidenceIndex(by_key=by_key, by_claim_id=by_claim_id)


def _mark_evidence_records_matched(
    records: Sequence[Mapping[str, Any]],
    *,
    matched_keys: set[tuple[str, int]],
    matched_claim_ids: set[str],
    default_run: str | None,
) -> None:
    for record in records:
        try:
            matched_keys.add(_record_key(record, default_run=default_run, record_name="evidence"))
        except ValueError:
            pass
        claim_id = _claim_id_from_evidence_record(record)
        if claim_id is not None:
            matched_claim_ids.add(claim_id)


def _claim_id_from_evidence_record(record: Mapping[str, Any]) -> str | None:
    metadata = _mapping(record.get("metadata"))
    claim_payload = _mapping(record.get("claim"))
    return _first_text(
        record.get("claim_id"),
        metadata.get("claim_id"),
        claim_payload.get("claim_id"),
        claim_payload.get("id"),
    )


def _default_run(records: Sequence[Mapping[str, Any]], *, run_name: str | None) -> str | None:
    if run_name is not None:
        return run_name
    runs = {str(record.get("run", "")) for record in records if str(record.get("run", ""))}
    if len(runs) == 1:
        return next(iter(runs))
    return None


def _record_key(record: Mapping[str, Any], *, default_run: str | None, record_name: str) -> tuple[str, int]:
    metadata = _mapping(record.get("metadata"))
    run = str(record.get("run", metadata.get("run", ""))).strip()
    if not run:
        if default_run is None:
            raise ValueError(f"{record_name} record run is required when multiple or unknown runs are present.")
        run = default_run
    record_index = _record_index(record, record_name=record_name)
    return run, record_index


def _record_index(record: Mapping[str, Any], *, record_name: str) -> int:
    metadata = _mapping(record.get("metadata"))
    value = record.get("record_index", metadata.get("record_index"))
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{record_name} record_index must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record_name} record_index must be an integer.") from exc
    if numeric < 0:
        raise ValueError(f"{record_name} record_index must be non-negative.")
    return numeric


def _aggregate_evidence_alignment_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    record_count = sum(int(summary.get("record_count", 0) or 0) for summary in summaries)
    aligned_count = sum(int(summary.get("aligned_count", 0) or 0) for summary in summaries)
    misaligned_count = sum(int(summary.get("misaligned_count", 0) or 0) for summary in summaries)
    insufficient_count = sum(int(summary.get("insufficient_evidence_count", 0) or 0) for summary in summaries)
    issue_count = sum(int(summary.get("issue_count", 0) or 0) for summary in summaries)
    reference_count = sum(int(summary.get("citation_reference_count", 0) or 0) for summary in summaries)
    matched_reference_count = sum(
        int(summary.get("matched_citation_reference_count", 0) or 0)
        for summary in summaries
    )
    cited_evidence_count = sum(int(summary.get("cited_evidence_count", 0) or 0) for summary in summaries)
    return {
        "record_count": record_count,
        "aligned_count": aligned_count,
        "misaligned_count": misaligned_count,
        "insufficient_evidence_count": insufficient_count,
        "alignment_rate": _safe_div(aligned_count, record_count),
        "misalignment_rate": _safe_div(misaligned_count, record_count) or 0.0,
        "insufficient_evidence_rate": _safe_div(insufficient_count, record_count) or 0.0,
        "keyword_overlap_mean": _mean_optional(_optional_rates(summaries, "keyword_overlap_mean")),
        "number_recall_mean": _mean_optional(_optional_rates(summaries, "number_recall_mean")),
        "entity_recall_mean": _mean_optional(_optional_rates(summaries, "entity_recall_mean")),
        "citation_reference_count": reference_count,
        "matched_citation_reference_count": matched_reference_count,
        "cited_evidence_count": cited_evidence_count,
        "citation_reference_coverage_rate": _safe_div(matched_reference_count, reference_count),
        "issue_count": issue_count,
    }


def _optional_rates(summaries: Sequence[Mapping[str, Any]], key: str) -> tuple[float, ...]:
    values = []
    for summary in summaries:
        value = summary.get(key)
        if value is None:
            continue
        values.append(_unit_interval(value, name=key))
    return tuple(values)


def _load_jsonl_records(path: Path, *, record_name: str) -> tuple[dict[str, Any], ...]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{record_name} line {line_number} must be a JSON object.")
            records.append(dict(payload))
    if not records:
        raise ValueError(f"{record_name} JSONL must contain at least one record.")
    return tuple(records)


def _records_from_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...] | None:
    if _payload_has_join_key(payload):
        return None
    records = _first_sequence(
        payload.get("records"),
        payload.get("evidence_records"),
        payload.get("items"),
        payload.get("source_documents"),
        payload.get("documents"),
        payload.get("hits"),
        payload.get("evidence"),
    )
    if records is None:
        return None
    return _records_from_sequence(records, record_name="evidence")


def _payload_has_join_key(payload: Mapping[str, Any]) -> bool:
    metadata = _mapping(payload.get("metadata"))
    return any(
        key in payload or key in metadata
        for key in ("run", "record_index", "claim_id")
    )


def _records_from_sequence(values: Sequence[Any], *, record_name: str) -> tuple[dict[str, Any], ...]:
    records = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"{record_name} record {index} must be a JSON object.")
        records.append(dict(value))
    if not records:
        raise ValueError(f"{record_name} records must contain at least one record.")
    return tuple(records)


def _first_sequence(*values: Any) -> Sequence[Any] | None:
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return value
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _unit_interval(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return numeric


def _mean_optional(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(numerator) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich eval_verifier_ensemble JSONL sidecars with evidence-alignment reports"
    )
    parser.add_argument("--verified-records-jsonl", required=True, help="input eval_verifier_ensemble sidecar JSONL")
    parser.add_argument(
        "--evidence",
        required=True,
        help="JSON/JSONL evidence records with run/record_index metadata or claim_id",
    )
    parser.add_argument("--output", required=True, help="output enriched verified-record sidecar JSONL")
    parser.add_argument("--run-name", default=None, help="optional run to enrich when sidecar has multiple runs")
    parser.add_argument("--min-keyword-overlap", type=float, default=0.2)
    parser.add_argument("--min-number-recall", type=float, default=1.0)
    parser.add_argument("--min-entity-recall", type=float, default=0.5)
    parser.add_argument(
        "--require-cited-evidence",
        action="store_true",
        help="require local evidence whose citation/source id matches the claim citation reference",
    )
    parser.add_argument("--allow-missing", action="store_true", help="copy records without matching evidence records")
    parser.add_argument("--overwrite", action="store_true", help="replace existing evidence_alignment payloads")
    parser.add_argument("--json", default=None, help="optional summary report path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
