"""Audit whether verifier routes correct detectability blind spots.

This helper joins a row-level DECK blind-spot report with an
``eval_verifier_ensemble.py --verified-records-jsonl`` sidecar. It does not run
models or verifiers. The default audit asks whether the promoted
``retrieval_structured_qa`` route refutes high-consistency/high-confidence false
answers that output-level uncertainty is expected to miss.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

DEFAULT_TARGET_ROUTE = "retrieval_structured_qa"


def load_blind_spot_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load blind-spot records from ``analyze_detectability_blind_spots.py`` output."""
    payload = _load_json_object(path)
    records = payload.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise ValueError("blind spot report must contain a records array.")
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"blind spot records[{index}] must be a JSON object.")
        normalized.append(dict(record))
    if not normalized:
        raise ValueError("blind spot report did not contain any records.")
    return tuple(normalized)


def load_verified_records_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load per-record verifier sidecar rows from ``eval_verifier_ensemble.py``."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"verified records line {line_no} is not a JSON object.")
            if payload.get("record_index") is None:
                raise ValueError(f"verified records line {line_no} is missing record_index.")
            records.append(dict(payload))
    if not records:
        raise ValueError("verified records JSONL did not contain any records.")
    return tuple(records)


def audit_blind_spot_correction_routes(
    blind_spots: Sequence[Mapping[str, Any]],
    verified_records: Sequence[Mapping[str, Any]],
    *,
    target_route: str = DEFAULT_TARGET_ROUTE,
    max_examples_per_bucket: int = 5,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a route-correction audit for blind-spot rows."""
    if not blind_spots:
        raise ValueError("blind_spots must not be empty.")
    if not verified_records:
        raise ValueError("verified_records must not be empty.")
    if max_examples_per_bucket < 0:
        raise ValueError("max_examples_per_bucket must be non-negative.")
    target_route = str(target_route).strip()
    if not target_route:
        raise ValueError("target_route must be non-empty.")

    verified_by_index = _verified_records_by_index(verified_records)
    status_counts: Counter[str] = Counter()
    selected_route_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    unresolved_question_type_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    unresolved_feature_counts: Counter[str] = Counter()
    hit_source_counts: Counter[str] = Counter()
    hit_count_total = 0
    records_with_hits = 0
    matched_count = 0
    target_selected_count = 0
    target_refuted_count = 0
    target_supported_count = 0
    target_insufficient_count = 0
    any_refuted_count = 0
    any_supported_count = 0
    unresolved_count = 0
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records = []

    for blind in blind_spots:
        record_index = _required_int(blind.get("record_index"), field_name="blind spot record_index")
        verified = verified_by_index.get(record_index)
        question_type = str(blind.get("question_type", "unknown"))
        question_type_counts[question_type] += 1
        _update_feature_counts(feature_counts, _mapping(blind.get("features")))

        if verified is None:
            status = "missing_verified_record"
            selected_route = "missing_verified_record"
            hit_count = 0
            hit_sources: tuple[str, ...] = ()
            final_explanation = None
            decision_rule = None
            matched = False
        else:
            matched = True
            matched_count += 1
            record_payload = _record_payload(verified)
            final = _mapping(record_payload.get("final"))
            route = _mapping(record_payload.get("route"))
            status = str(final.get("status", "unknown"))
            selected_route = str(route.get("selected_route", "unknown"))
            decision_rule = _optional_str(_mapping(final.get("metadata")).get("decision_rule"))
            final_explanation = _optional_str(final.get("explanation"))
            retrieval_hits = _retrieval_hits(record_payload)
            hit_count = len(retrieval_hits)
            hit_count_total += hit_count
            if hit_count:
                records_with_hits += 1
            hit_sources = tuple(
                str(hit["source"])
                for hit in retrieval_hits
                if hit.get("source") is not None
            )
            for source in hit_sources:
                hit_source_counts[source] += 1

        selected_route_counts[selected_route] += 1
        status_counts[status] += 1
        if selected_route == target_route:
            target_selected_count += 1
            if status == "refuted":
                target_refuted_count += 1
            elif status == "supported":
                target_supported_count += 1
            elif status == "insufficient_evidence":
                target_insufficient_count += 1
        if status == "refuted":
            any_refuted_count += 1
        if status == "supported":
            any_supported_count += 1

        outcome = _outcome_for_record(
            matched=matched,
            status=status,
            selected_route=selected_route,
            target_route=target_route,
        )
        outcome_counts[outcome] += 1
        is_unresolved = outcome != "corrected_refuted"
        if is_unresolved:
            unresolved_count += 1
            unresolved_question_type_counts[question_type] += 1
            _update_feature_counts(unresolved_feature_counts, _mapping(blind.get("features")))

        record_report = {
            "record_index": record_index,
            "label": _optional_int(blind.get("label")),
            "question_type": question_type,
            "question": blind.get("question"),
            "answer": blind.get("answer"),
            "text": blind.get("text"),
            "cell": blind.get("cell"),
            "cell_margin": blind.get("cell_margin"),
            "selected_route": selected_route,
            "target_route_selected": selected_route == target_route,
            "final_status": status,
            "outcome": outcome,
            "hit_count": hit_count,
            "hit_sources": hit_sources,
            "decision_rule": decision_rule,
            "explanation": final_explanation,
        }
        records.append(record_report)
        if len(examples[outcome]) < max_examples_per_bucket:
            examples[outcome].append(record_report)

    n_blind = len(blind_spots)
    summary = {
        "blind_spot_count": n_blind,
        "matched_verified_record_count": matched_count,
        "matched_verified_record_rate": _rate(matched_count, n_blind),
        "target_route": target_route,
        "target_route_selected_count": target_selected_count,
        "target_route_selected_rate": _rate(target_selected_count, n_blind),
        "target_route_refuted_count": target_refuted_count,
        "target_route_refuted_rate": _rate(target_refuted_count, n_blind),
        "target_route_refuted_when_selected_rate": _rate(target_refuted_count, target_selected_count),
        "target_route_supported_count": target_supported_count,
        "target_route_supported_when_selected_rate": _rate(target_supported_count, target_selected_count),
        "target_route_insufficient_count": target_insufficient_count,
        "any_route_refuted_count": any_refuted_count,
        "any_route_refuted_rate": _rate(any_refuted_count, n_blind),
        "any_route_supported_count": any_supported_count,
        "any_route_supported_rate": _rate(any_supported_count, n_blind),
        "records_with_retrieval_hits": records_with_hits,
        "records_with_retrieval_hits_rate": _rate(records_with_hits, n_blind),
        "total_retrieval_hits": hit_count_total,
        "average_retrieval_hits_per_record": hit_count_total / n_blind,
        "unresolved_count": unresolved_count,
        "unresolved_rate": _rate(unresolved_count, n_blind),
        "selected_route_counts": dict(sorted(selected_route_counts.items())),
        "final_status_counts": dict(sorted(status_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "unresolved_question_type_counts": dict(sorted(unresolved_question_type_counts.items())),
        "feature_counts": dict(sorted(feature_counts.items())),
        "unresolved_feature_counts": dict(sorted(unresolved_feature_counts.items())),
        "top_hit_sources": _counter_top(hit_source_counts, limit=20),
    }
    status = "complete" if matched_count == n_blind else "partial"
    return {
        "schema_version": 1,
        "workflow": "blind_spot_correction_route_audit",
        "status": status,
        "config": {
            "target_route": target_route,
            "max_examples_per_bucket": int(max_examples_per_bucket),
        },
        "summary": summary,
        "examples": {key: value for key, value in sorted(examples.items())},
        "records": records,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    blind_spots_path: str | Path,
    verified_records_jsonl_path: str | Path,
    output_path: str | Path,
    target_route: str = DEFAULT_TARGET_ROUTE,
    max_examples_per_bucket: int = 5,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    verifier_report_path: str | Path | None = None,
    claims_path: str | Path | None = None,
    route_report_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, save, optionally manifest, and optionally register the audit."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    report = audit_blind_spot_correction_routes(
        load_blind_spot_records(blind_spots_path),
        load_verified_records_jsonl(verified_records_jsonl_path),
        target_route=target_route,
        max_examples_per_bucket=max_examples_per_bucket,
        metadata=metadata,
    )
    report["source"] = {
        "blind_spots_path": str(blind_spots_path),
        "verified_records_jsonl_path": str(verified_records_jsonl_path),
        "verifier_report_path": None if verifier_report_path is None else str(verifier_report_path),
        "claims_path": None if claims_path is None else str(claims_path),
        "route_report_path": None if route_report_path is None else str(route_report_path),
    }
    if artifact_manifest_path is not None:
        report["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    output = Path(output_path)
    _write_json(output, report, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = _build_manifest(
            report,
            output_path=output,
            blind_spots_path=Path(blind_spots_path),
            verified_records_jsonl_path=Path(verified_records_jsonl_path),
            verifier_report_path=None if verifier_report_path is None else Path(verifier_report_path),
            claims_path=None if claims_path is None else Path(claims_path),
            route_report_path=None if route_report_path is None else Path(route_report_path),
            artifact_manifest_path=manifest_path,
        )
        _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output,
            metadata={
                "workflow": "blind_spot_correction_route_audit",
                "status": report.get("status"),
                "target_route": target_route,
                "blind_spot_count": _nested(report, "summary", "blind_spot_count"),
                "matched_verified_record_count": _nested(report, "summary", "matched_verified_record_count"),
                "target_route_selected_count": _nested(report, "summary", "target_route_selected_count"),
                "target_route_refuted_count": _nested(report, "summary", "target_route_refuted_count"),
                "target_route_refuted_rate": _nested(report, "summary", "target_route_refuted_rate"),
                "unresolved_count": _nested(report, "summary", "unresolved_count"),
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return report


def _verified_records_by_index(verified_records: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in verified_records:
        record_index = _required_int(row.get("record_index"), field_name="verified record_index")
        if record_index in by_index:
            raise ValueError(f"duplicate verified record_index: {record_index}")
        by_index[record_index] = row
    return by_index


def _outcome_for_record(
    *,
    matched: bool,
    status: str,
    selected_route: str,
    target_route: str,
) -> str:
    if not matched:
        return "missing_verified_record"
    if status == "refuted":
        return "corrected_refuted" if selected_route == target_route else "corrected_by_other_route"
    if selected_route != target_route:
        return "not_selected_by_target_route"
    if status == "supported":
        return "false_supported"
    if status == "insufficient_evidence":
        return "insufficient_evidence"
    if status == "error":
        return "route_error"
    return "other_unresolved"


def _record_payload(item: Mapping[str, Any]) -> Mapping[str, Any]:
    record = item.get("record", item)
    return record if isinstance(record, Mapping) else {}


def _retrieval_hits(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    hits = record.get("retrieval_hits", ())
    if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes, bytearray)):
        return ()
    return tuple(hit for hit in hits if isinstance(hit, Mapping))


def _update_feature_counts(counter: Counter[str], features: Mapping[str, Any]) -> None:
    for name, enabled in features.items():
        if bool(enabled):
            counter[str(name)] += 1


def _counter_top(counter: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _required_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
        if value is default:
            return default
    return value


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _build_manifest(
    report: Mapping[str, Any],
    *,
    output_path: Path,
    blind_spots_path: Path,
    verified_records_jsonl_path: Path,
    verifier_report_path: Path | None,
    claims_path: Path | None,
    route_report_path: Path | None,
    artifact_manifest_path: Path,
) -> dict[str, Any]:
    artifacts: dict[str, Path] = {
        "blind_spot_correction_route_audit": output_path,
        "blind_spots": blind_spots_path,
        "verified_records_jsonl": verified_records_jsonl_path,
    }
    if verifier_report_path is not None:
        artifacts["verifier_report"] = verifier_report_path
    if claims_path is not None:
        artifacts["claims"] = claims_path
    if route_report_path is not None:
        artifacts["route_report"] = route_report_path
    return build_artifact_manifest(
        artifacts,
        root=artifact_manifest_path.parent,
        metadata={
            "runner": "audit_blind_spot_correction_routes",
            "status": report.get("status"),
            "target_route": _nested(report, "summary", "target_route"),
            "blind_spot_count": _nested(report, "summary", "blind_spot_count"),
            "matched_verified_record_count": _nested(report, "summary", "matched_verified_record_count"),
            "target_route_selected_count": _nested(report, "summary", "target_route_selected_count"),
            "target_route_refuted_count": _nested(report, "summary", "target_route_refuted_count"),
            "unresolved_count": _nested(report, "summary", "unresolved_count"),
        },
    )


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
    parser.add_argument("--blind-spots", required=True, help="detectability blind-spot report JSON")
    parser.add_argument("--verified-records-jsonl", required=True, help="eval_verifier_ensemble sidecar JSONL")
    parser.add_argument("--target-route", default=DEFAULT_TARGET_ROUTE)
    parser.add_argument("--max-examples-per-bucket", type=int, default=5)
    parser.add_argument("--json", required=True, help="output audit report")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--verifier-report", default=None, help="optional verifier ensemble report for provenance")
    parser.add_argument("--claims", default=None, help="optional claims fixture for provenance")
    parser.add_argument(
        "--route-report",
        default=None,
        help="optional route comparison/promotion report for provenance",
    )
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        blind_spots_path=args.blind_spots,
        verified_records_jsonl_path=args.verified_records_jsonl,
        output_path=args.json,
        target_route=args.target_route,
        max_examples_per_bucket=args.max_examples_per_bucket,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        verifier_report_path=args.verifier_report,
        claims_path=args.claims,
        route_report_path=args.route_report,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "blind_spot_correction_route_audit_ok "
        f"target_route={summary['target_route']} "
        f"blind_spots={summary['blind_spot_count']} "
        f"selected={summary['target_route_selected_count']} "
        f"refuted={summary['target_route_refuted_count']} "
        f"unresolved={summary['unresolved_count']}"
    )


if __name__ == "__main__":
    main()
