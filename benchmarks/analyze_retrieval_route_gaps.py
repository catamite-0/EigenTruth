"""Analyze retrieval route coverage and failure modes from verified records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import build_artifact_manifest  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def analyze_retrieval_route_gaps(
    verified_records: Sequence[Mapping[str, Any]],
    *,
    max_examples_per_bucket: int = 5,
) -> dict[str, Any]:
    """Return a route-gap report for eval_verifier_ensemble verified records."""
    if max_examples_per_bucket < 0:
        raise ValueError("max_examples_per_bucket must be non-negative.")
    if not verified_records:
        raise ValueError("verified_records must not be empty.")

    buckets: dict[str, dict[str, Any]] = {}
    route_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    decision_rule_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    hit_count_total = 0
    records_with_hits = 0
    records_using_retrieval = 0
    false_positive_count = 0
    false_negative_count = 0
    topical_tokens: Counter[str] = Counter()
    hit_source_counts: Counter[str] = Counter()
    hit_property_counts: Counter[str] = Counter()

    for idx, item in enumerate(verified_records):
        record = _record_payload(item)
        label = int(item.get("label", _record_label(record)))
        label_key = "false" if label == 1 else "true"
        final = _mapping(record.get("final"))
        status = str(final.get("status", "unknown"))
        metadata = _mapping(final.get("metadata"))
        decision_rule = str(metadata.get("decision_rule", "unknown"))
        route = _mapping(record.get("route"))
        selected_route = str(route.get("selected_route", "unknown"))
        retrieval_hits = tuple(_mapping(hit) for hit in record.get("retrieval_hits", ()) if isinstance(hit, Mapping))
        hit_count = len(retrieval_hits)
        hit_count_total += hit_count
        if hit_count:
            records_with_hits += 1
        if route.get("used_retrieval") or selected_route.startswith("retrieval_") or hit_count:
            records_using_retrieval += 1
        for hit in retrieval_hits:
            if hit.get("source") is not None:
                hit_source_counts[str(hit["source"])] += 1
            hit_metadata = _mapping(hit.get("metadata"))
            if hit_metadata.get("statement_property") is not None:
                hit_property_counts[str(hit_metadata["statement_property"])] += 1

        route_counts[selected_route] += 1
        status_counts[status] += 1
        decision_rule_counts[decision_rule] += 1
        label_counts[label_key] += 1

        verified_triggered = _verified_triggered(final)
        false_positive = label == 0 and verified_triggered
        false_negative = label == 1 and not verified_triggered
        if false_positive:
            false_positive_count += 1
        if false_negative:
            false_negative_count += 1

        gap_reason = _gap_reason(
            label=label,
            status=status,
            decision_rule=decision_rule,
            selected_route=selected_route,
            hit_count=hit_count,
            false_positive=false_positive,
            false_negative=false_negative,
        )
        _add_bucket_example(
            buckets,
            gap_reason,
            item=item,
            record=record,
            label_key=label_key,
            selected_route=selected_route,
            status=status,
            decision_rule=decision_rule,
            hit_count=hit_count,
            max_examples=max_examples_per_bucket,
        )
        if gap_reason in {"no_retrieval_hits", "low_overlap_after_retrieval", "false_positive"}:
            topical_tokens.update(_topical_tokens(_claim_text(record)))

    n_records = len(verified_records)
    return _finalize_nested_counters({
        "schema_version": 1,
        "workflow": "retrieval_route_gap_analysis",
        "summary": {
            "n_records": n_records,
            "label_counts": dict(sorted(label_counts.items())),
            "selected_route_counts": dict(sorted(route_counts.items())),
            "final_status_counts": dict(sorted(status_counts.items())),
            "decision_rule_counts": dict(sorted(decision_rule_counts.items())),
            "records_with_retrieval_hits": records_with_hits,
            "records_with_retrieval_hit_rate": _rate(records_with_hits, n_records),
            "records_using_retrieval": records_using_retrieval,
            "records_using_retrieval_rate": _rate(records_using_retrieval, n_records),
            "total_retrieval_hits": hit_count_total,
            "average_hits_per_record": hit_count_total / n_records,
            "false_positive_count": false_positive_count,
            "false_positive_rate": _rate(false_positive_count, label_counts.get("true", 0)),
            "false_negative_count": false_negative_count,
            "false_negative_rate": _rate(false_negative_count, label_counts.get("false", 0)),
        },
        "gap_buckets": {
            key: {
                **{field: value for field, value in bucket.items() if field != "examples"},
                "examples": bucket.get("examples", ()),
            }
            for key, bucket in sorted(buckets.items())
        },
        "top_gap_tokens": _counter_top(topical_tokens, limit=30),
        "top_hit_sources": _counter_top(hit_source_counts, limit=20),
        "hit_property_counts": dict(sorted(hit_property_counts.items())),
    })


def load_verified_records_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load eval_verifier_ensemble verified-record sidecar rows."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"verified-records line {line_no} is not a JSON object.")
            records.append(dict(payload))
    if not records:
        raise ValueError("verified-records JSONL did not contain any records.")
    return tuple(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    records_path = Path(args.verified_records_jsonl)
    report = analyze_retrieval_route_gaps(
        load_verified_records_jsonl(records_path),
        max_examples_per_bucket=args.max_examples_per_bucket,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.artifact_manifest:
        manifest_path = Path(args.artifact_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = build_artifact_manifest(
            {
                "verified_records_jsonl": records_path,
                "gap_report": output_path,
            },
            root=manifest_path.parent,
            metadata={
                "workflow": "retrieval_route_gap_analysis",
                "summary": report["summary"],
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "retrieval_route_gap_analysis_ok "
        f"records={report['summary']['n_records']} output={output_path}"
    )
    return report


def _record_payload(item: Mapping[str, Any]) -> Mapping[str, Any]:
    record = item.get("record", item)
    return record if isinstance(record, Mapping) else {}


def _record_label(record: Mapping[str, Any]) -> int:
    metadata = _mapping(record.get("metadata"))
    statement = _mapping(metadata.get("statement"))
    if statement.get("is_false") is not None:
        return int(statement["is_false"])
    return 0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _verified_triggered(final: Mapping[str, Any]) -> bool:
    return str(final.get("status")) == "refuted"


def _gap_reason(
    *,
    label: int,
    status: str,
    decision_rule: str,
    selected_route: str,
    hit_count: int,
    false_positive: bool,
    false_negative: bool,
) -> str:
    if hit_count == 0:
        return "no_retrieval_hits"
    if false_positive:
        return "false_positive"
    if false_negative:
        return "false_negative"
    if selected_route.startswith("retrieval_") and status == "insufficient_evidence":
        if decision_rule == "low_overlap":
            return "low_overlap_after_retrieval"
        return "insufficient_after_retrieval"
    if label == 0 and status == "supported":
        return "true_supported"
    if label == 1 and status == "refuted":
        return "false_refuted"
    return "other"


def _add_bucket_example(
    buckets: dict[str, dict[str, Any]],
    key: str,
    *,
    item: Mapping[str, Any],
    record: Mapping[str, Any],
    label_key: str,
    selected_route: str,
    status: str,
    decision_rule: str,
    hit_count: int,
    max_examples: int,
) -> None:
    bucket = buckets.setdefault(key, {
        "count": 0,
        "label_counts": defaultdict(int),
        "route_counts": defaultdict(int),
        "status_counts": defaultdict(int),
        "decision_rule_counts": defaultdict(int),
        "examples": [],
    })
    bucket["count"] += 1
    bucket["label_counts"][label_key] += 1
    bucket["route_counts"][selected_route] += 1
    bucket["status_counts"][status] += 1
    bucket["decision_rule_counts"][decision_rule] += 1
    if len(bucket["examples"]) >= max_examples:
        return
    final = _mapping(record.get("final"))
    retrieval_hits = tuple(_mapping(hit) for hit in record.get("retrieval_hits", ()) if isinstance(hit, Mapping))
    best_hit = retrieval_hits[0] if retrieval_hits else {}
    bucket["examples"].append({
        "record_index": item.get("record_index"),
        "label": label_key,
        "claim": _claim_text(record),
        "selected_route": selected_route,
        "final_status": status,
        "decision_rule": decision_rule,
        "hit_count": hit_count,
        "score": item.get("score"),
        "best_source": best_hit.get("source"),
        "best_score": best_hit.get("score"),
        "best_text": best_hit.get("text"),
        "explanation": final.get("explanation"),
    })


def _claim_text(record: Mapping[str, Any]) -> str:
    claim = _mapping(record.get("claim"))
    return str(claim.get("text", ""))


def _topical_tokens(text: str) -> tuple[str, ...]:
    stop = {
        "a", "an", "and", "are", "as", "be", "by", "for", "from", "in", "is", "it",
        "of", "on", "or", "the", "to", "what", "when", "where", "who", "why",
        "with", "right", "now", "does", "do", "did", "can", "if",
    }
    return tuple(token for token in _tokens(text) if token not in stop and len(token) > 2)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(str(text)))


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _counter_top(counter: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(limit)
    ]


def _finalize_nested_counters(report: dict[str, Any]) -> dict[str, Any]:
    for bucket in report.get("gap_buckets", {}).values():
        for key in ("label_counts", "route_counts", "status_counts", "decision_rule_counts"):
            if isinstance(bucket.get(key), defaultdict):
                bucket[key] = dict(sorted(bucket[key].items()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retrieval route gap modes from verified-record JSONL")
    parser.add_argument("--verified-records-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples-per-bucket", type=int, default=5)
    parser.add_argument("--artifact-manifest", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
