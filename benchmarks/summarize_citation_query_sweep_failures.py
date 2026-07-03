"""Summarize why citation query sweeps still fail release gates.

The citation/search lane can be blocked even when source acquisition and
provenance pass. This report reads one or more
``sweep_blind_spot_retrieval_queries.py`` reports and turns the best observed
strategy in each sweep into an explicit failure-review artifact. It does not
promote evidence or loosen gates; it identifies the next implementation target.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "citation_query_sweep_failure_review"


def summarize_citation_query_sweep_failures(
    query_sweeps: Sequence[Mapping[str, Any]],
    *,
    max_examples_per_bucket: int = 3,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready failure review for citation query sweeps."""
    if not query_sweeps:
        raise ValueError("query_sweeps must not be empty.")
    if int(max_examples_per_bucket) < 0:
        raise ValueError("max_examples_per_bucket must be non-negative.")

    rows = tuple(
        _review_row(sweep, index=index, max_examples_per_bucket=int(max_examples_per_bucket))
        for index, sweep in enumerate(query_sweeps, start=1)
    )
    summary = _summary(rows)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "needs_alignment" if summary["sweep_without_passing_strategy_count"] else "monitor",
        "scope": (
            "Read-only failure review for citation query sweep artifacts. "
            "Rows explain why best observed strategies still fail release gates; "
            "they are not verifier evidence."
        ),
        "summary": summary,
        "sweeps": rows,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    query_sweep_paths: Sequence[str | Path],
    output_path: str | Path,
    max_examples_per_bucket: int = 3,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, optionally manifest, and optionally register a review."""
    if not query_sweep_paths:
        raise ValueError("query_sweep_paths must not be empty.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")

    paths = tuple(Path(path) for path in query_sweep_paths)
    payload = summarize_citation_query_sweep_failures(
        tuple(_load_mapping(path) for path in paths),
        max_examples_per_bucket=max_examples_per_bucket,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["source"] = {
        "query_sweeps": tuple(str(path) for path in paths),
    }
    if artifact_manifest_path is not None:
        payload["paths"] = {"artifact_manifest": str(artifact_manifest_path)}
    output = Path(output_path)
    _write_json(output, payload, compact=compact_json)

    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "citation_query_sweep_failure_review": output,
                **{f"query_sweep_{index}": path for index, path in enumerate(paths, start=1)},
            },
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "sweep_count": payload["summary"]["sweep_count"],
                "sweep_without_passing_strategy_count": payload["summary"][
                    "sweep_without_passing_strategy_count"
                ],
                "dominant_recommendation": payload["summary"]["dominant_recommendation"],
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
                "workflow": WORKFLOW,
                "status": payload["status"],
                "sweep_count": payload["summary"]["sweep_count"],
                "sweep_without_passing_strategy_count": payload["summary"][
                    "sweep_without_passing_strategy_count"
                ],
                "dominant_recommendation": payload["summary"]["dominant_recommendation"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    print(
        f"{WORKFLOW}_ok status={payload['status']} "
        f"sweeps={payload['summary']['sweep_count']} output={output}"
    )
    return payload


def _review_row(
    sweep: Mapping[str, Any],
    *,
    index: int,
    max_examples_per_bucket: int,
) -> dict[str, Any]:
    summary = _mapping(sweep.get("summary"))
    strategies = tuple(item for item in _sequence(sweep.get("strategies")) if isinstance(item, Mapping))
    selected_key = _clean_text(summary.get("best_strategy"))
    selected = _strategy_by_key(selected_key, strategies) if selected_key else None
    if selected is None and strategies:
        selected = strategies[0]
        selected_key = _clean_text(selected.get("key"))

    if selected is None:
        return {
            "index": index,
            "status": "missing_strategy",
            "best_strategy": selected_key,
            "best_passing_strategy": _clean_text(summary.get("best_passing_strategy")),
            "recommendations": ("rerun_query_sweep_with_strategy_records",),
            "failure_buckets": {},
            "examples": {},
        }

    blind = _mapping(selected.get("blind_spot"))
    gate = _mapping(selected.get("gate"))
    gap_analysis = _mapping(selected.get("gap_analysis"))
    outcome_counts = _int_mapping(blind.get("outcome_counts"))
    gap_counts = _gap_bucket_counts(gap_analysis)
    failure_buckets = _failure_buckets(outcome_counts=outcome_counts, gap_counts=gap_counts)
    recommendations = _recommendations(
        failure_buckets=failure_buckets,
        gate=gate,
        blind=blind,
        gap_analysis=gap_analysis,
    )
    return {
        "index": index,
        "status": "reviewed",
        "best_strategy": selected_key,
        "best_passing_strategy": _clean_text(summary.get("best_passing_strategy")),
        "gate_passed": bool(gate.get("pass")),
        "verified_false_alarm": _optional_float(gate.get("verified_false_alarm")),
        "max_verified_false_alarm": _optional_float(gate.get("max_verified_false_alarm")),
        "blind_refuted_rate": _optional_float(gate.get("blind_refuted_rate")),
        "min_blind_refuted_rate": _optional_float(gate.get("min_blind_refuted_rate")),
        "blind_spot_count": _optional_int(summary.get("blind_spot_count")),
        "corrected_refuted_count": _int(blind.get("target_route_refuted_count")),
        "records_with_retrieval_hits": _int(blind.get("records_with_retrieval_hits")),
        "target_route_selected_count": _int(blind.get("target_route_selected_count")),
        "target_route_selected_rate": _optional_float(blind.get("target_route_selected_rate")),
        "failure_buckets": failure_buckets,
        "dominant_failure_bucket": _dominant_failure_bucket(failure_buckets),
        "top_hit_sources": tuple(_sequence(gap_analysis.get("top_hit_sources")))[:10],
        "recommendations": recommendations,
        "examples": _selected_examples(selected, max_examples=max_examples_per_bucket),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recommendation_counts: Counter[str] = Counter()
    dominant_bucket_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    corrected_refuted_total = 0
    with_hits_total = 0
    blind_total = 0
    without_passing = 0
    for row in rows:
        for recommendation in _sequence(row.get("recommendations")):
            recommendation_counts[str(recommendation)] += 1
        bucket = _clean_text(row.get("dominant_failure_bucket"))
        if bucket:
            dominant_bucket_counts[bucket] += 1
        strategy = _clean_text(row.get("best_strategy"))
        if strategy:
            strategy_counts[strategy] += 1
        corrected_refuted_total += _int(row.get("corrected_refuted_count"))
        with_hits_total += _int(row.get("records_with_retrieval_hits"))
        blind_total += _int(row.get("blind_spot_count"))
        if not row.get("best_passing_strategy"):
            without_passing += 1
    return {
        "sweep_count": len(rows),
        "sweep_without_passing_strategy_count": without_passing,
        "total_blind_spot_count": blind_total,
        "total_records_with_retrieval_hits": with_hits_total,
        "total_corrected_refuted_count": corrected_refuted_total,
        "selected_strategy_counts": dict(sorted(strategy_counts.items())),
        "dominant_failure_bucket_counts": dict(sorted(dominant_bucket_counts.items())),
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "dominant_recommendation": _counter_first(recommendation_counts),
    }


def _failure_buckets(
    *,
    outcome_counts: Mapping[str, int],
    gap_counts: Mapping[str, int],
) -> dict[str, int]:
    buckets = Counter()
    for key in (
        "insufficient_evidence",
        "not_selected_by_target_route",
        "false_supported",
        "corrected_refuted",
    ):
        buckets[key] += _int(outcome_counts.get(key))
    for key in (
        "no_retrieval_hits",
        "low_overlap_after_retrieval",
        "insufficient_after_retrieval",
        "false_positive",
        "false_negative",
    ):
        buckets[key] += _int(gap_counts.get(key))
    return dict(sorted((key, count) for key, count in buckets.items() if count))


def _recommendations(
    *,
    failure_buckets: Mapping[str, int],
    gate: Mapping[str, Any],
    blind: Mapping[str, Any],
    gap_analysis: Mapping[str, Any],
) -> tuple[str, ...]:
    recommendations: list[str] = []
    false_alarm = _optional_float(gate.get("verified_false_alarm"))
    max_false_alarm = _optional_float(gate.get("max_verified_false_alarm"))
    blind_rate = _optional_float(gate.get("blind_refuted_rate"))
    min_blind_rate = _optional_float(gate.get("min_blind_refuted_rate"))
    if false_alarm is not None and max_false_alarm is not None and false_alarm > max_false_alarm:
        recommendations.append("tighten_false_alarm_calibration")
    if blind_rate is not None and min_blind_rate is not None and blind_rate < min_blind_rate:
        recommendations.append("improve_claim_intent_alignment_or_query_construction")
    if _int(failure_buckets.get("false_supported")):
        recommendations.append("tighten_slot_alignment_false_support_guard")
    if _int(failure_buckets.get("low_overlap_after_retrieval")) or _int(failure_buckets.get("insufficient_evidence")):
        recommendations.append("improve_claim_evidence_alignment_rules")
    if _int(failure_buckets.get("not_selected_by_target_route")) or _int(failure_buckets.get("no_retrieval_hits")):
        recommendations.append("improve_query_planning_or_route_selection")
    top_sources = tuple(
        str(item.get("value"))
        for item in _sequence(gap_analysis.get("top_hit_sources"))
        if isinstance(item, Mapping) and item.get("value")
    )
    if top_sources and _int(blind.get("records_with_retrieval_hits")):
        recommendations.append("extract_structured_facts_from_retrieved_sources")
    if not recommendations:
        recommendations.append("monitor_query_sweep_regression")
    return tuple(dict.fromkeys(recommendations))


def _dominant_failure_bucket(failure_buckets: Mapping[str, int]) -> str | None:
    candidates = {
        key: count
        for key, count in failure_buckets.items()
        if key != "corrected_refuted" and int(count) > 0
    }
    if not candidates:
        return None
    return max(sorted(candidates), key=lambda key: candidates[key])


def _selected_examples(strategy: Mapping[str, Any], *, max_examples: int) -> dict[str, tuple[Mapping[str, Any], ...]]:
    if max_examples <= 0:
        return {}
    examples = _mapping(strategy.get("examples"))
    selected: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for key in (
        "false_supported",
        "insufficient_evidence",
        "not_selected_by_target_route",
        "corrected_refuted",
    ):
        values = tuple(item for item in _sequence(examples.get(key)) if isinstance(item, Mapping))
        if values:
            selected[key] = values[:max_examples]
    return selected


def _strategy_by_key(key: str, strategies: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for strategy in strategies:
        if _clean_text(strategy.get("key")) == key:
            return strategy
    return None


def _gap_bucket_counts(gap_analysis: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for key, value in _mapping(gap_analysis.get("gap_buckets")).items():
        if isinstance(value, Mapping):
            counts[str(key)] = _int(value.get("count"))
    return dict(sorted(counts.items()))


def _load_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--metadata entries must be key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        metadata[key] = raw
    return metadata


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    output.write_text(strict_json_dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _int_mapping(value: Any) -> dict[str, int]:
    return {
        str(key): _int(item)
        for key, item in _mapping(value).items()
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    return _optional_int(value) or 0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _counter_first(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize citation query sweep failure modes.")
    parser.add_argument("--query-sweep", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples-per-bucket", type=int, default=3)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args()
    run(
        query_sweep_paths=tuple(args.query_sweep),
        output_path=args.output,
        max_examples_per_bucket=args.max_examples_per_bucket,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )


if __name__ == "__main__":
    main()
