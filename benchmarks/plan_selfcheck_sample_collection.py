"""Plan self-check sample collection before selfcheck signal fusion.

This preflight keeps the direct selfcheck workflow fail-closed: it aligns any
existing sampled responses to a statement-bearing score dump, reports which
records still lack enough samples, and writes a machine-readable rerun plan.
It does not load models, regenerate samples, or call external services.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_selfcheck_fixture import (  # noqa: E402
    SAMPLE_KEYS,
    build_selfcheck_fixture,
    load_sample_payloads,
    load_score_dump,
)


@dataclass(frozen=True)
class SelfcheckSampleCollectionPlanConfig:
    """Configuration for selfcheck sample-collection preflight."""

    scores: Path
    output: Path
    sample_paths: Sequence[Path] = ()
    min_samples: int = 2
    target_samples_per_record: int | None = None
    max_records: int | None = None
    include_ready_records: bool = False
    sample_quality_min_coverage: float = 0.50
    sample_quality_min_average_samples_per_record: float = 1.0
    sample_quality_min_records_meeting_min_samples: int | None = None
    compact_json: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", Path(self.scores))
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "sample_paths", tuple(Path(path) for path in self.sample_paths))
        min_samples = int(self.min_samples)
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        object.__setattr__(self, "min_samples", min_samples)
        target = min_samples if self.target_samples_per_record is None else int(self.target_samples_per_record)
        if target < min_samples:
            raise ValueError("target_samples_per_record must be >= min_samples.")
        object.__setattr__(self, "target_samples_per_record", target)
        if self.max_records is not None:
            max_records = int(self.max_records)
            if max_records < 0:
                raise ValueError("max_records must be >= 0 when set.")
            object.__setattr__(self, "max_records", max_records)
        min_coverage = float(self.sample_quality_min_coverage)
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("sample_quality_min_coverage must be in [0, 1].")
        object.__setattr__(self, "sample_quality_min_coverage", min_coverage)
        min_average = float(self.sample_quality_min_average_samples_per_record)
        if min_average < 0.0:
            raise ValueError("sample_quality_min_average_samples_per_record must be >= 0.")
        object.__setattr__(self, "sample_quality_min_average_samples_per_record", min_average)
        if self.sample_quality_min_records_meeting_min_samples is not None:
            min_records = int(self.sample_quality_min_records_meeting_min_samples)
            if min_records < 0:
                raise ValueError("sample_quality_min_records_meeting_min_samples must be >= 0 when set.")
            object.__setattr__(self, "sample_quality_min_records_meeting_min_samples", min_records)


def build_selfcheck_sample_collection_plan(
    config: SelfcheckSampleCollectionPlanConfig,
) -> dict[str, Any]:
    """Build a sample-collection plan from a score dump and optional samples."""
    dump = load_score_dump(config.scores)
    sample_payloads = load_sample_payloads(config.sample_paths)
    fixture = build_selfcheck_fixture(
        dump,
        sample_payloads,
        min_samples=int(config.min_samples),
        include_empty_records=True,
    )
    records = tuple(
        _record_plan(record, target_samples=int(config.target_samples_per_record))
        for record in fixture["records"]
    )
    summary = _summary_from_records(records, fixture["summary"], config)
    quality_projection = _sample_quality_projection(summary, config)
    collection_status = "ready" if summary["records_below_target_samples"] == 0 else "needs_samples"
    detail_records = (
        records
        if bool(config.include_ready_records)
        else tuple(record for record in records if int(record["sample_deficit"]) > 0)
    )
    limited_records, truncated_count = _limit_records(detail_records, config.max_records)
    return {
        "schema_version": 1,
        "workflow": "selfcheck_sample_collection_plan",
        "status": collection_status,
        "input": {
            "scores": str(config.scores),
            "sample_paths": [str(path) for path in config.sample_paths],
            "sample_keys": list(SAMPLE_KEYS),
        },
        "config": {
            "min_samples": int(config.min_samples),
            "target_samples_per_record": int(config.target_samples_per_record),
            "include_ready_records": bool(config.include_ready_records),
            "max_records": config.max_records,
        },
        "summary": summary,
        "sample_quality_gate_projection": quality_projection,
        "collection_plan": {
            "status": collection_status,
            "records_to_collect_count": int(summary["records_below_target_samples"]),
            "recommended_min_new_samples": int(summary["sample_deficit_total"]),
            "target_samples_per_record": int(config.target_samples_per_record),
            "recommendation": (
                "existing samples satisfy the selfcheck sample target"
                if collection_status == "ready"
                else "collect or recover more aligned samples before promoting selfcheck signals"
            ),
            "rerun": _rerun_plan(config),
        },
        "records": list(limited_records),
        "records_truncated": truncated_count > 0,
        "records_truncated_count": truncated_count,
    }


def write_selfcheck_sample_collection_plan(
    config: SelfcheckSampleCollectionPlanConfig,
) -> dict[str, Any]:
    """Build and write a sample-collection plan."""
    payload = build_selfcheck_sample_collection_plan(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":")) if config.compact_json else (
        json.dumps(payload, indent=2, sort_keys=True)
    )
    config.output.write_text(text + "\n", encoding="utf-8")
    return payload


def _record_plan(record: Mapping[str, Any], *, target_samples: int) -> dict[str, Any]:
    metadata = _mapping(record.get("metadata"))
    selfcheck = _mapping(metadata.get("selfcheck"))
    n_samples = int(_number(selfcheck.get("n_samples"), default=len(_sequence(record.get("selfcheck_samples")))))
    index = int(_number(metadata.get("index"), default=-1))
    deficit = max(0, int(target_samples) - n_samples)
    return {
        "index": index,
        "claim_id": str(record.get("claim_id", f"c{index + 1}")),
        "claim": str(record.get("claim", "")),
        "current_samples": n_samples,
        "required_samples": int(target_samples),
        "sample_deficit": deficit,
        "meets_min_samples": bool(selfcheck.get("meets_min_samples", False)),
        "meets_target_samples": deficit == 0,
        "sample_source": str(selfcheck.get("source", "score_dump_or_external_samples")),
    }


def _summary_from_records(
    records: Sequence[Mapping[str, Any]],
    fixture_summary: Mapping[str, Any],
    config: SelfcheckSampleCollectionPlanConfig,
) -> dict[str, Any]:
    n_records = int(_number(fixture_summary.get("n_records"), default=len(records)))
    records_with_samples = sum(1 for record in records if int(record["current_samples"]) > 0)
    records_meeting_min = sum(1 for record in records if bool(record["meets_min_samples"]))
    records_meeting_target = sum(1 for record in records if bool(record["meets_target_samples"]))
    total_samples = sum(int(record["current_samples"]) for record in records)
    sample_deficit_total = sum(int(record["sample_deficit"]) for record in records)
    return {
        "n_records": n_records,
        "records_with_samples": records_with_samples,
        "records_meeting_min_samples": records_meeting_min,
        "records_below_min_samples": max(0, n_records - records_meeting_min),
        "records_meeting_target_samples": records_meeting_target,
        "records_below_target_samples": max(0, n_records - records_meeting_target),
        "total_samples": total_samples,
        "average_samples_per_record": float(total_samples) / float(n_records) if n_records else 0.0,
        "sample_presence_rate": float(records_with_samples) / float(n_records) if n_records else 0.0,
        "min_sample_coverage": float(records_meeting_min) / float(n_records) if n_records else 0.0,
        "target_sample_coverage": float(records_meeting_target) / float(n_records) if n_records else 0.0,
        "sample_deficit_total": sample_deficit_total,
        "min_samples": int(config.min_samples),
        "target_samples_per_record": int(config.target_samples_per_record),
    }


def _sample_quality_projection(
    summary: Mapping[str, Any],
    config: SelfcheckSampleCollectionPlanConfig,
) -> dict[str, Any]:
    thresholds = {
        "min_coverage": float(config.sample_quality_min_coverage),
        "min_average_samples_per_record": float(config.sample_quality_min_average_samples_per_record),
        "min_records_meeting_min_samples": config.sample_quality_min_records_meeting_min_samples,
    }
    failures = []
    coverage = float(summary["min_sample_coverage"])
    if coverage < thresholds["min_coverage"]:
        failures.append({
            "metric": "coverage",
            "value": coverage,
            "threshold": thresholds["min_coverage"],
            "rule": "value >= threshold",
        })
    average_samples = float(summary["average_samples_per_record"])
    if average_samples < thresholds["min_average_samples_per_record"]:
        failures.append({
            "metric": "average_samples_per_record",
            "value": average_samples,
            "threshold": thresholds["min_average_samples_per_record"],
            "rule": "value >= threshold",
        })
    min_records = thresholds["min_records_meeting_min_samples"]
    if min_records is not None and int(summary["records_meeting_min_samples"]) < int(min_records):
        failures.append({
            "metric": "records_meeting_min_samples",
            "value": int(summary["records_meeting_min_samples"]),
            "threshold": int(min_records),
            "rule": "value >= threshold",
        })
    return {
        "status": "pass" if not failures else "fail",
        "passed": not failures,
        "thresholds": thresholds,
        "metrics": {
            "coverage": coverage,
            "min_sample_coverage": coverage,
            "average_samples_per_record": average_samples,
            "records_meeting_min_samples": int(summary["records_meeting_min_samples"]),
        },
        "unavailable_until_signal_build": [
            "not_applicable_rate",
            "best_overlap_mean",
        ],
        "failures": failures,
    }


def _rerun_plan(config: SelfcheckSampleCollectionPlanConfig) -> dict[str, Any]:
    target = str(int(config.target_samples_per_record))
    min_samples = str(int(config.min_samples))
    scores = str(config.scores)
    return {
        "eval_truthfulqa_sample_flags": [
            "--inside-samples",
            target,
            "--dump-inside-samples",
            "--inside-diagnostics-cache",
            "<inside-diagnostics-cache.json>",
        ],
        "export_inside_diagnostics_samples": [
            "python",
            "benchmarks/export_inside_diagnostics_samples.py",
            "--scores",
            scores,
            "--inside-diagnostics-cache",
            "<inside-diagnostics-cache.json>",
            "--output",
            "<selfcheck-samples.json>",
            "--min-samples",
            min_samples,
        ],
        "run_selfcheck_signal_fusion_workflow": [
            "python",
            "benchmarks/run_selfcheck_signal_fusion_workflow.py",
            "--scores",
            f"run={scores}",
            "--samples",
            "<selfcheck-samples.json>",
            "--output-dir",
            "<selfcheck-fusion-dir>",
            "--min-samples",
            min_samples,
        ],
    }


def _limit_records(
    records: Sequence[Mapping[str, Any]],
    max_records: int | None,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    if max_records is None:
        return tuple(records), 0
    if max_records == 0:
        return (), len(records)
    limited = tuple(records[:max_records])
    return limited, max(0, len(records) - len(limited))


def _number(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from CLI-style arguments."""
    config = SelfcheckSampleCollectionPlanConfig(
        scores=Path(args.scores),
        output=Path(args.output),
        sample_paths=tuple(Path(path) for path in args.samples or ()),
        min_samples=args.min_samples,
        target_samples_per_record=args.target_samples_per_record,
        max_records=args.max_records,
        include_ready_records=args.include_ready_records,
        sample_quality_min_coverage=args.sample_quality_min_coverage,
        sample_quality_min_average_samples_per_record=args.sample_quality_min_average_samples_per_record,
        sample_quality_min_records_meeting_min_samples=args.sample_quality_min_records_meeting_min_samples,
        compact_json=args.compact_json,
    )
    payload = write_selfcheck_sample_collection_plan(config)
    summary = payload["summary"]
    print(
        "selfcheck_sample_collection_plan_ok "
        f"status={payload['status']} "
        f"records_to_collect={summary['records_below_target_samples']} "
        f"sample_deficit={summary['sample_deficit_total']} "
        f"output={config.output}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan selfcheck sample collection before signal fusion")
    parser.add_argument("--scores", required=True, help="statement-bearing score dump")
    parser.add_argument("--samples", action="append", default=None,
                        help="optional sampled responses JSON/JSONL; repeatable")
    parser.add_argument("--output", required=True, help="path to write the sample collection plan JSON")
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--target-samples-per-record", type=int, default=None,
                        help="desired samples per record; defaults to --min-samples")
    parser.add_argument("--max-records", type=int, default=None,
                        help="maximum record details to include; 0 keeps only summary")
    parser.add_argument("--include-ready-records", action="store_true",
                        help="include records that already meet the target")
    parser.add_argument("--sample-quality-min-coverage", type=float, default=0.50)
    parser.add_argument("--sample-quality-min-average-samples-per-record", type=float, default=1.0)
    parser.add_argument("--sample-quality-min-records-meeting-min-samples", type=int, default=None)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-needs-samples", action="store_true",
                        help="exit with status 2 when additional samples are needed")
    args = parser.parse_args()
    payload = run(args)
    if args.fail_on_needs_samples and payload["status"] == "needs_samples":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
