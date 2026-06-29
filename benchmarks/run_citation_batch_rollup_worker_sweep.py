"""Sweep bounded worker counts for citation/search batch evidence rollups."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.config_utils import strict_bool, strict_positive_int  # noqa: E402
from benchmarks.rollup_citation_search_batch_evidence import (  # noqa: E402
    DEFAULT_EXPECTED_REQUEST_TYPE,
    rollup_citation_search_batch_evidence,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "citation_search_batch_rollup_worker_sweep"


@dataclass(frozen=True)
class CitationBatchRollupWorkerSweepConfig:
    """Configuration for replaying citation batch rollup across worker counts."""

    report_paths: Sequence[str | Path]
    output_dir: str | Path
    worker_counts: Sequence[int] = (1, 2, 4)
    queue_report_path: str | Path | None = None
    expected_batch_ids: Sequence[str] = ()
    expected_request_type: str = DEFAULT_EXPECTED_REQUEST_TYPE
    require_child_manifests: bool = True
    recursive_child_manifest_verification: bool = True
    compact_json: bool = False
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        report_paths = tuple(Path(path) for path in self.report_paths)
        if not report_paths:
            raise ValueError("report_paths must not be empty.")
        object.__setattr__(self, "report_paths", report_paths)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        worker_counts = tuple(strict_positive_int(value, name="worker_counts") for value in self.worker_counts)
        if not worker_counts:
            raise ValueError("worker_counts must not be empty.")
        if len(worker_counts) != len(set(worker_counts)):
            raise ValueError("worker_counts must not contain duplicates.")
        object.__setattr__(self, "worker_counts", worker_counts)
        if self.queue_report_path is not None:
            object.__setattr__(self, "queue_report_path", Path(self.queue_report_path))
        expected_batch_ids = tuple(dict.fromkeys(str(item) for item in self.expected_batch_ids if str(item)))
        object.__setattr__(self, "expected_batch_ids", expected_batch_ids)
        object.__setattr__(
            self,
            "require_child_manifests",
            strict_bool(self.require_child_manifests, name="require_child_manifests"),
        )
        object.__setattr__(
            self,
            "recursive_child_manifest_verification",
            strict_bool(
                self.recursive_child_manifest_verification,
                name="recursive_child_manifest_verification",
            ),
        )
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
            if not self.name or not self.version:
                raise ValueError("registry_path requires name and version.")
        if (self.name is None) != (self.version is None):
            raise ValueError("registry recording requires both name and version.")
        object.__setattr__(self, "metadata", dict(self.metadata))


def run_citation_batch_rollup_worker_sweep(
    config: CitationBatchRollupWorkerSweepConfig,
) -> dict[str, Any]:
    """Replay the same rollup under each worker count and recommend a setting."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = tuple(_run_worker_sample(config, worker_count) for worker_count in config.worker_counts)
    leaderboard = _leaderboard(rows)
    decision = _decision(leaderboard)
    status = (
        "blocked"
        if decision["recommended_worker_count"] is None
        else ("promote" if decision["recommendation_basis"] == "promotion_ready" else "complete")
    )
    report_path = output_dir / "citation-batch-rollup-worker-sweep.json"
    manifest_path = output_dir / "artifact-manifest.json"
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "decision": decision,
        "leaderboard": leaderboard,
        "rows": rows,
        "summary": _summary(rows),
        "source": {
            "reports": tuple(str(path) for path in config.report_paths),
            "queue_report": None if config.queue_report_path is None else str(config.queue_report_path),
        },
        "config": {
            "worker_counts": tuple(config.worker_counts),
            "expected_batch_ids": tuple(config.expected_batch_ids),
            "expected_request_type": config.expected_request_type,
            "require_child_manifests": config.require_child_manifests,
            "recursive_child_manifest_verification": config.recursive_child_manifest_verification,
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
        "paths": {
            "report": str(report_path),
            "artifact_manifest": str(manifest_path),
            "registry": None if config.registry_path is None else str(config.registry_path),
        },
        "metadata": dict(config.metadata),
        "registry_record": None if config.registry_path is None else f"report:{config.name}:{config.version}",
    }
    _write_json(report_path, payload, compact=config.compact_json)
    manifest = build_artifact_manifest(
        _manifest_artifacts(config, rows=rows, report_path=report_path),
        root=output_dir,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "recommended_worker_count": decision["recommended_worker_count"],
            "recommended_wall_clock_seconds": decision["recommended_wall_clock_seconds"],
            "recommendation_basis": decision["recommendation_basis"],
            "worker_count_count": len(tuple(config.worker_counts)),
            "promoted_worker_count": payload["summary"]["promoted_worker_count"],
            "passed_worker_count": payload["summary"]["passed_worker_count"],
            **dict(config.metadata),
        },
        max_workers=decision["recommended_worker_count"] or 1,
    )
    _write_json(manifest_path, manifest, compact=config.compact_json)
    if config.registry_path is not None:
        _record_registry(config, payload)
    return payload


def _run_worker_sample(
    config: CitationBatchRollupWorkerSweepConfig,
    worker_count: int,
) -> dict[str, Any]:
    worker_dir = Path(config.output_dir) / f"workers_{worker_count}"
    started_at = time.perf_counter()
    rollup = rollup_citation_search_batch_evidence(
        report_paths=config.report_paths,
        queue_report_path=config.queue_report_path,
        expected_batch_ids=config.expected_batch_ids,
        expected_request_type=config.expected_request_type,
        report_json_path=worker_dir / "citation-search-batch-rollup.json",
        artifact_manifest_path=worker_dir / "artifact-manifest.json",
        require_child_manifests=config.require_child_manifests,
        recursive_child_manifest_verification=config.recursive_child_manifest_verification,
        max_workers=worker_count,
        metadata={
            "worker_sweep_workflow": WORKFLOW,
            "worker_count": worker_count,
            **dict(config.metadata),
        },
        compact_json=config.compact_json,
    )
    wall_clock_seconds = _round_seconds(time.perf_counter() - started_at)
    gate = _mapping(rollup.get("gate"))
    summary = _mapping(rollup.get("summary"))
    paths = _mapping(rollup.get("paths"))
    execution = _mapping(rollup.get("execution"))
    return {
        "worker_count": int(worker_count),
        "status": rollup.get("status"),
        "passed": bool(gate.get("passed")),
        "promotion_ready": bool(gate.get("promotion_ready")),
        "wall_clock_seconds": wall_clock_seconds,
        "blocking_reason_count": len(_sequence(gate.get("blocking_reasons", ()))),
        "report_count": _optional_int(summary.get("report_count")) or 0,
        "expected_batch_count": _optional_int(summary.get("expected_batch_count")) or 0,
        "observed_batch_count": _optional_int(summary.get("observed_batch_count")) or 0,
        "missing_expected_batch_count": _optional_int(summary.get("missing_expected_batch_count")) or 0,
        "blocked_report_count": _optional_int(summary.get("blocked_report_count")) or 0,
        "child_manifest_failed_count": _optional_int(summary.get("child_manifest_failed_count")) or 0,
        "parallel_child_report_count": _optional_int(execution.get("parallel_child_report_count")) or 0,
        "rollup_report": paths.get("report"),
        "artifact_manifest": paths.get("artifact_manifest"),
    }


def _leaderboard(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    baseline_seconds = _baseline_seconds(rows)
    ranked = []
    for row in rows:
        seconds = _finite_float(row.get("wall_clock_seconds"))
        ranked.append({
            "worker_count": _optional_int(row.get("worker_count")),
            "wall_clock_seconds": seconds,
            "passed": bool(row.get("passed")),
            "promotion_ready": bool(row.get("promotion_ready")),
            "status": row.get("status"),
            "speedup_vs_worker_1": (
                None
                if baseline_seconds in (None, 0.0) or seconds in (None, 0.0)
                else baseline_seconds / seconds
            ),
            "rollup_report": row.get("rollup_report"),
            "artifact_manifest": row.get("artifact_manifest"),
        })
    return tuple(sorted(
        ranked,
        key=lambda item: (
            not bool(item.get("promotion_ready")),
            not bool(item.get("passed")),
            math.inf if item.get("wall_clock_seconds") is None else float(item["wall_clock_seconds"]),
            math.inf if item.get("worker_count") is None else int(item["worker_count"]),
        ),
    ))


def _decision(leaderboard: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recommended = next((row for row in leaderboard if bool(row.get("promotion_ready"))), None)
    basis = "promotion_ready"
    if recommended is None:
        recommended = next((row for row in leaderboard if bool(row.get("passed"))), None)
        basis = "passed"
    if recommended is None:
        return {
            "recommended_worker_count": None,
            "recommended_wall_clock_seconds": None,
            "recommendation_basis": "none",
            "speedup_vs_worker_1": None,
            "rollup_report": None,
            "artifact_manifest": None,
            "blocking_reasons": ("No worker count produced a passing citation batch rollup.",),
        }
    return {
        "recommended_worker_count": recommended.get("worker_count"),
        "recommended_wall_clock_seconds": recommended.get("wall_clock_seconds"),
        "recommendation_basis": basis,
        "speedup_vs_worker_1": recommended.get("speedup_vs_worker_1"),
        "rollup_report": recommended.get("rollup_report"),
        "artifact_manifest": recommended.get("artifact_manifest"),
        "blocking_reasons": (),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "worker_count_count": len(rows),
        "worker_counts": tuple(row.get("worker_count") for row in rows),
        "promoted_worker_count": sum(1 for row in rows if bool(row.get("promotion_ready"))),
        "passed_worker_count": sum(1 for row in rows if bool(row.get("passed"))),
        "blocked_worker_count": sum(1 for row in rows if row.get("status") == "blocked"),
        "fastest_worker_count": (
            None
            if not rows
            else min(
                rows,
                key=lambda row: (
                    math.inf
                    if _finite_float(row.get("wall_clock_seconds")) is None
                    else float(row.get("wall_clock_seconds")),
                    int(row.get("worker_count", 0)),
                ),
            ).get("worker_count")
        ),
    }


def _record_registry(
    config: CitationBatchRollupWorkerSweepConfig,
    payload: Mapping[str, Any],
) -> None:
    decision = _mapping(payload.get("decision"))
    summary = _mapping(payload.get("summary"))
    assert config.registry_path is not None
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=str(config.name),
        version=str(config.version),
        path=Path(config.output_dir) / "citation-batch-rollup-worker-sweep.json",
        metadata={
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "recommended_worker_count": decision.get("recommended_worker_count"),
            "recommended_wall_clock_seconds": decision.get("recommended_wall_clock_seconds"),
            "recommendation_basis": decision.get("recommendation_basis"),
            "speedup_vs_worker_1": decision.get("speedup_vs_worker_1"),
            "worker_count_count": summary.get("worker_count_count"),
            "promoted_worker_count": summary.get("promoted_worker_count"),
            "passed_worker_count": summary.get("passed_worker_count"),
            "artifact_manifest": str(Path(config.output_dir) / "artifact-manifest.json"),
            **dict(config.metadata),
        },
    )
    registry.save_json()


def _manifest_artifacts(
    config: CitationBatchRollupWorkerSweepConfig,
    *,
    rows: Sequence[Mapping[str, Any]],
    report_path: Path,
) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "worker_sweep_report": report_path,
        "queue_report": None if config.queue_report_path is None else Path(config.queue_report_path),
    }
    for index, path in enumerate(config.report_paths, start=1):
        artifacts[f"source_batch_report_{index}"] = Path(path)
    for index, row in enumerate(rows, start=1):
        artifacts[f"worker_rollup_report_{index}"] = _path_or_none(row.get("rollup_report"))
        artifacts[f"worker_artifact_manifest_{index}"] = _path_or_none(row.get("artifact_manifest"))
    return artifacts


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value)
    return None if not text else Path(text)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 6)


def _baseline_seconds(rows: Sequence[Mapping[str, Any]]) -> float | None:
    baseline = next((row for row in rows if int(row.get("worker_count", 0)) == 1), None)
    if baseline is None:
        return None
    return _finite_float(baseline.get("wall_clock_seconds"))


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _optional_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(strict_positive_int(part.strip(), name="workers") for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("workers must contain at least one positive integer.")
    if len(values) != len(set(values)):
        raise ValueError("workers must not contain duplicates.")
    return values


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the citation batch rollup worker sweep from parsed CLI args."""
    payload = run_citation_batch_rollup_worker_sweep(
        CitationBatchRollupWorkerSweepConfig(
            report_paths=tuple(args.batch_report or ()),
            output_dir=args.output_dir,
            worker_counts=args.workers,
            queue_report_path=args.queue,
            expected_batch_ids=tuple(args.expected_batch_id or ()),
            expected_request_type=args.expected_request_type,
            require_child_manifests=not bool(args.allow_missing_child_manifest),
            recursive_child_manifest_verification=not bool(args.no_recursive_child_manifest_verification),
            compact_json=bool(args.compact_json),
            registry_path=args.registry,
            name=args.name,
            version=args.version,
            metadata=_parse_metadata(args.metadata or ()),
        )
    )
    decision = payload["decision"]
    print(
        "citation_batch_rollup_worker_sweep="
        f"{payload['status']} "
        f"recommended_workers={decision.get('recommended_worker_count')} "
        f"speedup_vs_worker_1={decision.get('speedup_vs_worker_1')}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--queue", default=None)
    parser.add_argument("--expected-batch-id", action="append", default=[])
    parser.add_argument("--expected-request-type", default=DEFAULT_EXPECTED_REQUEST_TYPE)
    parser.add_argument(
        "--workers",
        type=_parse_int_list,
        default=(1, 2, 4),
        help="comma-separated worker counts to test; default: 1,2,4",
    )
    parser.add_argument("--allow-missing-child-manifest", action="store_true")
    parser.add_argument("--no-recursive-child-manifest-verification", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
