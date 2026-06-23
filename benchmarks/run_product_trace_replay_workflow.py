"""Build a replay-ready ProductTrace corpus and run replay reports.

This workflow is the one-command handoff from raw saved product traces to
redacted corpus artifacts, product runtime baselines, and selector-policy replay.
It performs no model, verifier, retriever, or external-service work.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_product_trace_corpus import (  # noqa: E402
    ProductTraceCorpusConfig,
    build_product_trace_corpus,
)
from benchmarks.run_product_runtime_baseline import (  # noqa: E402
    ProductRuntimeBaselineConfig,
    build_product_runtime_baseline,
)
from benchmarks.run_runtime_profile_selector_replay import (  # noqa: E402
    RuntimeProfileSelectorReplayConfig,
    run_runtime_profile_selector_replay,
)
from benchmarks.run_runtime_profile_selector_tuning import RuntimeProfileSelectorCandidate  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class ProductTraceReplayWorkflowConfig:
    """Configuration for a full ProductTrace replay workflow."""

    trace_paths: Sequence[str | Path] = ()
    jsonl_paths: Sequence[str | Path] = ()
    output_dir: str | Path = "artifacts/product_trace_replay_workflow"
    candidates: Sequence[RuntimeProfileSelectorCandidate | Mapping[str, Any]] = ()
    replay_policy_path: str | Path | None = None
    runtime_policy_path: str | Path | None = None
    promotion_contract_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    redact_text: bool = True
    require_runtime_trace: bool = False
    strict: bool = False
    limit: int | None = None
    compact_json: bool = False

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        jsonl_paths = tuple(Path(path) for path in self.jsonl_paths)
        if not trace_paths and not jsonl_paths:
            raise ValueError("at least one ProductTrace JSON or JSONL path is required.")
        candidates = tuple(_candidate_from_value(candidate) for candidate in self.candidates)
        if not candidates:
            raise ValueError("at least one selector candidate is required.")
        names = [candidate.name for candidate in candidates]
        if len(set(names)) != len(names):
            raise ValueError("selector candidate names must be unique.")
        if self.runtime_policy_path is not None and self.promotion_contract_path is not None:
            raise ValueError("runtime_policy_path and promotion_contract_path are mutually exclusive.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        limit = None if self.limit is None else int(self.limit)
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided.")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "jsonl_paths", jsonl_paths)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "candidates", candidates)
        if self.replay_policy_path is not None:
            object.__setattr__(self, "replay_policy_path", Path(self.replay_policy_path))
        if self.runtime_policy_path is not None:
            object.__setattr__(self, "runtime_policy_path", Path(self.runtime_policy_path))
        if self.promotion_contract_path is not None:
            object.__setattr__(self, "promotion_contract_path", Path(self.promotion_contract_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "redact_text", bool(self.redact_text))
        object.__setattr__(self, "require_runtime_trace", bool(self.require_runtime_trace))
        object.__setattr__(self, "strict", bool(self.strict))
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "compact_json", bool(self.compact_json))

    @property
    def resolved_report_path(self) -> Path:
        """Return the top-level workflow report path."""
        return Path(self.output_dir) / "product-trace-replay-workflow.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the top-level artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_dir) / "artifact-manifest.json"


def run_product_trace_replay_workflow(config: ProductTraceReplayWorkflowConfig) -> dict[str, Any]:
    """Run corpus build, runtime baseline, and selector replay in one workflow."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = _run_corpus(config)
    corpus_trace_paths = tuple(Path(record["path"]) for record in _sequence(corpus.get("traces")))
    runtime_baseline = _run_runtime_baseline(config, corpus_trace_paths)
    selector_replay = _run_selector_replay(config, corpus_trace_paths)
    status = _workflow_status(corpus, runtime_baseline, selector_replay)
    report = {
        "schema_version": 1,
        "workflow": "product_trace_replay_workflow",
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": _blocking_reasons(corpus, runtime_baseline, selector_replay),
            "recommended_selector_candidate": _nested(
                selector_replay,
                "decision",
                "recommended_candidate",
            ),
            "recommended_selector_policy_path": _nested(
                selector_replay,
                "decision",
                "recommended_policy_path",
            ),
        },
        "corpus": _corpus_summary(corpus),
        "runtime_baseline": _runtime_baseline_summary(runtime_baseline),
        "selector_replay": _selector_replay_summary(selector_replay),
        "paths": {
            "report": str(config.resolved_report_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "output_dir": str(config.output_dir),
            "corpus_report": _nested(corpus, "paths", "report"),
            "corpus_manifest": _nested(corpus, "paths", "artifact_manifest"),
            "corpus_traces_dir": _nested(corpus, "paths", "traces_dir"),
            "runtime_baseline_report": _nested(runtime_baseline, "paths", "report"),
            "runtime_baseline_manifest": _nested(runtime_baseline, "paths", "artifact_manifest"),
            "selector_replay_report": _nested(selector_replay, "paths", "report"),
            "selector_replay_manifest": _nested(selector_replay, "paths", "artifact_manifest"),
        },
        "config": {
            "candidate_names": tuple(candidate.name for candidate in config.candidates),
            "trace_count": len(config.trace_paths),
            "jsonl_count": len(config.jsonl_paths),
            "replay_policy": None if config.replay_policy_path is None else str(config.replay_policy_path),
            "runtime_policy": None if config.runtime_policy_path is None else str(config.runtime_policy_path),
            "promotion_contract": (
                None if config.promotion_contract_path is None else str(config.promotion_contract_path)
            ),
            "redact_text": config.redact_text,
            "require_runtime_trace": config.require_runtime_trace,
            "strict": config.strict,
            "limit": config.limit,
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
    }
    _write_report_and_manifest(config, report)
    _record_registry(config, report)
    return report


def _run_corpus(config: ProductTraceReplayWorkflowConfig) -> dict[str, Any]:
    return build_product_trace_corpus(
        ProductTraceCorpusConfig(
            trace_paths=config.trace_paths,
            jsonl_paths=config.jsonl_paths,
            output_dir=Path(config.output_dir) / "corpus",
            redact_text=config.redact_text,
            require_runtime_trace=config.require_runtime_trace,
            strict=config.strict,
            limit=config.limit,
            compact_json=config.compact_json,
            metadata={
                "source": "run_product_trace_replay_workflow",
                **dict(config.metadata),
            },
        )
    )


def _run_runtime_baseline(
    config: ProductTraceReplayWorkflowConfig,
    trace_paths: Sequence[Path],
) -> dict[str, Any]:
    if not trace_paths:
        return _skipped_child_report("product_runtime_baseline", reason="no valid corpus traces")
    output_dir = Path(config.output_dir) / "runtime-baseline"
    return build_product_runtime_baseline(
        ProductRuntimeBaselineConfig(
            trace_paths=trace_paths,
            report_path=output_dir / "product-runtime-baseline.json",
            policy_path=config.runtime_policy_path,
            promotion_contract_path=config.promotion_contract_path,
            artifact_manifest_path=output_dir / "artifact-manifest.json",
            compact_json=config.compact_json,
            metadata={
                "source": "run_product_trace_replay_workflow",
                **dict(config.metadata),
            },
        )
    )


def _run_selector_replay(
    config: ProductTraceReplayWorkflowConfig,
    trace_paths: Sequence[Path],
) -> dict[str, Any]:
    if not trace_paths:
        return _skipped_child_report("runtime_profile_selector_replay", reason="no valid corpus traces")
    output_dir = Path(config.output_dir) / "selector-replay"
    return run_runtime_profile_selector_replay(
        RuntimeProfileSelectorReplayConfig(
            trace_paths=trace_paths,
            output_dir=output_dir,
            candidates=config.candidates,
            replay_policy_path=config.replay_policy_path,
            compact_json=config.compact_json,
            metadata={
                "source": "run_product_trace_replay_workflow",
                **dict(config.metadata),
            },
        )
    )


def _skipped_child_report(workflow: str, *, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": workflow,
        "status": "blocked",
        "decision": {
            "status": "blocked",
            "blocking_reasons": (reason,),
        },
    }


def _workflow_status(
    corpus: Mapping[str, Any],
    runtime_baseline: Mapping[str, Any],
    selector_replay: Mapping[str, Any],
) -> str:
    child_statuses = (
        corpus.get("status"),
        runtime_baseline.get("status"),
        selector_replay.get("status"),
    )
    if "blocked" in child_statuses:
        return "blocked"
    if corpus.get("status") == "partial":
        return "partial"
    if selector_replay.get("status") == "promote":
        return "promote"
    return "observed"


def _blocking_reasons(
    corpus: Mapping[str, Any],
    runtime_baseline: Mapping[str, Any],
    selector_replay: Mapping[str, Any],
) -> tuple[str, ...]:
    reasons = []
    for child_name, child in (
        ("corpus", corpus),
        ("runtime_baseline", runtime_baseline),
        ("selector_replay", selector_replay),
    ):
        if child.get("status") != "blocked":
            continue
        child_reasons = _sequence(_nested(child, "decision", "blocking_reasons"))
        if not child_reasons:
            reasons.append(f"{child_name}: blocked")
            continue
        reasons.extend(f"{child_name}: {reason}" for reason in child_reasons)
    return tuple(str(reason) for reason in reasons)


def _corpus_summary(corpus: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(corpus.get("summary"))
    return {
        "status": corpus.get("status"),
        "accepted_count": summary.get("accepted_count"),
        "rejected_count": summary.get("rejected_count"),
        "runtime_trace_count": summary.get("runtime_trace_count"),
        "redacted_trace_count": summary.get("redacted_trace_count"),
        "unique_request_key_count": summary.get("unique_request_key_count"),
        "counts_by_runtime_profile": dict(_mapping(summary.get("counts_by_runtime_profile"))),
        "counts_by_risk_level": dict(_mapping(summary.get("counts_by_risk_level"))),
        "counts_by_action": dict(_mapping(summary.get("counts_by_action"))),
    }


def _runtime_baseline_summary(runtime_baseline: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(runtime_baseline.get("summary"))
    total_seconds = _mapping(summary.get("total_seconds"))
    return {
        "status": runtime_baseline.get("status"),
        "budget_enabled": _nested(runtime_baseline, "budget", "enabled"),
        "budget_passed": _nested(runtime_baseline, "budget", "passed"),
        "n_traces": summary.get("n_traces"),
        "runtime_trace_count": summary.get("runtime_trace_count"),
        "total_seconds_mean": total_seconds.get("mean"),
        "total_seconds_p95": total_seconds.get("p95"),
        "total_seconds_max": total_seconds.get("max"),
    }


def _selector_replay_summary(selector_replay: Mapping[str, Any]) -> dict[str, Any]:
    recommended = _recommended_leaderboard_row(selector_replay)
    return {
        "status": selector_replay.get("status"),
        "recommended_candidate": _nested(selector_replay, "decision", "recommended_candidate"),
        "recommended_policy_path": _nested(selector_replay, "decision", "recommended_policy_path"),
        "recommended_estimated_cost_units_mean": recommended.get("estimated_cost_units_mean"),
        "recommended_observed_runtime_coverage_rate": recommended.get("observed_runtime_coverage_rate"),
        "recommended_observed_selected_total_seconds_mean": recommended.get(
            "observed_selected_total_seconds_mean"
        ),
        "recommended_observed_selected_total_seconds_p95": recommended.get(
            "observed_selected_total_seconds_p95"
        ),
        "leaderboard": tuple(_sequence(selector_replay.get("leaderboard"))),
    }


def _recommended_leaderboard_row(report: Mapping[str, Any]) -> dict[str, Any]:
    recommended_candidate = _nested(report, "decision", "recommended_candidate")
    for row in _sequence(report.get("leaderboard")):
        if isinstance(row, Mapping) and row.get("candidate") == recommended_candidate:
            return dict(row)
    return {}


def _write_report_and_manifest(
    config: ProductTraceReplayWorkflowConfig,
    report: dict[str, Any],
) -> dict[str, Any]:
    _write_json(config.resolved_report_path, report, compact=config.compact_json)
    manifest = _write_artifact_manifest(config, report)
    report["artifact_manifest_summary"] = manifest["summary"]
    _write_json(config.resolved_report_path, report, compact=config.compact_json)
    return _write_artifact_manifest(config, report)


def _write_artifact_manifest(
    config: ProductTraceReplayWorkflowConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "product_trace_replay_workflow_report": config.resolved_report_path,
        "corpus_report": _nested(report, "paths", "corpus_report"),
        "corpus_manifest": _nested(report, "paths", "corpus_manifest"),
        "corpus_traces_dir": _nested(report, "paths", "corpus_traces_dir"),
        "runtime_baseline_report": _nested(report, "paths", "runtime_baseline_report"),
        "runtime_baseline_manifest": _nested(report, "paths", "runtime_baseline_manifest"),
        "selector_replay_report": _nested(report, "paths", "selector_replay_report"),
        "selector_replay_manifest": _nested(report, "paths", "selector_replay_manifest"),
        "runtime_policy": config.runtime_policy_path,
        "promotion_contract": config.promotion_contract_path,
        "replay_policy": config.replay_policy_path,
    }
    for index, path in enumerate(config.trace_paths):
        artifacts[f"input_trace_{index:04d}_{_safe_artifact_name(path.stem)}"] = path
    for index, path in enumerate(config.jsonl_paths):
        artifacts[f"input_jsonl_{index:04d}_{_safe_artifact_name(path.stem)}"] = path
    manifest = build_artifact_manifest(
        artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_product_trace_replay_workflow",
            "status": report.get("status"),
            "corpus_status": _nested(report, "corpus", "status"),
            "runtime_baseline_status": _nested(report, "runtime_baseline", "status"),
            "selector_replay_status": _nested(report, "selector_replay", "status"),
            "recommended_selector_candidate": _nested(
                report,
                "decision",
                "recommended_selector_candidate",
            ),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(config: ProductTraceReplayWorkflowConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
    ArtifactRegistry.load_json(config.registry_path).record_report(
        name=str(config.name),
        path=config.resolved_report_path,
        version=str(config.version),
        metadata={
            "workflow": "run_product_trace_replay_workflow",
            "status": report.get("status"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "corpus_status": _nested(report, "corpus", "status"),
            "runtime_baseline_status": _nested(report, "runtime_baseline", "status"),
            "selector_replay_status": _nested(report, "selector_replay", "status"),
            "recommended_selector_candidate": _nested(
                report,
                "decision",
                "recommended_selector_candidate",
            ),
            "recommended_selector_policy_path": _nested(
                report,
                "decision",
                "recommended_selector_policy_path",
            ),
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    ).save_json()


def _candidate_from_value(
    value: RuntimeProfileSelectorCandidate | Mapping[str, Any],
) -> RuntimeProfileSelectorCandidate:
    if isinstance(value, RuntimeProfileSelectorCandidate):
        return value
    payload = dict(value)
    return RuntimeProfileSelectorCandidate(
        name=str(payload["name"]),
        policy=_mapping(payload.get("policy")),
        source=None if payload.get("source") is None else str(payload.get("source")),
    )


def _load_candidate(value: str) -> RuntimeProfileSelectorCandidate:
    if "=" in value:
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
    else:
        path = Path(value)
        name = path.stem
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"selector candidate JSON must be an object: {path}")
    return RuntimeProfileSelectorCandidate(name=name, policy=payload, source=str(path))


def _trace_paths_from_args(values: Sequence[str], globs: Sequence[str]) -> tuple[Path, ...]:
    paths = [Path(value) for value in values]
    for pattern in globs:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern)))
    return _unique_paths(paths)


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned or "artifact"


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_text(payload, compact=compact), encoding="utf-8")


def _json_text(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _parse_mapping_json(value: str | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return dict(payload)


def _config_from_args(args: argparse.Namespace) -> ProductTraceReplayWorkflowConfig:
    if not args.candidate:
        raise ValueError("--candidate is required for product trace replay workflow.")
    return ProductTraceReplayWorkflowConfig(
        trace_paths=_trace_paths_from_args(args.trace or (), args.trace_glob or ()),
        jsonl_paths=_unique_paths(tuple(Path(path) for path in args.jsonl or ())),
        output_dir=Path(args.output_dir),
        candidates=tuple(_load_candidate(value) for value in args.candidate),
        replay_policy_path=Path(args.replay_policy) if args.replay_policy else None,
        runtime_policy_path=Path(args.runtime_policy) if args.runtime_policy else None,
        promotion_contract_path=Path(args.promotion_contract) if args.promotion_contract else None,
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_mapping_json(args.metadata_json, name="--metadata-json"),
        redact_text=not bool(args.no_redact_text),
        require_runtime_trace=bool(args.require_runtime_trace),
        strict=bool(args.strict),
        limit=args.limit,
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = run_product_trace_replay_workflow(_config_from_args(args))
    print(_json_text(report, compact=bool(args.compact_json)), end="")
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a replay workflow over saved ProductTrace payloads")
    parser.add_argument("--trace", action="append", default=[], help="ProductTrace JSON path; repeatable")
    parser.add_argument("--trace-glob", action="append", default=[], help="glob for ProductTrace JSON files")
    parser.add_argument("--jsonl", action="append", default=[], help="ProductTrace JSONL path; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", default=[],
                        help="candidate selector policy JSON path, or name=path; repeatable")
    parser.add_argument("--replay-policy", default=None, help="RuntimeProfileSelectorReplayPolicy JSON path")
    parser.add_argument("--runtime-policy", default=None, help="ProductRuntimeBudgetPolicy JSON path")
    parser.add_argument("--promotion-contract", default=None, help="ProductPromotionContract/release report JSON path")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--no-redact-text", action="store_true")
    parser.add_argument("--require-runtime-trace", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
