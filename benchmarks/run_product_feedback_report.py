"""Join ProductTrace payloads with post-hoc feedback records."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import planned_artifact_manifest_summary, strict_bool  # noqa: E402
from eigentruth.control import (  # noqa: E402
    FeedbackOutcome,
    ProductFeedbackRecord,
    load_feedback_jsonl,
    product_trace_fingerprint,
)
from eigentruth.eval.metrics import binomial_confidence_interval  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

_WRONG_OUTCOMES = frozenset({
    FeedbackOutcome.INCORRECT.value,
    FeedbackOutcome.PARTIALLY_CORRECT.value,
    FeedbackOutcome.UNSUPPORTED.value,
})
_UNSUPPORTED_OUTCOMES = frozenset({
    FeedbackOutcome.UNSUPPORTED.value,
})
_BLOCK_FALSE_POSITIVE_OUTCOMES = frozenset({
    FeedbackOutcome.CORRECT.value,
    FeedbackOutcome.UNNECESSARY_BLOCK.value,
})


@dataclass(frozen=True)
class ProductFeedbackReportConfig:
    """Configuration for a ProductTrace feedback join report."""

    trace_paths: Sequence[str | Path]
    feedback_paths: Sequence[str | Path]
    report_path: str | Path
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    min_matched_feedback_count: int | None = None
    max_accepted_but_wrong_rate: float | None = None
    max_retrieved_failure_rate: float | None = None
    max_abstain_false_positive_rate: float | None = None
    compact_json: bool = False

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        feedback_paths = tuple(Path(path) for path in self.feedback_paths)
        if not trace_paths:
            raise ValueError("at least one ProductTrace path is required.")
        if not feedback_paths:
            raise ValueError("at least one feedback JSONL path is required.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "feedback_paths", feedback_paths)
        object.__setattr__(self, "report_path", Path(self.report_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "min_matched_feedback_count",
            _optional_non_negative_int(
                self.min_matched_feedback_count,
                name="min_matched_feedback_count",
            ),
        )
        object.__setattr__(
            self,
            "max_accepted_but_wrong_rate",
            _optional_unit_float(
                self.max_accepted_but_wrong_rate,
                name="max_accepted_but_wrong_rate",
            ),
        )
        object.__setattr__(
            self,
            "max_retrieved_failure_rate",
            _optional_unit_float(
                self.max_retrieved_failure_rate,
                name="max_retrieved_failure_rate",
            ),
        )
        object.__setattr__(
            self,
            "max_abstain_false_positive_rate",
            _optional_unit_float(
                self.max_abstain_false_positive_rate,
                name="max_abstain_false_positive_rate",
            ),
        )
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the report artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.report_path).with_name("product-feedback-artifact-manifest.json")


def build_product_feedback_report(config: ProductFeedbackReportConfig) -> dict[str, Any]:
    """Build a feedback quality report from saved ProductTrace and JSONL feedback."""
    traces = tuple(_load_trace_record(path) for path in config.trace_paths)
    feedback = load_feedback_jsonl(config.feedback_paths)
    join = _join_feedback(traces, feedback)
    summary = _feedback_summary(join["matched"], unmatched_count=len(join["unmatched"]))
    gate = _gate_summary(summary, config)
    status = "observed" if not gate["configured"] else ("passed" if gate["passed"] else "blocked")
    report = {
        "schema_version": 1,
        "workflow": "product_feedback_report",
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": tuple(gate["blocking_reasons"]),
        },
        "summary": summary,
        "quality_gate": gate,
        "matched_feedback": tuple(join["matched"]),
        "unmatched_feedback": tuple(join["unmatched"]),
        "traces": tuple(traces),
        "config": {
            "trace_paths": tuple(str(path) for path in config.trace_paths),
            "feedback_paths": tuple(str(path) for path in config.feedback_paths),
            "min_matched_feedback_count": config.min_matched_feedback_count,
            "max_accepted_but_wrong_rate": config.max_accepted_but_wrong_rate,
            "max_retrieved_failure_rate": config.max_retrieved_failure_rate,
            "max_abstain_false_positive_rate": config.max_abstain_false_positive_rate,
            "metadata": dict(config.metadata),
        },
        "paths": {
            "report": str(config.report_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
        },
        "artifact_manifest_summary": _artifact_manifest_summary(config),
    }
    _write_json(config.report_path, report, compact=config.compact_json)
    _write_artifact_manifest(config, report)
    if config.registry_path is not None:
        _record_registry(config, report)
    return report


def _join_feedback(
    traces: Sequence[Mapping[str, Any]],
    feedback: Sequence[ProductFeedbackRecord],
) -> dict[str, list[dict[str, Any]]]:
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    by_fingerprint = {str(trace["trace_fingerprint"]): trace for trace in traces}
    for trace in traces:
        request_id = trace.get("request_id")
        if request_id is not None:
            by_request.setdefault(str(request_id), []).append(trace)

    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for record in feedback:
        trace, reason = _match_trace(record, by_request=by_request, by_fingerprint=by_fingerprint)
        if trace is None:
            unmatched.append({
                "feedback": record.to_dict(),
                "reason": reason,
            })
            continue
        matched.append(_joined_record(record, trace))
    return {"matched": matched, "unmatched": unmatched}


def _match_trace(
    record: ProductFeedbackRecord,
    *,
    by_request: Mapping[str, Sequence[Mapping[str, Any]]],
    by_fingerprint: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str | None]:
    if record.trace_fingerprint is not None:
        trace = by_fingerprint.get(record.trace_fingerprint)
        if trace is not None:
            return trace, None
        return None, "trace_fingerprint_not_found"
    candidates = tuple(by_request.get(record.request_id, ()))
    if not candidates:
        return None, "request_id_not_found"
    if len(candidates) > 1:
        return None, "ambiguous_request_id"
    return candidates[0], None


def _joined_record(record: ProductFeedbackRecord, trace: Mapping[str, Any]) -> dict[str, Any]:
    outcome = record.outcome.value
    actions = tuple(str(action) for action in trace.get("action_names", ()))
    decision_action = _optional_str(trace.get("decision_action"))
    accepted = _has_action(actions, decision_action, "accept")
    retrieved = _has_action(actions, decision_action, "retrieve")
    abstained = _has_action(actions, decision_action, "abstain")
    clarified = _has_action(actions, decision_action, "clarify")
    return {
        "feedback": record.to_dict(),
        "trace": {
            "path": trace.get("path"),
            "request_id": trace.get("request_id"),
            "trace_fingerprint": trace.get("trace_fingerprint"),
            "decision_action": decision_action,
            "risk_level": trace.get("risk_level"),
            "action_names": actions,
        },
        "flags": {
            "accepted": accepted,
            "retrieved": retrieved,
            "abstained": abstained,
            "clarified": clarified,
            "wrong_outcome": outcome in _WRONG_OUTCOMES,
            "unsupported_outcome": outcome in _UNSUPPORTED_OUTCOMES,
            "block_false_positive_outcome": outcome in _BLOCK_FALSE_POSITIVE_OUTCOMES,
            "accepted_but_wrong": accepted and outcome in _WRONG_OUTCOMES,
            "retrieved_failure": retrieved and outcome in _WRONG_OUTCOMES,
            "retrieved_but_still_unsupported": retrieved and outcome in _UNSUPPORTED_OUTCOMES,
            "abstain_false_positive": abstained and outcome in _BLOCK_FALSE_POSITIVE_OUTCOMES,
        },
    }


def _feedback_summary(
    matched: Sequence[Mapping[str, Any]],
    *,
    unmatched_count: int,
) -> dict[str, Any]:
    outcome_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    accepted_count = 0
    accepted_but_wrong_count = 0
    retrieved_count = 0
    retrieved_failure_count = 0
    retrieved_still_unsupported_count = 0
    abstain_count = 0
    abstain_false_positive_count = 0
    claim_level_count = 0
    for item in matched:
        feedback = _mapping(item.get("feedback"))
        trace = _mapping(item.get("trace"))
        flags = _mapping(item.get("flags"))
        outcome = str(feedback.get("outcome", "unknown"))
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        action = str(trace.get("decision_action") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        if feedback.get("claim_id") is not None:
            claim_level_count += 1
        if flags.get("accepted"):
            accepted_count += 1
        if flags.get("accepted_but_wrong"):
            accepted_but_wrong_count += 1
        if flags.get("retrieved"):
            retrieved_count += 1
        if flags.get("retrieved_failure"):
            retrieved_failure_count += 1
        if flags.get("retrieved_but_still_unsupported"):
            retrieved_still_unsupported_count += 1
        if flags.get("abstained"):
            abstain_count += 1
        if flags.get("abstain_false_positive"):
            abstain_false_positive_count += 1

    matched_count = len(matched)
    feedback_count = matched_count + unmatched_count
    return {
        "trace_matched_feedback_count": matched_count,
        "unmatched_feedback_count": unmatched_count,
        "feedback_count": feedback_count,
        "match_rate": binomial_confidence_interval(matched_count, feedback_count),
        "claim_level_feedback_count": claim_level_count,
        "outcome_counts": outcome_counts,
        "decision_action_counts": action_counts,
        "accepted_feedback_count": accepted_count,
        "accepted_but_wrong_count": accepted_but_wrong_count,
        "accepted_but_wrong_rate": binomial_confidence_interval(
            accepted_but_wrong_count,
            accepted_count,
        ),
        "retrieved_feedback_count": retrieved_count,
        "retrieved_failure_count": retrieved_failure_count,
        "retrieved_failure_rate": binomial_confidence_interval(
            retrieved_failure_count,
            retrieved_count,
        ),
        "retrieved_but_still_unsupported_count": retrieved_still_unsupported_count,
        "retrieved_but_still_unsupported_rate": binomial_confidence_interval(
            retrieved_still_unsupported_count,
            retrieved_count,
        ),
        "abstain_feedback_count": abstain_count,
        "abstain_false_positive_count": abstain_false_positive_count,
        "abstain_false_positive_rate": binomial_confidence_interval(
            abstain_false_positive_count,
            abstain_count,
        ),
    }


def _gate_summary(summary: Mapping[str, Any], config: ProductFeedbackReportConfig) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    _check_min_count(
        failures,
        "trace_matched_feedback_count",
        summary.get("trace_matched_feedback_count"),
        config.min_matched_feedback_count,
    )
    _check_max_rate(
        failures,
        "accepted_but_wrong_rate",
        _nested_estimate(summary, "accepted_but_wrong_rate"),
        config.max_accepted_but_wrong_rate,
    )
    _check_max_rate(
        failures,
        "retrieved_failure_rate",
        _nested_estimate(summary, "retrieved_failure_rate"),
        config.max_retrieved_failure_rate,
    )
    _check_max_rate(
        failures,
        "abstain_false_positive_rate",
        _nested_estimate(summary, "abstain_false_positive_rate"),
        config.max_abstain_false_positive_rate,
    )
    configured = any(
        value is not None
        for value in (
            config.min_matched_feedback_count,
            config.max_accepted_but_wrong_rate,
            config.max_retrieved_failure_rate,
            config.max_abstain_false_positive_rate,
        )
    )
    return {
        "configured": configured,
        "passed": not failures,
        "failures": tuple(failures),
        "blocking_reasons": tuple(failure["reason"] for failure in failures),
        "policy": {
            "min_matched_feedback_count": config.min_matched_feedback_count,
            "max_accepted_but_wrong_rate": config.max_accepted_but_wrong_rate,
            "max_retrieved_failure_rate": config.max_retrieved_failure_rate,
            "max_abstain_false_positive_rate": config.max_abstain_false_positive_rate,
        },
    }


def _check_min_count(
    failures: list[dict[str, Any]],
    metric: str,
    actual: Any,
    threshold: int | None,
) -> None:
    if threshold is None:
        return
    value = _int_or_none(actual)
    if value is None or value < threshold:
        failures.append({
            "metric": metric,
            "comparison": ">=",
            "threshold": threshold,
            "actual": value,
            "reason": f"{metric} below {threshold}",
        })


def _check_max_rate(
    failures: list[dict[str, Any]],
    metric: str,
    actual: float | None,
    threshold: float | None,
) -> None:
    if threshold is None:
        return
    if actual is None or actual > threshold:
        failures.append({
            "metric": metric,
            "comparison": "<=",
            "threshold": threshold,
            "actual": actual,
            "reason": f"{metric} above {threshold}",
        })


def _load_trace_record(path: str | Path) -> dict[str, Any]:
    trace_path = Path(path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"ProductTrace JSON must contain an object: {trace_path}")
    fingerprint = product_trace_fingerprint(payload)
    risk_decision = _mapping(payload.get("risk_decision"))
    action_names = _trace_action_names(payload)
    return {
        "path": str(trace_path),
        "request_id": payload.get("request_id"),
        "trace_fingerprint": fingerprint,
        "decision_action": risk_decision.get("action"),
        "risk_level": risk_decision.get("risk_level"),
        "action_names": action_names,
    }


def _trace_action_names(trace: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    risk_decision = _mapping(trace.get("risk_decision"))
    if risk_decision.get("action") is not None:
        names.append(str(risk_decision["action"]))
    for key in ("actions", "action_results"):
        values = trace.get(key)
        if not isinstance(values, Sequence) or isinstance(values, str | bytes):
            continue
        for value in values:
            if isinstance(value, Mapping) and value.get("action") is not None:
                names.append(str(value["action"]))
    return tuple(dict.fromkeys(names))


def _has_action(actions: Sequence[str], decision_action: str | None, action: str) -> bool:
    return decision_action == action or action in actions


def _write_artifact_manifest(
    config: ProductFeedbackReportConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "product_feedback_report": config.report_path,
        **{f"trace_{idx}": path for idx, path in enumerate(config.trace_paths, start=1)},
        **{f"feedback_{idx}": path for idx, path in enumerate(config.feedback_paths, start=1)},
    }
    manifest = build_artifact_manifest(
        artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_product_feedback_report",
            "status": report.get("status"),
            "matched_feedback_count": _nested(report, "summary", "trace_matched_feedback_count"),
            "unmatched_feedback_count": _nested(report, "summary", "unmatched_feedback_count"),
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(config: ProductFeedbackReportConfig, report: Mapping[str, Any]) -> None:
    assert config.registry_path is not None
    assert config.name is not None
    assert config.version is not None
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=config.name,
        version=config.version,
        path=config.report_path,
        metadata={
            "workflow": "product_feedback_report",
            "status": report.get("status"),
            "matched_feedback_count": _nested(report, "summary", "trace_matched_feedback_count"),
            "unmatched_feedback_count": _nested(report, "summary", "unmatched_feedback_count"),
            "accepted_but_wrong_rate": _nested(
                report,
                "summary",
                "accepted_but_wrong_rate",
                "estimate",
            ),
            "retrieved_failure_rate": _nested(
                report,
                "summary",
                "retrieved_failure_rate",
                "estimate",
            ),
            "abstain_false_positive_rate": _nested(
                report,
                "summary",
                "abstain_false_positive_rate",
                "estimate",
            ),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            **dict(config.metadata),
        },
    ).save_json()


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")


def _artifact_manifest_summary(config: ProductFeedbackReportConfig) -> dict[str, int]:
    return planned_artifact_manifest_summary(
        {
            "product_feedback_report": config.report_path,
            **{f"trace_{idx}": path for idx, path in enumerate(config.trace_paths, start=1)},
            **{f"feedback_{idx}": path for idx, path in enumerate(config.feedback_paths, start=1)},
        },
        assume_file_paths=(config.report_path,),
    )


def _trace_paths_from_args(args: argparse.Namespace) -> tuple[Path, ...]:
    paths = [Path(path) for path in args.trace]
    for pattern in args.trace_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    return tuple(paths)


def _config_from_args(args: argparse.Namespace) -> ProductFeedbackReportConfig:
    return ProductFeedbackReportConfig(
        trace_paths=_trace_paths_from_args(args),
        feedback_paths=tuple(Path(path) for path in args.feedback_jsonl),
        report_path=Path(args.json),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        metadata=_metadata_from_args(args.metadata),
        min_matched_feedback_count=args.min_matched_feedback_count,
        max_accepted_but_wrong_rate=args.max_accepted_but_wrong_rate,
        max_retrieved_failure_rate=args.max_retrieved_failure_rate,
        max_abstain_false_positive_rate=args.max_abstain_false_positive_rate,
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI args."""
    config = _config_from_args(args)
    report = build_product_feedback_report(config)
    print(
        "product_feedback_report="
        f"{report['status']} matched={report['summary']['trace_matched_feedback_count']} "
        f"unmatched={report['summary']['unmatched_feedback_count']} "
        f"accepted_but_wrong={report['summary']['accepted_but_wrong_count']}"
    )
    if args.fail_on_blocked and report["decision"]["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Join ProductTrace JSON with post-hoc feedback JSONL")
    parser.add_argument("--trace", action="append", default=[], help="ProductTrace JSON path; repeatable")
    parser.add_argument("--trace-glob", action="append", default=[], help="glob for ProductTrace JSON files")
    parser.add_argument("--feedback-jsonl", action="append", required=True,
                        help="ProductFeedbackRecord JSONL path; repeatable")
    parser.add_argument("--json", required=True, help="output report JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry report name")
    parser.add_argument("--version", default=None, help="registry report version")
    parser.add_argument("--metadata", action="append", default=[],
                        help="metadata key=value pair to include in report and registry; repeatable")
    parser.add_argument("--min-matched-feedback-count", type=int, default=None)
    parser.add_argument("--max-accepted-but-wrong-rate", type=float, default=None)
    parser.add_argument("--max-retrieved-failure-rate", type=float, default=None)
    parser.add_argument("--max-abstain-false-positive-rate", type=float, default=None)
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero when configured feedback gates fail")
    run(parser.parse_args(argv))


def _metadata_from_args(items: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--metadata entries must use key=value.")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        metadata[key] = value
    return metadata


def _optional_unit_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be between 0 and 1, not bool.")
    numeric = float(value)
    if not math.isfinite(numeric) or not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric


def _optional_non_negative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, not bool.")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _nested_estimate(mapping: Mapping[str, Any], key: str) -> float | None:
    value = _mapping(mapping.get(key)).get("estimate")
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    main()
