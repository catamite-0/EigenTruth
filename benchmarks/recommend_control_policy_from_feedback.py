"""Recommend control-policy updates from ProductTrace feedback reports."""

from __future__ import annotations

import argparse
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
from eigentruth.control import ControlAction, ControlPolicyConfig, RiskLevel  # noqa: E402
from eigentruth.eval.metrics import binomial_confidence_interval  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

_RATE_STATISTICS = frozenset({"estimate", "upper"})
_SENSITIVE_FEATURE_FLAGS = (
    "has_number",
    "has_citation",
    "is_time_sensitive",
    "has_calculation",
)
_SENSITIVE_METADATA_KEYS = (
    "requires_verification",
    "requires_current_facts",
    "requires_retrieval",
)


@dataclass(frozen=True)
class FeedbackPolicyRecommendationConfig:
    """Configuration for feedback-driven control-policy recommendations."""

    feedback_report_paths: Sequence[str | Path]
    output_path: str | Path
    base_control_policy_path: str | Path | None = None
    save_control_policy_path: str | Path | None = None
    save_control_defaults_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    min_matched_feedback_count: int = 20
    max_accepted_but_wrong_rate: float = 0.05
    max_retrieved_failure_rate: float = 0.10
    max_abstain_false_positive_rate: float = 0.20
    max_final_answered_but_wrong_rate: float | None = None
    max_final_answer_false_block_rate: float | None = None
    rate_statistic: str = "upper"
    compact_json: bool = False

    def __post_init__(self) -> None:
        feedback_report_paths = tuple(Path(path) for path in self.feedback_report_paths)
        if not feedback_report_paths:
            raise ValueError("at least one feedback report path is required.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        rate_statistic = str(self.rate_statistic)
        if rate_statistic not in _RATE_STATISTICS:
            choices = ", ".join(sorted(_RATE_STATISTICS))
            raise ValueError(f"rate_statistic must be one of: {choices}.")
        object.__setattr__(self, "feedback_report_paths", feedback_report_paths)
        object.__setattr__(self, "output_path", Path(self.output_path))
        _set_optional_path(self, "base_control_policy_path", self.base_control_policy_path)
        _set_optional_path(self, "save_control_policy_path", self.save_control_policy_path)
        _set_optional_path(self, "save_control_defaults_path", self.save_control_defaults_path)
        _set_optional_path(self, "artifact_manifest_path", self.artifact_manifest_path)
        _set_optional_path(self, "registry_path", self.registry_path)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "min_matched_feedback_count",
            _non_negative_int(
                self.min_matched_feedback_count,
                name="min_matched_feedback_count",
            ),
        )
        object.__setattr__(
            self,
            "max_accepted_but_wrong_rate",
            _unit_float(
                self.max_accepted_but_wrong_rate,
                name="max_accepted_but_wrong_rate",
            ),
        )
        object.__setattr__(
            self,
            "max_retrieved_failure_rate",
            _unit_float(
                self.max_retrieved_failure_rate,
                name="max_retrieved_failure_rate",
            ),
        )
        object.__setattr__(
            self,
            "max_abstain_false_positive_rate",
            _unit_float(
                self.max_abstain_false_positive_rate,
                name="max_abstain_false_positive_rate",
            ),
        )
        final_answered_threshold = (
            self.max_accepted_but_wrong_rate
            if self.max_final_answered_but_wrong_rate is None
            else self.max_final_answered_but_wrong_rate
        )
        object.__setattr__(
            self,
            "max_final_answered_but_wrong_rate",
            _unit_float(
                final_answered_threshold,
                name="max_final_answered_but_wrong_rate",
            ),
        )
        final_block_threshold = (
            self.max_abstain_false_positive_rate
            if self.max_final_answer_false_block_rate is None
            else self.max_final_answer_false_block_rate
        )
        object.__setattr__(
            self,
            "max_final_answer_false_block_rate",
            _unit_float(
                final_block_threshold,
                name="max_final_answer_false_block_rate",
            ),
        )
        object.__setattr__(self, "rate_statistic", rate_statistic)
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the report artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_path).with_name("feedback-policy-artifact-manifest.json")


def build_feedback_policy_recommendation(
    config: FeedbackPolicyRecommendationConfig,
) -> dict[str, Any]:
    """Build a control-policy recommendation from feedback report summaries."""
    reports = tuple(_load_json_object(path) for path in config.feedback_report_paths)
    aggregate = _aggregate_feedback_summaries(reports)
    base_policy = _load_base_policy(config)
    recommendation = _recommend_policy(
        aggregate,
        base_policy=base_policy,
        config=config,
    )
    report = {
        "schema_version": 1,
        "workflow": "feedback_policy_recommendation",
        "status": recommendation["status"],
        "decision": {
            "status": recommendation["status"],
            "reasons": tuple(recommendation["reasons"]),
            "tradeoffs": tuple(recommendation["tradeoffs"]),
        },
        "aggregate_feedback": aggregate,
        "recommendation": recommendation,
        "source_reports": tuple(
            _source_report_summary(path, report)
            for path, report in zip(config.feedback_report_paths, reports)
        ),
        "config": {
            "feedback_report_paths": tuple(str(path) for path in config.feedback_report_paths),
            "base_control_policy_path": (
                None if config.base_control_policy_path is None else str(config.base_control_policy_path)
            ),
            "min_matched_feedback_count": config.min_matched_feedback_count,
            "max_accepted_but_wrong_rate": config.max_accepted_but_wrong_rate,
            "max_retrieved_failure_rate": config.max_retrieved_failure_rate,
            "max_abstain_false_positive_rate": config.max_abstain_false_positive_rate,
            "max_final_answered_but_wrong_rate": config.max_final_answered_but_wrong_rate,
            "max_final_answer_false_block_rate": config.max_final_answer_false_block_rate,
            "rate_statistic": config.rate_statistic,
            "metadata": dict(config.metadata),
        },
        "paths": {
            "report": str(config.output_path),
            "control_policy": (
                None
                if config.save_control_policy_path is None
                else str(config.save_control_policy_path)
            ),
            "control_defaults": (
                None
                if config.save_control_defaults_path is None
                else str(config.save_control_defaults_path)
            ),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
        },
        "artifact_manifest_summary": _artifact_manifest_summary(config),
    }
    _write_optional_outputs(config, recommendation)
    _write_json(config.output_path, report, compact=config.compact_json)
    _write_artifact_manifest(config, report)
    if config.registry_path is not None:
        _record_registry(config, report)
    return report


def _recommend_policy(
    aggregate: Mapping[str, Any],
    *,
    base_policy: ControlPolicyConfig,
    config: FeedbackPolicyRecommendationConfig,
) -> dict[str, Any]:
    matched_count = int(aggregate.get("trace_matched_feedback_count", 0))
    rates = _mapping(aggregate.get("rates"))
    signals = {
        "accepted_but_wrong": _rate_signal(
            rates,
            "accepted_but_wrong_rate",
            threshold=config.max_accepted_but_wrong_rate,
            statistic=config.rate_statistic,
        ),
        "retrieved_failure": _rate_signal(
            rates,
            "retrieved_failure_rate",
            threshold=config.max_retrieved_failure_rate,
            statistic=config.rate_statistic,
        ),
        "abstain_false_positive": _rate_signal(
            rates,
            "abstain_false_positive_rate",
            threshold=config.max_abstain_false_positive_rate,
            statistic=config.rate_statistic,
        ),
        "final_answered_but_wrong": _rate_signal(
            rates,
            "final_answered_but_wrong_rate",
            threshold=config.max_final_answered_but_wrong_rate,
            statistic=config.rate_statistic,
        ),
        "final_answer_false_block": _rate_signal(
            rates,
            "final_answer_false_block_rate",
            threshold=config.max_final_answer_false_block_rate,
            statistic=config.rate_statistic,
        ),
    }
    insufficient_evidence = matched_count < config.min_matched_feedback_count
    policy = base_policy.to_dict()
    control_defaults: dict[str, Any] = {}
    reasons: list[str] = []
    tradeoffs: list[str] = []

    accepted_high = bool(signals["accepted_but_wrong"]["triggered"])
    retrieved_high = bool(signals["retrieved_failure"]["triggered"])
    abstain_fp_high = bool(signals["abstain_false_positive"]["triggered"])
    final_answered_high = bool(signals["final_answered_but_wrong"]["triggered"])
    final_block_high = bool(signals["final_answer_false_block"]["triggered"])
    safety_high = accepted_high or retrieved_high or final_answered_high
    overblock_high = abstain_fp_high or final_block_high

    if insufficient_evidence:
        reasons.append(
            f"matched feedback count {matched_count} below minimum {config.min_matched_feedback_count}"
        )

    if accepted_high:
        reasons.append("accepted answers have excessive wrong/unsupported feedback")
    if final_answered_high:
        reasons.append("final answered responses have excessive wrong/unsupported feedback")
    if accepted_high or final_answered_high:
        policy["compound_verification_escalates"] = True
        policy["compound_risk_action"] = ControlAction.ABSTAIN.value
        policy["compound_risk_level"] = RiskLevel.HIGH.value
        control_defaults.update({
            "staged_verification": True,
            "stage_verify_claim_feature_flags": _SENSITIVE_FEATURE_FLAGS,
            "stage_verify_claim_metadata_keys": _SENSITIVE_METADATA_KEYS,
            "stage_verify_triggered_claims_only": True,
            "max_verifier_route_attempts": 2,
        })

    if retrieved_high:
        reasons.append("retrieval actions still receive wrong/unsupported feedback")
        policy["unsupported_action"] = ControlAction.CLARIFY.value
        policy["unsupported_risk_level"] = RiskLevel.HIGH.value
        policy["compound_risk_action"] = ControlAction.ABSTAIN.value
        policy["compound_risk_level"] = RiskLevel.HIGH.value
        control_defaults["max_verifier_route_attempts"] = max(
            2,
            int(control_defaults.get("max_verifier_route_attempts", 0)),
        )

    if abstain_fp_high:
        reasons.append("abstain actions have excessive correct/unnecessary-block feedback")
    if final_block_high:
        reasons.append("final blocking responses have excessive correct/unnecessary-block feedback")
    if overblock_high and not safety_high:
        policy["compound_verification_escalates"] = False
        policy["compound_risk_action"] = ControlAction.RETRIEVE.value
        policy["unsupported_action"] = ControlAction.RETRIEVE.value
        policy["unsupported_risk_level"] = RiskLevel.MEDIUM.value
        control_defaults.update({
            "staged_verification": True,
            "stage_verify_claim_feature_flags": _SENSITIVE_FEATURE_FLAGS,
            "stage_verify_claim_metadata_keys": _SENSITIVE_METADATA_KEYS,
            "stage_verify_triggered_claims_only": True,
            "max_verifier_route_attempts": 1,
        })
    elif overblock_high:
        tradeoffs.append(
            "block false positives are elevated, but safety feedback is also elevated; "
            "recommendation keeps the stricter safety posture"
        )

    if not reasons:
        reasons.append("feedback rates are within configured thresholds")

    if insufficient_evidence:
        status = "needs_evidence"
    elif safety_high or overblock_high:
        status = "recommend"
    else:
        status = "observed"

    return {
        "status": status,
        "base_control_policy_config": base_policy.to_dict(),
        "candidate_control_policy_config": policy,
        "candidate_control_defaults": control_defaults,
        "rate_signals": signals,
        "reasons": tuple(reasons),
        "tradeoffs": tuple(tradeoffs),
        "next_workflows": (
            "examples/calibrated_control_demo.py",
            "benchmarks/run_product_runtime_profile_sweep.py",
            "benchmarks/run_product_trace_replay_workflow.py",
        ),
    }


def _aggregate_feedback_summaries(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        "trace_matched_feedback_count": 0,
        "unmatched_feedback_count": 0,
        "feedback_count": 0,
        "claim_level_feedback_count": 0,
        "accepted_feedback_count": 0,
        "accepted_but_wrong_count": 0,
        "retrieved_feedback_count": 0,
        "retrieved_failure_count": 0,
        "retrieved_but_still_unsupported_count": 0,
        "abstain_feedback_count": 0,
        "abstain_false_positive_count": 0,
        "final_answer_available_feedback_count": 0,
        "final_answer_unavailable_feedback_count": 0,
        "final_answered_feedback_count": 0,
        "final_answerable_feedback_count": 0,
        "final_answer_block_feedback_count": 0,
        "final_answered_but_wrong_count": 0,
        "final_answer_false_block_count": 0,
    }
    outcome_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    final_answer_status_counts: dict[str, int] = {}
    final_answer_action_counts: dict[str, int] = {}
    for report in reports:
        summary = _mapping(report.get("summary"))
        for key in counts:
            counts[key] += _int_value(summary.get(key))
        _merge_counts(outcome_counts, _mapping(summary.get("outcome_counts")))
        _merge_counts(action_counts, _mapping(summary.get("decision_action_counts")))
        _merge_counts(final_answer_status_counts, _mapping(summary.get("final_answer_status_counts")))
        _merge_counts(final_answer_action_counts, _mapping(summary.get("final_answer_action_counts")))
    return {
        **counts,
        "match_rate": binomial_confidence_interval(
            counts["trace_matched_feedback_count"],
            counts["feedback_count"],
        ),
        "outcome_counts": outcome_counts,
        "decision_action_counts": action_counts,
        "final_answer_status_counts": final_answer_status_counts,
        "final_answer_action_counts": final_answer_action_counts,
        "rates": {
            "accepted_but_wrong_rate": binomial_confidence_interval(
                counts["accepted_but_wrong_count"],
                counts["accepted_feedback_count"],
            ),
            "retrieved_failure_rate": binomial_confidence_interval(
                counts["retrieved_failure_count"],
                counts["retrieved_feedback_count"],
            ),
            "retrieved_but_still_unsupported_rate": binomial_confidence_interval(
                counts["retrieved_but_still_unsupported_count"],
                counts["retrieved_feedback_count"],
            ),
            "abstain_false_positive_rate": binomial_confidence_interval(
                counts["abstain_false_positive_count"],
                counts["abstain_feedback_count"],
            ),
            "final_answered_but_wrong_rate": binomial_confidence_interval(
                counts["final_answered_but_wrong_count"],
                counts["final_answered_feedback_count"],
            ),
            "final_answer_false_block_rate": binomial_confidence_interval(
                counts["final_answer_false_block_count"],
                counts["final_answer_block_feedback_count"],
            ),
        },
    }


def _rate_signal(
    rates: Mapping[str, Any],
    key: str,
    *,
    threshold: float,
    statistic: str,
) -> dict[str, Any]:
    interval = _mapping(rates.get(key))
    value = _finite_float(interval.get(statistic))
    estimate = _finite_float(interval.get("estimate"))
    total = _int_or_none(interval.get("total"))
    triggered = value is not None and value > threshold
    return {
        "metric": key,
        "threshold": threshold,
        "statistic": statistic,
        "statistic_value": value,
        "estimate": estimate,
        "interval": interval,
        "denominator": total,
        "triggered": triggered,
    }


def _load_base_policy(config: FeedbackPolicyRecommendationConfig) -> ControlPolicyConfig:
    if config.base_control_policy_path is None:
        return ControlPolicyConfig()
    payload = _load_json_object(config.base_control_policy_path)
    return ControlPolicyConfig.from_dict(payload)


def _write_optional_outputs(
    config: FeedbackPolicyRecommendationConfig,
    recommendation: Mapping[str, Any],
) -> None:
    if config.save_control_policy_path is not None:
        _write_json(
            config.save_control_policy_path,
            _mapping(recommendation.get("candidate_control_policy_config")),
            compact=config.compact_json,
        )
    if config.save_control_defaults_path is not None:
        _write_json(
            config.save_control_defaults_path,
            _mapping(recommendation.get("candidate_control_defaults")),
            compact=config.compact_json,
        )


def _write_artifact_manifest(
    config: FeedbackPolicyRecommendationConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = _artifact_paths(config)
    manifest = build_artifact_manifest(
        artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "recommend_control_policy_from_feedback",
            "status": report.get("status"),
            "matched_feedback_count": _nested(
                report,
                "aggregate_feedback",
                "trace_matched_feedback_count",
            ),
            "recommendation_count": len(_mapping(_nested(report, "recommendation", "candidate_control_defaults"))),
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(
    config: FeedbackPolicyRecommendationConfig,
    report: Mapping[str, Any],
) -> None:
    assert config.registry_path is not None
    assert config.name is not None
    assert config.version is not None
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=config.name,
        version=config.version,
        path=config.output_path,
        metadata={
            "workflow": "feedback_policy_recommendation",
            "status": report.get("status"),
            "matched_feedback_count": _nested(
                report,
                "aggregate_feedback",
                "trace_matched_feedback_count",
            ),
            "accepted_but_wrong_rate": _nested(
                report,
                "aggregate_feedback",
                "rates",
                "accepted_but_wrong_rate",
                "estimate",
            ),
            "retrieved_failure_rate": _nested(
                report,
                "aggregate_feedback",
                "rates",
                "retrieved_failure_rate",
                "estimate",
            ),
            "abstain_false_positive_rate": _nested(
                report,
                "aggregate_feedback",
                "rates",
                "abstain_false_positive_rate",
                "estimate",
            ),
            "final_answered_but_wrong_rate": _nested(
                report,
                "aggregate_feedback",
                "rates",
                "final_answered_but_wrong_rate",
                "estimate",
            ),
            "final_answer_false_block_rate": _nested(
                report,
                "aggregate_feedback",
                "rates",
                "final_answer_false_block_rate",
                "estimate",
            ),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "saved_control_policy": (
                None
                if config.save_control_policy_path is None
                else str(config.save_control_policy_path)
            ),
            "saved_control_defaults": (
                None
                if config.save_control_defaults_path is None
                else str(config.save_control_defaults_path)
            ),
            **dict(config.metadata),
        },
    ).save_json()


def _artifact_manifest_summary(config: FeedbackPolicyRecommendationConfig) -> dict[str, int]:
    return planned_artifact_manifest_summary(
        _artifact_paths(config),
        assume_file_paths=(
            config.output_path,
            *(() if config.save_control_policy_path is None else (config.save_control_policy_path,)),
            *(() if config.save_control_defaults_path is None else (config.save_control_defaults_path,)),
        ),
    )


def _artifact_paths(config: FeedbackPolicyRecommendationConfig) -> dict[str, str | Path | None]:
    return {
        "feedback_policy_recommendation": config.output_path,
        "candidate_control_policy": config.save_control_policy_path,
        "candidate_control_defaults": config.save_control_defaults_path,
        **{f"feedback_report_{idx}": path for idx, path in enumerate(config.feedback_report_paths, start=1)},
        "base_control_policy": config.base_control_policy_path,
    }


def _source_report_summary(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "workflow": report.get("workflow"),
        "status": report.get("status"),
        "matched_feedback_count": _nested(report, "summary", "trace_matched_feedback_count"),
        "unmatched_feedback_count": _nested(report, "summary", "unmatched_feedback_count"),
    }


def _config_from_args(args: argparse.Namespace) -> FeedbackPolicyRecommendationConfig:
    return FeedbackPolicyRecommendationConfig(
        feedback_report_paths=tuple(Path(path) for path in args.feedback_report),
        output_path=Path(args.json),
        base_control_policy_path=None if args.base_control_policy is None else Path(args.base_control_policy),
        save_control_policy_path=None if args.save_control_policy is None else Path(args.save_control_policy),
        save_control_defaults_path=None if args.save_control_defaults is None else Path(args.save_control_defaults),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        metadata=_metadata_from_args(args.metadata),
        min_matched_feedback_count=args.min_matched_feedback_count,
        max_accepted_but_wrong_rate=args.max_accepted_but_wrong_rate,
        max_retrieved_failure_rate=args.max_retrieved_failure_rate,
        max_abstain_false_positive_rate=args.max_abstain_false_positive_rate,
        max_final_answered_but_wrong_rate=args.max_final_answered_but_wrong_rate,
        max_final_answer_false_block_rate=args.max_final_answer_false_block_rate,
        rate_statistic=args.rate_statistic,
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI args."""
    config = _config_from_args(args)
    report = build_feedback_policy_recommendation(config)
    print(
        "feedback_policy_recommendation="
        f"{report['status']} matched={report['aggregate_feedback']['trace_matched_feedback_count']} "
        f"reasons={len(report['decision']['reasons'])}"
    )
    if args.fail_on_needs_evidence and report["decision"]["status"] == "needs_evidence":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recommend control policy updates from ProductTrace feedback reports"
    )
    parser.add_argument("--feedback-report", action="append", required=True,
                        help="product feedback report JSON path; repeatable")
    parser.add_argument("--json", required=True, help="output recommendation JSON path")
    parser.add_argument("--base-control-policy", default=None,
                        help="optional base ControlPolicyConfig JSON path")
    parser.add_argument("--save-control-policy", default=None,
                        help="optional path for candidate ControlPolicyConfig JSON")
    parser.add_argument("--save-control-defaults", default=None,
                        help="optional path for candidate runtime control defaults JSON")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry report name")
    parser.add_argument("--version", default=None, help="registry report version")
    parser.add_argument("--metadata", action="append", default=[],
                        help="metadata key=value pair to include in report and registry; repeatable")
    parser.add_argument("--min-matched-feedback-count", type=int, default=20)
    parser.add_argument("--max-accepted-but-wrong-rate", type=float, default=0.05)
    parser.add_argument("--max-retrieved-failure-rate", type=float, default=0.10)
    parser.add_argument("--max-abstain-false-positive-rate", type=float, default=0.20)
    parser.add_argument("--max-final-answered-but-wrong-rate", type=float, default=None)
    parser.add_argument("--max-final-answer-false-block-rate", type=float, default=None)
    parser.add_argument("--rate-statistic", choices=sorted(_RATE_STATISTICS), default="upper",
                        help="rate field used for recommendation triggers")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    parser.add_argument("--fail-on-needs-evidence", action="store_true",
                        help="exit non-zero when matched feedback is below the configured minimum")
    run(parser.parse_args(argv))


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.write_text(text, encoding="utf-8")


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


def _set_optional_path(instance: Any, field_name: str, value: str | Path | None) -> None:
    if value is not None:
        object.__setattr__(instance, field_name, Path(value))


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, not bool.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdecimal():
            raise ValueError(f"{name} must be a non-negative integer.")
        parsed = int(stripped)
    else:
        raise ValueError(f"{name} must be a non-negative integer.")
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _unit_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be between 0 and 1, not bool.")
    numeric = float(value)
    if not math.isfinite(numeric) or not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: Any) -> int:
    parsed = _int_or_none(value)
    return 0 if parsed is None else parsed


def _merge_counts(target: dict[str, int], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        target[str(key)] = target.get(str(key), 0) + _int_value(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


if __name__ == "__main__":
    main()
