"""Run the post-hoc feedback-to-policy workflow end to end.

This workflow chains:

1. ProductTrace + ProductFeedbackRecord join report.
2. Feedback-derived ControlPolicyConfig recommendation.
3. Counterfactual replay audit over historical feedback labels.

It performs no model, verifier, retrieval, database, or network work.
"""

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

from benchmarks.audit_feedback_policy_replay import (  # noqa: E402
    FeedbackPolicyReplayAuditConfig,
    build_feedback_policy_replay_audit,
)
from benchmarks.config_utils import planned_artifact_manifest_summary, strict_bool  # noqa: E402
from benchmarks.recommend_control_policy_from_feedback import (  # noqa: E402
    FeedbackPolicyRecommendationConfig,
    build_feedback_policy_recommendation,
)
from benchmarks.run_product_feedback_report import (  # noqa: E402
    ProductFeedbackReportConfig,
    build_product_feedback_report,
)
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

_RATE_STATISTICS = frozenset({"estimate", "upper"})
_PASSING_FEEDBACK_STATUSES = frozenset({"observed", "passed"})


@dataclass(frozen=True)
class FeedbackPolicyWorkflowConfig:
    """Configuration for the full post-hoc feedback policy workflow."""

    trace_paths: Sequence[str | Path] = ()
    feedback_paths: Sequence[str | Path] = ()
    feedback_report_path: str | Path | None = None
    output_dir: str | Path = "artifacts/feedback_policy_workflow"
    workflow_report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    base_control_policy_path: str | Path | None = None
    candidate_control_policy_path: str | Path | None = None
    candidate_control_defaults_path: str | Path | None = None
    product_feedback_manifest_path: str | Path | None = None
    policy_recommendation_manifest_path: str | Path | None = None
    replay_audit_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    feedback_min_matched_feedback_count: int | None = None
    feedback_max_accepted_but_wrong_rate: float | None = None
    feedback_max_retrieved_failure_rate: float | None = None
    feedback_max_abstain_false_positive_rate: float | None = None
    recommendation_min_matched_feedback_count: int = 20
    recommendation_max_accepted_but_wrong_rate: float = 0.05
    recommendation_max_retrieved_failure_rate: float = 0.10
    recommendation_max_abstain_false_positive_rate: float = 0.20
    replay_min_matched_feedback_count: int = 20
    min_safety_coverage: float = 0.50
    max_unknown_safety_issue_rate: float = 0.50
    rate_statistic: str = "upper"
    compact_json: bool = False

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        feedback_paths = tuple(Path(path) for path in self.feedback_paths)
        feedback_report_path = None if self.feedback_report_path is None else Path(self.feedback_report_path)
        if feedback_report_path is None:
            if not trace_paths:
                raise ValueError("trace_paths are required when feedback_report_path is not provided.")
            if not feedback_paths:
                raise ValueError("feedback_paths are required when feedback_report_path is not provided.")
        elif trace_paths or feedback_paths:
            raise ValueError("feedback_report_path is mutually exclusive with trace_paths and feedback_paths.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        rate_statistic = str(self.rate_statistic)
        if rate_statistic not in _RATE_STATISTICS:
            choices = ", ".join(sorted(_RATE_STATISTICS))
            raise ValueError(f"rate_statistic must be one of: {choices}.")

        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "feedback_paths", feedback_paths)
        object.__setattr__(self, "feedback_report_path", feedback_report_path)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for field_name in (
            "workflow_report_path",
            "artifact_manifest_path",
            "base_control_policy_path",
            "candidate_control_policy_path",
            "candidate_control_defaults_path",
            "product_feedback_manifest_path",
            "policy_recommendation_manifest_path",
            "replay_audit_manifest_path",
            "registry_path",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "feedback_min_matched_feedback_count",
            _optional_non_negative_int(
                self.feedback_min_matched_feedback_count,
                name="feedback_min_matched_feedback_count",
            ),
        )
        object.__setattr__(
            self,
            "feedback_max_accepted_but_wrong_rate",
            _optional_unit_float(
                self.feedback_max_accepted_but_wrong_rate,
                name="feedback_max_accepted_but_wrong_rate",
            ),
        )
        object.__setattr__(
            self,
            "feedback_max_retrieved_failure_rate",
            _optional_unit_float(
                self.feedback_max_retrieved_failure_rate,
                name="feedback_max_retrieved_failure_rate",
            ),
        )
        object.__setattr__(
            self,
            "feedback_max_abstain_false_positive_rate",
            _optional_unit_float(
                self.feedback_max_abstain_false_positive_rate,
                name="feedback_max_abstain_false_positive_rate",
            ),
        )
        object.__setattr__(
            self,
            "recommendation_min_matched_feedback_count",
            _non_negative_int(
                self.recommendation_min_matched_feedback_count,
                name="recommendation_min_matched_feedback_count",
            ),
        )
        object.__setattr__(
            self,
            "recommendation_max_accepted_but_wrong_rate",
            _unit_float(
                self.recommendation_max_accepted_but_wrong_rate,
                name="recommendation_max_accepted_but_wrong_rate",
            ),
        )
        object.__setattr__(
            self,
            "recommendation_max_retrieved_failure_rate",
            _unit_float(
                self.recommendation_max_retrieved_failure_rate,
                name="recommendation_max_retrieved_failure_rate",
            ),
        )
        object.__setattr__(
            self,
            "recommendation_max_abstain_false_positive_rate",
            _unit_float(
                self.recommendation_max_abstain_false_positive_rate,
                name="recommendation_max_abstain_false_positive_rate",
            ),
        )
        object.__setattr__(
            self,
            "replay_min_matched_feedback_count",
            _non_negative_int(
                self.replay_min_matched_feedback_count,
                name="replay_min_matched_feedback_count",
            ),
        )
        object.__setattr__(
            self,
            "min_safety_coverage",
            _unit_float(self.min_safety_coverage, name="min_safety_coverage"),
        )
        object.__setattr__(
            self,
            "max_unknown_safety_issue_rate",
            _unit_float(
                self.max_unknown_safety_issue_rate,
                name="max_unknown_safety_issue_rate",
            ),
        )
        object.__setattr__(self, "rate_statistic", rate_statistic)
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))

    @property
    def resolved_workflow_report_path(self) -> Path:
        """Return the top-level workflow report path."""
        if self.workflow_report_path is not None:
            return Path(self.workflow_report_path)
        return Path(self.output_dir) / "feedback-policy-workflow.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the top-level workflow artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_dir) / "artifact-manifest.json"

    @property
    def resolved_product_feedback_report_path(self) -> Path:
        """Return the generated or reused product feedback report path."""
        if self.feedback_report_path is not None:
            return Path(self.feedback_report_path)
        return Path(self.output_dir) / "product-feedback-report.json"

    @property
    def resolved_product_feedback_manifest_path(self) -> Path | None:
        """Return the generated product feedback manifest path, if this workflow creates one."""
        if self.feedback_report_path is not None:
            return None
        if self.product_feedback_manifest_path is not None:
            return Path(self.product_feedback_manifest_path)
        return Path(self.output_dir) / "product-feedback-manifest.json"

    @property
    def resolved_policy_recommendation_path(self) -> Path:
        """Return the feedback policy recommendation report path."""
        return Path(self.output_dir) / "feedback-policy-recommendation.json"

    @property
    def resolved_policy_recommendation_manifest_path(self) -> Path:
        """Return the feedback policy recommendation manifest path."""
        if self.policy_recommendation_manifest_path is not None:
            return Path(self.policy_recommendation_manifest_path)
        return Path(self.output_dir) / "feedback-policy-recommendation-manifest.json"

    @property
    def resolved_replay_audit_path(self) -> Path:
        """Return the replay audit report path."""
        return Path(self.output_dir) / "feedback-policy-replay-audit.json"

    @property
    def resolved_replay_audit_manifest_path(self) -> Path:
        """Return the replay audit manifest path."""
        if self.replay_audit_manifest_path is not None:
            return Path(self.replay_audit_manifest_path)
        return Path(self.output_dir) / "feedback-policy-replay-manifest.json"

    @property
    def resolved_candidate_control_policy_path(self) -> Path:
        """Return the candidate ControlPolicyConfig artifact path."""
        if self.candidate_control_policy_path is not None:
            return Path(self.candidate_control_policy_path)
        return Path(self.output_dir) / "candidate-control-policy.json"

    @property
    def resolved_candidate_control_defaults_path(self) -> Path:
        """Return the candidate runtime control defaults artifact path."""
        if self.candidate_control_defaults_path is not None:
            return Path(self.candidate_control_defaults_path)
        return Path(self.output_dir) / "candidate-control-defaults.json"


def run_feedback_policy_workflow(config: FeedbackPolicyWorkflowConfig) -> dict[str, Any]:
    """Run feedback report, policy recommendation, and replay audit in sequence."""
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    metadata = {"parent_workflow": "feedback_policy_workflow", **dict(config.metadata)}
    feedback_report = _build_or_load_feedback_report(config, metadata=metadata)
    feedback_report_path = config.resolved_product_feedback_report_path
    recommendation = build_feedback_policy_recommendation(
        FeedbackPolicyRecommendationConfig(
            feedback_report_paths=(feedback_report_path,),
            output_path=config.resolved_policy_recommendation_path,
            base_control_policy_path=config.base_control_policy_path,
            save_control_policy_path=config.resolved_candidate_control_policy_path,
            save_control_defaults_path=config.resolved_candidate_control_defaults_path,
            artifact_manifest_path=config.resolved_policy_recommendation_manifest_path,
            metadata=metadata,
            min_matched_feedback_count=config.recommendation_min_matched_feedback_count,
            max_accepted_but_wrong_rate=config.recommendation_max_accepted_but_wrong_rate,
            max_retrieved_failure_rate=config.recommendation_max_retrieved_failure_rate,
            max_abstain_false_positive_rate=config.recommendation_max_abstain_false_positive_rate,
            rate_statistic=config.rate_statistic,
            compact_json=config.compact_json,
        )
    )
    replay_audit = build_feedback_policy_replay_audit(
        FeedbackPolicyReplayAuditConfig(
            feedback_report_path=feedback_report_path,
            policy_recommendation_path=config.resolved_policy_recommendation_path,
            output_path=config.resolved_replay_audit_path,
            control_policy_path=config.resolved_candidate_control_policy_path,
            control_defaults_path=config.resolved_candidate_control_defaults_path,
            artifact_manifest_path=config.resolved_replay_audit_manifest_path,
            metadata=metadata,
            min_matched_feedback_count=config.replay_min_matched_feedback_count,
            min_safety_coverage=config.min_safety_coverage,
            max_unknown_safety_issue_rate=config.max_unknown_safety_issue_rate,
            compact_json=config.compact_json,
        )
    )
    decision = _workflow_decision(feedback_report, recommendation, replay_audit)
    report = {
        "schema_version": 1,
        "workflow": "feedback_policy_workflow",
        "status": decision["status"],
        "decision": decision,
        "children": {
            "product_feedback_report": _child_summary(
                config.resolved_product_feedback_report_path,
                feedback_report,
            ),
            "feedback_policy_recommendation": _child_summary(
                config.resolved_policy_recommendation_path,
                recommendation,
            ),
            "feedback_policy_replay_audit": _child_summary(
                config.resolved_replay_audit_path,
                replay_audit,
            ),
        },
        "feedback_summary": feedback_report.get("summary"),
        "recommendation_summary": {
            "status": recommendation.get("status"),
            "reasons": tuple(_sequence(_nested(recommendation, "decision", "reasons"))),
            "tradeoffs": tuple(_sequence(_nested(recommendation, "decision", "tradeoffs"))),
            "rate_signals": _nested(recommendation, "recommendation", "rate_signals"),
        },
        "replay_summary": replay_audit.get("summary"),
        "config": {
            "feedback_report_path": str(config.feedback_report_path) if config.feedback_report_path else None,
            "trace_paths": tuple(str(path) for path in config.trace_paths),
            "feedback_paths": tuple(str(path) for path in config.feedback_paths),
            "base_control_policy_path": (
                None if config.base_control_policy_path is None else str(config.base_control_policy_path)
            ),
            "feedback_gates": {
                "min_matched_feedback_count": config.feedback_min_matched_feedback_count,
                "max_accepted_but_wrong_rate": config.feedback_max_accepted_but_wrong_rate,
                "max_retrieved_failure_rate": config.feedback_max_retrieved_failure_rate,
                "max_abstain_false_positive_rate": config.feedback_max_abstain_false_positive_rate,
            },
            "recommendation_gates": {
                "min_matched_feedback_count": config.recommendation_min_matched_feedback_count,
                "max_accepted_but_wrong_rate": config.recommendation_max_accepted_but_wrong_rate,
                "max_retrieved_failure_rate": config.recommendation_max_retrieved_failure_rate,
                "max_abstain_false_positive_rate": config.recommendation_max_abstain_false_positive_rate,
                "rate_statistic": config.rate_statistic,
            },
            "replay_gates": {
                "min_matched_feedback_count": config.replay_min_matched_feedback_count,
                "min_safety_coverage": config.min_safety_coverage,
                "max_unknown_safety_issue_rate": config.max_unknown_safety_issue_rate,
            },
            "metadata": dict(config.metadata),
        },
        "paths": _workflow_paths(config, feedback_report),
        "artifact_manifest_summary": _artifact_manifest_summary(config, feedback_report),
    }
    _write_json(config.resolved_workflow_report_path, report, compact=config.compact_json)
    _write_artifact_manifest(config, report, feedback_report)
    if config.registry_path is not None:
        _record_registry(config, report)
    return report


def _build_or_load_feedback_report(
    config: FeedbackPolicyWorkflowConfig,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if config.feedback_report_path is not None:
        return _load_json_object(config.feedback_report_path)
    product_manifest_path = config.resolved_product_feedback_manifest_path
    assert product_manifest_path is not None
    return build_product_feedback_report(
        ProductFeedbackReportConfig(
            trace_paths=config.trace_paths,
            feedback_paths=config.feedback_paths,
            report_path=config.resolved_product_feedback_report_path,
            artifact_manifest_path=product_manifest_path,
            metadata=metadata,
            min_matched_feedback_count=config.feedback_min_matched_feedback_count,
            max_accepted_but_wrong_rate=config.feedback_max_accepted_but_wrong_rate,
            max_retrieved_failure_rate=config.feedback_max_retrieved_failure_rate,
            max_abstain_false_positive_rate=config.feedback_max_abstain_false_positive_rate,
            compact_json=config.compact_json,
        )
    )


def _workflow_decision(
    feedback_report: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    replay_audit: Mapping[str, Any],
) -> dict[str, Any]:
    feedback_status = str(feedback_report.get("status", "unknown"))
    recommendation_status = str(recommendation.get("status", "unknown"))
    replay_status = str(replay_audit.get("status", "unknown"))
    blocking_reasons: list[str] = []
    evidence_reasons: list[str] = []

    if feedback_status == "blocked":
        blocking_reasons.extend(_decision_reasons(feedback_report, fallback="feedback report blocked"))
    elif feedback_status not in _PASSING_FEEDBACK_STATUSES:
        blocking_reasons.append(f"feedback report status is {feedback_status}")
    if replay_status == "blocked":
        blocking_reasons.extend(_decision_reasons(replay_audit, fallback="replay audit blocked"))
    elif replay_status not in {"passed", "needs_evidence"}:
        blocking_reasons.append(f"replay audit status is {replay_status}")
    if recommendation_status == "needs_evidence":
        evidence_reasons.extend(_decision_reasons(recommendation, fallback="policy recommendation needs evidence"))
    elif recommendation_status not in {"recommend", "observed"}:
        blocking_reasons.append(f"policy recommendation status is {recommendation_status}")
    if replay_status == "needs_evidence":
        evidence_reasons.extend(_decision_reasons(replay_audit, fallback="replay audit needs evidence"))

    if blocking_reasons:
        status = "blocked"
        promotion_decision = "do_not_promote"
    elif evidence_reasons:
        status = "needs_evidence"
        promotion_decision = "collect_more_feedback"
    elif recommendation_status == "recommend":
        status = "recommend"
        promotion_decision = "promote_candidate_policy"
    else:
        status = "observed"
        promotion_decision = "keep_current_policy"

    return {
        "status": status,
        "promotion_decision": promotion_decision,
        "feedback_report_status": feedback_status,
        "policy_recommendation_status": recommendation_status,
        "replay_audit_status": replay_status,
        "candidate_control_policy": _nested(
            recommendation,
            "paths",
            "control_policy",
        ),
        "candidate_control_defaults": _nested(
            recommendation,
            "paths",
            "control_defaults",
        ),
        "matched_feedback_count": _nested(
            feedback_report,
            "summary",
            "trace_matched_feedback_count",
        ),
        "safety_coverage_rate": _nested(
            replay_audit,
            "summary",
            "safety_coverage_rate",
            "estimate",
        ),
        "unknown_safety_issue_rate": _nested(
            replay_audit,
            "summary",
            "unknown_safety_issue_rate",
            "estimate",
        ),
        "blocking_reasons": tuple(blocking_reasons),
        "evidence_reasons": tuple(evidence_reasons),
    }


def _decision_reasons(report: Mapping[str, Any], *, fallback: str) -> list[str]:
    decision = _mapping(report.get("decision"))
    reasons = []
    for key in ("blocking_reasons", "reasons"):
        reasons.extend(str(value) for value in _sequence(decision.get(key)))
    return reasons or [fallback]


def _child_summary(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "workflow": report.get("workflow"),
        "status": report.get("status"),
        "artifact_manifest": _nested(report, "paths", "artifact_manifest"),
    }


def _workflow_paths(
    config: FeedbackPolicyWorkflowConfig,
    feedback_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "workflow_report": str(config.resolved_workflow_report_path),
        "artifact_manifest": str(config.resolved_artifact_manifest_path),
        "product_feedback_report": str(config.resolved_product_feedback_report_path),
        "product_feedback_manifest": _optional_path_string(
            _resolved_feedback_manifest_path(config, feedback_report)
        ),
        "feedback_policy_recommendation": str(config.resolved_policy_recommendation_path),
        "candidate_control_policy": str(config.resolved_candidate_control_policy_path),
        "candidate_control_defaults": str(config.resolved_candidate_control_defaults_path),
        "feedback_policy_recommendation_manifest": str(
            config.resolved_policy_recommendation_manifest_path
        ),
        "feedback_policy_replay_audit": str(config.resolved_replay_audit_path),
        "feedback_policy_replay_manifest": str(config.resolved_replay_audit_manifest_path),
    }


def _write_artifact_manifest(
    config: FeedbackPolicyWorkflowConfig,
    report: Mapping[str, Any],
    feedback_report: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(config, feedback_report),
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "run_feedback_policy_workflow",
            "status": report.get("status"),
            "promotion_decision": _nested(report, "decision", "promotion_decision"),
            "matched_feedback_count": _nested(report, "decision", "matched_feedback_count"),
            "accepted_but_wrong_rate": _nested(
                report,
                "feedback_summary",
                "accepted_but_wrong_rate",
                "estimate",
            ),
            "retrieved_failure_rate": _nested(
                report,
                "feedback_summary",
                "retrieved_failure_rate",
                "estimate",
            ),
            "abstain_false_positive_rate": _nested(
                report,
                "feedback_summary",
                "abstain_false_positive_rate",
                "estimate",
            ),
            "safety_coverage_rate": _nested(report, "decision", "safety_coverage_rate"),
            "unknown_safety_issue_rate": _nested(report, "decision", "unknown_safety_issue_rate"),
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(config: FeedbackPolicyWorkflowConfig, report: Mapping[str, Any]) -> None:
    assert config.registry_path is not None
    assert config.name is not None
    assert config.version is not None
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=config.name,
        version=config.version,
        path=config.resolved_workflow_report_path,
        metadata={
            "workflow": "feedback_policy_workflow",
            "status": report.get("status"),
            "promotion_decision": _nested(report, "decision", "promotion_decision"),
            "feedback_report_status": _nested(report, "decision", "feedback_report_status"),
            "policy_recommendation_status": _nested(report, "decision", "policy_recommendation_status"),
            "replay_audit_status": _nested(report, "decision", "replay_audit_status"),
            "matched_feedback_count": _nested(report, "decision", "matched_feedback_count"),
            "accepted_but_wrong_rate": _nested(
                report,
                "feedback_summary",
                "accepted_but_wrong_rate",
                "estimate",
            ),
            "retrieved_failure_rate": _nested(
                report,
                "feedback_summary",
                "retrieved_failure_rate",
                "estimate",
            ),
            "abstain_false_positive_rate": _nested(
                report,
                "feedback_summary",
                "abstain_false_positive_rate",
                "estimate",
            ),
            "safety_coverage_rate": _nested(report, "decision", "safety_coverage_rate"),
            "unknown_safety_issue_rate": _nested(report, "decision", "unknown_safety_issue_rate"),
            "candidate_control_policy": _nested(report, "paths", "candidate_control_policy"),
            "candidate_control_defaults": _nested(report, "paths", "candidate_control_defaults"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            **dict(config.metadata),
        },
    ).save_json()


def _artifact_manifest_summary(
    config: FeedbackPolicyWorkflowConfig,
    feedback_report: Mapping[str, Any],
) -> dict[str, int]:
    return planned_artifact_manifest_summary(
        _artifact_paths(config, feedback_report),
        assume_file_paths=(
            config.resolved_workflow_report_path,
            config.resolved_policy_recommendation_path,
            config.resolved_candidate_control_policy_path,
            config.resolved_candidate_control_defaults_path,
            config.resolved_replay_audit_path,
            *(() if config.feedback_report_path is not None else (config.resolved_product_feedback_report_path,)),
            *(
                ()
                if config.resolved_product_feedback_manifest_path is None
                else (config.resolved_product_feedback_manifest_path,)
            ),
            config.resolved_policy_recommendation_manifest_path,
            config.resolved_replay_audit_manifest_path,
        ),
    )


def _artifact_paths(
    config: FeedbackPolicyWorkflowConfig,
    feedback_report: Mapping[str, Any],
) -> dict[str, str | Path | None]:
    return {
        "feedback_policy_workflow": config.resolved_workflow_report_path,
        "product_feedback_report": config.resolved_product_feedback_report_path,
        "product_feedback_manifest": _resolved_feedback_manifest_path(config, feedback_report),
        "feedback_policy_recommendation": config.resolved_policy_recommendation_path,
        "candidate_control_policy": config.resolved_candidate_control_policy_path,
        "candidate_control_defaults": config.resolved_candidate_control_defaults_path,
        "feedback_policy_recommendation_manifest": config.resolved_policy_recommendation_manifest_path,
        "feedback_policy_replay_audit": config.resolved_replay_audit_path,
        "feedback_policy_replay_manifest": config.resolved_replay_audit_manifest_path,
    }


def _resolved_feedback_manifest_path(
    config: FeedbackPolicyWorkflowConfig,
    feedback_report: Mapping[str, Any],
) -> str | Path | None:
    generated_path = config.resolved_product_feedback_manifest_path
    if generated_path is not None:
        return generated_path
    path = _nested(feedback_report, "paths", "artifact_manifest")
    return None if path is None else str(path)


def _optional_path_string(path: str | Path | None) -> str | None:
    return None if path is None else str(path)


def _config_from_args(args: argparse.Namespace) -> FeedbackPolicyWorkflowConfig:
    trace_paths = _expand_paths(args.trace, args.trace_glob)
    return FeedbackPolicyWorkflowConfig(
        trace_paths=trace_paths,
        feedback_paths=tuple(Path(path) for path in args.feedback_jsonl),
        feedback_report_path=None if args.feedback_report is None else Path(args.feedback_report),
        output_dir=Path(args.output_dir),
        workflow_report_path=None if args.json is None else Path(args.json),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        base_control_policy_path=None if args.base_control_policy is None else Path(args.base_control_policy),
        candidate_control_policy_path=(
            None if args.save_control_policy is None else Path(args.save_control_policy)
        ),
        candidate_control_defaults_path=(
            None if args.save_control_defaults is None else Path(args.save_control_defaults)
        ),
        product_feedback_manifest_path=(
            None if args.product_feedback_manifest is None else Path(args.product_feedback_manifest)
        ),
        policy_recommendation_manifest_path=(
            None
            if args.policy_recommendation_manifest is None
            else Path(args.policy_recommendation_manifest)
        ),
        replay_audit_manifest_path=(
            None if args.replay_audit_manifest is None else Path(args.replay_audit_manifest)
        ),
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        metadata=_metadata_from_args(args.metadata),
        feedback_min_matched_feedback_count=args.feedback_min_matched_feedback_count,
        feedback_max_accepted_but_wrong_rate=args.feedback_max_accepted_but_wrong_rate,
        feedback_max_retrieved_failure_rate=args.feedback_max_retrieved_failure_rate,
        feedback_max_abstain_false_positive_rate=args.feedback_max_abstain_false_positive_rate,
        recommendation_min_matched_feedback_count=args.recommendation_min_matched_feedback_count,
        recommendation_max_accepted_but_wrong_rate=args.recommendation_max_accepted_but_wrong_rate,
        recommendation_max_retrieved_failure_rate=args.recommendation_max_retrieved_failure_rate,
        recommendation_max_abstain_false_positive_rate=args.recommendation_max_abstain_false_positive_rate,
        replay_min_matched_feedback_count=args.replay_min_matched_feedback_count,
        min_safety_coverage=args.min_safety_coverage,
        max_unknown_safety_issue_rate=args.max_unknown_safety_issue_rate,
        rate_statistic=args.rate_statistic,
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI args."""
    report = run_feedback_policy_workflow(_config_from_args(args))
    print(
        "feedback_policy_workflow="
        f"{report['status']} promotion={report['decision']['promotion_decision']} "
        f"feedback={report['decision']['feedback_report_status']} "
        f"recommendation={report['decision']['policy_recommendation_status']} "
        f"replay={report['decision']['replay_audit_status']}"
    )
    if args.fail_on_blocked and report["decision"]["status"] == "blocked":
        raise SystemExit(1)
    if args.fail_on_needs_evidence and report["decision"]["status"] == "needs_evidence":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the ProductTrace feedback-to-policy workflow end to end"
    )
    parser.add_argument("--trace", action="append", default=[],
                        help="ProductTrace JSON path; repeatable")
    parser.add_argument("--trace-glob", action="append", default=[],
                        help="glob for ProductTrace JSON paths; repeatable")
    parser.add_argument("--feedback-jsonl", action="append", default=[],
                        help="ProductFeedbackRecord JSONL path; repeatable")
    parser.add_argument("--feedback-report", default=None,
                        help="reuse an existing product feedback report instead of joining traces")
    parser.add_argument("--output-dir", required=True, help="directory for generated workflow artifacts")
    parser.add_argument("--json", default=None, help="optional top-level workflow report path")
    parser.add_argument("--artifact-manifest", default=None, help="optional top-level artifact manifest path")
    parser.add_argument("--base-control-policy", default=None,
                        help="optional base ControlPolicyConfig JSON path")
    parser.add_argument("--save-control-policy", default=None,
                        help="optional path for candidate ControlPolicyConfig JSON")
    parser.add_argument("--save-control-defaults", default=None,
                        help="optional path for candidate runtime control defaults JSON")
    parser.add_argument("--product-feedback-manifest", default=None,
                        help="optional child product feedback manifest path")
    parser.add_argument("--policy-recommendation-manifest", default=None,
                        help="optional child policy recommendation manifest path")
    parser.add_argument("--replay-audit-manifest", default=None,
                        help="optional child replay audit manifest path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry report name")
    parser.add_argument("--version", default=None, help="registry report version")
    parser.add_argument("--metadata", action="append", default=[],
                        help="metadata key=value pair for report and registry; repeatable")
    parser.add_argument("--feedback-min-matched-feedback-count", type=int, default=None)
    parser.add_argument("--feedback-max-accepted-but-wrong-rate", type=float, default=None)
    parser.add_argument("--feedback-max-retrieved-failure-rate", type=float, default=None)
    parser.add_argument("--feedback-max-abstain-false-positive-rate", type=float, default=None)
    parser.add_argument("--recommendation-min-matched-feedback-count", type=int, default=20)
    parser.add_argument("--recommendation-max-accepted-but-wrong-rate", type=float, default=0.05)
    parser.add_argument("--recommendation-max-retrieved-failure-rate", type=float, default=0.10)
    parser.add_argument("--recommendation-max-abstain-false-positive-rate", type=float, default=0.20)
    parser.add_argument("--replay-min-matched-feedback-count", type=int, default=20)
    parser.add_argument("--min-safety-coverage", type=float, default=0.50)
    parser.add_argument("--max-unknown-safety-issue-rate", type=float, default=0.50)
    parser.add_argument("--rate-statistic", choices=sorted(_RATE_STATISTICS), default="upper")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero when the workflow blocks promotion")
    parser.add_argument("--fail-on-needs-evidence", action="store_true",
                        help="exit non-zero when feedback evidence is insufficient")
    run(parser.parse_args(argv))


def _expand_paths(paths: Sequence[str], patterns: Sequence[str]) -> tuple[Path, ...]:
    resolved = [Path(path) for path in paths]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"trace glob matched no files: {pattern}")
        resolved.extend(Path(path) for path in matches)
    return tuple(dict.fromkeys(resolved))


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _optional_non_negative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name=name)


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


def _optional_unit_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _unit_float(value, name=name)


def _unit_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float in [0, 1], not bool.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float in [0, 1].") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1].")
    return parsed


if __name__ == "__main__":
    main()
