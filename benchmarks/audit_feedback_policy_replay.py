"""Audit a feedback-derived control policy against historical feedback labels."""

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
from eigentruth.control import ControlPolicyConfig  # noqa: E402
from eigentruth.eval.metrics import binomial_confidence_interval  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

_SAFER_UNSUPPORTED_ACTIONS = frozenset({"abstain", "clarify"})
_DEESCALATING_ACTIONS = frozenset({"retrieve", "rewrite", "clarify"})
_DEFAULT_SENSITIVE_FEATURE_FLAGS = (
    "has_number",
    "has_citation",
    "is_time_sensitive",
    "has_calculation",
)
_DEFAULT_SENSITIVE_METADATA_KEYS = (
    "requires_verification",
    "requires_current_facts",
    "requires_retrieval",
)


@dataclass(frozen=True)
class FeedbackPolicyReplayAuditConfig:
    """Configuration for policy replay audit over product feedback reports."""

    feedback_report_path: str | Path
    policy_recommendation_path: str | Path
    output_path: str | Path
    control_policy_path: str | Path | None = None
    control_defaults_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    min_matched_feedback_count: int = 20
    min_safety_coverage: float = 0.50
    max_unknown_safety_issue_rate: float = 0.50
    compact_json: bool = False

    def __post_init__(self) -> None:
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        object.__setattr__(self, "feedback_report_path", Path(self.feedback_report_path))
        object.__setattr__(self, "policy_recommendation_path", Path(self.policy_recommendation_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        _set_optional_path(self, "control_policy_path", self.control_policy_path)
        _set_optional_path(self, "control_defaults_path", self.control_defaults_path)
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
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the report artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_path).with_name("feedback-policy-replay-artifact-manifest.json")


def build_feedback_policy_replay_audit(
    config: FeedbackPolicyReplayAuditConfig,
) -> dict[str, Any]:
    """Build a replay-style audit for a feedback-derived control policy."""
    feedback_report = _load_json_object(config.feedback_report_path)
    policy_recommendation = _load_json_object(config.policy_recommendation_path)
    candidate_policy, candidate_defaults = _load_candidate_policy(config, policy_recommendation)
    matched = tuple(_mapping(item) for item in _sequence(feedback_report.get("matched_feedback")))
    audited = tuple(
        _audit_matched_feedback(
            item,
            feedback_report_path=config.feedback_report_path,
            candidate_policy=candidate_policy,
            candidate_defaults=candidate_defaults,
        )
        for item in matched
    )
    summary = _audit_summary(audited)
    gate = _quality_gate(summary, config)
    status = (
        "needs_evidence"
        if gate["needs_evidence"]
        else ("passed" if gate["passed"] else "blocked")
    )
    report = {
        "schema_version": 1,
        "workflow": "feedback_policy_replay_audit",
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": tuple(gate["blocking_reasons"]),
        },
        "summary": summary,
        "quality_gate": gate,
        "audited_feedback": audited,
        "candidate": {
            "control_policy_config": candidate_policy,
            "control_defaults": candidate_defaults,
            "policy_recommendation_status": policy_recommendation.get("status"),
            "policy_recommendation_path": str(config.policy_recommendation_path),
        },
        "source_feedback_report": {
            "path": str(config.feedback_report_path),
            "workflow": feedback_report.get("workflow"),
            "status": feedback_report.get("status"),
            "matched_feedback_count": _nested(
                feedback_report,
                "summary",
                "trace_matched_feedback_count",
            ),
        },
        "config": {
            "min_matched_feedback_count": config.min_matched_feedback_count,
            "min_safety_coverage": config.min_safety_coverage,
            "max_unknown_safety_issue_rate": config.max_unknown_safety_issue_rate,
            "metadata": dict(config.metadata),
        },
        "paths": {
            "report": str(config.output_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
        },
        "artifact_manifest_summary": _artifact_manifest_summary(config),
    }
    _write_json(config.output_path, report, compact=config.compact_json)
    _write_artifact_manifest(config, report)
    if config.registry_path is not None:
        _record_registry(config, report)
    return report


def _audit_matched_feedback(
    item: Mapping[str, Any],
    *,
    feedback_report_path: Path,
    candidate_policy: Mapping[str, Any],
    candidate_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    feedback = _mapping(item.get("feedback"))
    trace_summary = _mapping(item.get("trace"))
    flags = _mapping(item.get("flags"))
    trace_payload, trace_error = _load_trace_payload(
        trace_summary.get("path"),
        feedback_report_path=feedback_report_path,
    )
    outcome = str(feedback.get("outcome", "unknown"))
    decision_action = _optional_str(trace_summary.get("decision_action"))
    effect = "unchanged"
    safety_covered = False
    safety_unknown = False
    overblock_relieved = False
    residual_issue = False
    evidence: dict[str, Any] = {}

    if flags.get("accepted_but_wrong"):
        sensitive = _sensitive_claim_match(
            trace_payload,
            feedback_claim_id=_optional_str(feedback.get("claim_id")),
            control_defaults=candidate_defaults,
        )
        evidence["sensitive_claim_match"] = sensitive
        if sensitive["matched"]:
            effect = "candidate_adds_sensitive_claim_verification"
            safety_covered = True
        elif sensitive["unknown"]:
            effect = "needs_claim_metadata_for_verification_replay"
            safety_unknown = True
        else:
            effect = "residual_accepted_wrong"
            residual_issue = True

    elif flags.get("retrieved_failure"):
        unsupported_action = str(candidate_policy.get("unsupported_action", "retrieve"))
        evidence["candidate_unsupported_action"] = unsupported_action
        if unsupported_action in _SAFER_UNSUPPORTED_ACTIONS:
            effect = f"candidate_routes_unsupported_to_{unsupported_action}"
            safety_covered = True
        else:
            effect = "residual_retrieval_failure"
            residual_issue = True

    elif flags.get("abstain_false_positive"):
        compound_escalates = _bool_value(
            candidate_policy.get("compound_verification_escalates"),
            default=True,
        )
        compound_action = str(candidate_policy.get("compound_risk_action", "abstain"))
        unsupported_action = str(candidate_policy.get("unsupported_action", "retrieve"))
        evidence.update({
            "candidate_compound_verification_escalates": compound_escalates,
            "candidate_compound_risk_action": compound_action,
            "candidate_unsupported_action": unsupported_action,
        })
        if not compound_escalates and compound_action in _DEESCALATING_ACTIONS:
            effect = "candidate_deescalates_overblocking"
            overblock_relieved = True
        else:
            effect = "residual_abstain_false_positive"
            residual_issue = True

    return {
        "feedback": feedback,
        "trace": trace_summary,
        "baseline": {
            "decision_action": decision_action,
            "outcome": outcome,
            "flags": flags,
        },
        "candidate_effect": {
            "effect": effect,
            "safety_covered": safety_covered,
            "safety_unknown": safety_unknown,
            "overblock_relieved": overblock_relieved,
            "residual_issue": residual_issue,
            "evidence": evidence,
        },
        "trace_replay": {
            "loaded": trace_payload is not None,
            "error": trace_error,
        },
    }


def _audit_summary(audited: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matched_count = len(audited)
    safety_issue_count = 0
    safety_covered_count = 0
    safety_unknown_count = 0
    residual_safety_issue_count = 0
    overblock_issue_count = 0
    overblock_relief_count = 0
    residual_overblock_count = 0
    trace_load_failure_count = 0
    effect_counts: dict[str, int] = {}
    for item in audited:
        flags = _mapping(_nested(item, "baseline", "flags"))
        effect = _mapping(item.get("candidate_effect"))
        effect_name = str(effect.get("effect", "unknown"))
        effect_counts[effect_name] = effect_counts.get(effect_name, 0) + 1
        if _nested(item, "trace_replay", "loaded") is not True:
            trace_load_failure_count += 1
        if flags.get("accepted_but_wrong") or flags.get("retrieved_failure"):
            safety_issue_count += 1
            if effect.get("safety_covered"):
                safety_covered_count += 1
            elif effect.get("safety_unknown"):
                safety_unknown_count += 1
            else:
                residual_safety_issue_count += 1
        if flags.get("abstain_false_positive"):
            overblock_issue_count += 1
            if effect.get("overblock_relieved"):
                overblock_relief_count += 1
            else:
                residual_overblock_count += 1
    return {
        "matched_feedback_count": matched_count,
        "safety_issue_count": safety_issue_count,
        "safety_covered_count": safety_covered_count,
        "safety_unknown_count": safety_unknown_count,
        "residual_safety_issue_count": residual_safety_issue_count,
        "safety_coverage_rate": binomial_confidence_interval(
            safety_covered_count,
            safety_issue_count,
        ),
        "unknown_safety_issue_rate": binomial_confidence_interval(
            safety_unknown_count,
            safety_issue_count,
        ),
        "overblock_issue_count": overblock_issue_count,
        "overblock_relief_count": overblock_relief_count,
        "residual_overblock_count": residual_overblock_count,
        "overblock_relief_rate": binomial_confidence_interval(
            overblock_relief_count,
            overblock_issue_count,
        ),
        "trace_load_failure_count": trace_load_failure_count,
        "effect_counts": effect_counts,
    }


def _quality_gate(
    summary: Mapping[str, Any],
    config: FeedbackPolicyReplayAuditConfig,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    matched_count = _int_value(summary.get("matched_feedback_count"))
    safety_issue_count = _int_value(summary.get("safety_issue_count"))
    coverage = _finite_float(_nested(summary, "safety_coverage_rate", "estimate"))
    unknown_rate = _finite_float(_nested(summary, "unknown_safety_issue_rate", "estimate"))
    needs_evidence = matched_count < config.min_matched_feedback_count
    if needs_evidence:
        failures.append({
            "metric": "matched_feedback_count",
            "comparison": ">=",
            "threshold": config.min_matched_feedback_count,
            "actual": matched_count,
            "reason": (
                f"matched feedback count below {config.min_matched_feedback_count}"
            ),
        })
    if safety_issue_count > 0 and (coverage is None or coverage < config.min_safety_coverage):
        failures.append({
            "metric": "safety_coverage_rate",
            "comparison": ">=",
            "threshold": config.min_safety_coverage,
            "actual": coverage,
            "reason": "candidate policy does not cover enough historical safety issues",
        })
    if safety_issue_count > 0 and (unknown_rate is None or unknown_rate > config.max_unknown_safety_issue_rate):
        failures.append({
            "metric": "unknown_safety_issue_rate",
            "comparison": "<=",
            "threshold": config.max_unknown_safety_issue_rate,
            "actual": unknown_rate,
            "reason": "too many safety issues lack claim metadata for replay",
        })
    blocking = tuple(failure["reason"] for failure in failures if failure["metric"] != "matched_feedback_count")
    return {
        "configured": True,
        "needs_evidence": needs_evidence,
        "passed": not failures,
        "failures": tuple(failures),
        "blocking_reasons": blocking,
        "policy": {
            "min_matched_feedback_count": config.min_matched_feedback_count,
            "min_safety_coverage": config.min_safety_coverage,
            "max_unknown_safety_issue_rate": config.max_unknown_safety_issue_rate,
        },
    }


def _sensitive_claim_match(
    trace: Mapping[str, Any] | None,
    *,
    feedback_claim_id: str | None,
    control_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    if trace is None:
        return {"matched": False, "unknown": True, "reason": "trace_not_loaded"}
    if _bool_value(control_defaults.get("staged_verification"), default=False) is not True:
        return {"matched": False, "unknown": False, "reason": "staged_verification_disabled"}
    claims = tuple(_mapping(claim) for claim in _sequence(trace.get("claims")))
    if feedback_claim_id is not None:
        claims = tuple(claim for claim in claims if str(claim.get("claim_id")) == feedback_claim_id)
    if not claims:
        return {"matched": False, "unknown": True, "reason": "claim_metadata_missing"}
    feature_flags = _string_tuple(
        control_defaults.get("stage_verify_claim_feature_flags"),
        default=_DEFAULT_SENSITIVE_FEATURE_FLAGS,
    )
    metadata_keys = _string_tuple(
        control_defaults.get("stage_verify_claim_metadata_keys"),
        default=_DEFAULT_SENSITIVE_METADATA_KEYS,
    )
    matched_claims: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        metadata = _mapping(claim.get("metadata"))
        features = _mapping(metadata.get("features"))
        matched_features = tuple(
            flag for flag in feature_flags if _truthy(features.get(flag)) or _truthy(metadata.get(flag))
        )
        matched_metadata = tuple(key for key in metadata_keys if _truthy(_nested(metadata, *key.split("."))))
        if matched_features or matched_metadata:
            matched_claims.append({
                "claim_id": claim.get("claim_id") or f"c{index + 1}",
                "matched_features": matched_features,
                "matched_metadata": matched_metadata,
            })
    return {
        "matched": bool(matched_claims),
        "unknown": False,
        "matched_claims": tuple(matched_claims),
        "checked_claim_count": len(claims),
        "feature_flags": feature_flags,
        "metadata_keys": metadata_keys,
    }


def _load_candidate_policy(
    config: FeedbackPolicyReplayAuditConfig,
    recommendation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.control_policy_path is not None:
        policy_payload = _load_json_object(config.control_policy_path)
    else:
        policy_payload = _mapping(_nested(recommendation, "recommendation", "candidate_control_policy_config"))
    if config.control_defaults_path is not None:
        defaults_payload = _load_json_object(config.control_defaults_path)
    else:
        defaults_payload = _mapping(_nested(recommendation, "recommendation", "candidate_control_defaults"))
    return ControlPolicyConfig.from_dict(policy_payload).to_dict(), dict(defaults_payload)


def _load_trace_payload(
    trace_path_value: Any,
    *,
    feedback_report_path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    if trace_path_value is None:
        return None, "trace_path_missing"
    trace_path = Path(str(trace_path_value))
    candidates = [trace_path]
    if not trace_path.is_absolute():
        candidates.append(feedback_report_path.parent / trace_path)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, f"trace_load_error: {exc}"
        if not isinstance(payload, Mapping):
            return None, "trace_json_not_object"
        return dict(payload), None
    return None, f"trace_path_not_found: {trace_path}"


def _write_artifact_manifest(
    config: FeedbackPolicyReplayAuditConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = _artifact_paths(config)
    manifest = build_artifact_manifest(
        artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "audit_feedback_policy_replay",
            "status": report.get("status"),
            "matched_feedback_count": _nested(report, "summary", "matched_feedback_count"),
            "safety_coverage_rate": _nested(report, "summary", "safety_coverage_rate", "estimate"),
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(config: FeedbackPolicyReplayAuditConfig, report: Mapping[str, Any]) -> None:
    assert config.registry_path is not None
    assert config.name is not None
    assert config.version is not None
    registry = ArtifactRegistry.load_json(config.registry_path)
    registry.record_report(
        name=config.name,
        version=config.version,
        path=config.output_path,
        metadata={
            "workflow": "feedback_policy_replay_audit",
            "status": report.get("status"),
            "matched_feedback_count": _nested(report, "summary", "matched_feedback_count"),
            "safety_coverage_rate": _nested(report, "summary", "safety_coverage_rate", "estimate"),
            "unknown_safety_issue_rate": _nested(
                report,
                "summary",
                "unknown_safety_issue_rate",
                "estimate",
            ),
            "overblock_relief_rate": _nested(report, "summary", "overblock_relief_rate", "estimate"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            **dict(config.metadata),
        },
    ).save_json()


def _artifact_manifest_summary(config: FeedbackPolicyReplayAuditConfig) -> dict[str, int]:
    return planned_artifact_manifest_summary(
        _artifact_paths(config),
        assume_file_paths=(config.output_path,),
    )


def _artifact_paths(config: FeedbackPolicyReplayAuditConfig) -> dict[str, str | Path | None]:
    return {
        "feedback_policy_replay_audit": config.output_path,
        "feedback_report": config.feedback_report_path,
        "policy_recommendation": config.policy_recommendation_path,
        "control_policy": config.control_policy_path,
        "control_defaults": config.control_defaults_path,
    }


def _config_from_args(args: argparse.Namespace) -> FeedbackPolicyReplayAuditConfig:
    return FeedbackPolicyReplayAuditConfig(
        feedback_report_path=Path(args.feedback_report),
        policy_recommendation_path=Path(args.policy_recommendation),
        output_path=Path(args.json),
        control_policy_path=None if args.control_policy is None else Path(args.control_policy),
        control_defaults_path=None if args.control_defaults is None else Path(args.control_defaults),
        artifact_manifest_path=None if args.artifact_manifest is None else Path(args.artifact_manifest),
        registry_path=None if args.registry is None else Path(args.registry),
        name=args.name,
        version=args.version,
        metadata=_metadata_from_args(args.metadata),
        min_matched_feedback_count=args.min_matched_feedback_count,
        min_safety_coverage=args.min_safety_coverage,
        max_unknown_safety_issue_rate=args.max_unknown_safety_issue_rate,
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI args."""
    config = _config_from_args(args)
    report = build_feedback_policy_replay_audit(config)
    print(
        "feedback_policy_replay_audit="
        f"{report['status']} matched={report['summary']['matched_feedback_count']} "
        f"safety_coverage={report['summary']['safety_coverage_rate']['estimate']}"
    )
    if args.fail_on_blocked and report["decision"]["status"] == "blocked":
        raise SystemExit(1)
    if args.fail_on_needs_evidence and report["decision"]["status"] == "needs_evidence":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit a feedback-derived control policy against historical feedback labels"
    )
    parser.add_argument("--feedback-report", required=True, help="product feedback report JSON path")
    parser.add_argument("--policy-recommendation", required=True,
                        help="feedback policy recommendation JSON path")
    parser.add_argument("--json", required=True, help="output replay audit JSON path")
    parser.add_argument("--control-policy", default=None,
                        help="optional candidate ControlPolicyConfig JSON override")
    parser.add_argument("--control-defaults", default=None,
                        help="optional candidate runtime control defaults JSON override")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry report name")
    parser.add_argument("--version", default=None, help="registry report version")
    parser.add_argument("--metadata", action="append", default=[],
                        help="metadata key=value pair to include in report and registry; repeatable")
    parser.add_argument("--min-matched-feedback-count", type=int, default=20)
    parser.add_argument("--min-safety-coverage", type=float, default=0.50)
    parser.add_argument("--max-unknown-safety-issue-rate", type=float, default=0.50)
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    parser.add_argument("--fail-on-blocked", action="store_true",
                        help="exit non-zero when replay audit gates fail")
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


def _int_value(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bool_value(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return False


def _string_tuple(value: Any, *, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(str(item) for item in default)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return tuple(str(item) for item in default)


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    main()
