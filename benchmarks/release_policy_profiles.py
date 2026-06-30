"""Named release-gate policy defaults shared by release workflows."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_CANDIDATE_RELEASE_POLICY_DEFAULTS: Mapping[str, Any] = {
    "min_best_quality_auroc": 0.70,
    "max_uncached_forward_seconds": 20.0,
    "min_selected": 4,
    "min_decision_accuracy": 0.95,
    "max_false_supported_rate": 0.05,
    "min_false_refuted_rate": 0.50,
}

_STRICT_STRUCTURED_FACT_DEFAULTS: Mapping[str, Any] = {
    **_CANDIDATE_RELEASE_POLICY_DEFAULTS,
    "require_structured_fact_robustness": True,
    "min_decision_accuracy": 0.99,
    "max_false_supported_rate": 0.0,
    "min_false_refuted_rate": 0.99,
    "required_route_min_selected": 200,
    "required_route_min_decision_accuracy": 0.99,
    "required_route_max_false_supported_rate": 0.0,
    "required_route_min_false_refuted_rate": 0.99,
    "structured_fact_robustness_min_selected": 700,
    "structured_fact_robustness_min_decision_accuracy": 0.99,
    "structured_fact_robustness_max_false_supported_rate": 0.0,
    "structured_fact_robustness_min_false_refuted_rate": 0.99,
    "structured_fact_robustness_min_covered_fact_properties": 3,
    "structured_fact_robustness_min_covered_fact_property_records": 2,
    "structured_fact_robustness_min_covered_fact_property_source_documents": 1,
    "structured_fact_robustness_min_covered_fact_property_decision_accuracy": 0.99,
    "structured_fact_robustness_max_covered_fact_property_false_supported_rate": 0.0,
    "structured_fact_robustness_min_covered_fact_property_false_refuted_rate": 0.99,
}
FRONTIER_EXTERNAL_EVIDENCE_BASELINE_COMPARISON_KEY = (
    "report:covered-facts-external-evidence-handoff:0.4"
)
FRONTIER_TRIPLE_EXTRACTION_FIXTURE_MATRIX_KEY = "report:triple-extraction-fixture-matrix:0.1"
FRONTIER_MECHANISM_HANDOFF_EVIDENCE_BUNDLE_KEY = (
    "report:truthfulqa-frontier-smollm2-l80-mechanism-handoff-evidence-bundle:0.1"
)

RELEASE_POLICY_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "research_smoke": {
        "min_best_quality_auroc": 0.50,
        "min_selected": 1,
    },
    "candidate_release": _CANDIDATE_RELEASE_POLICY_DEFAULTS,
    "strict_structured_fact": _STRICT_STRUCTURED_FACT_DEFAULTS,
    "frontier_audit": {
        **_STRICT_STRUCTURED_FACT_DEFAULTS,
        "max_uncached_forward_seconds": None,
        "max_recommended_runtime_seconds": 1.0,
        "adapter_family_profile": "strict_audit",
        "require_state_transition_world_model": True,
        "require_product_runtime_drift_promotion_evidence": True,
        "require_product_runtime_drift_pre_generation_evidence": True,
        "require_product_runtime_drift_counterfactual_evidence": True,
        "require_product_runtime_drift_triple_audit_evidence": True,
        "require_product_runtime_drift_covered_fact_property_evidence": True,
        "require_product_runtime_drift_action_gate_evidence": True,
        "require_product_runtime_drift_trajectory_audit_evidence": True,
        "require_product_runtime_drift_evidence_handoff_evidence": True,
        "require_product_runtime_drift_world_model_evidence": True,
        "require_product_runtime_drift_context_sensitivity_evidence": True,
        "require_product_runtime_drift_counterfactual_robustness_evidence": True,
        "require_product_runtime_drift_frontier_release_evidence": True,
        "require_product_trace_action_audit_gate": True,
        "require_product_trace_action_execution_gate": True,
        "external_evidence_baseline_comparison_key": (
            FRONTIER_EXTERNAL_EVIDENCE_BASELINE_COMPARISON_KEY
        ),
        "triple_extraction_fixture_matrix_key": FRONTIER_TRIPLE_EXTRACTION_FIXTURE_MATRIX_KEY,
        "mechanism_handoff_evidence_bundle_key": FRONTIER_MECHANISM_HANDOFF_EVIDENCE_BUNDLE_KEY,
        "min_triple_extraction_corpora": 2,
        "min_triple_extraction_distinct_predicates": 6,
        "min_triple_extraction_external_prediction_count": 2,
        "min_triple_extraction_external_prediction_corpora": 2,
        "min_triple_extraction_mean_best_external_f1": 0.90,
    },
}
RELEASE_POLICY_PROFILE_NAMES = tuple(sorted(RELEASE_POLICY_PROFILES))


def clean_optional_key(value: str | None) -> str | None:
    """Normalize optional registry keys, treating blank strings as missing."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def append_unique(existing: Sequence[str], additions: Sequence[str | None]) -> tuple[str, ...]:
    """Append non-empty values while preserving order and removing duplicates."""
    values = list(existing)
    seen = set(values)
    for raw in additions:
        value = clean_optional_key(raw)
        if value is None or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return tuple(values)


def normalize_release_policy_profile(value: str | None) -> str | None:
    """Normalize and validate a release policy profile name."""
    if value is None:
        return None
    profile = str(value).strip().lower().replace("-", "_")
    if profile not in RELEASE_POLICY_PROFILES:
        choices = ", ".join(RELEASE_POLICY_PROFILE_NAMES)
        raise ValueError(f"release_policy_profile must be one of: {choices}")
    return profile


def apply_release_policy_profile_defaults(
    release_policy_profile: str | None,
    values: Mapping[str, Any],
    *,
    disabled_defaults: Sequence[str] = (),
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    """Apply profile defaults to unset keys without overriding explicit values."""
    profile = normalize_release_policy_profile(release_policy_profile)
    merged = dict(values)
    applied: dict[str, Any] = {}
    disabled = set(disabled_defaults)
    if profile is None:
        return None, merged, applied
    for key, default in RELEASE_POLICY_PROFILES[profile].items():
        if key not in merged:
            continue
        if key in disabled:
            continue
        if not _profile_default_is_unset(merged[key], default):
            continue
        merged[key] = default
        applied[key] = default
    return profile, merged, applied


def _profile_default_is_unset(current: Any, default: Any) -> bool:
    if isinstance(default, bool):
        return current is False and default is True
    if isinstance(default, (tuple, list)):
        return not tuple(current or ())
    if isinstance(default, str):
        return clean_optional_key(current) is None
    return current is None
