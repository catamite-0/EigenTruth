"""No-model smoke checks for the default product promotion contract handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.control import (  # noqa: E402
    load_product_runtime_evidence_bundle,
    product_runtime_metrics,
)

DEFAULT_CONTRACT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "smollm2_product_promotion_contract_v1_9"
    / "product-promotion-contract.json"
)
DEFAULT_EVIDENCE_HANDOFF_MANIFEST_PATH = (
    REPO_ROOT
    / "artifacts"
    / "smollm2_product_promotion_evidence_handoff_v1_9_frontier_v7"
    / "artifact-manifest.json"
)
LEGACY_EVIDENCE_HANDOFF_MANIFEST_PATH = (
    REPO_ROOT
    / "artifacts"
    / "smollm2_product_promotion_contract_v1_9"
    / "evidence-handoff-artifact-manifest.json"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "artifacts" / "local-release-registry.json"
EXPECTED_REGISTRY_KEY = "product_promotion_contract:smollm2-product-promotion-contract:1.9"
LEGACY_REQUIRED_HANDOFF_GROUPS = {
    "action_gate",
    "counterfactual",
    "covered_fact_property",
    "frontier_release_evidence",
    "pre_generation",
    "promotion",
    "triple_audit",
}
DEFAULT_REQUIRED_HANDOFF_GROUPS = LEGACY_REQUIRED_HANDOFF_GROUPS | {
    "action_receipts",
    "receipt_claim_support",
}
DEFAULT_MIN_PRESENT_METRIC_COUNT = 77


def build_product_promotion_contract_smoke(
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    evidence_handoff_manifest_path: Path | None = DEFAULT_EVIDENCE_HANDOFF_MANIFEST_PATH,
    registry_path: Path | None = DEFAULT_REGISTRY_PATH,
    min_present_metric_count: int = DEFAULT_MIN_PRESENT_METRIC_COUNT,
    required_handoff_groups: Iterable[str] = DEFAULT_REQUIRED_HANDOFF_GROUPS,
) -> dict[str, Any]:
    """Load the default promoted contract and verify release-critical handoff fields."""
    if not contract_path.exists():
        raise AssertionError(f"default product promotion contract is missing: {contract_path}")
    if evidence_handoff_manifest_path is not None and not evidence_handoff_manifest_path.exists():
        raise AssertionError(
            "default product promotion evidence-handoff manifest is missing: "
            f"{evidence_handoff_manifest_path}"
        )
    bundle = load_product_runtime_evidence_bundle(
        contract_path,
        evidence_handoff_manifest_path=evidence_handoff_manifest_path,
        registry_path=registry_path if registry_path and registry_path.exists() else None,
        require_promoted=True,
    )
    if bundle is None:
        raise AssertionError("default product promotion contract did not load.")
    if bundle.manifest_path is None:
        raise AssertionError("default product promotion contract manifest was not discovered.")
    if bundle.evidence_handoff_manifest_path is None:
        raise AssertionError("default product promotion evidence-handoff manifest was not discovered.")

    metadata = bundle.runtime_metadata(
        budget_enabled=True,
        verify_manifest=True,
        verify_evidence_handoff_manifest=True,
    )
    metrics = product_runtime_metrics({"metadata": metadata})
    _assert_contract_metadata(
        metadata,
        min_present_metric_count=min_present_metric_count,
        required_handoff_groups=required_handoff_groups,
    )
    _assert_runtime_metrics(metrics)
    registry_record = bundle.registry_record()
    if registry_path is not None and registry_path.exists():
        if registry_record is None:
            raise AssertionError("default product promotion registry record was not discovered.")
        if registry_record.key() != EXPECTED_REGISTRY_KEY:
            raise AssertionError(
                "default product promotion registry record mismatch: "
                f"{registry_record.key()} != {EXPECTED_REGISTRY_KEY}"
            )

    return {
        "status": "pass",
        "workflow": "product_promotion_contract_smoke",
        "contract": str(contract_path),
        "registry": None if registry_path is None else str(registry_path),
        "registry_key": None if registry_record is None else registry_record.key(),
        "promotion_summary_status": metadata["promotion_contract_promotion_summary"]["status"],
        "recommended_runtime_seconds": metadata["promotion_contract_recommended_runtime_seconds"],
        "evidence_handoff_manifest": metadata["promotion_contract_evidence_handoff_manifest"],
        "evidence_handoff_status": metadata["promotion_contract_evidence_handoff_status"],
        "evidence_handoff_expected_metric_count": (
            metadata["promotion_contract_evidence_handoff_expected_metric_count"]
        ),
        "evidence_handoff_present_metric_count": (
            metadata["promotion_contract_evidence_handoff_present_metric_count"]
        ),
        "evidence_handoff_missing_metric_count": (
            metadata["promotion_contract_evidence_handoff_missing_metric_count"]
        ),
        "frontier_release_evidence_decision_status": metadata[
            "promotion_contract_frontier_release_evidence"
        ]["decision_status"],
        "product_runtime_drift_status": metadata[
            "promotion_contract_product_runtime_drift_status"
        ],
        "product_runtime_drift_compared_metric_count": metadata[
            "promotion_contract_product_runtime_drift_compared_metric_count"
        ],
        "runtime_budget": {
            "max_mean_attempted_route_count": (
                bundle.contract.runtime_budget_policy.max_mean_attempted_route_count
            ),
            "max_retrieval_use_rate": bundle.contract.runtime_budget_policy.max_retrieval_use_rate,
        },
        "manifest_checked": metadata["promotion_contract_manifest_verification"]["checked"],
        "evidence_handoff_manifest_checked": metadata[
            "promotion_contract_evidence_handoff_manifest_verification"
        ]["checked"],
        "metrics": {
            "promotion_contract_available": metrics["promotion_contract_available"],
            "promotion_contract_evidence_handoff_available": metrics[
                "promotion_contract_evidence_handoff_available"
            ],
            "promotion_contract_frontier_release_evidence_available": metrics[
                "promotion_contract_frontier_release_evidence_available"
            ],
        },
    }


def _assert_contract_metadata(
    metadata: Mapping[str, Any],
    *,
    min_present_metric_count: int,
    required_handoff_groups: Iterable[str],
) -> None:
    summary = _mapping(metadata.get("promotion_contract_promotion_summary"))
    if summary.get("status") != "promote":
        raise AssertionError("default product promotion summary did not promote.")
    if summary.get("blocking_gate_count") != 0:
        raise AssertionError("default product promotion summary has blocking gates.")

    if metadata.get("promotion_contract_recommended_runtime_seconds") != 0.191662:
        raise AssertionError("default product promotion runtime recommendation changed.")
    if metadata.get("promotion_contract_recommended_runtime_cost_source") != "cache_only_total_seconds":
        raise AssertionError("default product promotion runtime cost source changed.")

    manifest_verification = _mapping(metadata.get("promotion_contract_manifest_verification"))
    if manifest_verification.get("passed") is not True:
        raise AssertionError("default product promotion manifest verification failed.")
    if _int(manifest_verification.get("checked")) < 2:
        raise AssertionError("default product promotion manifest checked too few artifacts.")

    handoff_verification = _mapping(
        metadata.get("promotion_contract_evidence_handoff_manifest_verification")
    )
    if handoff_verification.get("passed") is not True:
        raise AssertionError("default product promotion handoff manifest verification failed.")
    if _int(handoff_verification.get("checked")) < 11:
        raise AssertionError("default product promotion handoff manifest checked too few artifacts.")

    if metadata.get("promotion_contract_evidence_handoff_status") != "promote":
        raise AssertionError("default product promotion handoff did not promote.")
    if (
        _int(metadata.get("promotion_contract_evidence_handoff_present_metric_count"))
        < min_present_metric_count
    ):
        raise AssertionError(
            "default product promotion handoff lost required metrics: "
            f"expected at least {min_present_metric_count}."
        )
    if _int(metadata.get("promotion_contract_evidence_handoff_missing_metric_count")) != 0:
        raise AssertionError("default product promotion handoff has missing metrics.")
    if _int(metadata.get("promotion_contract_evidence_handoff_blocked_group_count")) != 0:
        raise AssertionError("default product promotion handoff has blocked groups.")
    required_groups = set(required_handoff_groups)
    group_statuses = _mapping(metadata.get("promotion_contract_evidence_handoff_group_statuses"))
    missing_groups = required_groups.difference(group_statuses)
    if missing_groups:
        raise AssertionError(f"default product promotion handoff missing groups: {sorted(missing_groups)}")
    blocked_groups = {
        group
        for group in required_groups
        if group_statuses.get(group) != "promote"
    }
    if blocked_groups:
        raise AssertionError(f"default product promotion handoff groups did not promote: {sorted(blocked_groups)}")

    frontier = _mapping(metadata.get("promotion_contract_frontier_release_evidence"))
    if frontier.get("decision_status") != "promote":
        raise AssertionError("default product promotion frontier release decision did not promote.")
    if frontier.get("verifier_track_status") != "promote":
        raise AssertionError("default product promotion frontier verifier track did not promote.")
    if frontier.get("abstention_track_status") != "promote":
        raise AssertionError("default product promotion frontier abstention track did not promote.")
    if frontier.get("frontier_rerun_rollup_track_status") != "promote":
        raise AssertionError("default product promotion frontier rerun rollup track did not promote.")

    if metadata.get("promotion_contract_product_runtime_drift_status") != "promote":
        raise AssertionError("default product promotion runtime drift did not promote.")
    if _int(metadata.get("promotion_contract_product_runtime_drift_compared_metric_count")) < 107:
        raise AssertionError("default product promotion runtime drift compared too few metrics.")
    if _int(metadata.get("promotion_contract_product_runtime_drift_blocked_metric_count")) != 0:
        raise AssertionError("default product promotion runtime drift has blocked metrics.")


def _assert_runtime_metrics(metrics: Mapping[str, Any]) -> None:
    if metrics.get("promotion_contract_available") is not True:
        raise AssertionError("product runtime metrics did not expose promotion contract availability.")
    if metrics.get("promotion_contract_evidence_handoff_available") is not True:
        raise AssertionError("product runtime metrics did not expose evidence-handoff availability.")
    if metrics.get("promotion_contract_evidence_handoff_manifest_verified") is not True:
        raise AssertionError("product runtime metrics did not expose verified handoff manifest.")
    if metrics.get("promotion_contract_frontier_release_evidence_available") is not True:
        raise AssertionError("product runtime metrics did not expose frontier release evidence.")
    if metrics.get("promotion_contract_frontier_release_evidence_decision_status") != "promote":
        raise AssertionError("product runtime metrics did not expose promoted frontier release decision.")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    if isinstance(value, bool):
        raise AssertionError("boolean value is not a valid integer smoke metric.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"invalid integer smoke metric: {value!r}") from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the no-model product promotion contract smoke check")
    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT_PATH),
        help="ProductPromotionContract JSON path",
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="optional local ArtifactRegistry JSON path",
    )
    parser.add_argument(
        "--evidence-handoff-manifest",
        default=str(DEFAULT_EVIDENCE_HANDOFF_MANIFEST_PATH),
        help=(
            "optional enriched evidence-handoff artifact manifest path; pass 'none' "
            "to fall back to the contract sibling manifest"
        ),
    )
    parser.add_argument(
        "--min-present-metrics",
        type=int,
        default=DEFAULT_MIN_PRESENT_METRIC_COUNT,
        help="minimum promoted evidence-handoff present metric count",
    )
    parser.add_argument(
        "--required-handoff-groups",
        default=",".join(sorted(DEFAULT_REQUIRED_HANDOFF_GROUPS)),
        help="comma-separated evidence-handoff groups that must promote",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="optional path to write the smoke report JSON",
    )
    args = parser.parse_args(argv)
    registry_path = None if args.registry in {"", "none", "None"} else Path(args.registry)
    evidence_handoff_manifest_path = (
        None
        if args.evidence_handoff_manifest in {"", "none", "None"}
        else Path(args.evidence_handoff_manifest)
    )
    required_handoff_groups = tuple(
        group.strip()
        for group in args.required_handoff_groups.split(",")
        if group.strip()
    )
    report = build_product_promotion_contract_smoke(
        contract_path=Path(args.contract),
        evidence_handoff_manifest_path=evidence_handoff_manifest_path,
        registry_path=registry_path,
        min_present_metric_count=args.min_present_metrics,
        required_handoff_groups=required_handoff_groups,
    )
    if args.json is not None:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_report(report)


def _print_report(report: Mapping[str, Any]) -> None:
    print(
        "product_promotion_contract_smoke_ok "
        f"status={report['status']} "
        f"handoff={report['evidence_handoff_status']} "
        f"expected_metrics={report['evidence_handoff_expected_metric_count']} "
        f"present_metrics={report['evidence_handoff_present_metric_count']} "
        f"runtime_drift={report['product_runtime_drift_status']} "
        f"compared_metrics={report['product_runtime_drift_compared_metric_count']} "
        f"registry_key={report['registry_key']}"
    )


if __name__ == "__main__":
    main()
