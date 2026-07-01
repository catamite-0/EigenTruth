"""No-model smoke checks for the active frontier status handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_frontier_status_report import (  # noqa: E402
    DEFAULT_EVIDENCE_GAP_PLAN,
    DEFAULT_PRODUCT_CONTRACT,
    DEFAULT_RELEASE_CANDIDATE,
    build_frontier_status_report,
)
from benchmarks.plan_frontier_research_queue_commands import (  # noqa: E402
    build_frontier_research_queue_command_plan,
)

CURRENT_FRONTIER_EVIDENCE_REPORT = (
    "artifacts/frontier-release-evidence/frontier-release-evidence-budget-target-sweep-v5.json"
)
CURRENT_RUNTIME_DRIFT_REPORT = (
    "artifacts/smollm2_product_runtime_drift_v1_16_receipts_frontier_v5/product-runtime-drift.json"
)
CURRENT_PRODUCT_HANDOFF_CONTRACT = (
    "artifacts/smollm2_product_promotion_evidence_handoff_v1_9_frontier_v7/"
    "product-promotion-contract-evidence-handoff.json"
)


def build_frontier_status_smoke(
    *,
    release_candidate: Path = DEFAULT_RELEASE_CANDIDATE,
    product_contract: Path = DEFAULT_PRODUCT_CONTRACT,
    evidence_gap_plan: Path = DEFAULT_EVIDENCE_GAP_PLAN,
) -> dict[str, Any]:
    """Verify current productized frontier status and inactive stale research queue."""
    status_report = build_frontier_status_report(
        release_candidate=release_candidate,
        product_contract=product_contract,
        evidence_gap_plan=evidence_gap_plan,
        refresh_research_queue=True,
    )
    active_only_plan = build_frontier_research_queue_command_plan(
        source=status_report,
        only_active_research_queue=True,
    )
    _assert_status_report(status_report, product_contract=product_contract)
    _assert_active_only_plan(active_only_plan)
    research_queue = _mapping(status_report.get("research_queue"))
    release = _mapping(_nested(status_report, "productized_status", "release_candidate"))
    contract = _mapping(_nested(status_report, "productized_status", "product_contract"))
    return {
        "status": "pass",
        "workflow": "frontier_status_smoke",
        "release_candidate": str(release_candidate),
        "product_contract": str(product_contract),
        "evidence_gap_plan": str(evidence_gap_plan),
        "frontier_release_evidence_report": release.get("frontier_release_evidence_report"),
        "product_runtime_drift_report": release.get("runtime_drift_report"),
        "contract_frontier_release_evidence_report": _nested(
            contract,
            "frontier_release_evidence",
            "report_path",
        ),
        "research_lifecycle_status": research_queue.get("lifecycle_status"),
        "research_source_alignment_status": _nested(
            research_queue,
            "source_alignment",
            "status",
        ),
        "research_action_count": research_queue.get("action_count"),
        "research_active_action_count": research_queue.get("active_action_count"),
        "active_only_command_plan_status": active_only_plan.get("status"),
        "active_only_command_count": _nested(active_only_plan, "summary", "command_count"),
    }


def _assert_status_report(
    report: Mapping[str, Any],
    *,
    product_contract: Path,
) -> None:
    if report.get("status") != "promote":
        raise AssertionError("frontier status report did not promote.")
    productized = _mapping(report.get("productized_status"))
    if productized.get("blocking_reasons") not in ((), []):
        raise AssertionError("frontier status report has productized blockers.")

    release = _mapping(productized.get("release_candidate"))
    if release.get("status") != "promote":
        raise AssertionError("frontier status release candidate did not promote.")
    if release.get("frontier_release_evidence_report") != CURRENT_FRONTIER_EVIDENCE_REPORT:
        raise AssertionError("frontier status release evidence report is not current v5.")
    if release.get("runtime_drift_report") != CURRENT_RUNTIME_DRIFT_REPORT:
        raise AssertionError("frontier status runtime drift report is not current v1.16.")
    if _int(release.get("blocked_gate_count")) != 0:
        raise AssertionError("frontier status release candidate has blocked gates.")
    frontier = _mapping(release.get("frontier_release_evidence"))
    if frontier.get("decision_status") != "promote":
        raise AssertionError("frontier status release evidence did not promote.")
    if _int(frontier.get("input_manifest_verified_count")) < 7:
        raise AssertionError("frontier status release evidence verified too few input manifests.")

    contract = _mapping(productized.get("product_contract"))
    if contract.get("status") != "promote":
        raise AssertionError("frontier status product contract did not promote.")
    if _repo_relative(product_contract) != CURRENT_PRODUCT_HANDOFF_CONTRACT:
        raise AssertionError("frontier status default product contract is not the v1.9/v7 handoff.")
    if _nested(contract, "frontier_release_evidence", "report_path") != CURRENT_FRONTIER_EVIDENCE_REPORT:
        raise AssertionError("frontier status product contract is not aligned to current v5 evidence.")
    if _int(contract.get("blocked_evidence_group_count")) != 0:
        raise AssertionError("frontier status product contract has blocked evidence groups.")
    if _int(contract.get("required_evidence_group_count")) < 12:
        raise AssertionError("frontier status product contract checked too few evidence groups.")

    research = _mapping(report.get("research_queue"))
    if research.get("refresh_status") != "refreshed":
        raise AssertionError("frontier status research queue was not refreshed.")
    if research.get("active") is not False:
        raise AssertionError("frontier status still treats stale research queue as active.")
    if research.get("lifecycle_status") != "superseded":
        raise AssertionError("frontier status research queue is not marked superseded.")
    if _int(research.get("active_action_count")) != 0:
        raise AssertionError("frontier status research queue still has active actions.")
    if _int(research.get("superseded_action_count")) < 1:
        raise AssertionError("frontier status research queue did not preserve superseded actions.")
    if _nested(research, "source_alignment", "status") != "stale":
        raise AssertionError("frontier status research queue source alignment is not stale.")
    if _nested(research, "source_alignment", "is_current") is not False:
        raise AssertionError("frontier status research queue source unexpectedly matches current evidence.")


def _assert_active_only_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("status") != "empty":
        raise AssertionError("active-only frontier research command plan was not empty.")
    if _nested(plan, "config", "only_active_research_queue") is not True:
        raise AssertionError("frontier status smoke did not request active-only command planning.")
    summary = _mapping(plan.get("summary"))
    for field in ("entry_count", "command_count", "placeholder_count", "missing_input_count"):
        if _int(summary.get(field)) != 0:
            raise AssertionError(f"active-only frontier research command plan has {field}.")


def _repo_relative(path: Path) -> str:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    try:
        return resolved.resolve(strict=False).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


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
    parser = argparse.ArgumentParser(description="Run the no-model frontier status smoke check")
    parser.add_argument(
        "--release-candidate",
        default=str(DEFAULT_RELEASE_CANDIDATE),
        help="release-candidate comparison JSON path",
    )
    parser.add_argument(
        "--product-contract",
        default=str(DEFAULT_PRODUCT_CONTRACT),
        help="ProductPromotionContract JSON path",
    )
    parser.add_argument(
        "--evidence-gap-plan",
        default=str(DEFAULT_EVIDENCE_GAP_PLAN),
        help="historical evidence-gap plan used to verify superseded queue handling",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="optional path to write the smoke report JSON",
    )
    args = parser.parse_args(argv)
    report = build_frontier_status_smoke(
        release_candidate=Path(args.release_candidate),
        product_contract=Path(args.product_contract),
        evidence_gap_plan=Path(args.evidence_gap_plan),
    )
    if args.json is not None:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_report(report)


def _print_report(report: Mapping[str, Any]) -> None:
    print(
        "frontier_status_smoke_ok "
        f"status={report['status']} "
        f"frontier={report['frontier_release_evidence_report']} "
        f"handoff={report['contract_frontier_release_evidence_report']} "
        f"research={report['research_lifecycle_status']} "
        f"active_actions={report['research_active_action_count']} "
        f"active_plan={report['active_only_command_plan_status']}"
    )


if __name__ == "__main__":
    main()
