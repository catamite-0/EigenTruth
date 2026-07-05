"""No-model smoke checks for the active promoted frontier release evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.lib.paths import ensure_repo_root_on_path

REPO_ROOT = ensure_repo_root_on_path()
from eigentruth.registry import load_and_verify_artifact_manifest  # noqa: E402

DEFAULT_CONTRACT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "smollm2_product_promotion_evidence_handoff_v1_9_frontier_v7"
    / "product-promotion-contract-evidence-handoff.json"
)
REQUIRED_TRACK_STATUSES = {
    "verifier_track_status": "promote",
    "abstention_track_status": "promote",
    "detectability_track_status": "promote",
    "multiple_testing_track_status": "promote",
    "frontier_rerun_rollup_track_status": "promote",
}
CONTRACT_REQUIRED_TRACK_STATUSES = {
    "verifier_track_status": "promote",
    "abstention_track_status": "promote",
    "multiple_testing_track_status": "promote",
    "frontier_rerun_rollup_track_status": "promote",
}
REQUIRED_RERUN_PROMOTED_TRACKS = {"detectability", "multiple_testing"}


def build_frontier_release_evidence_smoke(
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Verify the product contract's active frontier release-evidence report."""
    contract = _load_mapping(contract_path)
    frontier = _mapping(contract.get("frontier_release_evidence"))
    if not frontier:
        raise AssertionError("product promotion contract is missing frontier_release_evidence.")
    report_path = _repo_path(frontier.get("report_path"))
    manifest_path = _repo_path(frontier.get("manifest_path"))
    report = _load_mapping(report_path)
    manifest_verification = load_and_verify_artifact_manifest(manifest_path).to_dict()
    _assert_contract_frontier(frontier)
    _assert_frontier_report(report, expected_report_path=report_path, expected_manifest_path=manifest_path)
    _assert_manifest_verification(manifest_verification)
    return {
        "status": "pass",
        "workflow": "frontier_release_evidence_smoke",
        "contract": str(contract_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
        "decision_status": report["decision"]["status"],
        "track_statuses": {
            key: report["decision"].get(key)
            for key in REQUIRED_TRACK_STATUSES
        },
        "frontier_rerun_rollup_promoted_tracks": tuple(
            report["decision"].get("frontier_rerun_rollup_promoted_tracks", ())
        ),
        "run_count": len(report.get("run_decisions", ())),
        "manifest_checked": manifest_verification["checked"],
    }


def _assert_contract_frontier(frontier: Mapping[str, Any]) -> None:
    if frontier.get("workflow") != "frontier_release_evidence_comparison":
        raise AssertionError("contract frontier release evidence workflow changed.")
    if frontier.get("report_status") != "complete":
        raise AssertionError("contract frontier release evidence report is not complete.")
    if frontier.get("decision_status") != "promote":
        raise AssertionError("contract frontier release evidence did not promote.")
    if frontier.get("blocking_reasons") not in ((), []):
        raise AssertionError("contract frontier release evidence has blocking reasons.")
    for field, expected in CONTRACT_REQUIRED_TRACK_STATUSES.items():
        if frontier.get(field) != expected:
            raise AssertionError(f"contract frontier release evidence {field}={frontier.get(field)!r}.")
    promoted_tracks = set(frontier.get("frontier_rerun_rollup_promoted_tracks") or ())
    if not REQUIRED_RERUN_PROMOTED_TRACKS.issubset(promoted_tracks):
        raise AssertionError("contract frontier release evidence lost required rerun-promoted tracks.")
    if _int(frontier.get("frontier_rerun_rollup_report_count")) < 2:
        raise AssertionError("contract frontier release evidence has too few rerun rollup reports.")
    if _int(frontier.get("frontier_rerun_rollup_missing_report_count")) != 0:
        raise AssertionError("contract frontier release evidence has missing rerun rollup reports.")
    if _int(frontier.get("frontier_rerun_rollup_invalid_report_count")) != 0:
        raise AssertionError("contract frontier release evidence has invalid rerun rollup reports.")


def _assert_frontier_report(
    report: Mapping[str, Any],
    *,
    expected_report_path: Path,
    expected_manifest_path: Path,
) -> None:
    if report.get("workflow") != "frontier_release_evidence_comparison":
        raise AssertionError("frontier release evidence report workflow changed.")
    if report.get("status") != "complete":
        raise AssertionError("frontier release evidence report is not complete.")
    decision = _mapping(report.get("decision"))
    if decision.get("status") != "promote":
        raise AssertionError("frontier release evidence report did not promote.")
    if decision.get("blocking_reasons") not in ((), []):
        raise AssertionError("frontier release evidence report has blocking reasons.")
    for field, expected in REQUIRED_TRACK_STATUSES.items():
        if decision.get(field) != expected:
            raise AssertionError(f"frontier release evidence report {field}={decision.get(field)!r}.")
    promoted_tracks = set(decision.get("frontier_rerun_rollup_promoted_tracks") or ())
    if not REQUIRED_RERUN_PROMOTED_TRACKS.issubset(promoted_tracks):
        raise AssertionError("frontier release evidence report lost required rerun-promoted tracks.")
    paths = _mapping(report.get("paths"))
    manifest_path = paths.get("artifact_manifest")
    if manifest_path is not None and _repo_path(manifest_path) != expected_manifest_path:
        raise AssertionError("frontier release evidence manifest path does not match contract.")
    report_path = paths.get("frontier_release_evidence_report")
    if report_path is not None and _repo_path(report_path) != expected_report_path:
        raise AssertionError("frontier release evidence report path does not match contract.")
    if len(report.get("run_decisions", ())) < 2:
        raise AssertionError("frontier release evidence report checked too few runs.")


def _assert_manifest_verification(verification: Mapping[str, Any]) -> None:
    if verification.get("passed") is not True:
        raise AssertionError("frontier release evidence manifest verification failed.")
    if _int(verification.get("checked")) < 4:
        raise AssertionError("frontier release evidence manifest checked too few artifacts.")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise AssertionError(f"required frontier release evidence file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AssertionError(f"frontier release evidence payload is not an object: {path}")
    return payload


def _repo_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"invalid frontier release evidence path: {value!r}")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


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
    parser = argparse.ArgumentParser(description="Run the no-model frontier release evidence smoke check")
    parser.add_argument(
        "--contract",
        default=str(DEFAULT_CONTRACT_PATH),
        help="ProductPromotionContract JSON path",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="optional path to write the smoke report JSON",
    )
    args = parser.parse_args(argv)
    report = build_frontier_release_evidence_smoke(contract_path=Path(args.contract))
    if args.json is not None:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_report(report)


def _print_report(report: Mapping[str, Any]) -> None:
    print(
        "frontier_release_evidence_smoke_ok "
        f"status={report['status']} "
        f"decision={report['decision_status']} "
        f"runs={report['run_count']} "
        f"rerun_tracks={','.join(report['frontier_rerun_rollup_promoted_tracks'])} "
        f"manifest_checked={report['manifest_checked']}"
    )


if __name__ == "__main__":
    main()
