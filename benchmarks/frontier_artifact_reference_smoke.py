"""No-model smoke checks for active frontier artifact references."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.audit_frontier_artifact_references import (  # noqa: E402
    build_frontier_artifact_reference_audit,
)
from eigentruth.registry import ArtifactRegistry  # noqa: E402

SMOKE_NAME = "frontier-artifact-reference-smoke"
SMOKE_VERSION = "0.1"
SMOKE_RECORD_KEY = f"report:{SMOKE_NAME}:{SMOKE_VERSION}"


def build_frontier_artifact_reference_smoke(output_dir: Path) -> dict[str, Any]:
    """Verify active frontier/product artifact references from repository docs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "frontier-artifact-reference-audit.json"
    manifest_path = output_dir / "artifact-manifest.json"
    registry_path = output_dir / "registry.json"
    audit = build_frontier_artifact_reference_audit(
        root=REPO_ROOT,
        json_path=report_path,
        artifact_manifest_path=manifest_path,
        registry_path=registry_path,
        name=SMOKE_NAME,
        version=SMOKE_VERSION,
        metadata={"source": "frontier_artifact_reference_smoke"},
    )
    _assert_frontier_reference_audit(audit)
    registry_record = ArtifactRegistry.load_json(registry_path).get(SMOKE_RECORD_KEY)
    if registry_record.metadata.get("status") != "passed":
        raise AssertionError("frontier artifact reference smoke registry record did not pass.")
    return {
        "status": "pass",
        "workflow": "frontier_artifact_reference_smoke",
        "audit_report": str(report_path),
        "artifact_manifest": str(manifest_path),
        "registry": str(registry_path),
        "registry_record": registry_record.key(),
        "summary": audit["summary"],
    }


def _assert_frontier_reference_audit(audit: Mapping[str, Any]) -> None:
    if audit.get("status") != "passed":
        raise AssertionError("frontier artifact reference audit did not pass.")
    summary = _mapping(audit.get("summary"))
    if _int(summary.get("reference_count")) < 10:
        raise AssertionError("frontier artifact reference audit checked too few references.")
    if _int(summary.get("existing_count")) != _int(summary.get("reference_count")):
        raise AssertionError("frontier artifact reference audit has non-existing references.")
    if _int(summary.get("missing_count")) != 0:
        raise AssertionError("frontier artifact reference audit has missing references.")
    if _int(summary.get("manifest_verified_count")) < 3:
        raise AssertionError("frontier artifact reference audit verified too few manifests.")
    if _int(summary.get("manifest_failed_count")) != 0:
        raise AssertionError("frontier artifact reference audit has failed manifests.")
    if _int(summary.get("manifest_child_missing_count")) != 0:
        raise AssertionError("frontier artifact reference audit has missing manifest children.")
    if _int(summary.get("recommended_action_count")) != 0:
        raise AssertionError("frontier artifact reference audit still recommends repair actions.")
    manifest_summary = _mapping(audit.get("artifact_manifest_summary"))
    if _int(manifest_summary.get("missing_count")) != 0:
        raise AssertionError("frontier artifact reference audit manifest has missing artifacts.")
    if audit.get("recommended_actions") not in ((), []):
        raise AssertionError("frontier artifact reference audit emitted repair actions.")
    if audit.get("blocking_reasons") not in ((), []):
        raise AssertionError("frontier artifact reference audit emitted blocking reasons.")


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
    parser = argparse.ArgumentParser(description="Run the no-model frontier artifact reference smoke check")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="optional output directory; defaults to a temporary directory",
    )
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        report = build_frontier_artifact_reference_smoke(Path(args.output_dir))
        _print_report(report)
        return
    with tempfile.TemporaryDirectory(prefix="eigentruth-frontier-artifact-reference-smoke-") as tmpdir:
        report = build_frontier_artifact_reference_smoke(Path(tmpdir))
        _print_report(report)


def _print_report(report: Mapping[str, Any]) -> None:
    summary = _mapping(report["summary"])
    print(
        "frontier_artifact_reference_smoke_ok "
        f"status={report['status']} "
        f"references={summary.get('reference_count')} "
        f"manifests={summary.get('manifest_verified_count')} "
        f"missing={summary.get('missing_count')} "
        f"record={report['registry_record']}"
    )


if __name__ == "__main__":
    main()
