"""Build a next-evidence plan from blocked release-candidate reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.plan_frontier_multiple_testing_reruns import (  # noqa: E402
    build_frontier_multiple_testing_rerun_queue,
)
from eigentruth.control import plan_evidence_gaps_from_release_candidate  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry  # noqa: E402


def build_release_evidence_gap_plan(
    *,
    source: str | Path,
    json_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    multiple_testing_rerun_json_path: str | Path | None = None,
    multiple_testing_rerun_artifact_manifest_path: str | Path | None = None,
    multiple_testing_rerun_output_dir: str | Path | None = None,
    multiple_testing_rerun_name: str | None = None,
    multiple_testing_rerun_version: str | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Load a release report and optionally write/register its evidence-gap plan."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if multiple_testing_rerun_artifact_manifest_path is not None and multiple_testing_rerun_json_path is None:
        raise ValueError("multiple_testing_rerun_artifact_manifest_path requires multiple_testing_rerun_json_path.")
    if (multiple_testing_rerun_name or multiple_testing_rerun_version) and registry_path is None:
        raise ValueError("multiple_testing_rerun_name/version require registry_path.")
    if (multiple_testing_rerun_name is None) != (multiple_testing_rerun_version is None):
        raise ValueError("multiple_testing_rerun_name and multiple_testing_rerun_version must be provided together.")
    source_path = Path(source)
    payload = _load_json_object(source_path)
    plan = plan_evidence_gaps_from_release_candidate(
        payload,
        source_path=source_path,
        metadata=metadata,
    )
    output = plan.to_dict()
    derived_artifacts = _build_derived_artifacts(
        source_path=source_path,
        registry_path=None
        if multiple_testing_rerun_name is None or multiple_testing_rerun_version is None
        else registry_path,
        multiple_testing_rerun_json_path=multiple_testing_rerun_json_path,
        multiple_testing_rerun_artifact_manifest_path=multiple_testing_rerun_artifact_manifest_path,
        multiple_testing_rerun_output_dir=multiple_testing_rerun_output_dir,
        multiple_testing_rerun_name=multiple_testing_rerun_name,
        multiple_testing_rerun_version=multiple_testing_rerun_version,
        python_executable=python_executable,
    )
    if derived_artifacts:
        output = {**output, "derived_artifacts": derived_artifacts}
    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            strict_json_dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if registry_path is not None:
        ArtifactRegistry.load_json(registry_path).record_evidence_gap_plan(
            name=str(name),
            version=str(version),
            path=str(json_path) if json_path is not None else str(source_path),
            metadata={
                "source": str(source_path),
                "status": output["status"],
                "gap_count": output["summary"]["gap_count"],
                "action_count": output["summary"]["action_count"],
            },
        ).save_json()
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = build_release_evidence_gap_plan(
        source=args.source,
        json_path=args.json,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        multiple_testing_rerun_json_path=args.multiple_testing_rerun_json,
        multiple_testing_rerun_artifact_manifest_path=args.multiple_testing_rerun_artifact_manifest,
        multiple_testing_rerun_output_dir=args.multiple_testing_rerun_output_dir,
        multiple_testing_rerun_name=args.multiple_testing_rerun_name,
        multiple_testing_rerun_version=args.multiple_testing_rerun_version,
        python_executable=args.python,
    )
    summary = payload["summary"]
    print(
        "release_evidence_gap_plan="
        f"{payload['status']} "
        f"gaps={summary['gap_count']} "
        f"actions={summary['action_count']} "
        f"missing_metrics={summary['missing_metric_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a structured next-evidence plan from a blocked release report"
    )
    parser.add_argument("--source", required=True, help="release comparison or registry workflow JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--metadata", action="append", default=[], help="KEY=VALUE metadata; repeatable")
    parser.add_argument(
        "--multiple-testing-rerun-json",
        default=None,
        help="optional output JSON path for a derived frontier multiple-testing rerun queue",
    )
    parser.add_argument(
        "--multiple-testing-rerun-artifact-manifest",
        default=None,
        help="optional artifact manifest path for the derived multiple-testing rerun queue",
    )
    parser.add_argument(
        "--multiple-testing-rerun-output-dir",
        default=None,
        help="optional root directory for derived per-cell rerun outputs",
    )
    parser.add_argument(
        "--multiple-testing-rerun-name",
        default=None,
        help="optional registry name for the derived multiple-testing rerun queue",
    )
    parser.add_argument(
        "--multiple-testing-rerun-version",
        default=None,
        help="optional registry version for the derived multiple-testing rerun queue",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for generated rerun commands")
    run(parser.parse_args(argv))


def _build_derived_artifacts(
    *,
    source_path: Path,
    registry_path: str | Path | None,
    multiple_testing_rerun_json_path: str | Path | None,
    multiple_testing_rerun_artifact_manifest_path: str | Path | None,
    multiple_testing_rerun_output_dir: str | Path | None,
    multiple_testing_rerun_name: str | None,
    multiple_testing_rerun_version: str | None,
    python_executable: str,
) -> dict[str, Any]:
    if multiple_testing_rerun_json_path is None:
        return {}
    rerun_payload = build_frontier_multiple_testing_rerun_queue(
        source=source_path,
        json_path=multiple_testing_rerun_json_path,
        artifact_manifest_path=multiple_testing_rerun_artifact_manifest_path,
        registry_path=registry_path,
        name=multiple_testing_rerun_name,
        version=multiple_testing_rerun_version,
        output_dir=multiple_testing_rerun_output_dir,
        python_executable=python_executable,
    )
    summary = rerun_payload["summary"]
    return {
        "frontier_multiple_testing_rerun_queue": {
            "path": str(multiple_testing_rerun_json_path),
            "artifact_manifest": None
            if multiple_testing_rerun_artifact_manifest_path is None
            else str(multiple_testing_rerun_artifact_manifest_path),
            "status": rerun_payload["status"],
            "blocked_cell_count": summary["blocked_cell_count"],
            "command_count": summary["command_count"],
            "missing_command_count": summary["missing_command_count"],
        }
    }


def _load_json_object(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("source JSON must contain an object.")
    return data


def _parse_metadata(items: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"metadata must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key must be non-empty, got {item!r}")
        metadata[key] = value.strip()
    return metadata


if __name__ == "__main__":
    main()
