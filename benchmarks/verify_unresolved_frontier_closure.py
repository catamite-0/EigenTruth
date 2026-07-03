"""Verify that unresolved frontier targets are closed by promoted evidence.

This is a final coordination check for ``unresolved_frontier_evidence_summary``.
It does not create new verifier evidence. It fail-closes unless the summary has
only the terminal closure action left and the scoped covered-fact route covers
the unresolved target queue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "unresolved_frontier_closure_verification"
SOURCE_WORKFLOW = "unresolved_frontier_evidence_summary"
TERMINAL_ACTION = "verify_unresolved_targets_are_closed"


def verify_unresolved_frontier_closure(
    summary: Mapping[str, Any],
    *,
    summary_path: str | Path | None = None,
    min_coverage_rate: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a pass/block closure report for an unresolved frontier summary."""
    min_coverage_rate = _unit_float(min_coverage_rate, name="min_coverage_rate")
    checks = (
        _check_equal(
            "source_workflow",
            summary.get("workflow"),
            SOURCE_WORKFLOW,
        ),
        _check_in(
            "source_status",
            summary.get("status"),
            {"needs_evidence", "promote"},
        ),
        _check_terminal_actions(summary),
        _check_equal(
            "source_family_acquisition_status",
            _nested(summary, "lanes", "source_family_acquisition", "status"),
            "covered",
        ),
        _check_equal(
            "semantic_gap_review_status",
            _nested(summary, "lanes", "semantic_gap_review", "status"),
            "promote",
        ),
        _check_equal(
            "frontier_queue_execution_status",
            _nested(summary, "lanes", "frontier_queue_execution", "status"),
            "promote",
        ),
        _check_equal(
            "world_model_rules_status",
            _nested(summary, "lanes", "world_model_rules", "status"),
            "promote",
        ),
        _check_at_most(
            "semantic_gap_review_coverage_gap_count",
            _nested(summary, "summary", "semantic_gap_review_coverage_gap_count"),
            0,
        ),
        _check_at_least(
            "semantic_gap_review_coverage_rate",
            _nested(summary, "summary", "semantic_gap_review_coverage_rate"),
            min_coverage_rate,
        ),
        _check_at_least(
            "semantic_gap_review_covered_fact_route_identity_n_records",
            _nested(
                summary,
                "summary",
                "semantic_gap_review_covered_fact_route_identity_n_records",
            ),
            _int(_nested(summary, "summary", "unresolved_target_count")),
        ),
    )
    blocking_reasons = tuple(
        str(check["name"]) for check in checks if check.get("status") != "pass"
    )
    status = "pass" if not blocking_reasons else "blocked"
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "source_summary": {
            "path": None if summary_path is None else str(summary_path),
            "workflow": summary.get("workflow"),
            "status": summary.get("status"),
            "next_action_count": len(_sequence(summary.get("next_actions"))),
            "next_action_ids": tuple(
                str(action.get("action_id") or "")
                for action in _mapping_sequence(summary.get("next_actions"))
            ),
        },
        "config": {"min_coverage_rate": min_coverage_rate},
        "summary": {
            "unresolved_target_count": _int(
                _nested(summary, "summary", "unresolved_target_count")
            ),
            "semantic_gap_review_coverage_gap_count": _int(
                _nested(summary, "summary", "semantic_gap_review_coverage_gap_count")
            ),
            "semantic_gap_review_coverage_rate": _optional_float(
                _nested(summary, "summary", "semantic_gap_review_coverage_rate")
            ),
            "semantic_gap_review_covered_fact_route_identity_n_records": _int(
                _nested(
                    summary,
                    "summary",
                    "semantic_gap_review_covered_fact_route_identity_n_records",
                )
            ),
            "check_count": len(checks),
            "passed_check_count": sum(1 for check in checks if check["status"] == "pass"),
            "blocked_check_count": len(blocking_reasons),
        },
        "checks": checks,
        "decision": {
            "status": status,
            "blocking_reasons": blocking_reasons,
        },
        "metadata": dict(metadata or {}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    summary_path = Path(args.summary)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("summary must contain a JSON object.")
    metadata = _parse_metadata(args.metadata or ())
    report = verify_unresolved_frontier_closure(
        payload,
        summary_path=summary_path,
        min_coverage_rate=float(args.min_coverage_rate),
        metadata=metadata,
    )
    output_path = Path(args.json)
    manifest_path = None if args.artifact_manifest is None else Path(args.artifact_manifest)
    _write_json(output_path, report, compact=bool(args.compact_json))
    manifest = None
    if manifest_path is not None:
        manifest = build_artifact_manifest(
            {
                "closure_verification_report": output_path,
                "unresolved_frontier_evidence_summary": summary_path,
            },
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "status": report["status"],
                "source_summary_status": report["source_summary"]["status"],
                "source_next_action_count": report["source_summary"]["next_action_count"],
                "unresolved_target_count": report["summary"]["unresolved_target_count"],
                "semantic_gap_review_coverage_gap_count": report["summary"][
                    "semantic_gap_review_coverage_gap_count"
                ],
                "semantic_gap_review_coverage_rate": report["summary"][
                    "semantic_gap_review_coverage_rate"
                ],
                **metadata,
            },
        )
        _write_json(manifest_path, manifest, compact=False)
    if args.registry is not None:
        if not args.name or not args.version:
            raise ValueError("--registry requires --name and --version.")
        ArtifactRegistry.load_json(args.registry).record_report(
            name=str(args.name),
            version=str(args.version),
            path=output_path,
            metadata={
                "workflow": WORKFLOW,
                "status": report["status"],
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "source_summary": str(summary_path),
                "unresolved_target_count": report["summary"]["unresolved_target_count"],
                "semantic_gap_review_coverage_gap_count": report["summary"][
                    "semantic_gap_review_coverage_gap_count"
                ],
                **metadata,
            },
        ).save_json()
    print(
        "unresolved_frontier_closure_verification_ok "
        f"status={report['status']} output={output_path}"
    )
    return report


def _check_equal(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    status = "pass" if actual == expected else "blocked"
    return {"name": name, "status": status, "expected": expected, "actual": actual}


def _check_in(name: str, actual: Any, expected: set[str]) -> dict[str, Any]:
    status = "pass" if actual in expected else "blocked"
    return {
        "name": name,
        "status": status,
        "expected": tuple(sorted(expected)),
        "actual": actual,
    }


def _check_at_most(name: str, actual: Any, expected: float) -> dict[str, Any]:
    number = _optional_float(actual)
    status = "pass" if number is not None and number <= expected else "blocked"
    return {"name": name, "status": status, "expected_max": expected, "actual": number}


def _check_at_least(name: str, actual: Any, expected: float) -> dict[str, Any]:
    number = _optional_float(actual)
    status = "pass" if number is not None and number >= expected else "blocked"
    return {"name": name, "status": status, "expected_min": expected, "actual": number}


def _check_terminal_actions(summary: Mapping[str, Any]) -> dict[str, Any]:
    action_ids = tuple(
        str(action.get("action_id") or "")
        for action in _mapping_sequence(summary.get("next_actions"))
    )
    allowed = (TERMINAL_ACTION,)
    status = "pass" if not action_ids or action_ids == allowed else "blocked"
    return {
        "name": "terminal_next_actions",
        "status": status,
        "expected": allowed,
        "actual": action_ids,
    }


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _unit_float(value: Any, *, name: str) -> float:
    number = _optional_float(value)
    if number is None or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return number


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--min-coverage-rate", type=float, default=1.0)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
