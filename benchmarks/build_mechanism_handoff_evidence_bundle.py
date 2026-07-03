"""Bundle promoted mechanism handoff reports into release-gate evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "mechanism_handoff_evidence_bundle"
HANDOFF_WORKFLOW = "world_model_rule_candidate_handoff"


def build_mechanism_handoff_evidence_bundle(
    handoffs: Sequence[Mapping[str, Any]],
    *,
    handoff_paths: Sequence[str | Path | None] = (),
    handoff_manifest_paths: Sequence[str | Path | None] = (),
    product_traces: Sequence[Sequence[Mapping[str, Any]]] = (),
    expected_target_count: int | None = None,
    min_trace_count: int | None = None,
    min_supported_count: int | None = None,
    min_refuted_count: int | None = None,
    min_source_citation_count: int | None = None,
    require_action_execution_alignment: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a release-gate-ready bundle summary for promoted handoff reports."""
    handoff_paths = tuple(handoff_paths)
    handoff_manifest_paths = tuple(handoff_manifest_paths)
    product_traces = tuple(tuple(trace for trace in group) for group in product_traces)
    handoff_rows = []
    failures: list[str] = []
    total_trace_count = 0
    total_source_citation_count = 0
    total_blocked_candidate_count = 0
    verification_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    rule_family_counts: Counter[str] = Counter()
    source_family_counts: Counter[str] = Counter()
    target_ids: set[str] = set()

    if not handoffs:
        failures.append("mechanism handoff bundle has no handoff reports")

    for index, handoff in enumerate(handoffs, start=1):
        summary = _mapping(handoff.get("summary"))
        workflow = handoff.get("workflow")
        status = handoff.get("status")
        trace_count = _int(summary.get("trace_count")) or 0
        source_citation_count = _int(summary.get("source_citation_count")) or 0
        blocked_candidate_count = _int(summary.get("blocked_candidate_count")) or 0
        action_execution_alignment_passed = summary.get("action_execution_alignment_passed")
        report_path = _optional_indexed(handoff_paths, index - 1)
        manifest_path = _optional_indexed(handoff_manifest_paths, index - 1)
        trace_rows = _optional_indexed(product_traces, index - 1, default=())
        trace_target_ids = _trace_target_ids(trace_rows)
        trace_source_families = _trace_source_family_counts(trace_rows)

        if workflow != HANDOFF_WORKFLOW:
            failures.append(
                f"handoff {index} workflow is {workflow!r}, expected {HANDOFF_WORKFLOW!r}"
            )
        if status != "promote":
            failures.append(f"handoff {index} status is {status!r}, expected 'promote'")
        if blocked_candidate_count:
            failures.append(
                f"handoff {index} has blocked candidates: {blocked_candidate_count}"
            )
        if require_action_execution_alignment and action_execution_alignment_passed is not True:
            failures.append(f"handoff {index} action execution alignment did not pass")
        if manifest_path is None:
            failures.append(f"handoff {index} artifact manifest is missing")

        total_trace_count += trace_count
        total_source_citation_count += source_citation_count
        total_blocked_candidate_count += blocked_candidate_count
        verification_counts.update(_counter_mapping(summary.get("verification_status_counts")))
        action_counts.update(_counter_mapping(summary.get("action_counts")))
        rule_family_counts.update(_counter_mapping(summary.get("rule_family_counts")))
        source_family_counts.update(trace_source_families)
        target_ids.update(trace_target_ids)
        handoff_rows.append({
            "index": index,
            "report_path": None if report_path is None else str(report_path),
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "workflow": workflow,
            "status": status,
            "trace_count": trace_count,
            "blocked_candidate_count": blocked_candidate_count,
            "source_citation_count": source_citation_count,
            "verification_status_counts": _sorted_counter(
                _counter_mapping(summary.get("verification_status_counts"))
            ),
            "action_counts": _sorted_counter(_counter_mapping(summary.get("action_counts"))),
            "rule_family_counts": _sorted_counter(
                _counter_mapping(summary.get("rule_family_counts"))
            ),
            "source_family_counts": _sorted_counter(trace_source_families),
            "target_ids": tuple(sorted(trace_target_ids)),
            "action_execution_alignment_passed": action_execution_alignment_passed,
        })

    effective_min_trace_count = (
        expected_target_count
        if min_trace_count is None and expected_target_count is not None
        else min_trace_count
    )
    effective_min_trace_count = 1 if effective_min_trace_count is None else effective_min_trace_count
    effective_min_source_citation_count = (
        total_trace_count
        if min_source_citation_count is None
        else min_source_citation_count
    )
    if total_trace_count < effective_min_trace_count:
        failures.append(
            "mechanism handoff trace count below "
            f"{effective_min_trace_count}: {total_trace_count}"
        )
    if total_source_citation_count < effective_min_source_citation_count:
        failures.append(
            "mechanism handoff source citation count below "
            f"{effective_min_source_citation_count}: {total_source_citation_count}"
        )
    supported_count = int(verification_counts.get("supported", 0))
    refuted_count = int(verification_counts.get("refuted", 0))
    if min_supported_count is not None and supported_count < min_supported_count:
        failures.append(
            f"mechanism handoff supported count below {min_supported_count}: {supported_count}"
        )
    if min_refuted_count is not None and refuted_count < min_refuted_count:
        failures.append(
            f"mechanism handoff refuted count below {min_refuted_count}: {refuted_count}"
        )
    if expected_target_count is not None and len(target_ids) < expected_target_count:
        failures.append(
            "mechanism handoff target coverage below "
            f"{expected_target_count}: {len(target_ids)}"
        )

    status = "promote" if not failures else "blocked"
    target_coverage_rate = (
        None
        if expected_target_count is None
        else (len(target_ids) / expected_target_count if expected_target_count else 0.0)
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Release-gate bundle for target-specific mechanism world-model rule "
            "candidate handoffs. This is not an open-domain verifier claim."
        ),
        "config": {
            "expected_target_count": expected_target_count,
            "min_trace_count": effective_min_trace_count,
            "min_supported_count": min_supported_count,
            "min_refuted_count": min_refuted_count,
            "min_source_citation_count": effective_min_source_citation_count,
            "require_action_execution_alignment": require_action_execution_alignment,
        },
        "summary": {
            "handoff_count": len(handoffs),
            "trace_count": total_trace_count,
            "blocked_candidate_count": total_blocked_candidate_count,
            "source_citation_count": total_source_citation_count,
            "target_count": len(target_ids),
            "target_coverage_rate": target_coverage_rate,
            "verification_status_counts": _sorted_counter(verification_counts),
            "action_counts": _sorted_counter(action_counts),
            "rule_family_counts": _sorted_counter(rule_family_counts),
            "source_family_counts": _sorted_counter(source_family_counts),
            "target_ids": tuple(sorted(target_ids)),
        },
        "gate": {
            "passed": not failures,
            "status": status,
            "blocking_reasons": tuple(failures),
            "policy": {
                "expected_target_count": expected_target_count,
                "min_trace_count": effective_min_trace_count,
                "min_supported_count": min_supported_count,
                "min_refuted_count": min_refuted_count,
                "min_source_citation_count": effective_min_source_citation_count,
                "require_action_execution_alignment": require_action_execution_alignment,
            },
        },
        "handoffs": tuple(handoff_rows),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    handoff_paths: Sequence[str | Path],
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    expected_target_count: int | None = None,
    min_trace_count: int | None = None,
    min_supported_count: int | None = None,
    min_refuted_count: int | None = None,
    min_source_citation_count: int | None = None,
    require_action_execution_alignment: bool = True,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build, write, manifest, and optionally register a mechanism handoff bundle."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "mechanism-handoff-evidence-bundle.json")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    reports = tuple(_load_json_object(path) for path in handoff_paths)
    resolved_handoff_paths = tuple(Path(path) for path in handoff_paths)
    resolved_manifest_paths = tuple(
        _handoff_manifest_path(report, report_path=report_path)
        for report, report_path in zip(reports, resolved_handoff_paths, strict=False)
    )
    trace_paths = tuple(
        _resolve_optional_path(_nested(report, "paths", "product_traces"), base_path=path)
        for report, path in zip(reports, resolved_handoff_paths, strict=False)
    )
    action_result_paths = tuple(
        _resolve_optional_path(_nested(report, "paths", "action_results"), base_path=path)
        for report, path in zip(reports, resolved_handoff_paths, strict=False)
    )
    traces = tuple(
        _load_jsonl_mappings(path) if path is not None and path.exists() else ()
        for path in trace_paths
    )
    bundle = build_mechanism_handoff_evidence_bundle(
        reports,
        handoff_paths=resolved_handoff_paths,
        handoff_manifest_paths=resolved_manifest_paths,
        product_traces=traces,
        expected_target_count=expected_target_count,
        min_trace_count=min_trace_count,
        min_supported_count=min_supported_count,
        min_refuted_count=min_refuted_count,
        min_source_citation_count=min_source_citation_count,
        require_action_execution_alignment=require_action_execution_alignment,
        metadata=metadata,
    )
    bundle = dict(bundle)
    bundle["paths"] = {
        "report": str(report_path),
        "artifact_manifest": str(manifest_path),
        "handoff_reports": tuple(str(path) for path in resolved_handoff_paths),
        "handoff_manifests": tuple(
            None if path is None else str(path) for path in resolved_manifest_paths
        ),
        "product_traces": tuple(None if path is None else str(path) for path in trace_paths),
        "action_results": tuple(None if path is None else str(path) for path in action_result_paths),
    }
    _write_json(report_path, bundle, compact=compact_json)

    artifacts: dict[str, str | Path | None] = {
        "mechanism_handoff_evidence_bundle": report_path,
    }
    for index, path in enumerate(resolved_handoff_paths, start=1):
        artifacts[f"handoff_report_{index}"] = path
    for index, path in enumerate(resolved_manifest_paths, start=1):
        if path is not None:
            artifacts[f"handoff_manifest_{index}"] = path
    for index, path in enumerate(trace_paths, start=1):
        if path is not None:
            artifacts[f"handoff_product_traces_{index}"] = path
    for index, path in enumerate(action_result_paths, start=1):
        if path is not None:
            artifacts[f"handoff_action_results_{index}"] = path
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": bundle["status"],
            "trace_count": bundle["summary"]["trace_count"],
            "handoff_count": bundle["summary"]["handoff_count"],
            "target_count": bundle["summary"]["target_count"],
            "source_citation_count": bundle["summary"]["source_citation_count"],
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": bundle["status"],
                "trace_count": bundle["summary"]["trace_count"],
                "handoff_count": bundle["summary"]["handoff_count"],
                "target_count": bundle["summary"]["target_count"],
                "source_citation_count": bundle["summary"]["source_citation_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return bundle


def _handoff_manifest_path(report: Mapping[str, Any], *, report_path: Path) -> Path | None:
    raw_path = _nested(report, "paths", "artifact_manifest")
    if raw_path is None:
        sibling = report_path.parent / "artifact-manifest.json"
        return sibling if sibling.exists() else None
    return _resolve_path(raw_path, base_path=report_path)


def _trace_target_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        target_id = _nested(row, "metadata", "target_id")
        if target_id:
            values.add(str(target_id))
    return values


def _trace_source_family_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        for claim in _sequence_of_mappings(row.get("claims")):
            source_family = _nested(claim, "metadata", "source_family")
            if source_family:
                counter[str(source_family)] += 1
    return counter


def _counter_mapping(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, raw_count in value.items():
        count = _int(raw_count)
        if count is None:
            continue
        counter[str(key)] += count
    return counter


def _sorted_counter(counter: Mapping[str, int]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        current = _mapping(current).get(key)
    return current


def _sequence_of_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _optional_indexed(
    values: Sequence[Any],
    index: int,
    *,
    default: Any = None,
) -> Any:
    return values[index] if index < len(values) else default


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_optional_path(raw_path: Any, *, base_path: Path) -> Path | None:
    if raw_path is None:
        return None
    return _resolve_path(raw_path, base_path=base_path)


def _resolve_path(raw_path: Any, *, base_path: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base_path.parent / path).resolve()


def _load_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(dict(row))
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _parse_key_values(values: Sequence[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not values:
        return parsed
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"metadata item {item!r} must use key=value format.")
            key, raw = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("metadata key must be non-empty.")
            parsed[key] = raw.strip()
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", action="append", required=True, help="handoff report JSON path")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--json", default=None, help="optional bundle report path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--expected-target-count", type=_non_negative_int, default=None)
    parser.add_argument("--min-trace-count", type=_non_negative_int, default=None)
    parser.add_argument("--min-supported-count", type=_non_negative_int, default=None)
    parser.add_argument("--min-refuted-count", type=_non_negative_int, default=None)
    parser.add_argument("--min-source-citation-count", type=_non_negative_int, default=None)
    parser.add_argument(
        "--allow-action-execution-misalignment",
        action="store_true",
        help="do not require each handoff action-execution summary to pass alignment",
    )
    parser.add_argument("--metadata", action="append", default=None, help="key=value metadata")
    parser.add_argument("--compact-json", action="store_true", help="write minified JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run(
        handoff_paths=tuple(args.handoff),
        output_dir=args.output_dir,
        report_json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        expected_target_count=args.expected_target_count,
        min_trace_count=args.min_trace_count,
        min_supported_count=args.min_supported_count,
        min_refuted_count=args.min_refuted_count,
        min_source_citation_count=args.min_source_citation_count,
        require_action_execution_alignment=not args.allow_action_execution_misalignment,
        metadata=_parse_key_values(args.metadata),
        compact_json=bool(args.compact_json),
    )
    print(strict_json_dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
