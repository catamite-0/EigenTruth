"""Register covered-facts route manifests and compare them as external evidence.

This workflow is a thin reproducibility wrapper around
``compare_external_evidence_baselines.py``. It stages one or more saved
Wikidata covered-facts route manifests in a local ``ArtifactRegistry``, then
runs the fail-closed external-evidence comparator with the covered-facts gate
enabled.

It performs no model, network, retrieval, or database work.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.compare_external_evidence_baselines import (  # noqa: E402
    _record_registry,
    _verify_manifest,
    _write_artifact_manifest,
    compare_external_evidence_baselines,
)
from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

DEFAULT_WORKFLOW_NAME = "covered-facts-external-evidence-handoff"
DEFAULT_VERSION = "0.4"


@dataclass(frozen=True)
class CoveredFactsExternalEvidenceWorkflowConfig:
    """Configuration for the covered-facts external-evidence handoff workflow."""

    route_manifests: Mapping[str, str | Path]
    output_dir: str | Path = "artifacts/covered-facts-external-evidence-handoff"
    route_registry_path: str | Path | None = None
    comparison_report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    verification_report_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str = DEFAULT_WORKFLOW_NAME
    version: str = DEFAULT_VERSION
    route_record_name_prefix: str | None = None
    route_record_version: str | None = None
    covered_fact_routes: Sequence[str] = ("structured_fact",)
    min_covered_fact_records: int | None = 1
    min_covered_fact_source_documents: int | None = 1
    min_covered_fact_true: int | None = 1
    min_covered_fact_false: int | None = 1
    min_covered_fact_properties: int | None = None
    min_covered_fact_property_records: int | None = None
    min_covered_fact_property_source_documents: int | None = None
    min_covered_fact_property_decision_accuracy: float | None = None
    max_covered_fact_property_false_supported_rate: float | None = None
    min_covered_fact_property_false_refuted_rate: float | None = None
    recursive: bool = True
    allow_unverified: bool = False
    fail_on_blocked: bool = False
    manifest_fingerprint_workers: int = 1
    notes: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    compact_json: bool = False

    def __post_init__(self) -> None:
        if not self.route_manifests:
            raise ValueError("route_manifests must contain at least one label=manifest path.")
        route_manifests: dict[str, Path] = {}
        for raw_label, raw_path in self.route_manifests.items():
            label = _normalize_label(raw_label)
            if label in route_manifests:
                raise ValueError(f"duplicate route manifest label {label!r}.")
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"route manifest does not exist: {path}")
            route_manifests[label] = path

        object.__setattr__(self, "route_manifests", route_manifests)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for field_name in (
            "route_registry_path",
            "comparison_report_path",
            "artifact_manifest_path",
            "verification_report_path",
            "registry_path",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        object.__setattr__(self, "covered_fact_routes", tuple(str(route) for route in self.covered_fact_routes))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "manifest_fingerprint_workers",
            _positive_int(self.manifest_fingerprint_workers, name="manifest_fingerprint_workers"),
        )
        for field_name in (
            "min_covered_fact_records",
            "min_covered_fact_source_documents",
            "min_covered_fact_true",
            "min_covered_fact_false",
            "min_covered_fact_properties",
            "min_covered_fact_property_records",
            "min_covered_fact_property_source_documents",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_non_negative_int(getattr(self, field_name), name=field_name),
            )
        for field_name in (
            "min_covered_fact_property_decision_accuracy",
            "max_covered_fact_property_false_supported_rate",
            "min_covered_fact_property_false_refuted_rate",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_unit_float(getattr(self, field_name), name=field_name),
            )

    @property
    def resolved_route_registry_path(self) -> Path:
        if self.route_registry_path is not None:
            return Path(self.route_registry_path)
        return Path(self.output_dir) / "route-registry.json"

    @property
    def resolved_comparison_report_path(self) -> Path:
        if self.comparison_report_path is not None:
            return Path(self.comparison_report_path)
        return Path(self.output_dir) / "external-evidence-baseline-comparison.json"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_dir) / "artifact-manifest.json"

    @property
    def resolved_verification_report_path(self) -> Path:
        if self.verification_report_path is not None:
            return Path(self.verification_report_path)
        return Path(self.output_dir) / "manifest-verification.json"

    @property
    def resolved_route_record_version(self) -> str:
        return str(self.route_record_version or self.version)

    @property
    def resolved_route_record_name_prefix(self) -> str:
        return str(self.route_record_name_prefix or self.name)


def run_covered_facts_external_evidence_workflow(
    config: CoveredFactsExternalEvidenceWorkflowConfig,
) -> dict[str, Any]:
    """Run the full covered-facts registry handoff workflow."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_registry_path = config.resolved_route_registry_path
    route_registry_path.parent.mkdir(parents=True, exist_ok=True)
    route_keys = _register_route_manifests(config, route_registry_path=route_registry_path)

    payload = compare_external_evidence_baselines(
        route_registry_path=route_registry_path,
        route_baseline_keys=route_keys,
        require_route_baseline=True,
        recursive=config.recursive,
        allow_unverified=config.allow_unverified,
        require_covered_facts_route=True,
        covered_fact_routes=config.covered_fact_routes,
        min_covered_fact_records=config.min_covered_fact_records,
        min_covered_fact_source_documents=config.min_covered_fact_source_documents,
        min_covered_fact_true=config.min_covered_fact_true,
        min_covered_fact_false=config.min_covered_fact_false,
        min_covered_fact_properties=config.min_covered_fact_properties,
        min_covered_fact_property_records=config.min_covered_fact_property_records,
        min_covered_fact_property_source_documents=config.min_covered_fact_property_source_documents,
        min_covered_fact_property_decision_accuracy=config.min_covered_fact_property_decision_accuracy,
        max_covered_fact_property_false_supported_rate=(
            config.max_covered_fact_property_false_supported_rate
        ),
        min_covered_fact_property_false_refuted_rate=(
            config.min_covered_fact_property_false_refuted_rate
        ),
        notes=config.notes,
    )
    payload["workflow_runner"] = "run_covered_facts_external_evidence_workflow"
    payload["workflow_handoff"] = {
        "route_registry_path": str(route_registry_path),
        "registered_route_keys": list(route_keys),
        "route_manifest_labels": sorted(config.route_manifests),
        "metadata": dict(config.metadata),
    }
    payload["paths"] = {
        "route_registry": str(route_registry_path),
        "external_evidence_baseline_comparison_report": str(config.resolved_comparison_report_path),
        "artifact_manifest": str(config.resolved_artifact_manifest_path),
        "manifest_verification": str(config.resolved_verification_report_path),
    }

    report_path = config.resolved_comparison_report_path
    _write_json(report_path, payload, compact=config.compact_json)

    context = ArtifactVerificationContext()
    manifest = _write_artifact_manifest(
        context=context,
        report_path=report_path,
        output_path=config.resolved_artifact_manifest_path,
        payload=payload,
        max_workers=config.manifest_fingerprint_workers,
    )
    payload["artifact_manifest_summary"] = manifest.get("summary")
    _write_json(report_path, payload, compact=config.compact_json)
    manifest = _write_artifact_manifest(
        context=context,
        report_path=report_path,
        output_path=config.resolved_artifact_manifest_path,
        payload=payload,
        max_workers=config.manifest_fingerprint_workers,
    )
    verification = _verify_manifest(
        context=context,
        manifest_path=config.resolved_artifact_manifest_path,
        output_path=config.resolved_verification_report_path,
        recursive=config.recursive,
        max_workers=config.manifest_fingerprint_workers,
    )
    _record_registry(
        registry_path=None if config.registry_path is None else Path(config.registry_path),
        name=config.name,
        version=config.version,
        report_path=report_path,
        manifest_path=config.resolved_artifact_manifest_path,
        verification_path=config.resolved_verification_report_path,
        payload=payload,
        manifest=manifest,
        verification=verification,
    )
    if config.fail_on_blocked and _mapping(payload.get("decision")).get("status") != "promote":
        raise SystemExit(1)
    return payload


def _register_route_manifests(
    config: CoveredFactsExternalEvidenceWorkflowConfig,
    *,
    route_registry_path: Path,
) -> tuple[str, ...]:
    registry = ArtifactRegistry.load_json(route_registry_path)
    route_keys: list[str] = []
    for label, manifest_path in config.route_manifests.items():
        manifest = _load_json_mapping(manifest_path)
        manifest_metadata = _mapping(manifest.get("metadata"))
        record_name = f"{config.resolved_route_record_name_prefix}-{label}"
        metadata = {
            **dict(manifest_metadata),
            "workflow": manifest_metadata.get("workflow"),
            "status": manifest_metadata.get("status"),
            "route": manifest_metadata.get("route"),
            "source_route_manifest": str(manifest_path),
            "route_manifest_label": label,
            "manifest_metadata": dict(manifest_metadata),
            "manifest_summary": dict(_mapping(manifest.get("summary"))),
            **dict(config.metadata),
        }
        registry = registry.record_benchmark_manifest(
            name=record_name,
            path=manifest_path,
            version=config.resolved_route_record_version,
            metadata=metadata,
        )
        route_keys.append(f"benchmark_manifest:{record_name}:{config.resolved_route_record_version}")
    registry.save_json()
    return tuple(route_keys)


def _parse_route_manifest(value: str) -> tuple[str, Path]:
    text = str(value).strip()
    if not text:
        raise ValueError("--route-manifest must be non-empty.")
    if "=" in text:
        label, raw_path = text.split("=", 1)
        return _normalize_label(label), Path(raw_path)
    path = Path(text)
    return _normalize_label(path.parent.name or path.stem), path


def _parse_route_manifests(values: Sequence[str]) -> dict[str, Path]:
    route_manifests: dict[str, Path] = {}
    for value in values:
        label, path = _parse_route_manifest(value)
        if label in route_manifests:
            raise ValueError(f"duplicate --route-manifest label {label!r}.")
        route_manifests[label] = path
    return route_manifests


def _normalize_label(value: Any) -> str:
    label = str(value).strip().replace("_", "-")
    if not label:
        raise ValueError("route manifest labels must be non-empty.")
    return label


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _optional_non_negative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _positive_int(value: Any, *, name: str) -> int:
    numeric = int(value)
    if numeric < 1:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _optional_unit_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


def _parse_metadata(values: Sequence[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not values:
        return metadata
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
            metadata[key] = raw.strip()
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register covered-facts route manifests and compare them as external evidence"
    )
    parser.add_argument(
        "--route-manifest",
        action="append",
        required=True,
        help="covered-facts route manifest, either label=path or path",
    )
    parser.add_argument("--output-dir", default="artifacts/covered-facts-external-evidence-handoff")
    parser.add_argument("--route-registry", default=None)
    parser.add_argument("--json", default=None, help="comparison report output path")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--registry", default=None, help="optional registry for comparison handoff")
    parser.add_argument("--name", default=DEFAULT_WORKFLOW_NAME)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--route-record-name-prefix", default=None)
    parser.add_argument("--route-record-version", default=None)
    parser.add_argument("--covered-fact-route", action="append", default=None)
    parser.add_argument("--min-covered-fact-records", type=int, default=1)
    parser.add_argument("--min-covered-fact-source-documents", type=int, default=1)
    parser.add_argument("--min-covered-fact-true", type=int, default=1)
    parser.add_argument("--min-covered-fact-false", type=int, default=1)
    parser.add_argument("--min-covered-fact-properties", type=int, default=None)
    parser.add_argument("--min-covered-fact-property-records", type=int, default=None)
    parser.add_argument("--min-covered-fact-property-source-documents", type=int, default=None)
    parser.add_argument("--min-covered-fact-property-decision-accuracy", type=float, default=None)
    parser.add_argument("--max-covered-fact-property-false-supported-rate", type=float, default=None)
    parser.add_argument("--min-covered-fact-property-false-refuted-rate", type=float, default=None)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    parser.add_argument("--manifest-fingerprint-workers", type=int, default=1)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--metadata", action="append", default=None, help="optional key=value metadata")
    parser.add_argument("--compact-json", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    routes = tuple(args.covered_fact_route or ())
    config = CoveredFactsExternalEvidenceWorkflowConfig(
        route_manifests=_parse_route_manifests(args.route_manifest),
        output_dir=args.output_dir,
        route_registry_path=args.route_registry,
        comparison_report_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        verification_report_path=args.verification_report,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        route_record_name_prefix=args.route_record_name_prefix,
        route_record_version=args.route_record_version,
        covered_fact_routes=routes or ("structured_fact",),
        min_covered_fact_records=args.min_covered_fact_records,
        min_covered_fact_source_documents=args.min_covered_fact_source_documents,
        min_covered_fact_true=args.min_covered_fact_true,
        min_covered_fact_false=args.min_covered_fact_false,
        min_covered_fact_properties=args.min_covered_fact_properties,
        min_covered_fact_property_records=args.min_covered_fact_property_records,
        min_covered_fact_property_source_documents=args.min_covered_fact_property_source_documents,
        min_covered_fact_property_decision_accuracy=args.min_covered_fact_property_decision_accuracy,
        max_covered_fact_property_false_supported_rate=(
            args.max_covered_fact_property_false_supported_rate
        ),
        min_covered_fact_property_false_refuted_rate=args.min_covered_fact_property_false_refuted_rate,
        recursive=not args.no_recursive,
        allow_unverified=bool(args.allow_unverified),
        fail_on_blocked=bool(args.fail_on_blocked),
        manifest_fingerprint_workers=args.manifest_fingerprint_workers,
        notes=tuple(args.note or ()),
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    payload = run_covered_facts_external_evidence_workflow(config)
    decision = _mapping(payload.get("decision"))
    print(
        "covered_facts_external_evidence_workflow="
        f"{decision.get('status')} route={decision.get('recommended_route')} "
        f"record={decision.get('recommended_route_record')}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
