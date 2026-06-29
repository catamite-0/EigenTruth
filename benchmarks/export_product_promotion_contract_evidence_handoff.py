"""Export ProductPromotionContract runtime-evidence handoff fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.control import enrich_product_promotion_contract_evidence  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402


def export_product_promotion_contract_evidence_handoff(
    *,
    contract: str | Path,
    json_path: str | Path,
    audit_json_path: str | Path,
    pre_generation_probe_comparison: str | Path | None = None,
    triple_extraction_fixture_matrix: str | Path | None = None,
    counterfactual_verification: str | Path | None = None,
    product_trace_replay_workflow: str | Path | None = None,
    frontier_release_evidence: str | Path | None = None,
    runtime_baseline: str | Path | None = None,
    covered_fact_property_metrics: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an enriched contract and its evidence-handoff audit."""
    contract_path = Path(contract)
    output_path = Path(json_path)
    audit_path = Path(audit_json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    if (name or version) and (registry_path is None or name is None or version is None):
        raise ValueError("registry export requires registry_path, name, and version.")

    pre_generation_path = _optional_path(pre_generation_probe_comparison)
    matrix_path = _optional_path(triple_extraction_fixture_matrix)
    counterfactual_path = _optional_path(counterfactual_verification)
    workflow_path = _optional_path(product_trace_replay_workflow)
    frontier_path = _optional_path(frontier_release_evidence)
    runtime_path = _optional_path(runtime_baseline)
    covered_fact_path = _optional_path(covered_fact_property_metrics)
    result = enrich_product_promotion_contract_evidence(
        _load_json_object(contract_path),
        pre_generation_probe_comparison=_load_optional_object(pre_generation_path),
        pre_generation_probe_comparison_path=_path_str(pre_generation_path),
        triple_extraction_fixture_matrix=_load_optional_object(matrix_path),
        triple_extraction_fixture_matrix_path=_path_str(matrix_path),
        counterfactual_verification=_load_optional_object(counterfactual_path),
        counterfactual_verification_path=_path_str(counterfactual_path),
        product_trace_replay_workflow=_load_optional_object(workflow_path),
        product_trace_replay_workflow_path=_path_str(workflow_path),
        frontier_release_evidence=_load_optional_object(frontier_path),
        frontier_release_evidence_path=_path_str(frontier_path),
        runtime_baseline=_load_optional_object(runtime_path),
        runtime_baseline_path=_path_str(runtime_path),
        covered_fact_property_metrics=_load_optional_object(covered_fact_path),
        metadata={
            "source_contract": str(contract_path),
            **dict(metadata or {}),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        strict_json_dumps(result.contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_payload = result.after_audit.to_dict()
    audit_path.write_text(
        strict_json_dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = None
    if manifest_path is not None:
        artifacts: dict[str, Path] = {
            "source_contract": contract_path,
            "product_promotion_contract_evidence_handoff": output_path,
            "product_promotion_contract_evidence_handoff_audit": audit_path,
        }
        optional_artifacts = {
            "pre_generation_probe_comparison": pre_generation_path,
            "triple_extraction_fixture_matrix": matrix_path,
            "counterfactual_verification": counterfactual_path,
            "product_trace_replay_workflow": workflow_path,
            "frontier_release_evidence": frontier_path,
            "runtime_baseline": runtime_path,
            "covered_fact_property_metrics": covered_fact_path,
        }
        artifacts.update({name: path for name, path in optional_artifacts.items() if path is not None})
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "runner": "export_product_promotion_contract_evidence_handoff",
                "status": result.after_audit.status,
                "source_contract": str(contract_path),
                "before_missing_metric_count": result.summary["before_missing_metric_count"],
                "after_missing_metric_count": result.summary["after_missing_metric_count"],
                "resolved_missing_metric_count": result.summary["resolved_missing_metric_count"],
                "filled_groups": result.filled_groups,
                **dict(metadata or {}),
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if registry_path is not None and name is not None and version is not None:
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_product_promotion_contract(
            name=name,
            path=output_path,
            version=version,
            metadata={
                "workflow": "product_promotion_evidence_handoff_export",
                "source_contract": str(contract_path),
                "evidence_handoff_audit": str(audit_path),
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                "before_missing_metric_count": result.summary["before_missing_metric_count"],
                "after_missing_metric_count": result.summary["after_missing_metric_count"],
                "resolved_missing_metric_count": result.summary["resolved_missing_metric_count"],
                "filled_groups": result.filled_groups,
                **dict(metadata or {}),
            },
        )
        registry.record_product_promotion_evidence_audit(
            name=f"{name}-audit",
            path=audit_path,
            version=version,
            metadata={
                "workflow": audit_payload["workflow"],
                "status": audit_payload["status"],
                "source_contract": str(contract_path),
                "enriched_contract": str(output_path),
                "missing_metric_count": audit_payload["summary"]["missing_metric_count"],
                "blocked_group_count": audit_payload["summary"]["blocked_group_count"],
                "recommended_action_ids": audit_payload["recommended_action_ids"],
                **dict(metadata or {}),
            },
        )
        registry.save_json(registry_path)
    return result.to_dict()


def _optional_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


def _path_str(value: Path | None) -> str | None:
    return None if value is None else str(value)


def _load_optional_object(path: Path | None) -> Mapping[str, Any] | None:
    return None if path is None else _load_json_object(path)


def _load_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _parse_metadata(values: Sequence[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--metadata entries must be formatted as key=value.")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--metadata keys must be non-empty.")
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich a ProductPromotionContract with runtime evidence handoff fields from explicit local reports."
        )
    )
    parser.add_argument("--contract", required=True, help="source ProductPromotionContract JSON")
    parser.add_argument("--json", required=True, help="output enriched contract JSON")
    parser.add_argument("--audit-json", required=True, help="output evidence audit JSON")
    parser.add_argument("--pre-generation-probe-comparison", default=None)
    parser.add_argument("--triple-extraction-fixture-matrix", default=None)
    parser.add_argument("--counterfactual-verification", default=None)
    parser.add_argument("--product-trace-replay-workflow", default=None)
    parser.add_argument("--frontier-release-evidence", default=None)
    parser.add_argument("--runtime-baseline", default=None)
    parser.add_argument("--covered-fact-property-metrics", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON")
    parser.add_argument("--name", default=None, help="registry contract name")
    parser.add_argument("--version", default=None, help="registry contract version")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="metadata key=value; repeatable",
    )
    args = parser.parse_args(argv)
    payload = export_product_promotion_contract_evidence_handoff(
        contract=args.contract,
        json_path=args.json,
        audit_json_path=args.audit_json,
        pre_generation_probe_comparison=args.pre_generation_probe_comparison,
        triple_extraction_fixture_matrix=args.triple_extraction_fixture_matrix,
        counterfactual_verification=args.counterfactual_verification,
        product_trace_replay_workflow=args.product_trace_replay_workflow,
        frontier_release_evidence=args.frontier_release_evidence,
        runtime_baseline=args.runtime_baseline,
        covered_fact_property_metrics=args.covered_fact_property_metrics,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
    )
    print(
        "product_promotion_evidence_handoff_export="
        f"{payload['status']} "
        f"before_missing={payload['summary']['before_missing_metric_count']} "
        f"after_missing={payload['summary']['after_missing_metric_count']} "
        f"resolved={payload['summary']['resolved_missing_metric_count']} "
        f"filled_groups={len(payload['filled_groups'])}"
    )


if __name__ == "__main__":
    main()
