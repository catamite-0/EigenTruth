"""Tests for active frontier artifact reference audits."""

import json

from benchmarks.audit_frontier_artifact_references import (
    build_frontier_artifact_reference_audit,
)
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest


def test_frontier_artifact_reference_audit_reports_missing_refs_and_verified_manifest(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "frontier-audit-release-candidate-v6"
    artifact_dir.mkdir(parents=True)
    comparison_path = artifact_dir / "frontier-audit-comparison.json"
    comparison_path.write_text('{"workflow":"frontier_release_evidence_comparison"}\n', encoding="utf-8")
    manifest_path = artifact_dir / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {"frontier_audit_comparison": comparison_path},
                root=artifact_dir,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    doc_path = tmp_path / "docs" / "experiment-plan.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(
        "\n".join((
            "current artifacts:",
            "`artifacts/frontier-audit-release-candidate-v6/frontier-audit-comparison.json`",
            "`artifacts/frontier-audit-release-candidate-v6/artifact-manifest.json`",
            "`artifacts/smollm2_product_promotion_contract_v1_9/product-promotion-contract.json`",
        ))
        + "\n",
        encoding="utf-8",
    )
    cached_contract_path = (
        tmp_path
        / "artifacts"
        / "smollm2_product_promotion_contract_v1_9"
        / "product-promotion-contract.json"
    )
    json_cache_path = tmp_path / "artifact-json-cache.json"
    json_cache_path.write_text(
        json.dumps({
            f"{cached_contract_path}:10:20:30:40:cached": {
                "error": None,
                "payload": {"workflow": "product_promotion_contract", "source_status": "promote"},
            },
        }),
        encoding="utf-8",
    )
    report_path = tmp_path / "audit" / "frontier-artifact-reference-audit.json"
    audit_manifest_path = tmp_path / "audit" / "artifact-manifest.json"
    registry_path = tmp_path / "registry.json"

    payload = build_frontier_artifact_reference_audit(
        doc_paths=(doc_path,),
        root=tmp_path,
        include_regex="frontier-audit-release-candidate-v6|smollm2_product_promotion_contract_v1_9",
        json_cache_paths=(json_cache_path,),
        json_path=report_path,
        artifact_manifest_path=audit_manifest_path,
        registry_path=registry_path,
        name="frontier-artifact-reference-audit",
        version="0.1",
        metadata={"release": "test"},
    )

    references = {reference["path"]: reference for reference in payload["references"]}
    registry_record = ArtifactRegistry.load_json(registry_path).get(
        "report:frontier-artifact-reference-audit:0.1"
    )

    assert payload["status"] == "blocked"
    assert payload["summary"]["reference_count"] == 3
    assert payload["summary"]["missing_count"] == 1
    assert payload["summary"]["missing_recoverable_from_json_cache_count"] == 1
    assert payload["summary"]["missing_unrecoverable_count"] == 0
    assert payload["summary"]["manifest_verified_count"] == 1
    assert payload["summary"]["recommended_action_ids"] == (
        "restore_cached_json_artifacts",
        "export_product_promotion_contract_v1_9",
        "rerun_frontier_artifact_reference_audit",
    )
    assert payload["recommended_actions"][0]["action_id"] == "restore_cached_json_artifacts"
    assert payload["recommended_actions"][0]["affected_paths"] == (
        "artifacts/smollm2_product_promotion_contract_v1_9/product-promotion-contract.json",
    )
    assert payload["recommended_actions"][1]["action_id"] == "export_product_promotion_contract_v1_9"
    assert payload["recommended_actions"][1]["suggested_commands"][0].startswith(
        "python benchmarks/export_product_promotion_contract.py"
    )
    assert references[
        "artifacts/frontier-audit-release-candidate-v6/artifact-manifest.json"
    ]["manifest_verification"]["passed"] is True
    assert references[
        "artifacts/smollm2_product_promotion_contract_v1_9/product-promotion-contract.json"
    ]["status"] == "missing"
    assert references[
        "artifacts/smollm2_product_promotion_contract_v1_9/product-promotion-contract.json"
    ]["recoverable_from_json_cache"] is True
    assert references[
        "artifacts/smollm2_product_promotion_contract_v1_9/product-promotion-contract.json"
    ]["json_cache_sources"][0]["workflow"] == "product_promotion_contract"
    assert payload["artifact_manifest_summary"]["missing_count"] == 1
    assert payload["registry_record"] == "report:frontier-artifact-reference-audit:0.1"
    assert registry_record.metadata["status"] == "blocked"
    assert registry_record.metadata["missing_count"] == 1
    assert report_path.exists()
    assert audit_manifest_path.exists()


def test_frontier_artifact_reference_audit_passes_when_filtered_refs_exist(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "frontier-audit-release-candidate-v6"
    artifact_dir.mkdir(parents=True)
    report_path = artifact_dir / "frontier-audit-comparison.json"
    report_path.write_text("{}\n", encoding="utf-8")
    doc_path = tmp_path / "README.md"
    doc_path.write_text(
        "`artifacts/frontier-audit-release-candidate-v6/frontier-audit-comparison.json`\n",
        encoding="utf-8",
    )

    payload = build_frontier_artifact_reference_audit(
        doc_paths=(doc_path,),
        root=tmp_path,
        include_regex="frontier-audit-release-candidate-v6",
        verify_manifests=False,
    )

    assert payload["status"] == "passed"
    assert payload["summary"]["existing_count"] == 1
    assert payload["blocking_reasons"] == ()
    assert payload["recommended_actions"] == ()
