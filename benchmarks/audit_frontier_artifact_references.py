"""Audit active frontier artifact references in docs against local files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    ArtifactVerificationContext,
    build_artifact_manifest,
    load_json_cache,
)

DEFAULT_DOC_PATHS = (
    "README.md",
    "benchmarks/README.md",
    "docs/experiment-plan.md",
    "docs/frontier-research-notes.md",
)
DEFAULT_INCLUDE_REGEX = (
    "frontier-audit-release-candidate-v6|"
    "smollm2_product_promotion_contract_v1_9"
)
DEFAULT_JSON_CACHE_PATHS = (
    "artifacts/frontier-audit-release-candidate-v6/artifact-json-cache.json",
)
ARTIFACT_REFERENCE_RE = re.compile(r"artifacts/[^\s`'\"|)>\\]+")
FRONTIER_V6_PREFIX = "artifacts/frontier-audit-release-candidate-v6/"
PRODUCT_V19_PREFIX = "artifacts/smollm2_product_promotion_contract_v1_9/"
PRODUCT_V19_CONTRACT_PATH = f"{PRODUCT_V19_PREFIX}product-promotion-contract.json"
PRODUCT_V19_ARTIFACT_MANIFEST_PATH = f"{PRODUCT_V19_PREFIX}artifact-manifest.json"
PRODUCT_V19_HANDOFF_PATH = f"{PRODUCT_V19_PREFIX}product-promotion-contract-evidence-handoff.json"
PRODUCT_V19_HANDOFF_AUDIT_PATH = f"{PRODUCT_V19_PREFIX}product-promotion-contract-evidence-handoff-audit.json"
PRODUCT_V19_HANDOFF_MANIFEST_PATH = f"{PRODUCT_V19_PREFIX}evidence-handoff-artifact-manifest.json"
PRE_GENERATION_COMPARISON_PATH = (
    "artifacts/runtime_evidence/"
    "pre-generation-qwen-smollm2-l12-comparison/comparison.json"
)
TRIPLE_EXTRACTION_FIXTURE_MATRIX_PATH = (
    "artifacts/wikidata-cross-corpus-triple-extraction-adversarial-matrix-v1/"
    "triple-extraction-fixture-matrix.json"
)
COUNTERFACTUAL_VERIFICATION_PATH = (
    "artifacts/smollm2_product_counterfactual_structured_qa_audit_v0/"
    "counterfactual-verification-report.json"
)
PRODUCT_TRACE_REPLAY_WORKFLOW_PATH = (
    "artifacts/smollm2_product_trace_replay_workflow_action_gated_v0/"
    "product-trace-replay-workflow.json"
)
TRIPLE_AUDIT_ENRICHMENT_PATH = (
    "artifacts/smollm2_product_trace_triple_audit_enrichment_v1/"
    "product-trace-triple-audit-enrichment.json"
)
COVERED_FACT_PROPERTY_METRICS_PATH = (
    "artifacts/truthfulqa-frontier-smollm2-l80-source-family-structured-qa-fact-collection-route/"
    "structured-qa-route-summary.json"
)
FRONTIER_ARTIFACT_REFERENCE_AUDIT_COMMAND = (
    "python benchmarks/audit_frontier_artifact_references.py "
    "--json artifacts/frontier-artifact-reference-audit.json "
    "--artifact-manifest artifacts/frontier-artifact-reference-audit-manifest.json "
    "--registry artifacts/local-release-registry.json "
    "--name frontier-artifact-reference-audit "
    "--version 0.1 "
    "--no-fail"
)
RESTORE_CACHED_JSON_ARTIFACTS_COMMAND = (
    "python benchmarks/audit_frontier_artifact_references.py "
    "--restore-json-cache-artifacts "
    "--json artifacts/frontier-artifact-reference-audit.json "
    "--artifact-manifest artifacts/frontier-artifact-reference-audit-manifest.json "
    "--registry artifacts/local-release-registry.json "
    "--name frontier-artifact-reference-audit "
    "--version 0.1 "
    "--no-fail"
)
EXPORT_PRODUCT_V19_COMMAND = (
    "python benchmarks/export_product_promotion_contract.py "
    "--source artifacts/frontier-audit-release-candidate-v6/frontier-audit-registry-workflow.json "
    "--output artifacts/smollm2_product_promotion_contract_v1_9/product-promotion-contract.json "
    "--artifact-manifest artifacts/smollm2_product_promotion_contract_v1_9/artifact-manifest.json "
    "--registry artifacts/local-release-registry.json "
    "--name smollm2-product-promotion-contract "
    "--version 1.9 "
    "--metadata release=smollm2-v1.9 "
    "--metadata source_record=benchmark_manifest:smollm2-l8-frontier-audit-release-candidate:0.6 "
    "--compact-json"
)
EXPORT_PRODUCT_V19_HANDOFF_COMMAND = " ".join((
    "python",
    "benchmarks/export_product_promotion_contract_evidence_handoff.py",
    "--contract",
    PRODUCT_V19_CONTRACT_PATH,
    "--json",
    PRODUCT_V19_HANDOFF_PATH,
    "--audit-json",
    PRODUCT_V19_HANDOFF_AUDIT_PATH,
    "--pre-generation-probe-comparison",
    PRE_GENERATION_COMPARISON_PATH,
    "--triple-extraction-fixture-matrix",
    TRIPLE_EXTRACTION_FIXTURE_MATRIX_PATH,
    "--counterfactual-verification",
    COUNTERFACTUAL_VERIFICATION_PATH,
    "--product-trace-replay-workflow",
    PRODUCT_TRACE_REPLAY_WORKFLOW_PATH,
    "--triple-audit-enrichment",
    TRIPLE_AUDIT_ENRICHMENT_PATH,
    "--covered-fact-property-metrics",
    COVERED_FACT_PROPERTY_METRICS_PATH,
    "--artifact-manifest",
    PRODUCT_V19_HANDOFF_MANIFEST_PATH,
    "--registry",
    "artifacts/local-release-registry.json",
    "--name",
    "smollm2-product-promotion-contract-v1-9-evidence-handoff",
    "--version",
    "0.3",
    "--metadata",
    "release=smollm2-v1.9",
    "--metadata",
    "triple_audit_enrichment=triple_audit_v1",
))


def build_frontier_artifact_reference_audit(
    *,
    doc_paths: Sequence[str | Path] = DEFAULT_DOC_PATHS,
    root: str | Path = REPO_ROOT,
    include_regex: str | None = DEFAULT_INCLUDE_REGEX,
    exclude_regex: str | None = None,
    verify_manifests: bool = True,
    recursive_manifests: bool = False,
    max_workers: int = 1,
    json_cache_paths: Sequence[str | Path] = DEFAULT_JSON_CACHE_PATHS,
    restore_json_cache_artifacts: bool = False,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return and optionally persist a local audit of active artifact references."""
    root_path = Path(root)
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if (name or version) and (registry_path is None or name is None or version is None):
        raise ValueError("registry export requires registry_path, name, and version.")

    include_pattern = re.compile(include_regex) if include_regex else None
    exclude_pattern = re.compile(exclude_regex) if exclude_regex else None
    documents = tuple(_resolve_doc_path(path, root=root_path) for path in doc_paths)
    json_cache_sources = _load_json_cache_sources(json_cache_paths, root=root_path)
    references = _collect_references(
        documents,
        root=root_path,
        include_pattern=include_pattern,
        exclude_pattern=exclude_pattern,
        verify_manifests=verify_manifests,
        recursive_manifests=recursive_manifests,
        max_workers=max_workers,
        json_cache_sources=json_cache_sources,
    )
    restore_report = None
    if restore_json_cache_artifacts:
        restore_report = _restore_recoverable_json_artifacts(
            references,
            root=root_path,
            json_cache_sources=json_cache_sources,
        )
        if restore_report["restored_count"]:
            references = _collect_references(
                documents,
                root=root_path,
                include_pattern=include_pattern,
                exclude_pattern=exclude_pattern,
                verify_manifests=verify_manifests,
                recursive_manifests=recursive_manifests,
                max_workers=max_workers,
                json_cache_sources=json_cache_sources,
            )
    blocking_reasons = _blocking_reasons(references)
    if not references:
        blocking_reasons.append("no artifact references matched the configured include/exclude filters")
    recommended_actions = _recommended_actions(references)
    summary = _summary(references, document_count=len(documents))
    summary["recommended_action_count"] = len(recommended_actions)
    summary["recommended_action_ids"] = tuple(action["action_id"] for action in recommended_actions)
    status = "passed" if not blocking_reasons else "blocked"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "frontier_artifact_reference_audit",
        "status": status,
        "summary": summary,
        "config": {
            "root": str(root_path),
            "documents": tuple(_display_path(path, root=root_path) for path in documents),
            "include_regex": include_regex,
            "exclude_regex": exclude_regex,
            "verify_manifests": verify_manifests,
            "recursive_manifests": recursive_manifests,
            "max_workers": max_workers,
            "json_cache_paths": tuple(source["path"] for source in json_cache_sources),
            "restore_json_cache_artifacts": restore_json_cache_artifacts,
        },
        "references": references,
        "blocking_reasons": tuple(blocking_reasons),
        "recommended_actions": recommended_actions,
        "metadata": dict(metadata or {}),
    }
    if restore_report is not None:
        payload["restore_report"] = restore_report

    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    if manifest_path is not None:
        payload["artifact_manifest"] = str(manifest_path)
    registry_record_key = None
    if registry_path is not None and output_path is not None and name is not None and version is not None:
        registry_record_key = f"report:{name}:{version}"
        payload["registry_record"] = registry_record_key
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path, payload)
    if manifest_path is not None and output_path is not None:
        manifest = _write_artifact_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            documents=documents,
            references=references,
            root=root_path,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = manifest.get("summary", {})
        _write_json(output_path, payload)
        manifest = _write_artifact_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            documents=documents,
            references=references,
            root=root_path,
            payload=payload,
        )
        payload["artifact_manifest_summary"] = manifest.get("summary", {})
    if registry_path is not None and output_path is not None and name is not None and version is not None:
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_report(
            name=name,
            path=output_path,
            version=version,
            metadata={
                "workflow": payload["workflow"],
                "status": status,
                "artifact_manifest": None if manifest_path is None else str(manifest_path),
                **summary,
                **dict(metadata or {}),
            },
        )
        registry.save_json(registry_path)
    return payload


def _collect_references(
    documents: Sequence[Path],
    *,
    root: Path,
    include_pattern: re.Pattern[str] | None,
    exclude_pattern: re.Pattern[str] | None,
    verify_manifests: bool,
    recursive_manifests: bool,
    max_workers: int,
    json_cache_sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    refs: dict[str, dict[str, Any]] = {}
    for document in documents:
        if not document.exists():
            continue
        relative_doc = _display_path(document, root=root)
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
            for raw_reference in ARTIFACT_REFERENCE_RE.findall(line):
                reference = _clean_reference(raw_reference)
                if not reference:
                    continue
                if include_pattern is not None and include_pattern.search(reference) is None:
                    continue
                if exclude_pattern is not None and exclude_pattern.search(reference) is not None:
                    continue
                record = refs.setdefault(
                    reference,
                    {
                        "path": reference,
                        "documents": {},
                        "line_numbers": set(),
                    },
                )
                record["line_numbers"].add(line_number)
                record["documents"].setdefault(relative_doc, set()).add(line_number)

    context = ArtifactVerificationContext()
    return tuple(
        _reference_record(
            reference=reference,
            raw=raw,
            root=root,
            context=context,
            verify_manifests=verify_manifests,
            recursive_manifests=recursive_manifests,
            max_workers=max_workers,
            json_cache_sources=json_cache_sources,
        )
        for reference, raw in sorted(refs.items())
    )


def _reference_record(
    *,
    reference: str,
    raw: Mapping[str, Any],
    root: Path,
    context: ArtifactVerificationContext,
    verify_manifests: bool,
    recursive_manifests: bool,
    max_workers: int,
    json_cache_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual_path = root / reference
    exists = actual_path.exists()
    kind = "missing"
    if actual_path.is_file():
        kind = "file"
    elif actual_path.is_dir():
        kind = "directory"
    is_manifest = _is_artifact_manifest_path(actual_path)
    manifest_verification = None
    manifest_error = None
    if verify_manifests and exists and actual_path.is_file() and is_manifest:
        try:
            manifest_verification = context.load_and_verify_artifact_manifest(
                actual_path,
                recursive=recursive_manifests,
                max_workers=max_workers,
            ).to_dict()
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            manifest_error = f"{type(exc).__name__}: {exc}"

    status = "present"
    if not exists:
        status = "missing"
    elif manifest_error is not None:
        status = "manifest_error"
    elif manifest_verification is not None and not manifest_verification["passed"]:
        status = "manifest_failed"
    json_cache_hits = _json_cache_hits_for_reference(
        reference,
        root=root,
        json_cache_sources=json_cache_sources,
    )

    documents = {
        document: tuple(sorted(lines))
        for document, lines in sorted(raw["documents"].items())
    }
    record: dict[str, Any] = {
        "path": reference,
        "status": status,
        "exists": exists,
        "kind": kind,
        "is_manifest": is_manifest,
        "documents": documents,
        "line_numbers": tuple(sorted(raw["line_numbers"])),
    }
    if manifest_verification is not None:
        record["manifest_verification"] = manifest_verification
    if manifest_error is not None:
        record["manifest_error"] = manifest_error
    if status == "missing" and json_cache_hits:
        record["recoverable_from_json_cache"] = True
        record["json_cache_sources"] = json_cache_hits
    if manifest_verification is not None and not manifest_verification["passed"]:
        manifest_missing_json_cache_sources = _manifest_missing_json_cache_records(
            manifest_path=actual_path,
            manifest_verification=manifest_verification,
            root=root,
            json_cache_sources=json_cache_sources,
        )
        if manifest_missing_json_cache_sources:
            record["manifest_missing_json_cache_sources"] = manifest_missing_json_cache_sources
    return record


def _summary(references: Sequence[Mapping[str, Any]], *, document_count: int) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for reference in references:
        status = str(reference["status"])
        statuses[status] = statuses.get(status, 0) + 1
    manifest_refs = tuple(reference for reference in references if reference.get("is_manifest"))
    verified_manifests = tuple(
        reference
        for reference in manifest_refs
        if _mapping(reference.get("manifest_verification")).get("passed") is True
    )
    failed_manifests = tuple(
        reference
        for reference in manifest_refs
        if _mapping(reference.get("manifest_verification")).get("passed") is False
    )
    errored_manifests = tuple(reference for reference in manifest_refs if reference.get("manifest_error"))
    recoverable_missing = tuple(
        reference
        for reference in references
        if reference["status"] == "missing" and reference.get("recoverable_from_json_cache")
    )
    manifest_child_missing_records = tuple(
        child
        for reference in references
        for child in reference.get("manifest_missing_json_cache_sources", ())
    )
    manifest_child_recoverable_records = tuple(
        child
        for child in manifest_child_missing_records
        if child.get("recoverable_from_json_cache")
    )
    return {
        "document_count": document_count,
        "reference_count": len(references),
        "existing_count": sum(1 for reference in references if reference["exists"]),
        "missing_count": statuses.get("missing", 0),
        "file_count": sum(1 for reference in references if reference["kind"] == "file"),
        "directory_count": sum(1 for reference in references if reference["kind"] == "directory"),
        "blocking_reference_count": sum(1 for reference in references if reference["status"] != "present"),
        "status_counts": dict(sorted(statuses.items())),
        "manifest_reference_count": len(manifest_refs),
        "manifest_verified_count": len(verified_manifests),
        "manifest_failed_count": len(failed_manifests),
        "manifest_error_count": len(errored_manifests),
        "missing_recoverable_from_json_cache_count": len(recoverable_missing),
        "missing_unrecoverable_count": statuses.get("missing", 0) - len(recoverable_missing),
        "manifest_child_missing_count": len(manifest_child_missing_records),
        "manifest_child_recoverable_from_json_cache_count": len(manifest_child_recoverable_records),
        "manifest_child_unrecoverable_count": (
            len(manifest_child_missing_records) - len(manifest_child_recoverable_records)
        ),
    }


def _blocking_reasons(references: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for reference in references:
        status = reference["status"]
        path = reference["path"]
        if status == "missing":
            reasons.append(f"missing artifact reference: {path}")
        elif status == "manifest_failed":
            reasons.append(f"artifact manifest verification failed: {path}")
        elif status == "manifest_error":
            reasons.append(f"artifact manifest verification errored: {path}")
    return reasons


def _manifest_missing_json_cache_records(
    *,
    manifest_path: Path,
    manifest_verification: Mapping[str, Any],
    root: Path,
    json_cache_sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    failures = tuple(_mapping(failure) for failure in manifest_verification.get("failures", ()))
    for failure_mapping in failures:
        if failure_mapping.get("field") != "exists" or failure_mapping.get("actual") is not False:
            continue
        raw_path = failure_mapping.get("path")
        if not isinstance(raw_path, str):
            continue
        child_path = (manifest_path.parent / raw_path).resolve()
        display_path = _display_path(child_path, root=root)
        if display_path in seen_paths:
            continue
        seen_paths.add(display_path)
        json_cache_hits = _json_cache_hits_for_reference(
            display_path,
            root=root,
            json_cache_sources=json_cache_sources,
        )
        record: dict[str, Any] = {
            "name": failure_mapping.get("name"),
            "path": display_path,
            "manifest_relative_path": raw_path,
        }
        expected_sha256 = _expected_manifest_failure_value(failures, raw_path=raw_path, field="sha256")
        expected_size_bytes = _expected_manifest_failure_value(
            failures,
            raw_path=raw_path,
            field="size_bytes",
        )
        if expected_sha256 is not None:
            record["expected_sha256"] = expected_sha256
        if expected_size_bytes is not None:
            record["expected_size_bytes"] = expected_size_bytes
        cached_payload = _cached_payload_for_reference(
            display_path,
            root=root,
            json_cache_sources=json_cache_sources,
        )
        if json_cache_hits:
            record["json_cache_sources"] = json_cache_hits
        if cached_payload is not None:
            normalized_payload, normalized_absolute_path_count = _normalize_cached_json_payload(
                cached_payload["payload"],
                root=root.resolve(),
            )
            restore_variant, digest_mismatch = _matching_json_restore_variant(
                normalized_payload,
                candidate={
                    "expected_sha256": expected_sha256,
                    "expected_size_bytes": expected_size_bytes,
                },
            )
            if digest_mismatch is not None:
                record["json_cache_digest_mismatch"] = digest_mismatch
                record["normalized_absolute_path_count"] = normalized_absolute_path_count
                records.append(record)
                continue
            record["recoverable_from_json_cache"] = True
            record["normalized_absolute_path_count"] = normalized_absolute_path_count
            if restore_variant is not None:
                record["restore_serialization"] = restore_variant["serialization"]
        records.append(record)
    return tuple(records)


def _expected_manifest_failure_value(
    failures: Sequence[Mapping[str, Any]],
    *,
    raw_path: str,
    field: str,
) -> Any:
    for failure in failures:
        if failure.get("path") == raw_path and failure.get("field") == field:
            return failure.get("expected")
    return None


def _recommended_actions(references: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    missing_paths = tuple(
        sorted(str(reference["path"]) for reference in references if reference["status"] == "missing")
    )
    manifest_child_recoverable_paths = tuple(
        sorted({
            str(child["path"])
            for reference in references
            for child in reference.get("manifest_missing_json_cache_sources", ())
            if child.get("recoverable_from_json_cache")
        })
    )
    if not missing_paths and not manifest_child_recoverable_paths:
        return ()
    blocking_paths = tuple(sorted(set(missing_paths) | set(manifest_child_recoverable_paths)))
    actions: list[dict[str, Any]] = []
    recoverable_paths = tuple(
        sorted({
            str(reference["path"])
            for reference in references
            if reference["status"] == "missing" and reference.get("recoverable_from_json_cache")
        } | set(manifest_child_recoverable_paths))
    )
    if recoverable_paths:
        cache_paths = tuple(
            sorted({
                str(source["cache_path"])
                for reference in references
                for source in (
                    tuple(reference.get("json_cache_sources", ()))
                    + tuple(
                        source
                        for child in reference.get("manifest_missing_json_cache_sources", ())
                        if str(child["path"]) in recoverable_paths
                        for source in child.get("json_cache_sources", ())
                    )
                )
            })
        )
        actions.append({
            "action_id": "restore_cached_json_artifacts",
            "title": "Restore missing JSON artifacts available in local artifact cache",
            "action_type": "restore_cached_json_artifacts",
            "priority": 90,
            "rationale": (
                "Some missing JSON references have cached payloads in persisted "
                "artifact JSON caches and can be restored without rerunning model work."
            ),
            "affected_paths": recoverable_paths,
            "suggested_commands": (RESTORE_CACHED_JSON_ARTIFACTS_COMMAND,),
            "metadata": {
                "json_cache_paths": cache_paths,
                "notes": (
                    "Only restore cached JSON payloads after confirming the cache is "
                    "from the intended release run; unrecoverable references still need reruns."
                ),
            },
        })

    frontier_missing = tuple(path for path in missing_paths if path.startswith(FRONTIER_V6_PREFIX))
    if frontier_missing:
        actions.append({
            "action_id": "rebuild_frontier_audit_release_candidate_v6",
            "title": "Regenerate frontier audit release-candidate v6 artifacts",
            "action_type": "rerun_release_candidate_workflow",
            "priority": 100,
            "rationale": (
                "The active docs reference the v6 frontier-audit release candidate, "
                "but one or more source reports/manifests are missing locally."
            ),
            "affected_paths": frontier_missing,
            "suggested_commands": (),
            "metadata": {
                "required_output": (
                    "artifacts/frontier-audit-release-candidate-v6/"
                    "frontier-audit-registry-workflow.json"
                ),
                "release_policy_profile": "frontier_audit",
                "notes": (
                    "Rerun the release-candidate registry workflow that produced the "
                    "frontier-audit v6 candidate before exporting v1.9 product handoff artifacts."
                ),
            },
        })

    product_contract_missing = tuple(
        path
        for path in missing_paths
        if path
        in {
            PRODUCT_V19_PREFIX.rstrip("/"),
            PRODUCT_V19_PREFIX,
            PRODUCT_V19_CONTRACT_PATH,
            PRODUCT_V19_ARTIFACT_MANIFEST_PATH,
        }
    )
    if product_contract_missing:
        actions.append({
            "action_id": "export_product_promotion_contract_v1_9",
            "title": "Export product promotion contract v1.9 from frontier v6",
            "action_type": "export_product_promotion_contract",
            "priority": 80,
            "rationale": (
                "The demo prefers the v1.9 handoff when present, but the local "
                "product contract or its manifest is missing."
            ),
            "affected_paths": product_contract_missing,
            "suggested_commands": (EXPORT_PRODUCT_V19_COMMAND,),
            "metadata": {
                "depends_on_action_ids": (
                    ("rebuild_frontier_audit_release_candidate_v6",) if frontier_missing else ()
                ),
                "source": (
                    "artifacts/frontier-audit-release-candidate-v6/"
                    "frontier-audit-registry-workflow.json"
                ),
            },
        })

    handoff_missing = tuple(
        path
        for path in missing_paths
        if path
        in {
            PRODUCT_V19_HANDOFF_PATH,
            PRODUCT_V19_HANDOFF_AUDIT_PATH,
            PRODUCT_V19_HANDOFF_MANIFEST_PATH,
        }
    )
    if handoff_missing:
        actions.append({
            "action_id": "export_product_promotion_contract_v1_9_evidence_handoff",
            "title": "Export enriched v1.9 product evidence handoff",
            "action_type": "export_product_promotion_contract_evidence_handoff",
            "priority": 70,
            "rationale": (
                "The active docs reference enriched v1.9 evidence-handoff reports, "
                "but the handoff JSON, audit, or handoff manifest is missing locally."
            ),
            "affected_paths": handoff_missing,
            "suggested_commands": (EXPORT_PRODUCT_V19_HANDOFF_COMMAND,),
            "metadata": {
                "depends_on_action_ids": ("export_product_promotion_contract_v1_9",),
                "contract": PRODUCT_V19_CONTRACT_PATH,
            },
        })

    verification_missing = tuple(
        path
        for path in missing_paths
        if path.endswith("manifest-verification.json") or path.endswith("evidence-handoff-manifest-verification.json")
    )
    if verification_missing:
        actions.append({
            "action_id": "verify_frontier_artifact_manifests",
            "title": "Verify regenerated frontier artifact manifests",
            "action_type": "verify_artifact_manifests",
            "priority": 50,
            "rationale": (
                "The active docs reference saved manifest-verification reports that "
                "should be refreshed after the source manifests are regenerated."
            ),
            "affected_paths": verification_missing,
            "suggested_commands": (
                "python benchmarks/verify_artifact_manifest.py "
                "--manifest artifacts/frontier-audit-release-candidate-v6/artifact-manifest.json "
                "--recursive "
                "--json artifacts/frontier-audit-release-candidate-v6/manifest-verification.json",
                "python benchmarks/verify_artifact_manifest.py "
                "--manifest artifacts/smollm2_product_promotion_contract_v1_9/artifact-manifest.json "
                "--json artifacts/smollm2_product_promotion_contract_v1_9/manifest-verification.json",
                "python benchmarks/verify_artifact_manifest.py "
                "--manifest artifacts/smollm2_product_promotion_contract_v1_9/evidence-handoff-artifact-manifest.json "
                "--json artifacts/smollm2_product_promotion_contract_v1_9/evidence-handoff-manifest-verification.json",
            ),
            "metadata": {
                "depends_on_action_ids": (
                    "rebuild_frontier_audit_release_candidate_v6",
                    "export_product_promotion_contract_v1_9",
                    "export_product_promotion_contract_v1_9_evidence_handoff",
                ),
            },
        })

    actions.append({
        "action_id": "rerun_frontier_artifact_reference_audit",
        "title": "Rerun the frontier artifact reference audit",
        "action_type": "audit_frontier_artifact_references",
        "priority": 10,
        "rationale": "Recheck doc references after regenerating missing frontier artifacts.",
        "affected_paths": blocking_paths,
        "suggested_commands": (FRONTIER_ARTIFACT_REFERENCE_AUDIT_COMMAND,),
        "metadata": {},
    })
    return tuple(sorted(actions, key=lambda action: (-int(action["priority"]), str(action["action_id"]))))


def _restore_recoverable_json_artifacts(
    references: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    json_cache_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root_path = root.resolve()
    restored: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate in _cached_json_restore_candidates(references):
        reference_path = str(candidate["path"])
        output_path = root / reference_path
        resolved_output_path = output_path.resolve()
        if not resolved_output_path.is_relative_to(root_path):
            skipped.append({
                "path": reference_path,
                "source": candidate["source"],
                "reason": "path_outside_root",
            })
            continue
        if resolved_output_path.exists():
            skipped.append({
                "path": reference_path,
                "source": candidate["source"],
                "reason": "path_already_exists",
            })
            continue
        cached_payload = _cached_payload_for_reference(
            reference_path,
            root=root,
            json_cache_sources=json_cache_sources,
        )
        if cached_payload is None:
            skipped.append({
                "path": reference_path,
                "source": candidate["source"],
                "reason": "cached_payload_not_found",
            })
            continue
        normalized_payload, normalized_absolute_path_count = _normalize_cached_json_payload(
            cached_payload["payload"],
            root=root_path,
        )
        restore_variant, digest_mismatch = _matching_json_restore_variant(
            normalized_payload,
            candidate=_mapping(candidate.get("metadata")),
        )
        if digest_mismatch is not None:
            skipped.append({
                "path": reference_path,
                "source": candidate["source"],
                "reason": "manifest_digest_mismatch",
                **digest_mismatch,
                **_mapping(candidate.get("metadata")),
            })
            continue
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(
            str(restore_variant["text"]),
            encoding="utf-8",
        )
        restored.append({
            "path": reference_path,
            "source": candidate["source"],
            "cache_path": cached_payload["cache_path"],
            "entry_key": cached_payload["entry_key"],
            "payload_keys": tuple(sorted(str(key) for key in normalized_payload.keys())),
            "workflow": normalized_payload.get("workflow"),
            "status": normalized_payload.get("status"),
            "restore_serialization": restore_variant["serialization"],
            "normalized_absolute_path_count": normalized_absolute_path_count,
            **_mapping(candidate.get("metadata")),
        })
    return {
        "restored_count": len(restored),
        "skipped_count": len(skipped),
        "restored": tuple(restored),
        "skipped": tuple(skipped),
    }


def _cached_json_restore_candidates(references: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for reference in references:
        if reference["status"] == "missing" and reference.get("recoverable_from_json_cache"):
            reference_path = str(reference["path"])
            if reference_path not in seen_paths:
                candidates.append({
                    "path": reference_path,
                    "source": "document_reference",
                    "metadata": {},
                })
                seen_paths.add(reference_path)
        for child in reference.get("manifest_missing_json_cache_sources", ()):
            if not child.get("recoverable_from_json_cache"):
                continue
            child_path = str(child["path"])
            if child_path in seen_paths:
                continue
            candidates.append({
                "path": child_path,
                "source": "manifest_child",
                "metadata": {
                    "manifest_path": reference["path"],
                    "artifact_name": child.get("name"),
                    "expected_sha256": child.get("expected_sha256"),
                    "expected_size_bytes": child.get("expected_size_bytes"),
                },
            })
            seen_paths.add(child_path)
    return tuple(candidates)


def _manifest_digest_mismatch(restored_text: str, *, candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    expected_sha256 = candidate.get("expected_sha256")
    expected_size_bytes = candidate.get("expected_size_bytes")
    if expected_sha256 is None and expected_size_bytes is None:
        return None
    restored_bytes = restored_text.encode()
    restored_sha256 = hashlib.sha256(restored_bytes).hexdigest()
    restored_size_bytes = len(restored_bytes)
    sha_matches = expected_sha256 is None or restored_sha256 == expected_sha256
    size_matches = expected_size_bytes is None or restored_size_bytes == expected_size_bytes
    if sha_matches and size_matches:
        return None
    return {
        "expected_sha256": expected_sha256,
        "restored_sha256": restored_sha256,
        "expected_size_bytes": expected_size_bytes,
        "restored_size_bytes": restored_size_bytes,
    }


def _matching_json_restore_variant(
    payload: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    first_mismatch: dict[str, Any] | None = None
    for variant in _json_restore_variants(payload):
        digest_mismatch = _manifest_digest_mismatch(str(variant["text"]), candidate=candidate)
        if digest_mismatch is None:
            return variant, None
        if first_mismatch is None:
            first_mismatch = {
                **digest_mismatch,
                "restored_serialization": variant["serialization"],
            }
    return None, first_mismatch


def _json_restore_variants(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    variants = (
        {
            "serialization": "pretty",
            "text": strict_json_dumps(payload, indent=2, sort_keys=True) + "\n",
        },
        {
            "serialization": "compact",
            "text": strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        },
    )
    deduped: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for variant in variants:
        text = str(variant["text"])
        if text in seen_texts:
            continue
        seen_texts.add(text)
        deduped.append(variant)
    return tuple(deduped)


def _normalize_cached_json_payload(payload: Mapping[str, Any], *, root: Path) -> tuple[dict[str, Any], int]:
    normalized_count = 0

    def normalize_value(value: Any) -> Any:
        nonlocal normalized_count
        if isinstance(value, Mapping):
            return {
                normalize_value(key): normalize_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [normalize_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(normalize_value(item) for item in value)
        if isinstance(value, str):
            normalized = _normalize_root_absolute_path(value, root=root)
            if normalized != value:
                normalized_count += 1
            return normalized
        return value

    normalized = normalize_value(payload)
    return dict(_mapping(normalized)), normalized_count


def _normalize_root_absolute_path(value: str, *, root: Path) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    try:
        return str(candidate.resolve().relative_to(root))
    except ValueError:
        return value


def _load_json_cache_sources(
    json_cache_paths: Sequence[str | Path],
    *,
    root: Path,
) -> tuple[dict[str, Any], ...]:
    sources: list[dict[str, Any]] = []
    for path in json_cache_paths:
        cache_path = _resolve_doc_path(path, root=root)
        if not cache_path.exists():
            continue
        cache = load_json_cache(cache_path)
        if not cache:
            continue
        sources.append({
            "path": _display_path(cache_path, root=root),
            "absolute_path": str(cache_path),
            "cache": cache,
        })
    return tuple(sources)


def _json_cache_hits_for_reference(
    reference: str,
    *,
    root: Path,
    json_cache_sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    reference_path = str((root / reference).resolve())
    reference_suffix = "/" + reference.replace("\\", "/")
    hits: list[dict[str, Any]] = []
    for source in json_cache_sources:
        cache = _mapping(source.get("cache"))
        for key, entry in cache.items():
            key_path = _json_cache_key_path(str(key))
            if key_path != reference_path and not key_path.endswith(reference_suffix):
                continue
            entry_mapping = _mapping(entry)
            payload = _mapping(entry_mapping.get("payload"))
            error = entry_mapping.get("error")
            if error is not None or not payload:
                continue
            hits.append({
                "cache_path": str(source["path"]),
                "entry_key": str(key),
                "payload_keys": tuple(sorted(str(item) for item in payload.keys())),
                "workflow": payload.get("workflow"),
                "status": payload.get("status"),
            })
    return tuple(hits)


def _cached_payload_for_reference(
    reference: str,
    *,
    root: Path,
    json_cache_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    reference_path = str((root / reference).resolve())
    reference_suffix = "/" + reference.replace("\\", "/")
    for source in json_cache_sources:
        cache = _mapping(source.get("cache"))
        for key, entry in cache.items():
            key_path = _json_cache_key_path(str(key))
            if key_path != reference_path and not key_path.endswith(reference_suffix):
                continue
            entry_mapping = _mapping(entry)
            payload = _mapping(entry_mapping.get("payload"))
            error = entry_mapping.get("error")
            if error is not None or not payload:
                continue
            return {
                "cache_path": str(source["path"]),
                "entry_key": str(key),
                "payload": dict(payload),
            }
    return None


def _json_cache_key_path(key: str) -> str:
    return key.split(":", 1)[0]


def _write_artifact_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    documents: Sequence[Path],
    references: Sequence[Mapping[str, Any]],
    root: Path,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    artifacts: dict[str, Path] = {"frontier_artifact_reference_audit": output_path}
    for index, document in enumerate(documents, start=1):
        artifacts[f"document.{index:03d}"] = document
    for index, reference in enumerate(references, start=1):
        artifacts[f"reference.{index:03d}"] = root / str(reference["path"])
    manifest = build_artifact_manifest(
        artifacts,
        root=manifest_path.parent,
        metadata={
            "runner": "audit_frontier_artifact_references",
            "status": payload["status"],
            **_mapping(payload.get("summary")),
            **_mapping(payload.get("metadata")),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    return manifest


def _resolve_doc_path(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _display_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _clean_reference(reference: str) -> str:
    return reference.rstrip(".,;:]}").strip()


def _is_artifact_manifest_path(path: Path) -> bool:
    name = path.name
    return name.endswith(".json") and "manifest" in name and "verification" not in name


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(strict_json_dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        description="Audit active EigenTruth frontier artifact references against the local checkout"
    )
    parser.add_argument("--doc", action="append", default=None, help="doc path to scan; repeatable")
    parser.add_argument("--root", default=str(REPO_ROOT), help="repository root for relative references")
    parser.add_argument(
        "--include-regex",
        default=DEFAULT_INCLUDE_REGEX,
        help="only audit artifact paths matching this regex; empty string audits all artifact refs",
    )
    parser.add_argument("--exclude-regex", default=None, help="optional artifact path exclusion regex")
    parser.add_argument("--no-verify-manifests", action="store_true", help="skip manifest verification")
    parser.add_argument("--recursive-manifests", action="store_true", help="verify nested manifests")
    parser.add_argument("--max-workers", type=int, default=1, help="bounded manifest fingerprint workers")
    parser.add_argument(
        "--json-cache",
        action="append",
        default=None,
        help="artifact JSON cache to inspect for recoverable missing references; repeatable",
    )
    parser.add_argument("--no-json-cache", action="store_true", help="do not inspect artifact JSON caches")
    parser.add_argument(
        "--restore-json-cache-artifacts",
        action="store_true",
        help="write missing JSON references when a cached payload is available, then re-run the audit",
    )
    parser.add_argument("--json", default=None, help="optional audit JSON output path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest output path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="optional registry report name")
    parser.add_argument("--version", default=None, help="optional registry report version")
    parser.add_argument("--metadata", action="append", default=[], help="metadata key=value; repeatable")
    parser.add_argument("--no-fail", action="store_true", help="do not exit non-zero when blocked")
    args = parser.parse_args(argv)
    include_regex = args.include_regex if args.include_regex else None
    json_cache_paths = () if args.no_json_cache else tuple(args.json_cache or DEFAULT_JSON_CACHE_PATHS)
    payload = build_frontier_artifact_reference_audit(
        doc_paths=tuple(args.doc or DEFAULT_DOC_PATHS),
        root=args.root,
        include_regex=include_regex,
        exclude_regex=args.exclude_regex,
        verify_manifests=not args.no_verify_manifests,
        recursive_manifests=args.recursive_manifests,
        max_workers=args.max_workers,
        json_cache_paths=json_cache_paths,
        restore_json_cache_artifacts=args.restore_json_cache_artifacts,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
    )
    print(strict_json_dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
