"""Audit active frontier artifact references in docs against local files."""

from __future__ import annotations

import argparse
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
ARTIFACT_REFERENCE_RE = re.compile(r"artifacts/[^\s`'\"|)>\\]+")


def build_frontier_artifact_reference_audit(
    *,
    doc_paths: Sequence[str | Path] = DEFAULT_DOC_PATHS,
    root: str | Path = REPO_ROOT,
    include_regex: str | None = DEFAULT_INCLUDE_REGEX,
    exclude_regex: str | None = None,
    verify_manifests: bool = True,
    recursive_manifests: bool = False,
    max_workers: int = 1,
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
    references = _collect_references(
        documents,
        root=root_path,
        include_pattern=include_pattern,
        exclude_pattern=exclude_pattern,
        verify_manifests=verify_manifests,
        recursive_manifests=recursive_manifests,
        max_workers=max_workers,
    )
    blocking_reasons = _blocking_reasons(references)
    if not references:
        blocking_reasons.append("no artifact references matched the configured include/exclude filters")
    summary = _summary(references, document_count=len(documents))
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
        },
        "references": references,
        "blocking_reasons": tuple(blocking_reasons),
        "metadata": dict(metadata or {}),
    }

    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
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
        payload["artifact_manifest"] = str(manifest_path)
        payload["artifact_manifest_summary"] = manifest.get("summary", {})
        _write_json(output_path, payload)
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
        payload["registry_record"] = f"report:{name}:{version}"
        _write_json(output_path, payload)
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
    parser.add_argument("--json", default=None, help="optional audit JSON output path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest output path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="optional registry report name")
    parser.add_argument("--version", default=None, help="optional registry report version")
    parser.add_argument("--metadata", action="append", default=[], help="metadata key=value; repeatable")
    parser.add_argument("--no-fail", action="store_true", help="do not exit non-zero when blocked")
    args = parser.parse_args(argv)
    include_regex = args.include_regex if args.include_regex else None
    payload = build_frontier_artifact_reference_audit(
        doc_paths=tuple(args.doc or DEFAULT_DOC_PATHS),
        root=args.root,
        include_regex=include_regex,
        exclude_regex=args.exclude_regex,
        verify_manifests=not args.no_verify_manifests,
        recursive_manifests=args.recursive_manifests,
        max_workers=args.max_workers,
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
