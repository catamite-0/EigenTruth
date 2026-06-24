"""Promote a verified artifact manifest into a local registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    ArtifactVerificationContext,
    load_fingerprint_cache,
    save_fingerprint_cache,
)


def promote_artifact_manifest(
    *,
    manifest_path: str | Path,
    registry_path: str | Path,
    name: str,
    version: str,
    verification_report_path: str | Path | None = None,
    recursive: bool = True,
    allow_failures: bool = False,
    metadata: Mapping[str, Any] | None = None,
    fingerprint_cache: MutableMapping[str, dict[str, Any]] | None = None,
    fingerprint_cache_path: str | Path | None = None,
    verification_context: ArtifactVerificationContext | None = None,
    manifest_fingerprint_workers: int = 1,
) -> dict[str, Any]:
    """Verify a manifest and record it in a local artifact registry."""
    manifest_path = Path(manifest_path)
    registry_path = Path(registry_path)
    verification_report_path = Path(verification_report_path or manifest_path.with_name("manifest-verification.json"))
    cache = fingerprint_cache if fingerprint_cache is not None else load_fingerprint_cache(fingerprint_cache_path)
    context = verification_context or ArtifactVerificationContext(fingerprint_cache=cache)
    try:
        verification = context.load_and_verify_artifact_manifest(
            manifest_path,
            recursive=recursive,
            max_workers=manifest_fingerprint_workers,
        )
    finally:
        save_fingerprint_cache(fingerprint_cache_path, context.fingerprint_cache or {})
    verification_payload = verification.to_dict()
    verification_report_path.parent.mkdir(parents=True, exist_ok=True)
    if not verification.passed and not allow_failures:
        verification_report_path.write_text(
            json.dumps(verification_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise ValueError("artifact manifest verification failed; use --allow-failures to register anyway")

    manifest, manifest_error = context.load_json_object(manifest_path)
    if manifest_error is not None:
        raise ValueError(f"artifact manifest could not be loaded: {manifest_path}: {manifest_error}")
    manifest_metadata = dict(manifest.get("metadata", {})) if isinstance(manifest.get("metadata", {}), Mapping) else {}
    registry_metadata = {
        "verified": verification.passed,
        "verification_report": str(verification_report_path),
        "recursive": recursive,
        "checked": verification.checked,
        "nested_count": len(verification.nested),
        "failure_count": _failure_count(verification_payload),
        "artifact_json_cache": context.json_cache_summary(),
        "artifact_fingerprint_cache_entries": len(context.fingerprint_cache or {}),
        "manifest_fingerprint_workers": int(manifest_fingerprint_workers),
        "manifest_summary": (
            dict(manifest.get("summary", {})) if isinstance(manifest.get("summary", {}), Mapping) else {}
        ),
        "manifest_metadata": manifest_metadata,
    }
    if metadata is not None:
        registry_metadata.update(dict(metadata))

    verification_report_path.write_text(
        json.dumps(verification_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    registry = ArtifactRegistry.load_json(registry_path)
    registry.record_benchmark_manifest(
        name=name,
        path=manifest_path,
        version=version,
        metadata=registry_metadata,
    ).record_manifest_verification(
        name=f"{name}-verification",
        path=verification_report_path,
        version=version,
        metadata={
            "manifest_name": name,
            "manifest_path": str(manifest_path),
            "passed": verification.passed,
            "recursive": recursive,
        },
    ).save_json()

    return {
        "name": name,
        "version": version,
        "registry": str(registry_path),
        "manifest": str(manifest_path),
        "verification_report": str(verification_report_path),
        "verification": verification_payload,
        "records": {
            "benchmark_manifest": f"benchmark_manifest:{name}:{version}",
            "manifest_verification": f"manifest_verification:{name}-verification:{version}",
        },
    }


def _failure_count(verification_payload: Mapping[str, Any]) -> int:
    count = len(tuple(verification_payload.get("failures", ())))
    for nested in verification_payload.get("nested", ()):
        if isinstance(nested, Mapping):
            count += _failure_count(nested)
    return count


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata entry must be key=value: {value!r}")
        key, text = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key must not be empty: {value!r}")
        metadata[key] = text
    return metadata


def _parse_positive_int(value: str, *, flag: str) -> int:
    numeric = int(value)
    if numeric < 1:
        raise ValueError(f"{flag} must be a positive integer.")
    return numeric


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    payload = promote_artifact_manifest(
        manifest_path=args.manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        verification_report_path=args.verification_report,
        recursive=not args.no_recursive,
        allow_failures=bool(args.allow_failures),
        metadata=_parse_metadata(args.metadata or ()),
        fingerprint_cache_path=args.fingerprint_cache,
        manifest_fingerprint_workers=args.manifest_fingerprint_workers,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify and register an EigenTruth artifact manifest")
    parser.add_argument("--manifest", required=True, help="artifact-manifest.json to promote")
    parser.add_argument("--registry", required=True, help="local ArtifactRegistry JSON path")
    parser.add_argument("--name", required=True, help="registry artifact name")
    parser.add_argument("--version", required=True, help="registry artifact version")
    parser.add_argument("--verification-report", default=None, help="path for the verification report JSON")
    parser.add_argument("--metadata", action="append", default=[], help="extra registry metadata as key=value")
    parser.add_argument("--fingerprint-cache", default=None, help="optional JSON cache for manifest fingerprint reads")
    parser.add_argument(
        "--manifest-fingerprint-workers",
        type=lambda value: _parse_positive_int(value, flag="--manifest-fingerprint-workers"),
        default=1,
        help="bounded worker count for manifest artifact fingerprinting",
    )
    parser.add_argument("--no-recursive", action="store_true", help="only verify the root manifest")
    parser.add_argument("--allow-failures", action="store_true", help="register even when verification fails")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
