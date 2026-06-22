"""Content fingerprints for local EigenTruth artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ArtifactFingerprint:
    """Stable local metadata for a file, directory, or missing artifact."""

    path: str
    exists: bool
    kind: str
    sha256: str | None = None
    size_bytes: int | None = None
    file_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "path": self.path,
            "exists": self.exists,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
        }


@dataclass(frozen=True)
class ArtifactManifestMismatch:
    """One mismatch between a saved manifest record and current local state."""

    name: str
    path: str
    field: str
    expected: Any
    actual: Any

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "path": self.path,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class ArtifactManifestVerification:
    """Verification result for one artifact manifest."""

    manifest_path: str | None
    checked: int
    failures: tuple[ArtifactManifestMismatch, ...] = ()
    nested: tuple["ArtifactManifestVerification", ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether this manifest and all nested manifests matched."""
        return not self.failures and all(result.passed for result in self.nested)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "manifest_path": self.manifest_path,
            "passed": self.passed,
            "checked": self.checked,
            "failures": [failure.to_dict() for failure in self.failures],
            "nested": [result.to_dict() for result in self.nested],
        }


def fingerprint_path(path: str | Path, *, root: str | Path | None = None) -> ArtifactFingerprint:
    """Fingerprint one path with deterministic directory hashing."""
    artifact_path = Path(path)
    display_path = _display_path(artifact_path, root=root)
    if not artifact_path.exists():
        return ArtifactFingerprint(path=display_path, exists=False, kind="missing")
    if artifact_path.is_file():
        return ArtifactFingerprint(
            path=display_path,
            exists=True,
            kind="file",
            sha256=_sha256_file(artifact_path),
            size_bytes=artifact_path.stat().st_size,
            file_count=1,
        )
    if artifact_path.is_dir():
        digest, size_bytes, file_count = _sha256_directory(artifact_path)
        return ArtifactFingerprint(
            path=display_path,
            exists=True,
            kind="directory",
            sha256=digest,
            size_bytes=size_bytes,
            file_count=file_count,
        )
    return ArtifactFingerprint(path=display_path, exists=True, kind="other")


def build_artifact_manifest(
    artifacts: Mapping[str, str | Path | None],
    *,
    root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a dependency-free manifest with content fingerprints."""
    records = {
        str(name): fingerprint_path(path, root=root).to_dict()
        for name, path in sorted(artifacts.items())
        if path is not None
    }
    return {
        "schema_version": 1,
        "digest_algorithm": "sha256",
        "metadata": {} if metadata is None else dict(metadata),
        "artifacts": records,
        "summary": {
            "artifact_count": len(records),
            "missing_count": sum(1 for record in records.values() if not record["exists"]),
            "directory_count": sum(1 for record in records.values() if record["kind"] == "directory"),
            "file_count": sum(1 for record in records.values() if record["kind"] == "file"),
        },
    }


def verify_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    recursive: bool = False,
) -> ArtifactManifestVerification:
    """Verify current local artifact state against a saved manifest."""
    manifest_root = _verification_root(root=root, manifest_path=manifest_path)
    failures: list[ArtifactManifestMismatch] = []
    nested: list[ArtifactManifestVerification] = []
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifact manifest must contain an 'artifacts' mapping.")
    for name, expected_record in sorted(artifacts.items()):
        if not isinstance(expected_record, Mapping):
            failures.append(
                ArtifactManifestMismatch(str(name), "", "record", "mapping", type(expected_record).__name__)
            )
            continue
        expected_path = str(expected_record.get("path", ""))
        artifact_path = _resolve_manifest_path(expected_path, root=manifest_root)
        actual_record = fingerprint_path(artifact_path, root=manifest_root).to_dict()
        failures.extend(_compare_manifest_record(str(name), expected_record, actual_record))
        if recursive and _is_nested_manifest(str(name), expected_path) and Path(artifact_path).is_file():
            nested.extend(_verify_nested_manifest(artifact_path, recursive=recursive))
    return ArtifactManifestVerification(
        manifest_path=None if manifest_path is None else str(manifest_path),
        checked=sum(1 for value in artifacts.values() if isinstance(value, Mapping)),
        failures=tuple(failures),
        nested=tuple(nested),
    )


def load_and_verify_artifact_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path | None = None,
    recursive: bool = False,
) -> ArtifactManifestVerification:
    """Load and verify a UTF-8 JSON artifact manifest."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return verify_artifact_manifest(manifest, root=root, manifest_path=path, recursive=recursive)


def _display_path(path: Path, *, root: str | Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return path.relative_to(Path(root)).as_posix()
    except ValueError:
        return str(path)


def _verification_root(*, root: str | Path | None, manifest_path: str | Path | None) -> Path:
    if root is not None:
        return Path(root)
    if manifest_path is not None:
        return Path(manifest_path).parent
    return Path(".")


def _resolve_manifest_path(path: str, *, root: Path) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return artifact_path
    return root / artifact_path


def _compare_manifest_record(
    name: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> tuple[ArtifactManifestMismatch, ...]:
    failures = []
    path = str(expected.get("path", actual.get("path", "")))
    for field_name in ("exists", "kind", "sha256", "size_bytes", "file_count"):
        if expected.get(field_name) != actual.get(field_name):
            failures.append(
                ArtifactManifestMismatch(
                    name=name,
                    path=path,
                    field=field_name,
                    expected=expected.get(field_name),
                    actual=actual.get(field_name),
                )
            )
    return tuple(failures)


def _is_nested_manifest(name: str, path: str) -> bool:
    return name.endswith("manifest") or Path(path).name == "artifact-manifest.json"


def _verify_nested_manifest(path: Path, *, recursive: bool) -> tuple[ArtifactManifestVerification, ...]:
    try:
        return (load_and_verify_artifact_manifest(path, recursive=recursive),)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failure = ArtifactManifestMismatch(
            name="nested_manifest",
            path=str(path),
            field="load",
            expected="valid artifact manifest",
            actual=str(exc),
        )
        return (ArtifactManifestVerification(manifest_path=str(path), checked=0, failures=(failure,)),)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    file_count = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = child.relative_to(path).as_posix()
        child_digest = _sha256_file(child)
        child_size = child.stat().st_size
        digest.update(b"file\0")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(child_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(child_digest.encode("ascii"))
        digest.update(b"\0")
        size_bytes += child_size
        file_count += 1
    return digest.hexdigest(), size_bytes, file_count
