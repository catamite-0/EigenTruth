"""Content fingerprints for local EigenTruth artifacts."""

from __future__ import annotations

import hashlib
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


def _display_path(path: Path, *, root: str | Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return path.relative_to(Path(root)).as_posix()
    except ValueError:
        return str(path)


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
