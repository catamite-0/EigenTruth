"""Content fingerprints for local EigenTruth artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

_FingerprintCache = MutableMapping[str, dict[str, Any]]
_FILE_CACHE_SAMPLE_BYTES = 4096


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


def fingerprint_path(
    path: str | Path,
    *,
    root: str | Path | None = None,
    fingerprint_cache: _FingerprintCache | None = None,
) -> ArtifactFingerprint:
    """Fingerprint one path with deterministic directory hashing."""
    artifact_path = Path(path)
    display_path = _display_path(artifact_path, root=root)
    if not artifact_path.exists():
        cache_key = _fingerprint_cache_key(artifact_path, kind="missing")
        if fingerprint_cache is not None and cache_key in fingerprint_cache:
            return _fingerprint_from_cache(display_path, fingerprint_cache[cache_key])
        fingerprint = ArtifactFingerprint(path=display_path, exists=False, kind="missing")
        _store_fingerprint_cache(fingerprint_cache, cache_key, fingerprint)
        return fingerprint
    if artifact_path.is_file():
        stat = artifact_path.stat()
        cache_key = _fingerprint_cache_key(
            artifact_path,
            kind="file",
            signature=_file_cache_signature(artifact_path, stat=stat),
        )
        if fingerprint_cache is not None and cache_key in fingerprint_cache:
            return _fingerprint_from_cache(display_path, fingerprint_cache[cache_key])
        fingerprint = ArtifactFingerprint(
            path=display_path,
            exists=True,
            kind="file",
            sha256=_sha256_file(artifact_path),
            size_bytes=stat.st_size,
            file_count=1,
        )
        _store_fingerprint_cache(fingerprint_cache, cache_key, fingerprint)
        return fingerprint
    if artifact_path.is_dir():
        cache_key = _fingerprint_cache_key(
            artifact_path,
            kind="directory",
            signature=_directory_cache_signature(artifact_path),
        )
        if fingerprint_cache is not None and cache_key in fingerprint_cache:
            return _fingerprint_from_cache(display_path, fingerprint_cache[cache_key])
        digest, size_bytes, file_count = _sha256_directory(artifact_path)
        fingerprint = ArtifactFingerprint(
            path=display_path,
            exists=True,
            kind="directory",
            sha256=digest,
            size_bytes=size_bytes,
            file_count=file_count,
        )
        _store_fingerprint_cache(fingerprint_cache, cache_key, fingerprint)
        return fingerprint
    cache_key = _fingerprint_cache_key(artifact_path, kind="other")
    if fingerprint_cache is not None and cache_key in fingerprint_cache:
        return _fingerprint_from_cache(display_path, fingerprint_cache[cache_key])
    fingerprint = ArtifactFingerprint(path=display_path, exists=True, kind="other")
    _store_fingerprint_cache(fingerprint_cache, cache_key, fingerprint)
    return fingerprint


def build_artifact_manifest(
    artifacts: Mapping[str, str | Path | None],
    *,
    root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    fingerprint_cache: _FingerprintCache | None = None,
) -> dict[str, Any]:
    """Build a dependency-free manifest with content fingerprints."""
    records = {
        str(name): fingerprint_path(path, root=root, fingerprint_cache=fingerprint_cache).to_dict()
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
    fingerprint_cache: _FingerprintCache | None = None,
) -> ArtifactManifestVerification:
    """Verify current local artifact state against a saved manifest."""
    manifest_root = _verification_root(root=root, manifest_path=manifest_path)
    cache = fingerprint_cache if fingerprint_cache is not None else {}
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
        actual_record = fingerprint_path(
            artifact_path,
            root=manifest_root,
            fingerprint_cache=cache,
        ).to_dict()
        failures.extend(_compare_manifest_record(str(name), expected_record, actual_record))
        if recursive and _is_nested_manifest(str(name), expected_path) and Path(artifact_path).is_file():
            nested.extend(_verify_nested_manifest(
                artifact_path,
                recursive=recursive,
                fingerprint_cache=cache,
            ))
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
    fingerprint_cache: _FingerprintCache | None = None,
) -> ArtifactManifestVerification:
    """Load and verify a UTF-8 JSON artifact manifest."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return verify_artifact_manifest(
        manifest,
        root=root,
        manifest_path=path,
        recursive=recursive,
        fingerprint_cache=fingerprint_cache,
    )


def load_fingerprint_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load a JSON fingerprint cache, returning an empty cache when absent."""
    if path is None:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("fingerprint cache must be a JSON object.")
    cache: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"fingerprint cache entry must be an object: {key!r}")
        cache[str(key)] = dict(value)
    return cache


def save_fingerprint_cache(
    path: str | Path | None,
    cache: Mapping[str, Mapping[str, Any]],
    *,
    compact: bool = False,
) -> None:
    """Persist a JSON fingerprint cache."""
    if path is None:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(key): dict(value) for key, value in sorted(cache.items())}
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    cache_path.write_text(text, encoding="utf-8")


def _fingerprint_cache_key(path: Path, *, kind: str, signature: str = "") -> str:
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path.absolute())
    return f"{kind}:{resolved}:{signature}"


def _directory_cache_signature(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = child.relative_to(path).as_posix()
        stat = child.stat()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_ctime_ns).encode("ascii"))
        digest.update(b"\0")
        digest.update(_file_cache_sample_digest(child, size_bytes=stat.st_size).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_cache_signature(path: Path, *, stat: os.stat_result) -> str:
    sample_digest = _file_cache_sample_digest(path, size_bytes=stat.st_size)
    return f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}:{sample_digest}"


def _file_cache_sample_digest(path: Path, *, size_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if size_bytes <= _FILE_CACHE_SAMPLE_BYTES * 2:
            digest.update(stream.read())
        else:
            digest.update(stream.read(_FILE_CACHE_SAMPLE_BYTES))
            stream.seek(max(size_bytes - _FILE_CACHE_SAMPLE_BYTES, 0))
            digest.update(stream.read(_FILE_CACHE_SAMPLE_BYTES))
    return digest.hexdigest()


def _fingerprint_from_cache(path: str, cached: Mapping[str, Any]) -> ArtifactFingerprint:
    return ArtifactFingerprint(
        path=path,
        exists=bool(cached["exists"]),
        kind=str(cached["kind"]),
        sha256=cached.get("sha256"),
        size_bytes=cached.get("size_bytes"),
        file_count=cached.get("file_count"),
    )


def _store_fingerprint_cache(
    cache: _FingerprintCache | None,
    key: str,
    fingerprint: ArtifactFingerprint,
) -> None:
    if cache is None:
        return
    payload = fingerprint.to_dict()
    payload.pop("path", None)
    cache[key] = payload


def _display_path(path: Path, *, root: str | Path | None) -> str:
    if root is None:
        return str(path)
    root_path = Path(root)
    try:
        return path.relative_to(root_path).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(), start=root_path.resolve())).as_posix()


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


def _verify_nested_manifest(
    path: Path,
    *,
    recursive: bool,
    fingerprint_cache: _FingerprintCache,
) -> tuple[ArtifactManifestVerification, ...]:
    try:
        return (
            load_and_verify_artifact_manifest(
                path,
                recursive=recursive,
                fingerprint_cache=fingerprint_cache,
            ),
        )
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
