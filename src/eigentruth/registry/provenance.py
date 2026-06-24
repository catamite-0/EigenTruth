"""Content fingerprints for local EigenTruth artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

_FingerprintCache = MutableMapping[str, dict[str, Any]]
_FingerprintCacheStats = MutableMapping[str, int]
_JsonCache = MutableMapping[str, dict[str, Any]]
_JsonCacheStats = MutableMapping[str, int]
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


@dataclass
class ArtifactVerificationContext:
    """Shared local caches for artifact JSON and manifest verification."""

    fingerprint_cache: _FingerprintCache | None = None
    fingerprint_cache_stats: _FingerprintCacheStats | None = None
    json_cache: _JsonCache | None = None
    json_cache_stats: _JsonCacheStats | None = None
    fingerprint_cache_lock: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.fingerprint_cache is None:
            self.fingerprint_cache = {}
        if self.fingerprint_cache_stats is None:
            self.fingerprint_cache_stats = new_fingerprint_cache_stats()
        if self.fingerprint_cache_lock is None:
            self.fingerprint_cache_lock = threading.Lock()
        if self.json_cache is None:
            self.json_cache = {}
        if self.json_cache_stats is None:
            self.json_cache_stats = new_json_cache_stats()

    def fingerprint_path(
        self,
        path: str | Path,
        *,
        root: str | Path | None = None,
    ) -> ArtifactFingerprint:
        """Fingerprint one path using this context's fingerprint cache."""
        return fingerprint_path(
            path,
            root=root,
            fingerprint_cache=self.fingerprint_cache,
            fingerprint_cache_stats=self.fingerprint_cache_stats,
            fingerprint_cache_lock=self.fingerprint_cache_lock,
        )

    def build_artifact_manifest(
        self,
        artifacts: Mapping[str, str | Path | None],
        *,
        root: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
        max_workers: int = 1,
    ) -> dict[str, Any]:
        """Build an artifact manifest using this context's fingerprint cache."""
        return build_artifact_manifest(
            artifacts,
            root=root,
            metadata=metadata,
            fingerprint_cache=self.fingerprint_cache,
            fingerprint_cache_stats=self.fingerprint_cache_stats,
            fingerprint_cache_lock=self.fingerprint_cache_lock,
            max_workers=max_workers,
        )

    def verify_artifact_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        root: str | Path | None = None,
        manifest_path: str | Path | None = None,
        recursive: bool = False,
        max_workers: int = 1,
    ) -> ArtifactManifestVerification:
        """Verify a manifest using this context's fingerprint cache."""
        return verify_artifact_manifest(
            manifest,
            root=root,
            manifest_path=manifest_path,
            recursive=recursive,
            fingerprint_cache=self.fingerprint_cache,
            fingerprint_cache_stats=self.fingerprint_cache_stats,
            fingerprint_cache_lock=self.fingerprint_cache_lock,
            json_cache=self.json_cache,
            json_cache_stats=self.json_cache_stats,
            max_workers=max_workers,
        )

    def load_and_verify_artifact_manifest(
        self,
        manifest_path: str | Path,
        *,
        root: str | Path | None = None,
        recursive: bool = False,
        max_workers: int = 1,
    ) -> ArtifactManifestVerification:
        """Load a manifest through this context's JSON cache and verify it."""
        path = Path(manifest_path)
        manifest, error = self.load_json_object(path)
        if error is not None:
            raise ValueError(f"artifact manifest could not be loaded: {path}: {error}")
        return self.verify_artifact_manifest(
            manifest,
            root=root,
            manifest_path=path,
            recursive=recursive,
            max_workers=max_workers,
        )

    def load_json_object(self, path: str | Path) -> tuple[dict[str, Any], str | None]:
        """Load a JSON object using this context's path-signature cache."""
        return load_json_object(
            Path(path),
            json_cache=self.json_cache,
            json_cache_stats=self.json_cache_stats,
        )

    def json_cache_summary(self) -> dict[str, Any]:
        """Return JSON cache counters for reports."""
        return json_cache_summary(self.json_cache or {}, self.json_cache_stats or {})

    def cache_summary(self) -> dict[str, Any]:
        """Return a combined cache summary for reports."""
        return {
            "artifact_json_cache": self.json_cache_summary(),
            "artifact_fingerprint_cache": fingerprint_cache_summary(
                self.fingerprint_cache or {},
                self.fingerprint_cache_stats or {},
            ),
        }


def fingerprint_path(
    path: str | Path,
    *,
    root: str | Path | None = None,
    fingerprint_cache: _FingerprintCache | None = None,
    fingerprint_cache_stats: _FingerprintCacheStats | None = None,
    fingerprint_cache_lock: Any | None = None,
) -> ArtifactFingerprint:
    """Fingerprint one path with deterministic directory hashing."""
    _increment_fingerprint_cache_stat(fingerprint_cache_stats, "requests", fingerprint_cache_lock)
    artifact_path = Path(path)
    display_path = _display_path(artifact_path, root=root)
    if not artifact_path.exists():
        cache_key = _fingerprint_cache_key(artifact_path, kind="missing")
        cached = _load_fingerprint_cache_entry(
            fingerprint_cache,
            cache_key,
            display_path,
            stats=fingerprint_cache_stats,
            lock=fingerprint_cache_lock,
        )
        if cached is not None:
            return cached
        fingerprint = ArtifactFingerprint(path=display_path, exists=False, kind="missing")
        _store_fingerprint_cache(
            fingerprint_cache,
            cache_key,
            fingerprint,
            lock=fingerprint_cache_lock,
        )
        return fingerprint
    if artifact_path.is_file():
        stat = artifact_path.stat()
        cache_key = _fingerprint_cache_key(
            artifact_path,
            kind="file",
            signature=_file_cache_signature(artifact_path, stat=stat),
        )
        cached = _load_fingerprint_cache_entry(
            fingerprint_cache,
            cache_key,
            display_path,
            stats=fingerprint_cache_stats,
            lock=fingerprint_cache_lock,
        )
        if cached is not None:
            return cached
        fingerprint = ArtifactFingerprint(
            path=display_path,
            exists=True,
            kind="file",
            sha256=_sha256_file(artifact_path),
            size_bytes=stat.st_size,
            file_count=1,
        )
        _store_fingerprint_cache(
            fingerprint_cache,
            cache_key,
            fingerprint,
            lock=fingerprint_cache_lock,
        )
        return fingerprint
    if artifact_path.is_dir():
        cache_key = _fingerprint_cache_key(
            artifact_path,
            kind="directory",
            signature=_directory_cache_signature(artifact_path),
        )
        cached = _load_fingerprint_cache_entry(
            fingerprint_cache,
            cache_key,
            display_path,
            stats=fingerprint_cache_stats,
            lock=fingerprint_cache_lock,
        )
        if cached is not None:
            return cached
        digest, size_bytes, file_count = _sha256_directory(artifact_path)
        fingerprint = ArtifactFingerprint(
            path=display_path,
            exists=True,
            kind="directory",
            sha256=digest,
            size_bytes=size_bytes,
            file_count=file_count,
        )
        _store_fingerprint_cache(
            fingerprint_cache,
            cache_key,
            fingerprint,
            lock=fingerprint_cache_lock,
        )
        return fingerprint
    cache_key = _fingerprint_cache_key(artifact_path, kind="other")
    cached = _load_fingerprint_cache_entry(
        fingerprint_cache,
        cache_key,
        display_path,
        stats=fingerprint_cache_stats,
        lock=fingerprint_cache_lock,
    )
    if cached is not None:
        return cached
    fingerprint = ArtifactFingerprint(path=display_path, exists=True, kind="other")
    _store_fingerprint_cache(
        fingerprint_cache,
        cache_key,
        fingerprint,
        lock=fingerprint_cache_lock,
    )
    return fingerprint


def build_artifact_manifest(
    artifacts: Mapping[str, str | Path | None],
    *,
    root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    fingerprint_cache: _FingerprintCache | None = None,
    fingerprint_cache_stats: _FingerprintCacheStats | None = None,
    fingerprint_cache_lock: Any | None = None,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Build a dependency-free manifest with content fingerprints."""
    items = [
        (str(name), Path(path))
        for name, path in sorted(artifacts.items())
        if path is not None
    ]
    fingerprints = _fingerprint_artifacts(
        items,
        root=root,
        fingerprint_cache=fingerprint_cache,
        fingerprint_cache_stats=fingerprint_cache_stats,
        fingerprint_cache_lock=fingerprint_cache_lock,
        max_workers=max_workers,
    )
    records = {
        name: fingerprint.to_dict()
        for name, fingerprint in fingerprints
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
    fingerprint_cache_stats: _FingerprintCacheStats | None = None,
    fingerprint_cache_lock: Any | None = None,
    json_cache: _JsonCache | None = None,
    json_cache_stats: _JsonCacheStats | None = None,
    max_workers: int = 1,
) -> ArtifactManifestVerification:
    """Verify current local artifact state against a saved manifest."""
    manifest_root = _verification_root(root=root, manifest_path=manifest_path)
    cache = fingerprint_cache if fingerprint_cache is not None else {}
    failures: list[ArtifactManifestMismatch] = []
    nested: list[ArtifactManifestVerification] = []
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifact manifest must contain an 'artifacts' mapping.")
    fingerprint_items: list[tuple[str, Path]] = []
    expected_records: dict[str, Mapping[str, Any]] = {}
    expected_paths: dict[str, str] = {}
    for name, expected_record in sorted(artifacts.items()):
        if not isinstance(expected_record, Mapping):
            failures.append(
                ArtifactManifestMismatch(str(name), "", "record", "mapping", type(expected_record).__name__)
            )
            continue
        expected_path = str(expected_record.get("path", ""))
        artifact_path = _resolve_manifest_path(expected_path, root=manifest_root)
        name_key = str(name)
        expected_records[name_key] = expected_record
        expected_paths[name_key] = expected_path
        fingerprint_items.append((name_key, artifact_path))

    actual_records = {
        name: fingerprint.to_dict()
        for name, fingerprint in _fingerprint_artifacts(
            fingerprint_items,
            root=manifest_root,
            fingerprint_cache=cache,
            fingerprint_cache_stats=fingerprint_cache_stats,
            fingerprint_cache_lock=fingerprint_cache_lock,
            max_workers=max_workers,
        )
    }
    for name, _artifact_path in fingerprint_items:
        expected_record = expected_records[name]
        expected_path = expected_paths[name]
        actual_record = actual_records[name]
        failures.extend(_compare_manifest_record(name, expected_record, actual_record))
        artifact_path = _resolve_manifest_path(expected_path, root=manifest_root)
        if recursive and _is_nested_manifest(str(name), expected_path) and Path(artifact_path).is_file():
            nested.extend(_verify_nested_manifest(
                artifact_path,
                recursive=recursive,
                fingerprint_cache=cache,
                fingerprint_cache_stats=fingerprint_cache_stats,
                fingerprint_cache_lock=fingerprint_cache_lock,
                json_cache=json_cache,
                json_cache_stats=json_cache_stats,
                max_workers=max_workers,
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
    fingerprint_cache_stats: _FingerprintCacheStats | None = None,
    fingerprint_cache_lock: Any | None = None,
    json_cache: _JsonCache | None = None,
    json_cache_stats: _JsonCacheStats | None = None,
    max_workers: int = 1,
) -> ArtifactManifestVerification:
    """Load and verify a UTF-8 JSON artifact manifest."""
    path = Path(manifest_path)
    if json_cache is None and json_cache_stats is None:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest, error = load_json_object(path, json_cache=json_cache, json_cache_stats=json_cache_stats)
        if error is not None:
            raise ValueError(f"artifact manifest could not be loaded: {path}: {error}")
    return verify_artifact_manifest(
        manifest,
        root=root,
        manifest_path=path,
        recursive=recursive,
        fingerprint_cache=fingerprint_cache,
        fingerprint_cache_stats=fingerprint_cache_stats,
        fingerprint_cache_lock=fingerprint_cache_lock,
        json_cache=json_cache,
        json_cache_stats=json_cache_stats,
        max_workers=max_workers,
    )


def _fingerprint_artifacts(
    items: Sequence[tuple[str, Path]],
    *,
    root: str | Path | None,
    fingerprint_cache: _FingerprintCache | None,
    fingerprint_cache_stats: _FingerprintCacheStats | None,
    fingerprint_cache_lock: Any | None,
    max_workers: int,
) -> tuple[tuple[str, ArtifactFingerprint], ...]:
    workers = _normalize_max_workers(max_workers)
    if not items:
        return ()
    active_lock = fingerprint_cache_lock
    if workers > 1 and active_lock is None and (
        fingerprint_cache is not None or fingerprint_cache_stats is not None
    ):
        active_lock = threading.Lock()
    if workers <= 1 or len(items) <= 1:
        return tuple(
            (
                name,
                fingerprint_path(
                    path,
                    root=root,
                    fingerprint_cache=fingerprint_cache,
                    fingerprint_cache_stats=fingerprint_cache_stats,
                    fingerprint_cache_lock=active_lock,
                ),
            )
            for name, path in items
        )

    results: list[tuple[str, ArtifactFingerprint] | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
        futures = {
            executor.submit(
                fingerprint_path,
                path,
                root=root,
                fingerprint_cache=fingerprint_cache,
                fingerprint_cache_stats=fingerprint_cache_stats,
                fingerprint_cache_lock=active_lock,
            ): index
            for index, (_name, path) in enumerate(items)
        }
        for future in as_completed(futures):
            index = futures[future]
            name = items[index][0]
            results[index] = (name, future.result())
    return tuple(result for result in results if result is not None)


def _normalize_max_workers(max_workers: int) -> int:
    workers = int(max_workers)
    if workers < 1:
        raise ValueError("max_workers must be at least 1.")
    return workers


def load_json_object(
    path: str | Path,
    *,
    json_cache: _JsonCache | None = None,
    json_cache_stats: _JsonCacheStats | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Load a JSON object, optionally using a path-signature cache."""
    increment_json_cache_stat(json_cache_stats, "requests")
    cache_key = None if json_cache is None else json_cache_key(Path(path))
    if cache_key is not None:
        cached = json_cache.get(cache_key)
        if cached is not None:
            error = cached.get("error")
            increment_json_cache_stat(json_cache_stats, "hits")
            if error is not None:
                increment_json_cache_stat(json_cache_stats, "errors")
            return _mapping(cached.get("payload")), error
    increment_json_cache_stat(json_cache_stats, "misses")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error = str(exc)
        increment_json_cache_stat(json_cache_stats, "errors")
        if cache_key is not None:
            json_cache[cache_key] = {"payload": {}, "error": error}
        return {}, error
    if not isinstance(payload, dict):
        error = f"{path} did not contain a JSON object"
        increment_json_cache_stat(json_cache_stats, "errors")
        if cache_key is not None:
            json_cache[cache_key] = {"payload": {}, "error": error}
        return {}, error
    if cache_key is not None:
        json_cache[cache_key] = {"payload": dict(payload), "error": None}
    return payload, None


def json_cache_key(path: str | Path) -> str | None:
    """Return a cache key for the current path signature, or None if absent."""
    candidate = Path(path)
    try:
        stat = candidate.stat()
    except OSError:
        return None
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        resolved = candidate.absolute()
    return f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}:{getattr(stat, 'st_ino', 0)}"


def new_json_cache_stats() -> dict[str, int]:
    """Return initialized JSON cache counters."""
    return {
        "requests": 0,
        "hits": 0,
        "misses": 0,
        "errors": 0,
    }


def new_fingerprint_cache_stats() -> dict[str, int]:
    """Return initialized fingerprint cache counters."""
    return {
        "requests": 0,
        "hits": 0,
        "misses": 0,
    }


def increment_fingerprint_cache_stat(stats: _FingerprintCacheStats | None, key: str) -> None:
    """Increment one fingerprint cache counter if stats are being tracked."""
    if stats is None:
        return
    stats[key] = int(stats.get(key, 0)) + 1


def _increment_fingerprint_cache_stat(
    stats: _FingerprintCacheStats | None,
    key: str,
    lock: Any | None,
) -> None:
    if lock is None:
        increment_fingerprint_cache_stat(stats, key)
        return
    with lock:
        increment_fingerprint_cache_stat(stats, key)


def _load_fingerprint_cache_entry(
    cache: _FingerprintCache | None,
    key: str,
    display_path: str,
    *,
    stats: _FingerprintCacheStats | None,
    lock: Any | None,
) -> ArtifactFingerprint | None:
    if lock is None:
        if cache is not None and key in cache:
            increment_fingerprint_cache_stat(stats, "hits")
            return _fingerprint_from_cache(display_path, cache[key])
        increment_fingerprint_cache_stat(stats, "misses")
        return None
    with lock:
        if cache is not None and key in cache:
            increment_fingerprint_cache_stat(stats, "hits")
            return _fingerprint_from_cache(display_path, cache[key])
        increment_fingerprint_cache_stat(stats, "misses")
        return None


def fingerprint_cache_summary(
    fingerprint_cache: Mapping[str, Any],
    stats: Mapping[str, int],
) -> dict[str, Any]:
    """Return fingerprint cache counters with a hit-rate convenience field."""
    requests = int(stats.get("requests", 0))
    hits = int(stats.get("hits", 0))
    return {
        "requests": requests,
        "hits": hits,
        "misses": int(stats.get("misses", 0)),
        "entries": len(fingerprint_cache),
        "hit_rate": 0.0 if requests <= 0 else float(hits) / float(requests),
    }


def increment_json_cache_stat(stats: _JsonCacheStats | None, key: str) -> None:
    """Increment one JSON cache counter if stats are being tracked."""
    if stats is None:
        return
    stats[key] = int(stats.get(key, 0)) + 1


def json_cache_summary(
    json_cache: Mapping[str, Any],
    stats: Mapping[str, int],
) -> dict[str, Any]:
    """Return JSON cache counters with a hit-rate convenience field."""
    requests = int(stats.get("requests", 0))
    hits = int(stats.get("hits", 0))
    return {
        "requests": requests,
        "hits": hits,
        "misses": int(stats.get("misses", 0)),
        "errors": int(stats.get("errors", 0)),
        "entries": len(json_cache),
        "hit_rate": 0.0 if requests <= 0 else float(hits) / float(requests),
    }


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


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _store_fingerprint_cache(
    cache: _FingerprintCache | None,
    key: str,
    fingerprint: ArtifactFingerprint,
    *,
    lock: Any | None = None,
) -> None:
    if cache is None:
        return
    payload = fingerprint.to_dict()
    payload.pop("path", None)
    if lock is None:
        cache[key] = payload
        return
    with lock:
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
    fingerprint_cache_stats: _FingerprintCacheStats | None = None,
    fingerprint_cache_lock: Any | None = None,
    json_cache: _JsonCache | None = None,
    json_cache_stats: _JsonCacheStats | None = None,
    max_workers: int = 1,
) -> tuple[ArtifactManifestVerification, ...]:
    try:
        return (
            load_and_verify_artifact_manifest(
                path,
                recursive=recursive,
                fingerprint_cache=fingerprint_cache,
                fingerprint_cache_stats=fingerprint_cache_stats,
                fingerprint_cache_lock=fingerprint_cache_lock,
                json_cache=json_cache,
                json_cache_stats=json_cache_stats,
                max_workers=max_workers,
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
