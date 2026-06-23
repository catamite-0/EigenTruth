"""Shared helpers for benchmark workflow configuration parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def strict_bool(value: Any, *, name: str) -> bool:
    """Parse booleans without treating arbitrary non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or boolean string.")


def planned_artifact_manifest_summary(
    artifacts: Mapping[str, str | Path | None],
    *,
    assume_file_paths: tuple[str | Path, ...] = (),
) -> dict[str, int]:
    """Return manifest count fields without hashing artifact contents."""
    records = [
        _planned_artifact_kind(path, assume_file_paths=assume_file_paths)
        for _name, path in sorted(artifacts.items())
        if path is not None
    ]
    return {
        "artifact_count": len(records),
        "missing_count": sum(1 for record in records if not record["exists"]),
        "directory_count": sum(1 for record in records if record["kind"] == "directory"),
        "file_count": sum(1 for record in records if record["kind"] == "file"),
    }


def _planned_artifact_kind(
    path: str | Path,
    *,
    assume_file_paths: tuple[str | Path, ...],
) -> dict[str, Any]:
    artifact_path = Path(path)
    if _matches_any_path(artifact_path, assume_file_paths):
        return {"exists": True, "kind": "file"}
    if not artifact_path.exists():
        return {"exists": False, "kind": "missing"}
    if artifact_path.is_file():
        return {"exists": True, "kind": "file"}
    if artifact_path.is_dir():
        return {"exists": True, "kind": "directory"}
    return {"exists": True, "kind": "other"}


def _matches_any_path(path: Path, candidates: tuple[str | Path, ...]) -> bool:
    resolved = _safe_resolve(path)
    return any(resolved == _safe_resolve(Path(candidate)) for candidate in candidates)


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()
