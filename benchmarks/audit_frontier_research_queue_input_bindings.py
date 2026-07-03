"""Audit filled frontier input-binding sidecars before rule-input fill.

This workflow sits between ``scaffold_frontier_research_queue_input_bindings``
and the typed rule-input fill scripts. It validates edited sidecar rows,
reports which bindings are ready for fill, and never executes downstream
commands or treats bindings as verifier evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.scaffold_frontier_research_queue_input_bindings import (  # noqa: E402
    WORKFLOW as SCAFFOLD_WORKFLOW,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_research_queue_input_binding_audit"

RESERVED_FIELDS = {
    "answer",
    "answers",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "score_label",
}
READY_REVIEW_STATUSES = {"", "ready", "approved"}
SUBJECT_RESOLVABLE_REVIEW_STATUSES = {"ambiguous_subject", "subject_ambiguous"}
MECHANISM_STATUS_ALIASES = {
    "support": "supported",
    "supported": "supported",
    "supports": "supported",
    "refute": "refuted",
    "refuted": "refuted",
    "refutes": "refuted",
    "insufficient": "insufficient_evidence",
    "insufficient_evidence": "insufficient_evidence",
    "unknown": "insufficient_evidence",
}

SIDECAR_SPECS = {
    "numeric_bindings": {
        "path_key": "numeric_bindings",
        "label": "source_backed_numeric_bindings",
    },
    "subject_bindings": {
        "path_key": "subject_bindings",
        "label": "source_backed_subject_bindings",
    },
    "temporal_bindings": {
        "path_key": "temporal_bindings",
        "label": "source_backed_temporal_bindings",
    },
    "mechanism_bindings": {
        "path_key": "mechanism_bindings",
        "label": "source_backed_mechanism_bindings",
    },
    "entity_bindings": {
        "path_key": "entity_bindings",
        "label": "source_backed_entity_bindings",
    },
    "source_family_url_seeds": {
        "path_key": "source_family_url_seeds",
        "label": "source_family_url_seeds",
    },
}


def audit_frontier_research_queue_input_bindings(
    *,
    input_binding_scaffold: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    json_path: str | Path | None = None,
    binding_audit_rows_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit source-backed sidecar bindings without executing fill commands."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    scaffold_path, scaffold = _load_mapping_source(input_binding_scaffold)
    if scaffold.get("workflow") != SCAFFOLD_WORKFLOW:
        raise ValueError(f"input_binding_scaffold must have workflow={SCAFFOLD_WORKFLOW!r}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(json_path or output / "frontier-input-binding-audit.json")
    audit_rows_path = Path(binding_audit_rows_path or output / "binding-audit-rows.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    scaffold_root = None if scaffold_path is None else scaffold_path.parent
    sidecar_paths = _sidecar_paths(scaffold, scaffold_root=scaffold_root)
    loaded_sidecars = {
        key: _load_sidecar_rows(path) if path is not None and path.exists() else ()
        for key, path in sidecar_paths.items()
    }
    missing_sidecar_keys = tuple(
        key for key, path in sidecar_paths.items() if path is None or not path.exists()
    )

    subject_rows = loaded_sidecars.get("subject_bindings", ())
    subject_id_counts = _request_id_counts(subject_rows)
    subject_audits = tuple(
        _audit_subject_binding(
            row,
            line_no=index,
            duplicate=_request_id(row) in subject_id_counts and subject_id_counts[_request_id(row)] > 1,
        )
        for index, row in enumerate(subject_rows, start=1)
    )
    ready_subjects = {
        str(item["request_id"]): subject_rows[int(item["line_no"]) - 1]
        for item in subject_audits
        if item["status"] == "ready" and item["request_id"]
    }

    audit_rows: list[dict[str, Any]] = []
    for key in (
        "numeric_bindings",
        "subject_bindings",
        "temporal_bindings",
        "mechanism_bindings",
        "entity_bindings",
        "source_family_url_seeds",
    ):
        if key not in loaded_sidecars:
            continue
        rows = loaded_sidecars.get(key, ())
        id_counts = (
            _source_family_url_seed_key_counts(rows)
            if key == "source_family_url_seeds"
            else _request_id_counts(rows)
        )
        if key == "subject_bindings":
            audit_rows.extend(subject_audits)
            continue
        for index, row in enumerate(rows, start=1):
            row_key = (
                _source_family_url_seed_key(row)
                if key == "source_family_url_seeds"
                else _request_id(row)
            )
            duplicate = row_key in id_counts and id_counts[row_key] > 1
            if key == "numeric_bindings":
                audit_rows.append(
                    _audit_numeric_binding(
                        row,
                        line_no=index,
                        duplicate=duplicate,
                        ready_subject_binding=ready_subjects.get(_request_id(row)),
                    )
                )
            elif key == "temporal_bindings":
                audit_rows.append(_audit_temporal_binding(row, line_no=index, duplicate=duplicate))
            elif key == "mechanism_bindings":
                audit_rows.append(_audit_mechanism_binding(row, line_no=index, duplicate=duplicate))
            elif key == "entity_bindings":
                audit_rows.append(_audit_entity_binding(row, line_no=index, duplicate=duplicate))
            elif key == "source_family_url_seeds":
                audit_rows.append(_audit_source_family_url_seed(row, line_no=index, duplicate=duplicate))

    for key in missing_sidecar_keys:
        audit_rows.append(_missing_sidecar_row(key, sidecar_paths.get(key)))

    downstream_commands = tuple(_downstream_command_statuses(scaffold, audit_rows))
    summary = _summary(
        audit_rows=audit_rows,
        loaded_sidecars=loaded_sidecars,
        missing_sidecar_keys=missing_sidecar_keys,
        downstream_commands=downstream_commands,
    )
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Pre-execution audit for edited source-backed frontier input bindings. "
            "Ready rows may be passed to typed rule-input fill scripts; blocked rows "
            "need review, source values, or corrected provenance first."
        ),
        "source": {
            "input_binding_scaffold": None if scaffold_path is None else str(scaffold_path),
            "input_binding_scaffold_workflow": scaffold.get("workflow"),
            "input_binding_scaffold_status": scaffold.get("status"),
            "binding_skeleton_count": _nested(scaffold, "summary", "binding_skeleton_count"),
        },
        "label_usage": {
            "labels_used_for_binding_audit": False,
            "labels_allowed_in_sidecars": False,
            "model_answers_allowed_in_sidecars": False,
            "bindings_are_verifier_evidence": False,
            "audit_executes_commands": False,
        },
        "config": {
            "ready_review_statuses": tuple(sorted(status or "<empty>" for status in READY_REVIEW_STATUSES)),
            "subject_resolvable_review_statuses": tuple(sorted(SUBJECT_RESOLVABLE_REVIEW_STATUSES)),
        },
        "summary": summary,
        "paths": {
            "input_binding_scaffold": None if scaffold_path is None else str(scaffold_path),
            "report": str(report_path),
            "binding_audit_rows": str(audit_rows_path),
            "artifact_manifest": str(manifest_path),
            **{key: None if path is None else str(path) for key, path in sidecar_paths.items()},
        },
        "sidecar_files": _sidecar_file_statuses(sidecar_paths, loaded_sidecars),
        "binding_audit_rows": tuple(audit_rows),
        "downstream_commands": downstream_commands,
        "metadata": dict(metadata or {}),
    }

    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(audit_rows_path, audit_rows, compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "frontier_research_queue_input_binding_audit": report_path,
            "binding_audit_rows": audit_rows_path,
            "input_binding_scaffold": scaffold_path,
            **{key: path for key, path in sidecar_paths.items()},
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "ready_binding_count": summary["ready_binding_count"],
            "blocked_binding_count": summary["blocked_binding_count"],
            "missing_sidecar_count": summary["missing_sidecar_count"],
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "artifact_manifest": str(manifest_path),
                "ready_binding_count": summary["ready_binding_count"],
                "blocked_binding_count": summary["blocked_binding_count"],
                "missing_sidecar_count": summary["missing_sidecar_count"],
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _audit_numeric_binding(
    row: Mapping[str, Any],
    *,
    line_no: int,
    duplicate: bool,
    ready_subject_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    failures = list(_base_failures(row, duplicate=duplicate, duplicate_reason="duplicate_numeric_binding"))
    subject_applied = False
    review_status = _clean(row.get("review_status")).lower()
    subject = _clean(row.get("subject_entity"))
    if ready_subject_binding is not None and review_status in SUBJECT_RESOLVABLE_REVIEW_STATUSES:
        bound_subject = _clean(ready_subject_binding.get("subject_entity"))
        if subject and subject != bound_subject:
            failures.append("subject_binding_conflicts_with_numeric_subject")
        else:
            subject_applied = True
            failures = [failure for failure in failures if failure != "binding_requires_review"]
    elif review_status in SUBJECT_RESOLVABLE_REVIEW_STATUSES:
        failures = [failure for failure in failures if failure != "binding_requires_review"]
        failures.append("numeric_subject_binding_required")

    if not subject and not subject_applied:
        failures.append("missing_subject_entity")
    for key in ("unit", "reference_time", "source_citation"):
        if not _clean(row.get(key)):
            failures.append(f"missing_{key}")
    candidate = _float_or_none(row.get("candidate_numeric_value"))
    source = _float_or_none(row.get("source_numeric_value", row.get("numeric_value")))
    if candidate is None:
        failures.append("missing_or_invalid_candidate_numeric_value")
    if source is None:
        failures.append("missing_or_invalid_source_numeric_value")
    return _audit_row(
        row,
        sidecar_key="numeric_bindings",
        line_no=line_no,
        failures=failures,
        extra={
            "subject_binding_applied": subject_applied,
            "subject_binding_request_id": _request_id(ready_subject_binding or {}),
        },
    )


def _audit_subject_binding(row: Mapping[str, Any], *, line_no: int, duplicate: bool) -> dict[str, Any]:
    failures = list(_base_failures(row, duplicate=duplicate, duplicate_reason="duplicate_subject_binding"))
    for key in ("subject_entity", "source_citation"):
        if not _clean(row.get(key)):
            failures.append(f"missing_{key}")
    return _audit_row(row, sidecar_key="subject_bindings", line_no=line_no, failures=failures)


def _audit_temporal_binding(row: Mapping[str, Any], *, line_no: int, duplicate: bool) -> dict[str, Any]:
    failures = list(_base_failures(row, duplicate=duplicate, duplicate_reason="duplicate_temporal_binding"))
    for key in ("claim_time", "source_time", "retrieved_at", "source_citation"):
        if not _clean(row.get(key)):
            failures.append(f"missing_{key}")
    for key in ("claim_time", "source_time", "retrieved_at"):
        if _clean(row.get(key)) and _parse_temporal_value(row.get(key)) is None:
            failures.append(f"invalid_{key}")
    return _audit_row(row, sidecar_key="temporal_bindings", line_no=line_no, failures=failures)


def _audit_mechanism_binding(row: Mapping[str, Any], *, line_no: int, duplicate: bool) -> dict[str, Any]:
    failures = list(_base_failures(row, duplicate=duplicate, duplicate_reason="duplicate_mechanism_binding"))
    for key in ("mechanism", "precondition", "mechanism_status", "source_citation"):
        if not _clean(row.get(key)):
            failures.append(f"missing_{key}")
    if _clean(row.get("mechanism_status")) and _normalize_mechanism_status(row.get("mechanism_status")) is None:
        failures.append("invalid_mechanism_status")
    return _audit_row(row, sidecar_key="mechanism_bindings", line_no=line_no, failures=failures)


def _audit_entity_binding(row: Mapping[str, Any], *, line_no: int, duplicate: bool) -> dict[str, Any]:
    failures = list(_base_failures(row, duplicate=duplicate, duplicate_reason="duplicate_entity_binding"))
    for key in ("subject_entity", "answer_entity", "expected_entity", "requested_role", "source_citation"):
        if not _clean(row.get(key)):
            failures.append(f"missing_{key}")
    return _audit_row(row, sidecar_key="entity_bindings", line_no=line_no, failures=failures)


def _audit_source_family_url_seed(
    row: Mapping[str, Any],
    *,
    line_no: int,
    duplicate: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    if duplicate:
        failures.append("duplicate_source_family_url_seed")
    task_id = _clean(row.get("task_id") or row.get("collection_task_id"))
    url = _clean(row.get("url") or row.get("href"))
    if not task_id:
        failures.append("missing_task_id")
    if not url:
        failures.append("missing_url")
    elif not url.startswith(("http://", "https://")):
        failures.append("invalid_url")
    if row.get("not_verifier_evidence") is not True:
        failures.append("seed_not_marked_non_evidence")
    review_status = _clean(row.get("review_status")).lower()
    if review_status not in READY_REVIEW_STATUSES:
        failures.append("binding_requires_review")
    reserved = tuple(sorted(key for key in row if str(key) in RESERVED_FIELDS))
    if reserved:
        failures.append("reserved_fields_present")
    return _audit_row(
        row,
        sidecar_key="source_family_url_seeds",
        line_no=line_no,
        failures=failures,
        extra={
            "request_id": task_id,
            "target_id": task_id,
            "url": url,
            "reserved_fields": reserved,
        },
    )


def _base_failures(row: Mapping[str, Any], *, duplicate: bool, duplicate_reason: str) -> tuple[str, ...]:
    failures: list[str] = []
    if duplicate:
        failures.append(duplicate_reason)
    if not _request_id(row):
        failures.append("missing_request_id")
    if not _clean(row.get("target_id")):
        failures.append("missing_target_id")
    if row.get("not_verifier_evidence") is not True:
        failures.append("binding_not_marked_non_evidence")
    review_status = _clean(row.get("review_status")).lower()
    if review_status not in READY_REVIEW_STATUSES:
        failures.append("binding_requires_review")
    reserved = tuple(sorted(key for key in row if str(key) in RESERVED_FIELDS))
    if reserved:
        failures.append("reserved_fields_present")
    return tuple(failures)


def _audit_row(
    row: Mapping[str, Any],
    *,
    sidecar_key: str,
    line_no: int,
    failures: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    deduped_failures = tuple(dict.fromkeys(str(failure) for failure in failures if str(failure)))
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "sidecar_key": sidecar_key,
        "line_no": line_no,
        "binding_id": str(row.get("binding_id") or ""),
        "request_id": _request_id(row),
        "target_id": _clean(row.get("target_id")),
        "collection_request_id": _clean(row.get("collection_request_id")),
        "review_status": _clean(row.get("review_status")),
        "status": "ready" if not deduped_failures else "blocked",
        "failures": deduped_failures,
        "reserved_fields": tuple(sorted(key for key in row if str(key) in RESERVED_FIELDS)),
        "not_verifier_evidence": row.get("not_verifier_evidence") is True,
    }
    payload.update(dict(extra or {}))
    return payload


def _missing_sidecar_row(sidecar_key: str, path: Path | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "sidecar_key": sidecar_key,
        "line_no": 0,
        "binding_id": "",
        "request_id": "",
        "target_id": "",
        "collection_request_id": "",
        "review_status": "",
        "status": "blocked",
        "failures": ("missing_sidecar_file",),
        "reserved_fields": (),
        "not_verifier_evidence": False,
        "path": None if path is None else str(path),
    }


def _summary(
    *,
    audit_rows: Sequence[Mapping[str, Any]],
    loaded_sidecars: Mapping[str, Sequence[Mapping[str, Any]]],
    missing_sidecar_keys: Sequence[str],
    downstream_commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    sidecar_status_counts: dict[str, dict[str, int]] = {}
    ready_by_sidecar: Counter[str] = Counter()
    blocked_by_sidecar: Counter[str] = Counter()
    pending_review_count = 0
    reserved_field_count = 0
    for row in audit_rows:
        sidecar_key = str(row.get("sidecar_key") or "")
        status = str(row.get("status") or "")
        sidecar_status_counts.setdefault(sidecar_key, {"ready": 0, "blocked": 0})
        if status == "ready":
            sidecar_status_counts[sidecar_key]["ready"] += 1
            ready_by_sidecar[sidecar_key] += 1
        else:
            sidecar_status_counts[sidecar_key]["blocked"] += 1
            blocked_by_sidecar[sidecar_key] += 1
        failures = tuple(str(item) for item in _sequence(row.get("failures", ())))
        failure_counts.update(failures)
        if "binding_requires_review" in failures:
            pending_review_count += 1
        reserved_field_count += len(_sequence(row.get("reserved_fields", ())))
    sidecar_counts = {key: len(tuple(rows)) for key, rows in sorted(loaded_sidecars.items())}
    return {
        "binding_row_count": sum(sidecar_counts.values()),
        "audit_row_count": len(audit_rows),
        "ready_binding_count": sum(ready_by_sidecar.values()),
        "blocked_binding_count": sum(blocked_by_sidecar.values()),
        "pending_review_count": pending_review_count,
        "reserved_field_count": reserved_field_count,
        "missing_sidecar_count": len(tuple(missing_sidecar_keys)),
        "sidecar_counts": sidecar_counts,
        "sidecar_status_counts": sidecar_status_counts,
        "failure_counts": _sorted_counter(failure_counts),
        "ready_by_sidecar": _sorted_counter(ready_by_sidecar),
        "blocked_by_sidecar": _sorted_counter(blocked_by_sidecar),
        "downstream_command_count": len(downstream_commands),
        "downstream_ready_command_count": sum(
            1 for item in downstream_commands if item.get("ready_for_fill") is True
        ),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("blocked_binding_count", 0)) > 0 or int(summary.get("missing_sidecar_count", 0)) > 0:
        return "blocked"
    if int(summary.get("ready_binding_count", 0)) > 0:
        return "ready"
    return "empty"


def _downstream_command_statuses(
    scaffold: Mapping[str, Any],
    audit_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows_by_sidecar: dict[str, list[Mapping[str, Any]]] = {}
    for row in audit_rows:
        rows_by_sidecar.setdefault(str(row.get("sidecar_key") or ""), []).append(row)
    statuses = []
    for command in _mapping_sequence(scaffold.get("downstream_commands", ())):
        sidecar_key = str(command.get("sidecar_key") or "")
        rows = rows_by_sidecar.get(sidecar_key, [])
        blocked = sum(1 for row in rows if row.get("status") != "ready")
        ready = sum(1 for row in rows if row.get("status") == "ready")
        statuses.append({
            "request_id": str(command.get("request_id") or ""),
            "input_name": str(command.get("input_name") or ""),
            "sidecar_key": sidecar_key,
            "command": str(command.get("command") or ""),
            "ready_for_fill": ready > 0 and blocked == 0,
            "ready_binding_count": ready,
            "blocked_binding_count": blocked,
            "executes_commands": False,
        })
    return tuple(statuses)


def _sidecar_file_statuses(
    sidecar_paths: Mapping[str, Path | None],
    loaded_sidecars: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "path": None if path is None else str(path),
            "exists": bool(path is not None and path.exists()),
            "row_count": len(tuple(loaded_sidecars.get(key, ()))),
        }
        for key, path in sorted(sidecar_paths.items())
    }


def _sidecar_paths(scaffold: Mapping[str, Any], *, scaffold_root: Path | None) -> dict[str, Path | None]:
    paths = _mapping(scaffold.get("paths"))
    resolved = {}
    for key, spec in SIDECAR_SPECS.items():
        raw = paths.get(str(spec["path_key"]))
        if str(spec["path_key"]) not in paths:
            continue
        resolved[key] = None if not raw else _resolve_path(str(raw), source_root=scaffold_root)
    return resolved


def _load_mapping_source(source: str | Path | Mapping[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    if isinstance(source, Mapping):
        return None, dict(source)
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return path, dict(payload)


def _load_sidecar_rows(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append({str(key): value for key, value in row.items()})
    return tuple(rows)


def _request_id(row: Mapping[str, Any]) -> str:
    return _clean(row.get("request_id") or row.get("source_request_id"))


def _request_id_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(_request_id(row) for row in rows if _request_id(row))


def _source_family_url_seed_key(row: Mapping[str, Any]) -> str:
    task_id = _clean(row.get("task_id") or row.get("collection_task_id"))
    url = _clean(row.get("url") or row.get("href"))
    if not task_id or not url:
        return ""
    return f"{task_id}\n{url}"


def _source_family_url_seed_key_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(_source_family_url_seed_key(row) for row in rows if _source_family_url_seed_key(row))


def _normalize_mechanism_status(value: Any) -> str | None:
    return MECHANISM_STATUS_ALIASES.get(_clean(value).lower())


def _parse_temporal_value(value: Any) -> datetime | None:
    text = _clean(value)
    if not text:
        return None
    if re.fullmatch(r"\d{4}", text):
        return datetime(int(text), 1, 1, tzinfo=timezone.utc)
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = text.split("-")
        return datetime(int(year), int(month), 1, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            return None
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _resolve_path(path: str | Path, *, source_root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if source_root is not None and (source_root / candidate).exists():
        return source_root / candidate
    return ROOT / candidate


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if compact
        else strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (value,)
    return tuple(value)


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata value must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        metadata[key] = item
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-binding-scaffold", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--binding-audit-rows-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = audit_frontier_research_queue_input_bindings(
        input_binding_scaffold=args.input_binding_scaffold,
        output_dir=args.output_dir,
        json_path=args.json,
        binding_audit_rows_path=args.binding_audit_rows_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata),
        compact_json=args.compact_json,
    )
    print(strict_json_dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
