"""Build a replay-ready ProductTrace corpus from saved trace files.

This workflow validates and optionally redacts already-emitted ProductTrace JSON
payloads. It does not run models, verifiers, retrievers, or external services.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config_utils import (  # noqa: E402
    bounded_product_trace_reason,
    planned_artifact_manifest_summary,
    strict_bool,
)
from eigentruth.control import RUNTIME_PROFILE_NAMES  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest, fingerprint_path  # noqa: E402

_TEXT_KEYS = frozenset({
    "answer",
    "claim",
    "content",
    "evidence",
    "explanation",
    "generated_text",
    "input",
    "key",
    "matched_text",
    "output",
    "prompt",
    "question",
    "raw",
    "raw_output",
    "response",
    "retrieval_document",
    "retrieval_documents",
    "source_text",
    "statement",
    "text",
})


@dataclass(frozen=True)
class ProductTraceCorpusConfig:
    """Configuration for a replay-ready ProductTrace corpus."""

    trace_paths: Sequence[str | Path] = ()
    jsonl_paths: Sequence[str | Path] = ()
    output_dir: str | Path = "artifacts/product_trace_corpus"
    report_path: str | Path | None = None
    traces_dir: str | Path | None = None
    runtime_pair_index_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    registry_path: str | Path | None = None
    name: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_cache_path: str | Path | None = None
    refresh_source_cache: bool = False
    redact_text: bool = True
    require_runtime_trace: bool = False
    strict: bool = False
    limit: int | None = None
    compact_json: bool = False

    def __post_init__(self) -> None:
        trace_paths = tuple(Path(path) for path in self.trace_paths)
        jsonl_paths = tuple(Path(path) for path in self.jsonl_paths)
        if not trace_paths and not jsonl_paths:
            raise ValueError("at least one ProductTrace JSON or JSONL path is required.")
        if self.registry_path is not None and (not self.name or not self.version):
            raise ValueError("registry_path requires name and version.")
        limit = None if self.limit is None else int(self.limit)
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided.")
        object.__setattr__(self, "trace_paths", trace_paths)
        object.__setattr__(self, "jsonl_paths", jsonl_paths)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.report_path is not None:
            object.__setattr__(self, "report_path", Path(self.report_path))
        if self.traces_dir is not None:
            object.__setattr__(self, "traces_dir", Path(self.traces_dir))
        if self.runtime_pair_index_path is not None:
            object.__setattr__(self, "runtime_pair_index_path", Path(self.runtime_pair_index_path))
        if self.artifact_manifest_path is not None:
            object.__setattr__(self, "artifact_manifest_path", Path(self.artifact_manifest_path))
        if self.registry_path is not None:
            object.__setattr__(self, "registry_path", Path(self.registry_path))
        if self.source_cache_path is not None:
            object.__setattr__(self, "source_cache_path", Path(self.source_cache_path))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "refresh_source_cache",
            strict_bool(self.refresh_source_cache, name="refresh_source_cache"),
        )
        object.__setattr__(self, "redact_text", strict_bool(self.redact_text, name="redact_text"))
        object.__setattr__(
            self,
            "require_runtime_trace",
            strict_bool(self.require_runtime_trace, name="require_runtime_trace"),
        )
        object.__setattr__(self, "strict", strict_bool(self.strict, name="strict"))
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "compact_json", strict_bool(self.compact_json, name="compact_json"))

    @property
    def resolved_report_path(self) -> Path:
        """Return the corpus report path."""
        if self.report_path is not None:
            return Path(self.report_path)
        return Path(self.output_dir) / "product-trace-corpus.json"

    @property
    def resolved_traces_dir(self) -> Path:
        """Return the standardized trace output directory."""
        if self.traces_dir is not None:
            return Path(self.traces_dir)
        return Path(self.output_dir) / "traces"

    @property
    def resolved_artifact_manifest_path(self) -> Path:
        """Return the artifact manifest path."""
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_dir) / "artifact-manifest.json"

    @property
    def resolved_runtime_pair_index_path(self) -> Path:
        """Return the runtime pairing index artifact path."""
        if self.runtime_pair_index_path is not None:
            return Path(self.runtime_pair_index_path)
        return Path(self.output_dir) / "runtime-pair-index.json"


def build_product_trace_corpus(config: ProductTraceCorpusConfig) -> dict[str, Any]:
    """Validate, redact, and standardize saved ProductTrace payloads."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.resolved_traces_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_names: set[str] = set()
    source_cache = _load_source_cache(config)
    source_cache_files: dict[tuple[str, str], dict[str, Any]] = {}
    source_cache_stats = _source_cache_stats(config, loaded=source_cache is not None)

    for item in _iter_source_items(config, source_cache=source_cache):
        source = dict(item["source"])
        source_fingerprint = _mapping(item.get("source_fingerprint"))
        if item.get("cached") is True:
            source_cache_stats["hit_count"] += 1
            result = _mapping(item.get("result"))
            result_status = result.get("status")
            if result_status == "rejected":
                reason = str(result.get("reason") or "invalid cached ProductTrace")
                rejected.append(_rejected_record(source, reason=reason))
                if config.strict:
                    raise ValueError(f"invalid ProductTrace from {source['source_path']}: {reason}")
                _add_source_cache_result(
                    source_cache_files,
                    item,
                    status="rejected",
                    reason=reason,
                )
                continue
            trace = dict(_mapping(result.get("trace")))
        else:
            source_cache_stats["miss_count"] += 1
            payload = source["payload"]
            reason = _invalid_reason(payload, require_runtime_trace=config.require_runtime_trace)
            if reason is not None:
                rejected.append(_rejected_record(source, reason=reason))
                _add_source_cache_result(
                    source_cache_files,
                    item,
                    status="rejected",
                    reason=reason,
                )
                if config.strict:
                    raise ValueError(f"invalid ProductTrace from {source['source_path']}: {reason}")
                continue
            trace = _standardized_trace(source, redact_text=config.redact_text)
        output_path = _trace_output_path(
            config.resolved_traces_dir,
            trace,
            source=source,
            used_names=used_names,
        )
        _write_json(output_path, trace, compact=config.compact_json)
        accepted.append(_accepted_record(source, trace, output_path=output_path))
        _add_source_cache_result(
            source_cache_files,
            {
                **item,
                "source": source,
                "source_fingerprint": source_fingerprint,
            },
            status="accepted",
            trace=trace,
        )
        if config.limit is not None and len(accepted) >= config.limit:
            break

    _finalize_source_cache_stats(source_cache_stats)
    if _source_cache_enabled(config):
        _write_json(
            config.source_cache_path,
            _source_cache_payload(config, source_cache_files, source_cache_stats),
            compact=True,
        )
        source_cache_stats["cache_written"] = True

    status = _corpus_status(accepted, rejected)
    runtime_pair_index = _runtime_pair_index_payload(config, accepted)
    _write_json(
        config.resolved_runtime_pair_index_path,
        runtime_pair_index,
        compact=config.compact_json,
    )
    report = {
        "schema_version": 1,
        "workflow": "product_trace_corpus",
        "status": status,
        "decision": {
            "status": status,
            "blocking_reasons": () if accepted else ("no valid ProductTrace payloads",),
        },
        "summary": _corpus_summary(accepted, rejected),
        "runtime_pair_index": runtime_pair_index["summary"],
        "traces": accepted,
        "rejected": rejected,
        "paths": {
            "report": str(config.resolved_report_path),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "traces_dir": str(config.resolved_traces_dir),
            "runtime_pair_index": str(config.resolved_runtime_pair_index_path),
            "source_cache": None if config.source_cache_path is None else str(config.source_cache_path),
            "inputs": [str(path) for path in config.trace_paths],
            "jsonl_inputs": [str(path) for path in config.jsonl_paths],
        },
        "source_cache": source_cache_stats,
        "config": {
            "redact_text": config.redact_text,
            "require_runtime_trace": config.require_runtime_trace,
            "strict": config.strict,
            "limit": config.limit,
            "compact_json": config.compact_json,
            "source_cache": None if config.source_cache_path is None else str(config.source_cache_path),
            "refresh_source_cache": config.refresh_source_cache,
            "metadata": dict(config.metadata),
        },
    }
    _write_report_and_manifest(config, report)
    _record_registry(config, report)
    return report


def _iter_source_items(
    config: ProductTraceCorpusConfig,
    *,
    source_cache: Mapping[str, Any] | None,
) -> Iterator[dict[str, Any]]:
    cached_files = _source_cache_file_lookup(source_cache)
    for path in config.trace_paths:
        yield from _iter_path_source_items(path, kind="trace", cached_files=cached_files)
    for path in config.jsonl_paths:
        yield from _iter_path_source_items(path, kind="jsonl", cached_files=cached_files)


def _iter_path_source_items(
    path: Path,
    *,
    kind: str,
    cached_files: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    source_fingerprint = fingerprint_path(path).to_dict()
    cached_file = cached_files.get((kind, str(path)))
    if cached_file is not None and _fingerprint_matches(
        _mapping(cached_file.get("fingerprint")),
        source_fingerprint,
    ):
        for result in _sequence(cached_file.get("entries")):
            if not isinstance(result, Mapping):
                continue
            source = _mapping(result.get("source"))
            if not source:
                continue
            yield {
                "cached": True,
                "source_kind": kind,
                "source": source,
                "source_fingerprint": source_fingerprint,
                "result": dict(result),
            }
        return
    if kind == "trace":
        payload = _load_json(path)
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray, Mapping)):
            for index, item in enumerate(payload):
                yield _source_item(
                    path,
                    index,
                    item,
                    source_format="json_array",
                    source_kind=kind,
                    source_fingerprint=source_fingerprint,
                )
            return
        yield _source_item(
            path,
            0,
            payload,
            source_format="json",
            source_kind=kind,
            source_fingerprint=source_fingerprint,
        )
        return
    with Path(path).open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            yield _source_item(
                path,
                index,
                json.loads(line),
                source_format="jsonl",
                source_kind=kind,
                source_fingerprint=source_fingerprint,
            )


def _source_record(path: str | Path, index: int, payload: Any, *, source_format: str) -> dict[str, Any]:
    return {
        "source_path": str(path),
        "source_index": index,
        "source_format": source_format,
        "payload": payload,
    }


def _source_item(
    path: str | Path,
    index: int,
    payload: Any,
    *,
    source_format: str,
    source_kind: str,
    source_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cached": False,
        "source_kind": source_kind,
        "source_fingerprint": dict(source_fingerprint),
        "source": _source_record(path, index, payload, source_format=source_format),
    }


def _source_cache_enabled(config: ProductTraceCorpusConfig) -> bool:
    return config.source_cache_path is not None and config.limit is None


def _source_cache_stats(config: ProductTraceCorpusConfig, *, loaded: bool) -> dict[str, Any]:
    disabled_reason = None
    if config.source_cache_path is not None and config.limit is not None:
        disabled_reason = "limit"
    return {
        "enabled": _source_cache_enabled(config),
        "source": "source_cache" if loaded else "source_scan",
        "path": None if config.source_cache_path is None else str(config.source_cache_path),
        "cache_loaded": loaded,
        "cache_hit": False,
        "cache_partial_hit": False,
        "cache_written": False,
        "hit_count": 0,
        "miss_count": 0,
        "refresh": config.refresh_source_cache,
        "disabled_reason": disabled_reason,
    }


def _finalize_source_cache_stats(stats: dict[str, Any]) -> None:
    hit_count = int(stats.get("hit_count") or 0)
    miss_count = int(stats.get("miss_count") or 0)
    if hit_count and miss_count:
        stats["source"] = "source_cache_mixed"
    elif hit_count:
        stats["source"] = "source_cache"
    elif stats.get("enabled"):
        stats["source"] = "source_scan"
    stats["cache_hit"] = bool(hit_count and not miss_count)
    stats["cache_partial_hit"] = bool(hit_count and miss_count)


def _load_source_cache(config: ProductTraceCorpusConfig) -> dict[str, Any] | None:
    if (
        not _source_cache_enabled(config)
        or config.refresh_source_cache
        or config.source_cache_path is None
        or not Path(config.source_cache_path).exists()
    ):
        return None
    try:
        payload = json.loads(Path(config.source_cache_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != 1:
        return None
    if payload.get("workflow") != "product_trace_corpus_source_cache":
        return None
    if _mapping(payload.get("config")).get("signature") != _source_cache_signature(config):
        return None
    return dict(payload)


def _source_cache_file_lookup(
    payload: Mapping[str, Any] | None,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if not payload:
        return {}
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in _sequence(payload.get("files")):
        if not isinstance(item, Mapping):
            continue
        kind = item.get("kind")
        path = item.get("path")
        if kind is None or path is None:
            continue
        lookup[(str(kind), str(path))] = item
    return lookup


def _add_source_cache_result(
    files: dict[tuple[str, str], dict[str, Any]],
    item: Mapping[str, Any],
    *,
    status: str,
    trace: Mapping[str, Any] | None = None,
    reason: str | None = None,
) -> None:
    source = _source_cache_source(_mapping(item.get("source")))
    source_kind = str(item.get("source_kind") or "trace")
    source_path = str(source.get("source_path"))
    key = (source_kind, source_path)
    file_record = files.setdefault(
        key,
        {
            "kind": source_kind,
            "path": source_path,
            "fingerprint": dict(_mapping(item.get("source_fingerprint"))),
            "entries": [],
        },
    )
    entry: dict[str, Any] = {
        "source": source,
        "status": status,
    }
    if status == "accepted":
        entry["trace"] = _jsonable(dict(trace or {}))
    else:
        entry["reason"] = str(reason)
    file_record["entries"].append(entry)


def _source_cache_source(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_path": str(source.get("source_path")),
        "source_index": source.get("source_index"),
        "source_format": source.get("source_format"),
    }


def _source_cache_payload(
    config: ProductTraceCorpusConfig,
    files: Mapping[tuple[str, str], Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    file_records = tuple(dict(record) for record in files.values())
    entries = [
        entry
        for record in file_records
        for entry in _sequence(record.get("entries"))
        if isinstance(entry, Mapping)
    ]
    return {
        "schema_version": 1,
        "workflow": "product_trace_corpus_source_cache",
        "config": {
            "signature": _source_cache_signature(config),
            "payload": _source_cache_config_payload(config),
        },
        "summary": {
            "file_count": len(file_records),
            "source_count": len(entries),
            "accepted_count": sum(1 for entry in entries if entry.get("status") == "accepted"),
            "rejected_count": sum(1 for entry in entries if entry.get("status") == "rejected"),
            "hit_count": stats.get("hit_count"),
            "miss_count": stats.get("miss_count"),
        },
        "files": file_records,
    }


def _source_cache_signature(config: ProductTraceCorpusConfig) -> str:
    return json.dumps(
        _source_cache_config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
    )


def _source_cache_config_payload(config: ProductTraceCorpusConfig) -> dict[str, Any]:
    return {
        "redact_text": config.redact_text,
        "require_runtime_trace": config.require_runtime_trace,
        "strict": config.strict,
    }


def _fingerprint_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return all(
        expected.get(field_name) == actual.get(field_name)
        for field_name in ("exists", "kind", "sha256", "size_bytes", "file_count")
    )


def _invalid_reason(payload: Any, *, require_runtime_trace: bool) -> str | None:
    if not isinstance(payload, Mapping):
        return "payload is not a JSON object"
    if (reason := bounded_product_trace_reason(payload)) is not None:
        return reason
    risk_decision = payload.get("risk_decision")
    if not isinstance(risk_decision, Mapping):
        return "missing risk_decision object"
    if risk_decision.get("risk_level") is None:
        return "missing risk_decision.risk_level"
    if risk_decision.get("action") is None:
        return "missing risk_decision.action"
    runtime_trace = payload.get("runtime_trace")
    if require_runtime_trace and not isinstance(runtime_trace, Mapping):
        return "missing runtime_trace object"
    if isinstance(runtime_trace, Mapping) and _float_or_none(runtime_trace.get("total_seconds")) is None:
        return "runtime_trace.total_seconds is missing or non-finite"
    return None


def _standardized_trace(source: Mapping[str, Any], *, redact_text: bool) -> dict[str, Any]:
    payload = dict(source["payload"])
    trace = _redact_payload(payload) if redact_text else _jsonable(payload)
    if not isinstance(trace, Mapping):
        raise ValueError("standardized trace must be an object")
    trace = dict(trace)
    metadata = dict(_mapping(trace.get("metadata")))
    metadata.setdefault("runtime_replay_key", _trace_request_key(trace))
    metadata["trace_corpus"] = {
        "source_path": source["source_path"],
        "source_index": source["source_index"],
        "source_format": source["source_format"],
        "redacted_text": redact_text,
    }
    trace["metadata"] = metadata
    return trace


def _accepted_record(source: Mapping[str, Any], trace: Mapping[str, Any], *, output_path: Path) -> dict[str, Any]:
    risk_decision = _mapping(trace.get("risk_decision"))
    runtime_trace = _mapping(trace.get("runtime_trace"))
    metadata = _mapping(trace.get("metadata"))
    return {
        "source_path": source["source_path"],
        "source_index": source["source_index"],
        "source_format": source["source_format"],
        "path": str(output_path),
        "request_id": trace.get("request_id"),
        "request_key": metadata.get("runtime_replay_key"),
        "runtime_profile": _trace_runtime_profile(trace),
        "risk_level": risk_decision.get("risk_level"),
        "action": risk_decision.get("action"),
        "claim_count": len(_sequence(trace.get("claims"))),
        "has_runtime_trace": bool(runtime_trace),
        "total_seconds": _float_or_none(runtime_trace.get("total_seconds")),
        "redacted_text": bool(_nested(trace, "metadata", "trace_corpus", "redacted_text")),
    }


def _rejected_record(source: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "source_path": source["source_path"],
        "source_index": source["source_index"],
        "source_format": source["source_format"],
        "reason": reason,
    }


def _corpus_summary(accepted: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "input_count": len(accepted) + len(rejected),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "runtime_trace_count": sum(1 for record in accepted if bool(record.get("has_runtime_trace"))),
        "redacted_trace_count": sum(1 for record in accepted if bool(record.get("redacted_text"))),
        "unique_request_key_count": len({
            str(record.get("request_key"))
            for record in accepted
            if record.get("request_key") is not None
        }),
        "counts_by_runtime_profile": _counts(record.get("runtime_profile") for record in accepted),
        "counts_by_risk_level": _counts(record.get("risk_level") for record in accepted),
        "counts_by_action": _counts(record.get("action") for record in accepted),
        "rejected_reasons": _counts(record.get("reason") for record in rejected),
    }


def _runtime_pair_index_payload(
    config: ProductTraceCorpusConfig,
    accepted: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = []
    for record in accepted:
        request_key = record.get("request_key")
        runtime_profile = record.get("runtime_profile")
        if request_key is None or runtime_profile is None:
            continue
        records.append({
            "request_key": str(request_key),
            "runtime_profile": str(runtime_profile),
            "path": record.get("path"),
            "total_seconds": _float_or_none(record.get("total_seconds")),
        })
    return {
        "schema_version": 1,
        "workflow": "product_trace_runtime_pair_index",
        "summary": {
            "record_count": len(records),
            "request_key_count": len({record["request_key"] for record in records}),
            "profile_counts": _counts(record["runtime_profile"] for record in records),
        },
        "records": records,
        "paths": {
            "corpus_report": str(config.resolved_report_path),
            "traces_dir": str(config.resolved_traces_dir),
        },
        "config": {
            "redact_text": config.redact_text,
            "require_runtime_trace": config.require_runtime_trace,
            "limit": config.limit,
            "compact_json": config.compact_json,
            "metadata": dict(config.metadata),
        },
    }


def _corpus_status(accepted: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> str:
    if not accepted:
        return "blocked"
    if rejected:
        return "partial"
    return "ready"


def _write_report_and_manifest(
    config: ProductTraceCorpusConfig,
    report: dict[str, Any],
) -> dict[str, Any]:
    artifacts = _artifact_paths(config)
    report["artifact_manifest_summary"] = planned_artifact_manifest_summary(
        artifacts,
        assume_file_paths=(config.resolved_report_path,),
    )
    _write_json(config.resolved_report_path, report, compact=config.compact_json)
    return _write_artifact_manifest(config, report, artifacts=artifacts)


def _artifact_paths(config: ProductTraceCorpusConfig) -> dict[str, str | Path | None]:
    artifacts: dict[str, str | Path | None] = {
        "product_trace_corpus_report": config.resolved_report_path,
        "product_trace_corpus_traces": config.resolved_traces_dir,
        "product_trace_runtime_pair_index": config.resolved_runtime_pair_index_path,
        "product_trace_source_cache": config.source_cache_path,
    }
    for index, path in enumerate(config.trace_paths):
        artifacts[f"input_trace_{index:04d}_{_safe_artifact_name(path.stem)}"] = path
    for index, path in enumerate(config.jsonl_paths):
        artifacts[f"input_jsonl_{index:04d}_{_safe_artifact_name(path.stem)}"] = path
    return artifacts


def _write_artifact_manifest(
    config: ProductTraceCorpusConfig,
    report: Mapping[str, Any],
    *,
    artifacts: Mapping[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        _artifact_paths(config) if artifacts is None else artifacts,
        root=config.resolved_artifact_manifest_path.parent,
        metadata={
            "runner": "build_product_trace_corpus",
            "status": report.get("status"),
            "accepted_count": _nested(report, "summary", "accepted_count"),
            "rejected_count": _nested(report, "summary", "rejected_count"),
            "runtime_trace_count": _nested(report, "summary", "runtime_trace_count"),
            "runtime_pair_index_record_count": _nested(report, "runtime_pair_index", "record_count"),
            "source_cache_path": _nested(report, "paths", "source_cache"),
            "source_cache_source": _nested(report, "source_cache", "source"),
            "source_cache_hit_count": _nested(report, "source_cache", "hit_count"),
            "source_cache_miss_count": _nested(report, "source_cache", "miss_count"),
            "source_cache_written": _nested(report, "source_cache", "cache_written"),
            "redact_text": config.redact_text,
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    )
    _write_json(config.resolved_artifact_manifest_path, manifest, compact=config.compact_json)
    return manifest


def _record_registry(config: ProductTraceCorpusConfig, report: Mapping[str, Any]) -> None:
    if config.registry_path is None:
        return
    ArtifactRegistry.load_json(config.registry_path).record_report(
        name=str(config.name),
        path=config.resolved_report_path,
        version=str(config.version),
        metadata={
            "workflow": "build_product_trace_corpus",
            "status": report.get("status"),
            "artifact_manifest": str(config.resolved_artifact_manifest_path),
            "accepted_count": _nested(report, "summary", "accepted_count"),
            "rejected_count": _nested(report, "summary", "rejected_count"),
            "runtime_trace_count": _nested(report, "summary", "runtime_trace_count"),
            "runtime_pair_index": _nested(report, "paths", "runtime_pair_index"),
            "runtime_pair_index_record_count": _nested(report, "runtime_pair_index", "record_count"),
            "source_cache_path": _nested(report, "paths", "source_cache"),
            "source_cache_source": _nested(report, "source_cache", "source"),
            "source_cache_hit_count": _nested(report, "source_cache", "hit_count"),
            "source_cache_miss_count": _nested(report, "source_cache", "miss_count"),
            "source_cache_written": _nested(report, "source_cache", "cache_written"),
            "redact_text": config.redact_text,
            "compact_json": config.compact_json,
            **dict(config.metadata),
        },
    ).save_json()


def _trace_output_path(
    traces_dir: Path,
    trace: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    used_names: set[str],
) -> Path:
    request_key = str(_nested(trace, "metadata", "runtime_replay_key") or source["source_index"])
    profile = _trace_runtime_profile(trace) or "profile"
    base = _safe_artifact_name(f"{profile}-{request_key}")[:96]
    name = f"{base}.json"
    counter = 2
    while name in used_names:
        name = f"{base}-{counter}.json"
        counter += 1
    used_names.add(name)
    return traces_dir / name


def _trace_runtime_profile(trace: Mapping[str, Any]) -> str | None:
    raw = _nested(trace, "metadata", "runtime_profile")
    if raw is None:
        return None
    profile = str(raw).strip().lower().replace("-", "_")
    return profile if profile in RUNTIME_PROFILE_NAMES else profile


def _trace_request_key(trace: Mapping[str, Any]) -> str:
    metadata_key = _nested(trace, "metadata", "runtime_replay_key")
    if metadata_key is not None and str(metadata_key).strip():
        return str(metadata_key).strip()
    request_id = trace.get("request_id")
    if request_id is not None and str(request_id).strip():
        normalized = str(request_id).strip()
        for prefix in (*RUNTIME_PROFILE_NAMES, "auto"):
            marker = f"{prefix}-"
            if normalized.startswith(marker):
                return normalized[len(marker):]
        return normalized
    return "request"


def _redact_payload(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_payload(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_payload(item, key=key) for item in value]
    if isinstance(value, str) and key is not None and key.lower() in _TEXT_KEYS:
        return _redacted_text(value)
    return _jsonable(value)


def _redacted_text(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"[redacted:sha256={digest}:chars={len(value)}]"


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_text(payload, compact=compact), encoding="utf-8")


def _json_text(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _trace_paths_from_args(values: Sequence[str], globs: Sequence[str]) -> tuple[Path, ...]:
    paths = [Path(value) for value in values]
    for pattern in globs:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern)))
    return _unique_paths(paths)


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def _counts(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned or "artifact"


def _parse_mapping_json(value: str | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return dict(payload)


def _config_from_args(args: argparse.Namespace) -> ProductTraceCorpusConfig:
    trace_paths = _trace_paths_from_args(args.trace or (), args.trace_glob or ())
    jsonl_paths = _unique_paths(tuple(Path(path) for path in args.jsonl or ()))
    return ProductTraceCorpusConfig(
        trace_paths=trace_paths,
        jsonl_paths=jsonl_paths,
        output_dir=Path(args.output_dir),
        report_path=Path(args.json) if args.json else None,
        traces_dir=Path(args.traces_dir) if args.traces_dir else None,
        runtime_pair_index_path=Path(args.runtime_pair_index) if args.runtime_pair_index else None,
        artifact_manifest_path=Path(args.artifact_manifest) if args.artifact_manifest else None,
        registry_path=Path(args.registry) if args.registry else None,
        name=args.name,
        version=args.version,
        metadata=_parse_mapping_json(args.metadata_json, name="--metadata-json"),
        source_cache_path=Path(args.source_cache_json) if args.source_cache_json else None,
        refresh_source_cache=bool(args.refresh_source_cache),
        redact_text=not bool(args.no_redact_text),
        require_runtime_trace=bool(args.require_runtime_trace),
        strict=bool(args.strict),
        limit=args.limit,
        compact_json=bool(args.compact_json),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    report = build_product_trace_corpus(_config_from_args(args))
    print(_json_text(report, compact=bool(args.compact_json)), end="")
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a replay-ready ProductTrace corpus")
    parser.add_argument("--trace", action="append", default=[], help="ProductTrace JSON path; repeatable")
    parser.add_argument("--trace-glob", action="append", default=[], help="glob for ProductTrace JSON files")
    parser.add_argument("--jsonl", action="append", default=[], help="ProductTrace JSONL path; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--traces-dir", default=None)
    parser.add_argument("--runtime-pair-index", default=None,
                        help="runtime pairing index artifact path")
    parser.add_argument("--json", default=None, help="top-level corpus report path")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata-json", default=None)
    parser.add_argument("--source-cache-json", default=None,
                        help="optional per-source cache for validated/redacted trace entries")
    parser.add_argument("--refresh-source-cache", action="store_true",
                        help="rebuild --source-cache-json even when a valid cache exists")
    parser.add_argument("--no-redact-text", action="store_true")
    parser.add_argument("--require-runtime-trace", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
