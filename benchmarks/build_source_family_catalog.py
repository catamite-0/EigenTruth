"""Normalize local source documents into a source-family adapter catalog.

The source-family citation/search adapter expects source documents with
top-level provenance fields such as ``provider``, ``url``, ``source_family``,
and timestamp fields. Earlier evidence collection steps often store those
fields inside ``metadata``. This helper lifts the safe provenance fields into a
catalog JSONL while rejecting reserved label/model-answer fields before the
catalog can be used by any citation/search workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify.search_planning import SOURCE_FAMILY_NAMES  # noqa: E402

WORKFLOW = "source_family_catalog_builder"
DEFAULT_SOURCE_FAMILY = "reference"
DEFAULT_PROVIDER = "local_source_catalog"
TEXT_FIELDS = ("text", "content", "document", "body", "snippet", "summary", "abstract")
RESERVED_SOURCE_FIELDS = {
    "answer",
    "claim_id",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "record_index",
    "row_index",
    "score_label",
    "source_index",
    "target_id",
}


def build_source_family_catalog(
    source_paths: Sequence[str | Path],
    *,
    output_path: str | Path,
    report_json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    default_source_family: str = DEFAULT_SOURCE_FAMILY,
    provider_source_families: Mapping[str, str] | None = None,
    default_provider: str = DEFAULT_PROVIDER,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an adapter-ready source-family catalog from local source docs."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if not source_paths:
        raise ValueError("source_paths must contain at least one path.")
    family_default = _normalize_source_family(default_source_family)
    family_by_provider = {
        _clean(provider).casefold(): _normalize_source_family(family)
        for provider, family in dict(provider_source_families or {}).items()
    }
    rows: list[dict[str, Any]] = []
    skipped_empty = 0
    input_count = 0
    for source_path in tuple(Path(path) for path in source_paths):
        for index, item in enumerate(_load_source_documents(source_path), start=1):
            input_count += 1
            row = _catalog_row(
                item,
                source_default=f"{source_path}:{index}",
                default_provider=default_provider,
                default_source_family=family_default,
                provider_source_families=family_by_provider,
            )
            if row is None:
                skipped_empty += 1
                continue
            rows.append(row)
    if not rows:
        raise ValueError("no source-family catalog documents were produced.")
    _write_jsonl(output_path, rows, compact=compact_json)
    summary = _summary(rows, input_count=input_count, skipped_empty_count=skipped_empty)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "input_paths": tuple(str(path) for path in source_paths),
        "output_path": str(output_path),
        "config": {
            "default_provider": default_provider,
            "default_source_family": family_default,
            "provider_source_families": dict(sorted(family_by_provider.items())),
        },
        "summary": summary,
        "metadata": dict(metadata or {}),
    }
    if report_json_path is not None:
        _write_json(report_json_path, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        artifacts: dict[str, str | Path] = {
            "source_family_catalog": Path(output_path),
        }
        if report_json_path is not None:
            artifacts["catalog_builder_report"] = Path(report_json_path)
        for index, path in enumerate(source_paths, start=1):
            artifacts[f"source_input_{index}"] = Path(path)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": WORKFLOW,
                "input_document_count": summary["input_document_count"],
                "catalog_document_count": summary["catalog_document_count"],
                "skipped_empty_count": summary["skipped_empty_count"],
                **dict(metadata or {}),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)
        payload["artifact_manifest"] = str(manifest_path)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_json_path or output_path,
            metadata={
                "workflow": WORKFLOW,
                "input_document_count": summary["input_document_count"],
                "catalog_document_count": summary["catalog_document_count"],
                "skipped_empty_count": summary["skipped_empty_count"],
                "artifact_manifest": payload.get("artifact_manifest"),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _catalog_row(
    item: Mapping[str, Any],
    *,
    source_default: str,
    default_provider: str,
    default_source_family: str,
    provider_source_families: Mapping[str, str],
) -> dict[str, Any] | None:
    _reject_reserved_fields(item, source=source_default)
    metadata = dict(_mapping(item.get("metadata")))
    _reject_reserved_fields(metadata, source=source_default)
    text = _first_nonempty(item, TEXT_FIELDS)
    if not text:
        return None
    provider = _clean(item.get("provider") or metadata.get("provider")) or default_provider
    source_family = _normalize_source_family(
        item.get("source_family")
        or item.get("source_family_name")
        or metadata.get("source_family")
        or provider_source_families.get(provider.casefold())
        or default_source_family
    )
    source = _clean(item.get("source") or metadata.get("source")) or source_default
    url = _clean(item.get("url") or item.get("href") or metadata.get("url") or metadata.get("href"))
    timestamp = _clean(
        item.get("timestamp")
        or item.get("retrieved_at")
        or metadata.get("timestamp")
        or metadata.get("retrieved_at")
    )
    published_at = _clean(
        item.get("published_at")
        or item.get("publication_date")
        or metadata.get("published_at")
        or metadata.get("publication_date")
    )
    title = _clean(item.get("title") or metadata.get("title")) or _title_from_metadata(metadata, fallback=source)
    catalog_metadata = {
        **metadata,
        "source_document_sha256": _sha256_json(item),
    }
    return {
        "text": _clean(text),
        "title": title,
        "source": source,
        "url": url,
        "provider": provider,
        "source_family": source_family,
        "published_at": published_at,
        "timestamp": timestamp,
        "metadata": _drop_empty(catalog_metadata),
    }


def _load_source_documents(path: Path) -> tuple[Mapping[str, Any], ...]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{path}:{line_no} must contain a JSON object.")
                rows.append(payload)
        return tuple(rows)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_rows = (
            payload.get("documents")
            or payload.get("source_documents")
            or payload.get("records")
            or payload.get("results")
            or ()
        )
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        raw_rows = payload
    else:
        raise ValueError(f"{path} must contain a JSON object or list.")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes, bytearray)):
        raise ValueError(f"{path} does not expose a source document list.")
    return tuple(item for item in raw_rows if isinstance(item, Mapping))


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_count: int,
    skipped_empty_count: int,
) -> dict[str, Any]:
    providers = Counter(str(row.get("provider")) for row in rows if row.get("provider"))
    families = Counter(str(row.get("source_family")) for row in rows if row.get("source_family"))
    timestamped = sum(1 for row in rows if row.get("timestamp") or row.get("published_at"))
    return {
        "input_document_count": int(input_count),
        "catalog_document_count": len(rows),
        "skipped_empty_count": int(skipped_empty_count),
        "timestamped_document_count": timestamped,
        "provider_counts": _sorted_counter(providers),
        "source_family_counts": _sorted_counter(families),
    }


def _reject_reserved_fields(payload: Mapping[str, Any], *, source: str) -> None:
    reserved = sorted(set(str(key) for key in payload) & RESERVED_SOURCE_FIELDS)
    if reserved:
        raise ValueError(f"source document {source!r} contains reserved fields: {', '.join(reserved)}")


def _title_from_metadata(metadata: Mapping[str, Any], *, fallback: str) -> str:
    subject = _clean(metadata.get("subject"))
    prop = _clean(metadata.get("statement_property_label") or metadata.get("property_label"))
    value = _clean(metadata.get("value"))
    if subject and prop:
        return f"{subject} - {prop}"
    if subject:
        return subject
    if value:
        return value
    return fallback


def _first_nonempty(payload: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = _clean(payload.get(field))
        if value:
            return value
    return ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _drop_empty(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items() if value not in (None, "")}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _normalize_source_family(value: Any) -> str:
    family = _clean(value).casefold().replace("-", "_")
    if not family:
        family = DEFAULT_SOURCE_FAMILY
    if family not in SOURCE_FAMILY_NAMES:
        choices = ", ".join(SOURCE_FAMILY_NAMES)
        raise ValueError(f"source_family must be one of: {choices}.")
    return family


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def _parse_provider_source_families(values: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"provider source family must be PROVIDER=FAMILY, got {value!r}.")
        provider, family = value.split("=", 1)
        provider = _clean(provider)
        if not provider:
            raise ValueError("provider source family key cannot be empty.")
        mapping[provider] = _normalize_source_family(family)
    return mapping


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="source JSON/JSONL path; repeatable")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--default-source-family", default=DEFAULT_SOURCE_FAMILY)
    parser.add_argument("--default-provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--provider-source-family", action="append", default=[],
                        help="map provider to source family, e.g. wikidata=reference; repeatable")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_source_family_catalog(
        tuple(args.source or ()),
        output_path=args.output,
        report_json_path=args.report_json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        default_source_family=args.default_source_family,
        provider_source_families=_parse_provider_source_families(args.provider_source_family or ()),
        default_provider=args.default_provider,
        compact_json=bool(args.compact_json),
        metadata=_parse_metadata(args.metadata or ()),
    )
    summary = payload["summary"]
    print(
        "source_family_catalog_builder_ok "
        f"input_docs={summary['input_document_count']} "
        f"catalog_docs={summary['catalog_document_count']} "
        f"families={summary['source_family_counts']}"
    )


if __name__ == "__main__":
    main()
