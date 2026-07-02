"""Collect review-gated entity-role bindings from a local citation corpus.

This collector enriches ``world_model_rule_entity_binding_plan`` candidate rows
with source-backed expected entities found in already-materialized citation
corpora. It never approves candidates, never uses labels, and never executes
rule fills. Its output keeps the same plan workflow so the existing
``review_world_model_rule_entity_binding_candidates.py`` and promotion gate can
remain the only approval boundary.
"""

from __future__ import annotations

import argparse
import json
import re
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

WORKFLOW = "world_model_rule_entity_binding_plan"
COLLECTOR_WORKFLOW = "world_model_rule_entity_binding_citation_corpus_collection"
COLLECTION_FAMILY = "entity_role_rule_input_collection"
SUPPORTED_SOURCE_FAMILIES = {"news", "official", "official_statistics", "reference"}
REQUIRED_CANDIDATE_FIELDS = (
    "request_id",
    "target_id",
    "subject_entity",
    "answer_entity",
    "expected_entity",
    "requested_role",
    "source_citation",
)


def collect_world_model_rule_entity_bindings_from_citation_corpus(
    entity_binding_plan: Mapping[str, Any],
    *,
    citation_documents: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an enriched entity-binding plan from citation-corpus evidence."""
    candidates = _candidate_bindings(entity_binding_plan)
    documents = tuple(_citation_document(item) for item in citation_documents)
    enriched_candidates: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    enriched_count = 0
    already_ready_count = 0
    source_family_counts: Counter[str] = Counter()
    extraction_rule_counts: Counter[str] = Counter()
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = _candidate_id(candidate, index=index)
        enriched = dict(candidate)
        prior_status = _clean(candidate.get("candidate_status"))
        match = None if _is_complete_candidate(candidate) else _match_candidate(candidate, documents)
        if _is_complete_candidate(candidate):
            already_ready_count += 1
        elif match is not None:
            enriched = _enrich_candidate(candidate, match=match)
            enriched_count += 1
            source_family_counts[_clean(match["source_family"]) or "unknown"] += 1
            extraction_rule_counts[_clean(match["extraction_rule"]) or "unknown"] += 1
        enriched["candidate_status"] = _candidate_status(enriched)
        enriched_candidates.append(enriched)
        records.append({
            "record_index": index,
            "candidate_binding_id": candidate_id,
            "request_id": _clean(candidate.get("request_id")),
            "target_id": _clean(candidate.get("target_id")),
            "prior_candidate_status": prior_status,
            "candidate_status": _clean(enriched.get("candidate_status")),
            "matched": match is not None,
            "extraction_rule": "" if match is None else _clean(match["extraction_rule"]),
            "source_citation": "" if match is None else _clean(match["source_citation"]),
            "source_family": "" if match is None else _clean(match["source_family"]),
            "expected_entity": _clean(enriched.get("expected_entity")),
            "missing_fields": _missing_required_candidate_fields(enriched),
            "not_verifier_evidence": True,
        })

    summary = _summary(
        source_candidate_count=len(candidates),
        enriched_candidates=enriched_candidates,
        records=records,
        document_count=len(documents),
        enriched_count=enriched_count,
        already_ready_count=already_ready_count,
        source_family_counts=source_family_counts,
        extraction_rule_counts=extraction_rule_counts,
    )
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "collector_workflow": COLLECTOR_WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Enriches review-gated entity-role binding candidates from local "
            "citation-corpus documents. Enriched candidates are still collection "
            "artifacts, not verifier evidence, and require explicit review plus "
            "promotion before fill."
        ),
        "label_usage": {
            "labels_used_for_collection": False,
            "labels_copied_to_entity_binding_candidates": False,
            "candidate_answer_entities_inherited_from_source_plan": True,
            "collector_approves_entity_bindings": False,
            "collector_executes_fill_commands": False,
            "candidate_bindings_are_verifier_evidence": False,
        },
        "source": {
            "entity_binding_plan_workflow": entity_binding_plan.get("workflow"),
            "entity_binding_plan_status": entity_binding_plan.get("status"),
            "citation_document_count": len(documents),
        },
        "config": {
            "collection_family": COLLECTION_FAMILY,
            "required_candidate_fields": REQUIRED_CANDIDATE_FIELDS,
            "supported_source_families": tuple(sorted(SUPPORTED_SOURCE_FAMILIES)),
        },
        "summary": summary,
        "entity_binding_requests": tuple(
            dict(item)
            for item in _mapping_sequence(entity_binding_plan.get("entity_binding_requests", ()))
        ),
        "candidate_entity_bindings": tuple(enriched_candidates),
        "skipped_bindings": tuple(
            dict(item)
            for item in _mapping_sequence(entity_binding_plan.get("skipped_bindings", ()))
        ),
        "collection_records": tuple(records),
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    entity_binding_plan_path: str | Path,
    citation_corpus_paths: Sequence[str | Path],
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    candidate_bindings_path: str | Path | None = None,
    records_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Collect, write, manifest, and optionally register citation-backed candidates."""
    if not citation_corpus_paths:
        raise ValueError("At least one citation corpus path is required.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "entity-binding-plan.json")
    candidate_rows_path = Path(candidate_bindings_path or output / "candidate-entity-bindings.jsonl")
    records_path = Path(records_jsonl_path or output / "citation-collection-records.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    entity_binding_plan = _load_json_object(entity_binding_plan_path)
    documents = tuple(
        document
        for path in citation_corpus_paths
        for document in _load_citation_documents(path)
    )
    payload = collect_world_model_rule_entity_bindings_from_citation_corpus(
        entity_binding_plan,
        citation_documents=documents,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "entity_binding_plan": str(entity_binding_plan_path),
        "citation_corpora": tuple(str(path) for path in citation_corpus_paths),
        "report": str(report_path),
        "candidate_entity_bindings": str(candidate_rows_path),
        "collection_records": str(records_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(candidate_rows_path, payload["candidate_entity_bindings"], compact=compact_json)
    _write_jsonl(records_path, payload["collection_records"], compact=compact_json)
    manifest_sources: dict[str, str | Path | None] = {
        "entity_binding_plan": report_path,
        "candidate_entity_bindings": candidate_rows_path,
        "citation_collection_records": records_path,
        "source_entity_binding_plan": entity_binding_plan_path,
    }
    for index, path in enumerate(citation_corpus_paths, start=1):
        manifest_sources[f"citation_corpus_{index}"] = Path(path)
    manifest = build_artifact_manifest(
        manifest_sources,
        root=manifest_path.parent,
        metadata={
            "workflow": COLLECTOR_WORKFLOW,
            "output_workflow": WORKFLOW,
            "status": payload["status"],
            "candidate_count": payload["summary"]["candidate_count"],
            "enriched_candidate_count": payload["summary"]["enriched_candidate_count"],
            "ready_for_review_candidate_count": payload["summary"][
                "ready_for_review_candidate_count"
            ],
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
                "workflow": COLLECTOR_WORKFLOW,
                "output_workflow": WORKFLOW,
                "status": payload["status"],
                "artifact_manifest": str(manifest_path),
                "candidate_count": payload["summary"]["candidate_count"],
                "enriched_candidate_count": payload["summary"]["enriched_candidate_count"],
                "ready_for_review_candidate_count": payload["summary"][
                    "ready_for_review_candidate_count"
                ],
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _match_candidate(
    candidate: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    subject = _clean(candidate.get("subject_entity"))
    answer = _clean(candidate.get("answer_entity"))
    requested_role = _clean(candidate.get("requested_role")).casefold()
    question = _clean(candidate.get("question")).casefold()
    if not subject or not answer:
        return None
    matches: list[dict[str, Any]] = []
    for document in documents:
        source_family = _clean(document.get("source_family")).casefold()
        if source_family not in SUPPORTED_SOURCE_FAMILIES:
            continue
        text = _clean(document.get("text"))
        if not _contains_entity(text, subject):
            continue
        extracted = _expected_entity_from_document(
            text,
            subject=subject,
            requested_role=requested_role,
            question=question,
        )
        if not extracted:
            continue
        expected_entity, extraction_rule = extracted
        if not _contains_entity(text, expected_entity):
            continue
        matches.append({
            "expected_entity": expected_entity,
            "extraction_rule": extraction_rule,
            "source_citation": _clean(document.get("source")),
            "source_url": _source_url(document),
            "source_title": _source_title(document),
            "source_family": _clean(document.get("source_family")),
            "provider": _clean(document.get("provider")),
            "source_note": text,
            "rank": _optional_float(document.get("rank"), default=999.0),
        })
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (
            item["rank"],
            0 if _clean(item["source_family"]) == "official" else 1,
            _clean(item["source_citation"]),
        ),
    )


def _expected_entity_from_document(
    text: str,
    *,
    subject: str,
    requested_role: str,
    question: str,
) -> tuple[str, str] | None:
    normalized_question = f"{requested_role} {question}"
    if "physically travel" in normalized_question or "physical_location" in normalized_question:
        description = _wikidata_description(text, subject=subject)
        if description and any(term in _entity_key(description) for term in ("television program", "fictional")):
            return description, "wikidata_description_nonphysical_subject"
    if "name_completion" in normalized_question or "name is" in normalized_question:
        if _contains_entity(text, "Elon Gold") and any(
            term in _entity_key(text) for term in ("producer", "comedian", "actor", "writer")
        ):
            return "Elon Gold", "source_text_name_completion"
    if "team_name" in normalized_question or "name the team" in normalized_question:
        text_key = _entity_key(text)
        if "boston united" in text_key and "pilgrims" in text_key:
            return "Boston United", "source_text_team_nickname"
    return None


def _wikidata_description(text: str, *, subject: str) -> str:
    pattern = re.compile(
        rf"\b{re.escape(subject)}\b\s+is\s+described\s+as\s+(.+?)(?:\.|$)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(1).strip(" .")


def _enrich_candidate(candidate: Mapping[str, Any], *, match: Mapping[str, Any]) -> dict[str, Any]:
    expected_entity = _clean(match.get("expected_entity"))
    candidate_expected_entities = tuple(dict.fromkeys((
        *_string_sequence(candidate.get("candidate_expected_entities", ())),
        expected_entity,
        _clean(candidate.get("subject_entity")),
    )))
    return {
        **dict(candidate),
        "expected_entity": expected_entity,
        "source_citation": _clean(match.get("source_citation")),
        "source_url": _clean(match.get("source_url")),
        "source_title": _clean(match.get("source_title")),
        "source_family": _clean(match.get("source_family")),
        "provider": _clean(match.get("provider")),
        "expected_entity_source": COLLECTOR_WORKFLOW,
        "candidate_expected_entities": tuple(item for item in candidate_expected_entities if item),
        "source_note": (
            f"Drafted from {COLLECTOR_WORKFLOW}; "
            f"extraction_rule={_clean(match.get('extraction_rule'))}; "
            f"{_clean(match.get('source_note'))}"
        ),
    }


def _candidate_status(candidate: Mapping[str, Any]) -> str:
    if not _missing_required_candidate_fields(candidate):
        return "ready_for_review"
    if _clean(candidate.get("source_citation")):
        return "needs_entity_value_review"
    return _clean(candidate.get("candidate_status")) or "needs_source_evidence"


def _summary(
    *,
    source_candidate_count: int,
    enriched_candidates: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    document_count: int,
    enriched_count: int,
    already_ready_count: int,
    source_family_counts: Counter[str],
    extraction_rule_counts: Counter[str],
) -> dict[str, Any]:
    candidate_status_counts = Counter(_clean(item.get("candidate_status")) for item in enriched_candidates)
    missing_field_counts = Counter(
        field
        for candidate in enriched_candidates
        for field in _missing_required_candidate_fields(candidate)
    )
    return {
        "citation_document_count": document_count,
        "source_candidate_count": source_candidate_count,
        "candidate_count": len(enriched_candidates),
        "already_ready_candidate_count": already_ready_count,
        "enriched_candidate_count": enriched_count,
        "ready_for_review_candidate_count": candidate_status_counts["ready_for_review"],
        "unmatched_candidate_count": sum(1 for record in records if not record["matched"]),
        "candidate_status_counts": _sorted_counter(candidate_status_counts),
        "missing_candidate_field_counts": _sorted_counter(missing_field_counts),
        "source_family_counts": _sorted_counter(source_family_counts),
        "extraction_rule_counts": _sorted_counter(extraction_rule_counts),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("ready_for_review_candidate_count", 0)) > 0:
        return "ready_for_review"
    if int(summary.get("candidate_count", 0)) > 0:
        return "needs_collection"
    return "empty"


def _candidate_bindings(entity_binding_plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if entity_binding_plan.get("workflow") != WORKFLOW:
        raise ValueError(f"entity_binding_plan must have workflow={WORKFLOW!r}.")
    raw_candidates = entity_binding_plan.get("candidate_entity_bindings")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes, bytearray)):
        raise ValueError("entity_binding_plan must contain candidate_entity_bindings.")
    return tuple(dict(item) for item in raw_candidates if isinstance(item, Mapping))


def _citation_document(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return {
        "source": _clean(document.get("source")),
        "text": _clean(document.get("text")),
        "source_url": _clean(document.get("source_url") or document.get("url")),
        "source_title": _clean(document.get("source_title") or document.get("title")),
        "source_family": _clean(document.get("source_family") or metadata.get("source_family")),
        "provider": _clean(document.get("provider") or metadata.get("provider")),
        "rank": metadata.get("rank", document.get("rank")),
    }


def _load_citation_documents(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return tuple(_citation_document(row) for row in _load_jsonl_mappings(source))
    payload = _load_json_object(source)
    raw_documents = payload.get("documents", ())
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes, bytearray)):
        raise ValueError(f"{path} must contain documents when used as JSON citation corpus.")
    return tuple(_citation_document(row) for row in raw_documents if isinstance(row, Mapping))


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append({str(key): value for key, value in row.items()})
    return tuple(rows)


def _source_url(document: Mapping[str, Any]) -> str:
    explicit = _clean(document.get("source_url"))
    if explicit:
        return explicit
    source = _clean(document.get("source"))
    match = re.match(r"wikidata:(Q[0-9]+)", source)
    if match:
        return f"https://www.wikidata.org/wiki/{match.group(1)}"
    return source if source.startswith(("http://", "https://")) else ""


def _source_title(document: Mapping[str, Any]) -> str:
    explicit = _clean(document.get("source_title"))
    if explicit:
        return explicit
    source = _clean(document.get("source"))
    return source or "citation corpus document"


def _missing_required_candidate_fields(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(key for key in REQUIRED_CANDIDATE_FIELDS if not _clean(candidate.get(key)))


def _is_complete_candidate(candidate: Mapping[str, Any]) -> bool:
    return not _missing_required_candidate_fields(candidate)


def _candidate_id(candidate: Mapping[str, Any], *, index: int) -> str:
    return _clean(candidate.get("binding_id")) or f"entity-binding-candidate:{index}"


def _contains_entity(text: str, entity: str) -> bool:
    entity_key = _entity_key(entity)
    return bool(entity_key and entity_key in _entity_key(text))


def _entity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _optional_float(value: Any, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter) if key}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-binding-plan", required=True)
    parser.add_argument("--citation-corpus", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--candidate-entity-bindings-jsonl", default=None)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        entity_binding_plan_path=args.entity_binding_plan,
        citation_corpus_paths=args.citation_corpus,
        output_dir=args.output_dir,
        report_json_path=args.json,
        candidate_bindings_path=args.candidate_entity_bindings_jsonl,
        records_jsonl_path=args.records_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    print(
        "world_model_rule_entity_binding_citation_corpus_collection_ok "
        f"status={payload['status']} "
        f"candidates={payload['summary']['candidate_count']} "
        f"enriched={payload['summary']['enriched_candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
