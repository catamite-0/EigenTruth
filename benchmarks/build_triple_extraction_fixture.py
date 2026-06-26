"""Build labeled triple-extraction fixtures from structured fact corpora.

The output feeds ``eval_triple_extraction.py``. This keeps extractor upgrades
measurable before they are promoted into structured-fact or triple-evidence
verification routes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.adapters.facts import StructuredFact  # noqa: E402
from eigentruth.registry import build_artifact_manifest, fingerprint_path  # noqa: E402

PREDICATE_OUTPUTS = {
    "capital": "capital_of",
    "capital_of": "capital_of",
    "p36": "capital_of",
    "official_language": "official_language_of",
    "official_languages": "official_language_of",
    "official_language_of": "official_language_of",
    "p37": "official_language_of",
    "currency": "currency_of",
    "currency_of": "currency_of",
    "p38": "currency_of",
}
DEFAULT_REGEX_PATTERNS = (
    {
        "pattern": r"^(?P<subject>.+?) has its capital at (?P<object>.+)$",
        "predicate": "capital_of",
        "source": "capital_has_at_template",
    },
    {
        "pattern": r"^The official language spoken in (?P<subject>.+?) is (?P<object>.+)$",
        "predicate": "official_language_of",
        "source": "official_language_spoken_in_template",
    },
    {
        "pattern": r"^(?P<object>.+?) is (?P<subject>.+?)(?:'s|\u2019s) currency$",
        "predicate": "currency_of",
        "source": "currency_possessive_inverse_template",
    },
)


def build_triple_extraction_fixture(
    source_records: Sequence[Mapping[str, Any]],
    *,
    max_facts: int | None = None,
) -> dict[str, Any]:
    """Build labeled extraction records from structured fact-like mappings."""
    if max_facts is not None and max_facts <= 0:
        raise ValueError("max_facts must be positive when provided.")
    facts, skipped = _coerce_structured_facts(source_records)
    if max_facts is not None:
        facts = facts[:max_facts]
    records = []
    by_predicate: dict[str, dict[str, int]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    for index, fact in enumerate(facts):
        predicate = _output_predicate(fact.predicate)
        if predicate is None:
            skipped["unsupported_predicate"] += 1
            continue
        templates = _templates_for_fact(fact, predicate=predicate)
        if not templates:
            skipped["unsupported_predicate"] += 1
            continue
        for template_id, text, expected_object in templates:
            key = (_normalize(text), _normalize(fact.subject), predicate, _normalize(expected_object))
            if key in seen:
                skipped["duplicate_record"] += 1
                continue
            seen.add(key)
            by_predicate.setdefault(predicate, {"fact_count": 0, "record_count": 0})
            by_predicate[predicate]["record_count"] += 1
            records.append({
                "id": f"{index + 1:04d}-{predicate}-{template_id}",
                "text": text,
                "expected_triples": [
                    {
                        "subject": fact.subject,
                        "predicate": predicate,
                        "object": expected_object,
                    }
                ],
                "metadata": {
                    "source_fact_index": index,
                    "template_id": template_id,
                    "predicate": predicate,
                    "source_fact": fact.to_dict(),
                },
            })
        by_predicate.setdefault(predicate, {"fact_count": 0, "record_count": 0})
        by_predicate[predicate]["fact_count"] += 1
    if not records:
        raise ValueError("no triple extraction fixture records were produced.")
    return {
        "schema_version": 1,
        "fixture_type": "triple_extraction_labeled_fixture",
        "description": (
            "Labeled subject-predicate-object extraction records generated from structured facts. "
            "Use with eval_triple_extraction.py to compare extractor variants."
        ),
        "summary": {
            "n_source_records": len(source_records),
            "n_facts": len(facts),
            "n_records": len(records),
            "by_predicate": by_predicate,
            "skipped": skipped,
        },
        "records": records,
    }


def build_default_regex_pattern_payload() -> dict[str, Any]:
    """Return default regex templates matching the generated non-default forms."""
    patterns = []
    for pattern in DEFAULT_REGEX_PATTERNS:
        patterns.append({
            **pattern,
            "metadata": {
                "builder": "build_triple_extraction_fixture",
                "template_family": "kg_core",
            },
        })
    return {
        "schema_version": 1,
        "pattern_type": "triple_extraction_regex_patterns",
        "description": (
            "Default regex patterns for generated KG-core triple extraction templates. "
            "Use with regex_rule_based or composite extractors."
        ),
        "patterns": patterns,
    }


def load_fact_records(paths: Sequence[str | Path]) -> tuple[Mapping[str, Any], ...]:
    """Load fact-like records from JSON or JSONL files."""
    records: list[Mapping[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() == ".jsonl":
            records.extend(_load_jsonl(path))
        else:
            records.extend(_load_json(path))
    if not records:
        raise ValueError("fact corpus paths did not contain any records.")
    return tuple(records)


def build_input_provenance(source_paths: Sequence[str | Path], *, max_facts: int | None) -> dict[str, Any]:
    """Return source fingerprints and builder settings."""
    return {
        "schema_version": 1,
        "builder": "build_triple_extraction_fixture",
        "sources": [fingerprint_path(path).to_dict() for path in source_paths],
        "config": {"max_facts": max_facts},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    source_paths = tuple(Path(path) for path in args.fact_corpus)
    records = load_fact_records(source_paths)
    fixture = build_triple_extraction_fixture(records, max_facts=args.max_facts)
    fixture["input_provenance"] = build_input_provenance(source_paths, max_facts=args.max_facts)

    output_records = Path(args.output_records)
    output_records.parent.mkdir(parents=True, exist_ok=True)
    output_records.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pattern_payload = None
    if args.output_patterns:
        pattern_payload = build_default_regex_pattern_payload()
        output_patterns = Path(args.output_patterns)
        output_patterns.parent.mkdir(parents=True, exist_ok=True)
        output_patterns.write_text(json.dumps(pattern_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.artifact_manifest:
        manifest_path = Path(args.artifact_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = {"records": output_records}
        if args.output_patterns:
            artifacts["patterns"] = Path(args.output_patterns)
        for idx, path in enumerate(source_paths, start=1):
            artifacts[f"source.{idx}.{path.stem}"] = path
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": "build_triple_extraction_fixture",
                "n_records": fixture["summary"]["n_records"],
                "n_facts": fixture["summary"]["n_facts"],
                "pattern_count": 0 if pattern_payload is None else len(pattern_payload["patterns"]),
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "triple_extraction_fixture_ok "
        f"facts={fixture['summary']['n_facts']} "
        f"records={fixture['summary']['n_records']} "
        f"output={output_records}"
    )
    return fixture


def _coerce_structured_facts(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[StructuredFact, ...], dict[str, int]]:
    facts = []
    skipped = {
        "invalid_fact": 0,
        "unsupported_predicate": 0,
        "duplicate_record": 0,
    }
    for record in records:
        try:
            facts.append(StructuredFact.from_mapping(record))
        except ValueError:
            skipped["invalid_fact"] += 1
    if not facts:
        raise ValueError("fact records did not contain any valid structured facts.")
    return tuple(facts), skipped


def _templates_for_fact(fact: StructuredFact, *, predicate: str) -> tuple[tuple[str, str, str], ...]:
    subject = fact.subject
    object_value = fact.object
    if predicate == "capital_of":
        return (
            ("capital-object-first", f"{object_value} is the capital of {subject}.", object_value),
            ("capital-subject-first", f"The capital of {subject} is {object_value}.", object_value),
            ("capital-possessive", f"{subject}'s capital is {object_value}.", object_value),
            ("capital-has-at", f"{subject} has its capital at {object_value}.", object_value),
        )
    if predicate == "official_language_of":
        return (
            ("official-language-object-first", f"{object_value} is an official language of {subject}.", object_value),
            ("official-language-subject-first", f"The official language of {subject} is {object_value}.", object_value),
            ("official-language-possessive", f"{subject}'s official language is {object_value}.", object_value),
            (
                "official-language-spoken-in",
                f"The official language spoken in {subject} is {object_value}.",
                object_value,
            ),
        )
    if predicate == "currency_of":
        return (
            ("currency-object-first", f"{object_value} is the currency of {subject}.", object_value),
            ("currency-subject-first", f"The currency of {subject} is {object_value}.", object_value),
            ("currency-uses", f"{subject} uses {object_value} as its currency.", object_value),
            ("currency-possessive-inverse", f"{object_value} is {subject}'s currency.", object_value),
        )
    return ()


def _output_predicate(value: Any) -> str | None:
    text = _normalize_predicate(value)
    return PREDICATE_OUTPUTS.get(text)


def _normalize_predicate(value: Any) -> str:
    text = _normalize(value)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    return text.strip("_")


def _normalize(value: Any) -> str:
    return " ".join(str(value).casefold().strip().split())


def _load_json(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_records = payload.get("facts", payload.get("documents", payload.get("records", ())))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        raw_records = payload
    else:
        raise ValueError(f"{path} must contain a JSON object or list.")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError(f"{path} records must be a list.")
    return [_coerce_mapping(item, path=path) for item in raw_records]


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records.append(_coerce_mapping(json.loads(line), path=path, line_no=line_no))
    return records


def _coerce_mapping(value: Any, *, path: Path, line_no: int | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        location = str(path) if line_no is None else f"{path}:{line_no}"
        raise ValueError(f"{location} contained a non-object fact record.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled triple-extraction fixtures from fact corpora")
    parser.add_argument("--fact-corpus", action="append", required=True, help="fact corpus JSON/JSONL; repeatable")
    parser.add_argument("--output-records", required=True, help="output labeled extraction records JSON")
    parser.add_argument("--output-patterns", default=None, help="optional output regex patterns JSON")
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
