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
    "headquarters": "headquarters_location_of",
    "headquarters_location": "headquarters_location_of",
    "headquarters_location_of": "headquarters_location_of",
    "headquarter_location": "headquarters_location_of",
    "headquarter_location_of": "headquarters_location_of",
    "p159": "headquarters_location_of",
    "manufacturer": "manufacturer_of",
    "manufacturer_of": "manufacturer_of",
    "manufactured_by": "manufacturer_of",
    "made_by": "manufacturer_of",
    "p176": "manufacturer_of",
    "founded": "inception_of",
    "founded_in": "inception_of",
    "founding_date": "inception_of",
    "inception": "inception_of",
    "inception_of": "inception_of",
    "p571": "inception_of",
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
    {
        "pattern": r"^(?P<subject>.+?) is headquartered in (?P<object>.+)$",
        "predicate": "headquarters_location_of",
        "source": "headquarters_headquartered_in_template",
    },
    {
        "pattern": r"^The headquarters of (?P<subject>.+?) (?:are|is) in (?P<object>.+)$",
        "predicate": "headquarters_location_of",
        "source": "headquarters_of_template",
    },
    {
        "pattern": r"^(?P<subject>.+?)(?:'s|\u2019s) headquarters (?:are|is) in (?P<object>.+)$",
        "predicate": "headquarters_location_of",
        "source": "headquarters_possessive_template",
    },
    {
        "pattern": r"^(?P<object>.+?) is the headquarters location of (?P<subject>.+)$",
        "predicate": "headquarters_location_of",
        "source": "headquarters_object_first_template",
    },
    {
        "pattern": r"^(?P<object>.+?) manufactures (?P<subject>.+)$",
        "predicate": "manufacturer_of",
        "source": "manufacturer_object_first_template",
    },
    {
        "pattern": r"^(?P<subject>.+?) is manufactured by (?P<object>.+)$",
        "predicate": "manufacturer_of",
        "source": "manufacturer_subject_first_template",
    },
    {
        "pattern": r"^(?P<subject>.+?) is made by (?P<object>.+)$",
        "predicate": "manufacturer_of",
        "source": "manufacturer_made_by_template",
    },
    {
        "pattern": r"^(?P<subject>.+?)(?:'s|\u2019s) manufacturer is (?P<object>.+)$",
        "predicate": "manufacturer_of",
        "source": "manufacturer_possessive_template",
    },
    {
        "pattern": r"^(?P<subject>.+?) was founded in (?P<object>.+)$",
        "predicate": "inception_of",
        "source": "inception_founded_in_template",
    },
    {
        "pattern": r"^The inception date of (?P<subject>.+?) is (?P<object>.+)$",
        "predicate": "inception_of",
        "source": "inception_date_subject_template",
    },
    {
        "pattern": r"^(?P<subject>.+?)(?:'s|\u2019s) inception date is (?P<object>.+)$",
        "predicate": "inception_of",
        "source": "inception_possessive_template",
    },
    {
        "pattern": r"^(?P<object>.+?) is the inception date of (?P<subject>.+)$",
        "predicate": "inception_of",
        "source": "inception_object_first_template",
    },
)


def build_triple_extraction_fixture(
    source_records: Sequence[Mapping[str, Any]],
    *,
    max_facts: int | None = None,
    adversarial_negatives_per_fact: int = 0,
    predicate_confusions_per_fact: int = 0,
    non_assertive_negatives_per_fact: int = 0,
) -> dict[str, Any]:
    """Build labeled extraction records from structured fact-like mappings."""
    if max_facts is not None and max_facts <= 0:
        raise ValueError("max_facts must be positive when provided.")
    if int(adversarial_negatives_per_fact) < 0:
        raise ValueError("adversarial_negatives_per_fact must be non-negative.")
    adversarial_negatives_per_fact = int(adversarial_negatives_per_fact)
    if int(predicate_confusions_per_fact) < 0:
        raise ValueError("predicate_confusions_per_fact must be non-negative.")
    predicate_confusions_per_fact = int(predicate_confusions_per_fact)
    if int(non_assertive_negatives_per_fact) < 0:
        raise ValueError("non_assertive_negatives_per_fact must be non-negative.")
    non_assertive_negatives_per_fact = int(non_assertive_negatives_per_fact)
    facts, skipped = _coerce_structured_facts(source_records)
    if max_facts is not None:
        facts = facts[:max_facts]
    records = []
    by_predicate: dict[str, dict[str, int]] = {}
    adversarial_negative_count = 0
    predicate_confusion_count = 0
    non_assertive_negative_count = 0
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
                        "record_type": "positive",
                        "source_fact_index": index,
                        "template_id": template_id,
                        "predicate": predicate,
                        "source_fact": fact.to_dict(),
                    },
                })
        for template_id, text in _adversarial_negative_templates_for_fact(
            fact,
            predicate=predicate,
            limit=adversarial_negatives_per_fact,
        ):
            key = (_normalize(text), _normalize(fact.subject), predicate, "")
            if key in seen:
                skipped["duplicate_record"] += 1
                continue
            seen.add(key)
            by_predicate.setdefault(predicate, {"fact_count": 0, "record_count": 0})
            by_predicate[predicate]["record_count"] += 1
            adversarial_negative_count += 1
            records.append({
                "id": f"{index + 1:04d}-{predicate}-adversarial-negative-{template_id}",
                "text": text,
                "expected_triples": [],
                "metadata": {
                    "record_type": "adversarial_negative",
                    "source_fact_index": index,
                    "template_id": template_id,
                    "predicate": predicate,
                    "adversarial_family": "negated_known_fact",
                    "source_fact": fact.to_dict(),
                },
            })
        for template_id, text, expected_predicate, expected_object in _predicate_confusion_templates_for_fact(
            fact,
            predicate=predicate,
            limit=predicate_confusions_per_fact,
        ):
            key = (_normalize(text), _normalize(fact.subject), expected_predicate, _normalize(expected_object))
            if key in seen:
                skipped["duplicate_record"] += 1
                continue
            seen.add(key)
            by_predicate.setdefault(expected_predicate, {"fact_count": 0, "record_count": 0})
            by_predicate[expected_predicate]["record_count"] += 1
            predicate_confusion_count += 1
            records.append({
                "id": f"{index + 1:04d}-{predicate}-predicate-confusion-{template_id}",
                "text": text,
                "expected_triples": [
                    {
                        "subject": fact.subject,
                        "predicate": expected_predicate,
                        "object": expected_object,
                    }
                ],
                "metadata": {
                    "record_type": "predicate_confusion",
                    "source_fact_index": index,
                    "template_id": template_id,
                    "predicate": expected_predicate,
                    "source_predicate": predicate,
                    "adversarial_family": "wrong_predicate_claim",
                    "source_fact": fact.to_dict(),
                },
            })
        for template_id, text in _non_assertive_negative_templates_for_fact(
            fact,
            predicate=predicate,
            positive_templates=templates,
            limit=non_assertive_negatives_per_fact,
        ):
            key = (_normalize(text), _normalize(fact.subject), predicate, "")
            if key in seen:
                skipped["duplicate_record"] += 1
                continue
            seen.add(key)
            by_predicate.setdefault(predicate, {"fact_count": 0, "record_count": 0})
            by_predicate[predicate]["record_count"] += 1
            non_assertive_negative_count += 1
            records.append({
                "id": f"{index + 1:04d}-{predicate}-non-assertive-negative-{template_id}",
                "text": text,
                "expected_triples": [],
                "metadata": {
                    "record_type": "non_assertive_negative",
                    "source_fact_index": index,
                    "template_id": template_id,
                    "predicate": predicate,
                    "adversarial_family": "quoted_or_questioned_fact",
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
            "n_adversarial_negative_records": adversarial_negative_count,
            "n_predicate_confusion_records": predicate_confusion_count,
            "n_non_assertive_negative_records": non_assertive_negative_count,
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


def build_input_provenance(
    source_paths: Sequence[str | Path],
    *,
    max_facts: int | None,
    adversarial_negatives_per_fact: int = 0,
    predicate_confusions_per_fact: int = 0,
    non_assertive_negatives_per_fact: int = 0,
) -> dict[str, Any]:
    """Return source fingerprints and builder settings."""
    return {
        "schema_version": 1,
        "builder": "build_triple_extraction_fixture",
        "sources": [fingerprint_path(path).to_dict() for path in source_paths],
        "config": {
            "max_facts": max_facts,
            "adversarial_negatives_per_fact": int(adversarial_negatives_per_fact),
            "predicate_confusions_per_fact": int(predicate_confusions_per_fact),
            "non_assertive_negatives_per_fact": int(non_assertive_negatives_per_fact),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    source_paths = tuple(Path(path) for path in args.fact_corpus)
    records = load_fact_records(source_paths)
    adversarial_negatives_per_fact = int(getattr(args, "adversarial_negatives_per_fact", 0))
    predicate_confusions_per_fact = int(getattr(args, "predicate_confusions_per_fact", 0))
    non_assertive_negatives_per_fact = int(getattr(args, "non_assertive_negatives_per_fact", 0))
    fixture = build_triple_extraction_fixture(
        records,
        max_facts=args.max_facts,
        adversarial_negatives_per_fact=adversarial_negatives_per_fact,
        predicate_confusions_per_fact=predicate_confusions_per_fact,
        non_assertive_negatives_per_fact=non_assertive_negatives_per_fact,
    )
    fixture["input_provenance"] = build_input_provenance(
        source_paths,
        max_facts=args.max_facts,
        adversarial_negatives_per_fact=adversarial_negatives_per_fact,
        predicate_confusions_per_fact=predicate_confusions_per_fact,
        non_assertive_negatives_per_fact=non_assertive_negatives_per_fact,
    )

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
                "n_adversarial_negative_records": fixture["summary"]["n_adversarial_negative_records"],
                "n_predicate_confusion_records": fixture["summary"]["n_predicate_confusion_records"],
                "n_non_assertive_negative_records": fixture["summary"]["n_non_assertive_negative_records"],
                "pattern_count": 0 if pattern_payload is None else len(pattern_payload["patterns"]),
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "triple_extraction_fixture_ok "
        f"facts={fixture['summary']['n_facts']} "
        f"records={fixture['summary']['n_records']} "
        f"adversarial_negatives={fixture['summary']['n_adversarial_negative_records']} "
        f"predicate_confusions={fixture['summary']['n_predicate_confusion_records']} "
        f"non_assertive_negatives={fixture['summary']['n_non_assertive_negative_records']} "
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
    if predicate == "headquarters_location_of":
        return (
            ("headquarters-headquartered-in", f"{subject} is headquartered in {object_value}.", object_value),
            ("headquarters-of", f"The headquarters of {subject} are in {object_value}.", object_value),
            ("headquarters-possessive", f"{subject}'s headquarters are in {object_value}.", object_value),
            (
                "headquarters-object-first",
                f"{object_value} is the headquarters location of {subject}.",
                object_value,
            ),
        )
    if predicate == "manufacturer_of":
        return (
            ("manufacturer-object-first", f"{object_value} manufactures {subject}.", object_value),
            ("manufacturer-subject-first", f"{subject} is manufactured by {object_value}.", object_value),
            ("manufacturer-made-by", f"{subject} is made by {object_value}.", object_value),
            ("manufacturer-possessive", f"{subject}'s manufacturer is {object_value}.", object_value),
        )
    if predicate == "inception_of":
        return (
            ("inception-founded-in", f"{subject} was founded in {object_value}.", object_value),
            ("inception-date-subject", f"The inception date of {subject} is {object_value}.", object_value),
            ("inception-possessive", f"{subject}'s inception date is {object_value}.", object_value),
            ("inception-object-first", f"{object_value} is the inception date of {subject}.", object_value),
        )
    return ()


def _adversarial_negative_templates_for_fact(
    fact: StructuredFact,
    *,
    predicate: str,
    limit: int,
) -> tuple[tuple[str, str], ...]:
    if limit <= 0:
        return ()
    subject = fact.subject
    object_value = fact.object
    templates: tuple[tuple[str, str], ...]
    if predicate == "capital_of":
        templates = (
            ("capital-negated-subject-first", f"The capital of {subject} is not {object_value}."),
            ("capital-negated-possessive", f"{subject}'s capital is not {object_value}."),
        )
    elif predicate == "official_language_of":
        templates = (
            (
                "official-language-negated-subject-first",
                f"The official language of {subject} is not {object_value}.",
            ),
            (
                "official-language-negated-object-first",
                f"{object_value} is not an official language of {subject}.",
            ),
        )
    elif predicate == "currency_of":
        templates = (
            ("currency-negated-subject-first", f"The currency of {subject} is not {object_value}."),
            ("currency-negated-usage", f"{subject} does not use {object_value} as its currency."),
        )
    elif predicate == "headquarters_location_of":
        templates = (
            ("headquarters-negated-headquartered-in", f"{subject} is not headquartered in {object_value}."),
            ("headquarters-negated-of", f"The headquarters of {subject} are not in {object_value}."),
        )
    elif predicate == "manufacturer_of":
        templates = (
            ("manufacturer-negated-subject-first", f"{subject} is not manufactured by {object_value}."),
            ("manufacturer-negated-object-first", f"{object_value} does not manufacture {subject}."),
        )
    elif predicate == "inception_of":
        templates = (
            ("inception-negated-founded-in", f"{subject} was not founded in {object_value}."),
            ("inception-negated-date", f"The inception date of {subject} is not {object_value}."),
        )
    else:
        templates = ()
    return templates[:limit]


def _predicate_confusion_templates_for_fact(
    fact: StructuredFact,
    *,
    predicate: str,
    limit: int,
) -> tuple[tuple[str, str, str, str], ...]:
    if limit <= 0:
        return ()
    subject = fact.subject
    object_value = fact.object
    templates: tuple[tuple[str, str, str, str], ...]
    if predicate == "capital_of":
        templates = (
            ("capital-as-currency", f"The currency of {subject} is {object_value}.", "currency_of", object_value),
            (
                "capital-as-official-language",
                f"The official language of {subject} is {object_value}.",
                "official_language_of",
                object_value,
            ),
        )
    elif predicate == "official_language_of":
        templates = (
            ("language-as-capital", f"The capital of {subject} is {object_value}.", "capital_of", object_value),
            ("language-as-currency", f"The currency of {subject} is {object_value}.", "currency_of", object_value),
        )
    elif predicate == "currency_of":
        templates = (
            (
                "currency-as-official-language",
                f"The official language of {subject} is {object_value}.",
                "official_language_of",
                object_value,
            ),
            ("currency-as-capital", f"The capital of {subject} is {object_value}.", "capital_of", object_value),
        )
    elif predicate == "headquarters_location_of":
        templates = (
            (
                "headquarters-as-manufacturer",
                f"{subject} is manufactured by {object_value}.",
                "manufacturer_of",
                object_value,
            ),
            ("headquarters-as-inception", f"{subject} was founded in {object_value}.", "inception_of", object_value),
        )
    elif predicate == "manufacturer_of":
        templates = (
            (
                "manufacturer-as-headquarters",
                f"The headquarters of {subject} are in {object_value}.",
                "headquarters_location_of",
                object_value,
            ),
            ("manufacturer-as-inception", f"{subject} was founded in {object_value}.", "inception_of", object_value),
        )
    elif predicate == "inception_of":
        templates = (
            (
                "inception-as-headquarters",
                f"{subject} is headquartered in {object_value}.",
                "headquarters_location_of",
                object_value,
            ),
            (
                "inception-as-manufacturer",
                f"{subject} is manufactured by {object_value}.",
                "manufacturer_of",
                object_value,
            ),
        )
    else:
        templates = ()
    return templates[:limit]


def _non_assertive_negative_templates_for_fact(
    fact: StructuredFact,
    *,
    predicate: str,
    positive_templates: Sequence[tuple[str, str, str]],
    limit: int,
) -> tuple[tuple[str, str], ...]:
    if limit <= 0 or not positive_templates:
        return ()
    _template_id, positive_text, _expected_object = positive_templates[0]
    claim_text = positive_text.rstrip(".")
    templates = (
        ("quoted-claim-reviewed", f'The claim "{claim_text}" was reviewed.'),
        ("question-asks-whether", f"A question asks whether {claim_text}."),
    )
    return templates[:limit]


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
    parser.add_argument(
        "--adversarial-negatives-per-fact",
        type=int,
        default=0,
        help="optional near-miss negated records with no expected triples per fact",
    )
    parser.add_argument(
        "--predicate-confusions-per-fact",
        type=int,
        default=0,
        help="optional wrong-predicate assertion records per fact",
    )
    parser.add_argument(
        "--non-assertive-negatives-per-fact",
        type=int,
        default=0,
        help="optional quoted/questioned records with no expected triples per fact",
    )
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
