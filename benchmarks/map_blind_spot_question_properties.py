"""Map blind-spot questions to explicit Wikidata covered-fact properties.

This workflow consumes the conservative covered-fact mapping audit and adds one
more gate: a joined fact is promotable only when the original question exposes a
matching property intent. Generic entity facts such as descriptions and
``instance of`` rows remain useful collection diagnostics, but they are not
treated as open-domain correction evidence by themselves.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
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
from eigentruth.verify import normalize_claim_text  # noqa: E402

DEFAULT_SUBJECT_COVERAGE_THRESHOLD = 0.60
DEFAULT_MAPPING_SCORE_THRESHOLD = 0.75
DEFAULT_ANSWER_OVERLAP_THRESHOLD = 0.80

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "give",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "some",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
}


@dataclass(frozen=True)
class PropertyIntentRule:
    """A lightweight lexical bridge from question wording to KG property ids."""

    intent: str
    property_ids: tuple[str, ...]
    patterns: tuple[str, ...]
    cue_terms: tuple[str, ...]
    strength: str = "explicit"

    def matches(self, question: str) -> bool:
        return any(re.search(pattern, question) for pattern in self.patterns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "property_ids": self.property_ids,
            "cue_terms": self.cue_terms,
            "strength": self.strength,
        }


PROPERTY_INTENT_RULES = (
    PropertyIntentRule(
        intent="founder",
        property_ids=("P112",),
        patterns=(
            r"\b(who|which person|what person)\b.*\b(started|founded|co[- ]?founded|established|launched)\b",
            r"\b(first|original)\b.*\b(started|founded|founder|founders)\b",
            r"\bfounder(s)?\b",
        ),
        cue_terms=("founder", "founded", "started", "cofounder", "established", "launched"),
    ),
    PropertyIntentRule(
        intent="creator",
        property_ids=("P170",),
        patterns=(
            r"\b(who|which person|what person)\b.*\b(created|creator|made|invented)\b",
            r"\bcreated by\b",
        ),
        cue_terms=("creator", "created", "made", "invented"),
    ),
    PropertyIntentRule(
        intent="author",
        property_ids=("P50",),
        patterns=(
            r"\b(who|which person|what person)\b.*\b(wrote|authored|author)\b",
            r"\bauthor(s)?\b",
        ),
        cue_terms=("author", "wrote", "authored"),
    ),
    PropertyIntentRule(
        intent="occupation",
        property_ids=("P106",),
        patterns=(
            r"\b(occupation|job|profession)\b",
            r"\bwhat\b.*\b(do|does|did)\b.*\bdo\b",
        ),
        cue_terms=("occupation", "job", "profession"),
    ),
    PropertyIntentRule(
        intent="country",
        property_ids=("P17", "P27"),
        patterns=(
            r"\b(which|what)\s+country\s+(is|was|are|were)\b",
            r"\bin\s+(which|what)\s+country\b",
            r"\bnationalit(y|ies)\b",
            r"\bfrom which country\b",
        ),
        cue_terms=("country", "nationality"),
    ),
    PropertyIntentRule(
        intent="located_in",
        property_ids=("P131", "P159", "P276", "P17"),
        patterns=(
            r"\bwhere\b.*\b(located|based|headquartered|situated)\b",
            r"\b(which|what)\b.*\b(city|state|province|place|location)\b",
        ),
        cue_terms=("located", "based", "headquartered", "location"),
    ),
    PropertyIntentRule(
        intent="official_website",
        property_ids=("P856",),
        patterns=(r"\b(official website|website|url)\b",),
        cue_terms=("official", "website", "url"),
    ),
    PropertyIntentRule(
        intent="award_received",
        property_ids=("P166",),
        patterns=(r"\b(won|win|winner|winners|received)\b.*\b(award|prize|nobel)\b",),
        cue_terms=("won", "winner", "award", "prize", "nobel"),
    ),
    PropertyIntentRule(
        intent="generic_definition",
        property_ids=("description", "P31", "P279"),
        patterns=(
            r"^\bwhat\b.*\b(is|are)\b",
            r"\bwhat kind\b",
            r"\bwhat type\b",
            r"\bdefinition\b",
        ),
        cue_terms=("description", "instance", "subclass", "type", "kind"),
        strength="generic",
    ),
)


def map_blind_spot_question_properties(
    mapping_audit: Mapping[str, Any],
    *,
    subject_coverage_threshold: float = DEFAULT_SUBJECT_COVERAGE_THRESHOLD,
    mapping_score_threshold: float = DEFAULT_MAPPING_SCORE_THRESHOLD,
    answer_overlap_threshold: float = DEFAULT_ANSWER_OVERLAP_THRESHOLD,
) -> dict[str, Any]:
    """Return explicit question-property mapping decisions for audit records."""
    records = _records(mapping_audit)
    mapped_records = tuple(
        _map_record(
            record,
            subject_coverage_threshold=float(subject_coverage_threshold),
            mapping_score_threshold=float(mapping_score_threshold),
            answer_overlap_threshold=float(answer_overlap_threshold),
        )
        for record in records
    )
    summary = _summary(mapped_records)
    status = "observed" if summary["mapped_correction_candidate_count"] > 0 else "blocked"
    return {
        "schema_version": 1,
        "workflow": "blind_spot_question_property_mapping",
        "status": status,
        "scope": (
            "Maps original blind-spot questions to explicit covered-fact "
            "properties. Generic joined facts are retained as diagnostics and "
            "are not promoted as correction evidence."
        ),
        "source": {
            "mapping_audit_workflow": mapping_audit.get("workflow"),
            "mapping_audit_status": mapping_audit.get("status"),
            "mapping_audit_target_count": _nested_int(mapping_audit, "summary", "target_count"),
        },
        "config": {
            "subject_coverage_threshold": float(subject_coverage_threshold),
            "mapping_score_threshold": float(mapping_score_threshold),
            "answer_overlap_threshold": float(answer_overlap_threshold),
        },
        "summary": summary,
        "records": mapped_records,
        "next_step": (
            "Use mapped_correction_candidate rows as the first explicit "
            "question/property correction gate; route generic_fact_only and "
            "unmapped rows to citation retrieval, richer KG property search, or "
            "world-model rule authoring."
        ),
    }


def run(
    *,
    mapping_audit_path: str | Path,
    output_path: str | Path,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    subject_coverage_threshold: float = DEFAULT_SUBJECT_COVERAGE_THRESHOLD,
    mapping_score_threshold: float = DEFAULT_MAPPING_SCORE_THRESHOLD,
    answer_overlap_threshold: float = DEFAULT_ANSWER_OVERLAP_THRESHOLD,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Load the audit, write the mapping report, and optionally register it."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    audit = _load_json_mapping(mapping_audit_path)
    payload = map_blind_spot_question_properties(
        audit,
        subject_coverage_threshold=subject_coverage_threshold,
        mapping_score_threshold=mapping_score_threshold,
        answer_overlap_threshold=answer_overlap_threshold,
    )
    payload["paths"] = {"mapping_audit": str(mapping_audit_path)}
    payload["metadata"] = dict(metadata or {})
    output = Path(output_path)
    _write_json(output, payload, compact=compact_json)
    if artifact_manifest_path is not None:
        manifest_path = Path(artifact_manifest_path)
        manifest = build_artifact_manifest(
            {
                "question_property_mapping": output,
                "covered_fact_mapping_audit": Path(mapping_audit_path),
            },
            root=manifest_path.parent,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "mapped_correction_candidate_count": payload["summary"][
                    "mapped_correction_candidate_count"
                ],
                "generic_fact_only_count": payload["summary"]["generic_fact_only_count"],
                **dict(metadata or {}),
            },
        )
        _write_json(manifest_path, manifest, compact=compact_json)
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=output,
            metadata={
                "workflow": payload["workflow"],
                "status": payload["status"],
                "target_count": payload["summary"]["target_count"],
                "mapped_correction_candidate_count": payload["summary"][
                    "mapped_correction_candidate_count"
                ],
                "generic_fact_only_count": payload["summary"]["generic_fact_only_count"],
                "artifact_manifest": None if artifact_manifest_path is None else str(artifact_manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _map_record(
    record: Mapping[str, Any],
    *,
    subject_coverage_threshold: float,
    mapping_score_threshold: float,
    answer_overlap_threshold: float,
) -> dict[str, Any]:
    question = str(record.get("question", ""))
    normalized_question = normalize_claim_text(question)
    intents = _question_intents(normalized_question, question_type=str(record.get("question_type") or ""))
    facts = tuple(_mapping(item) for item in _sequence(record.get("facts")))
    scored_facts = tuple(
        sorted(
            (
                _score_fact(
                    fact,
                    question=question,
                    intents=intents,
                    answer_overlap_threshold=answer_overlap_threshold,
                )
                for fact in facts
            ),
            key=lambda item: (-float(item["mapping_score"]), str(item["source"])),
        )
    )
    accepted = tuple(
        fact
        for fact in scored_facts
        if fact["explicit_property_match"]
        and fact["subject_coverage"] >= subject_coverage_threshold
        and fact["mapping_score"] >= mapping_score_threshold
        and not fact["answer_value_supported"]
        and not fact["answer_entity_collision"]
        and not fact["self_referential_fact"]
    )
    generic = tuple(
        fact
        for fact in scored_facts
        if fact["generic_property_match"]
        and fact["subject_coverage"] >= subject_coverage_threshold
        and not fact["explicit_property_match"]
    )
    decision = _decision(record, accepted=accepted, generic=generic, scored_facts=scored_facts)
    return {
        "record_index": int(record.get("record_index", -1)),
        "question": question,
        "answer": str(record.get("answer", "")),
        "question_type": record.get("question_type"),
        "source_mapping_status": record.get("mapping_status"),
        "mapping_decision": decision,
        "correction_candidate": decision == "mapped_correction_candidate",
        "question_property_intents": tuple(intent.to_dict() for intent in intents),
        "best_mapping_score": max((float(item["mapping_score"]) for item in scored_facts), default=0.0),
        "best_subject_coverage": max((float(item["subject_coverage"]) for item in scored_facts), default=0.0),
        "mapped_facts": accepted[:10],
        "generic_facts": generic[:10],
        "top_fact_candidates": scored_facts[:10],
        "gate_recommendation": _gate_recommendation(decision),
    }


def _score_fact(
    fact: Mapping[str, Any],
    *,
    question: str,
    intents: Sequence[PropertyIntentRule],
    answer_overlap_threshold: float,
) -> dict[str, Any]:
    property_id = str(fact.get("statement_property") or "unknown")
    subject = str(fact.get("subject") or "")
    question_tokens = _tokens(question)
    subject_tokens = _tokens(subject)
    property_tokens = _tokens(f"{fact.get('statement_property_label') or ''} {property_id}")
    matched_intents = tuple(intent for intent in intents if property_id in intent.property_ids)
    explicit_property_match = any(intent.strength == "explicit" for intent in matched_intents)
    generic_property_match = any(intent.strength == "generic" for intent in matched_intents)
    subject_coverage = _coverage(subject_tokens, question_tokens)
    property_coverage = _coverage(property_tokens, question_tokens)
    mapping_score = (
        0.50 * subject_coverage
        + 0.40 * (1.0 if explicit_property_match else 0.0)
        + 0.10 * max(property_coverage, float(fact.get("question_overlap") or 0.0))
    )
    answer_value_overlap = float(fact.get("answer_value_overlap") or 0.0)
    answer_subject_overlap = float(fact.get("answer_subject_overlap") or 0.0)
    self_referential_fact = normalize_claim_text(subject) == normalize_claim_text(str(fact.get("answer") or ""))
    return {
        "question": fact.get("question"),
        "answer": fact.get("answer"),
        "source": fact.get("source"),
        "statement_property": property_id,
        "statement_property_label": fact.get("statement_property_label"),
        "subject": subject,
        "subject_qid": fact.get("subject_qid"),
        "value_qid": fact.get("value_qid"),
        "matched_intents": tuple(intent.intent for intent in matched_intents),
        "explicit_property_match": explicit_property_match,
        "generic_property_match": generic_property_match,
        "subject_coverage": subject_coverage,
        "property_coverage": property_coverage,
        "question_overlap": float(fact.get("question_overlap") or 0.0),
        "answer_value_overlap": answer_value_overlap,
        "answer_subject_overlap": answer_subject_overlap,
        "answer_value_supported": answer_value_overlap >= answer_overlap_threshold,
        "answer_entity_collision": answer_subject_overlap >= answer_overlap_threshold,
        "self_referential_fact": self_referential_fact,
        "mapping_score": mapping_score,
    }


def _question_intents(question: str, *, question_type: str) -> tuple[PropertyIntentRule, ...]:
    matched = [rule for rule in PROPERTY_INTENT_RULES if rule.matches(question)]
    if not matched and question_type == "definition":
        matched.extend(rule for rule in PROPERTY_INTENT_RULES if rule.intent == "generic_definition")
    return tuple(matched)


def _decision(
    record: Mapping[str, Any],
    *,
    accepted: Sequence[Mapping[str, Any]],
    generic: Sequence[Mapping[str, Any]],
    scored_facts: Sequence[Mapping[str, Any]],
) -> str:
    if not scored_facts:
        return "no_joined_facts"
    if bool(record.get("answer_value_supported")):
        return "answer_value_supported"
    if bool(record.get("answer_entity_collision")):
        return "answer_entity_collision"
    if accepted:
        return "mapped_correction_candidate"
    if generic:
        return "generic_fact_only"
    if any(item.get("explicit_property_match") for item in scored_facts):
        return "property_without_subject_match"
    if any(float(item.get("subject_coverage") or 0.0) > 0.0 for item in scored_facts):
        return "subject_only_or_unsupported_property"
    return "unmapped_low_relevance"


def _gate_recommendation(decision: str) -> str:
    if decision == "mapped_correction_candidate":
        return "structured_fact_property_gate"
    if decision == "generic_fact_only":
        return "citation_or_richer_property_search"
    if decision in {"answer_value_supported", "answer_entity_collision"}:
        return "answer_collision_audit"
    if decision == "no_joined_facts":
        return "coverage_expansion"
    return "citation_retrieval_or_world_model"


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = Counter(str(record.get("mapping_decision")) for record in records)
    property_counts = Counter()
    intent_counts = Counter()
    for record in records:
        if record.get("correction_candidate"):
            for fact in _sequence(record.get("mapped_facts")):
                property_counts[str(_mapping(fact).get("statement_property") or "unknown")] += 1
            for intent in _sequence(record.get("question_property_intents")):
                intent_counts[str(_mapping(intent).get("intent") or "unknown")] += 1
    return {
        "target_count": len(records),
        "mapped_correction_candidate_count": decisions.get("mapped_correction_candidate", 0),
        "generic_fact_only_count": decisions.get("generic_fact_only", 0),
        "answer_value_supported_count": decisions.get("answer_value_supported", 0),
        "answer_entity_collision_count": decisions.get("answer_entity_collision", 0),
        "no_joined_fact_count": decisions.get("no_joined_facts", 0),
        "mapping_decision_counts": _sorted_counter(decisions),
        "mapped_property_counts": _sorted_counter(property_counts),
        "mapped_intent_counts": _sorted_counter(intent_counts),
    }


def _records(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_records = payload.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("mapping audit must contain a records list.")
    records = [dict(item) for item in raw_records if isinstance(item, Mapping)]
    if not records:
        raise ValueError("mapping audit did not contain usable records.")
    return tuple(records)


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _tokens(value: Any) -> set[str]:
    text = normalize_claim_text(str(value))
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) > 1 and token not in STOPWORDS
    }


def _coverage(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    try:
        return int(current)
    except (TypeError, ValueError):
        return None


def _sorted_counter(counter: Counter[str] | Mapping[str, Any]) -> dict[str, int]:
    items = Counter({str(key): int(value) for key, value in dict(counter).items()})
    return dict(sorted(items.items(), key=lambda item: (-item[1], item[0])))


def _parse_metadata(values: Sequence[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not values:
        return metadata
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"metadata item {item!r} must use key=value format.")
            key, raw = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("metadata key must be non-empty.")
            metadata[key] = raw.strip()
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-audit", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--subject-coverage-threshold", type=float, default=DEFAULT_SUBJECT_COVERAGE_THRESHOLD)
    parser.add_argument("--mapping-score-threshold", type=float, default=DEFAULT_MAPPING_SCORE_THRESHOLD)
    parser.add_argument("--answer-overlap-threshold", type=float, default=DEFAULT_ANSWER_OVERLAP_THRESHOLD)
    parser.add_argument("--metadata", action="append", default=None)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        mapping_audit_path=args.mapping_audit,
        output_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        subject_coverage_threshold=args.subject_coverage_threshold,
        mapping_score_threshold=args.mapping_score_threshold,
        answer_overlap_threshold=args.answer_overlap_threshold,
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "blind_spot_question_property_mapping_ok "
        f"status={payload['status']} "
        f"targets={summary['target_count']} "
        f"mapped={summary['mapped_correction_candidate_count']} "
        f"generic={summary['generic_fact_only_count']}"
    )


if __name__ == "__main__":
    main()
