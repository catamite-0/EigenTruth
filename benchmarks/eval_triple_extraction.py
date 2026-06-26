"""Evaluate claim triple extractors against labeled triples.

This script is intentionally dependency-free. It is a lightweight harness for
comparing rule-based, regex-augmented, and future extractor adapters before they
are trusted by structured-fact or triple-evidence verifier routes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.verify import (  # noqa: E402
    Claim,
    ClaimTriple,
    ClaimTripleExtractor,
    CompositeTripleExtractor,
    RegexTripleExtractor,
    RegexTriplePattern,
    RuleBasedTripleExtractor,
)


def evaluate_triple_extractor(
    records: Sequence[Mapping[str, Any]],
    extractor: ClaimTripleExtractor,
    *,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Evaluate exact triple extraction precision/recall/F1."""
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative.")
    total_expected = 0
    total_extracted = 0
    true_positive = 0
    examples = []
    for index, record in enumerate(records):
        claim = _record_claim(record, index=index)
        expected = tuple(_triple_key(triple) for triple in _record_expected_triples(record, claim))
        extracted_triples = tuple(extractor.extract(claim))
        extracted = tuple(_triple_key(triple) for triple in extracted_triples)
        expected_set = set(expected)
        extracted_set = set(extracted)
        matched = expected_set & extracted_set
        total_expected += len(expected_set)
        total_extracted += len(extracted_set)
        true_positive += len(matched)
        if len(examples) < max_examples and (expected_set != extracted_set):
            examples.append({
                "index": index,
                "claim_id": claim.claim_id,
                "text": claim.text,
                "expected": tuple(_key_dict(key) for key in sorted(expected_set)),
                "extracted": tuple(triple.to_dict() for triple in extracted_triples),
                "missing": tuple(_key_dict(key) for key in sorted(expected_set - extracted_set)),
                "extra": tuple(_key_dict(key) for key in sorted(extracted_set - expected_set)),
            })
    precision = _safe_div(true_positive, total_extracted)
    recall = _safe_div(true_positive, total_expected)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "record_count": len(records),
        "expected_triple_count": total_expected,
        "extracted_triple_count": total_extracted,
        "exact_match_count": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_examples": tuple(examples),
    }


def build_triple_extractor(
    extractor_name: str,
    *,
    patterns: Sequence[RegexTriplePattern | Mapping[str, Any]] = (),
) -> ClaimTripleExtractor:
    """Build a named extractor for benchmark use."""
    name = extractor_name.strip().casefold().replace("-", "_")
    rule_based = RuleBasedTripleExtractor()
    if name == "rule_based":
        return rule_based
    if name == "regex":
        return RegexTripleExtractor(patterns=patterns)
    if name in {"regex_rule_based", "regex_with_rule_based"}:
        return RegexTripleExtractor(patterns=patterns, fallback=rule_based)
    if name == "composite":
        return CompositeTripleExtractor((
            RegexTripleExtractor(patterns=patterns),
            rule_based,
        ))
    raise ValueError(
        "extractor must be one of rule_based, regex, regex_rule_based, or composite."
    )


def load_triple_extraction_records(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Load labeled extraction records from JSON or JSONL."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            records.append(payload)
        return tuple(records)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_records = payload.get("records", payload.get("examples"))
    else:
        raw_records = payload
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("triple extraction dataset must be a list or contain records/examples.")
    records = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"record {index} must be a mapping.")
        records.append(item)
    return tuple(records)


def load_regex_patterns(path: str | Path | None) -> tuple[RegexTriplePattern, ...]:
    """Load regex triple patterns from JSON."""
    if path is None:
        return ()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_patterns = payload.get("patterns") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_patterns, Sequence) or isinstance(raw_patterns, (str, bytes, bytearray)):
        raise ValueError("regex patterns must be a list or contain a patterns list.")
    return tuple(
        RegexTriplePattern.from_dict(_as_mapping(item, index=index))
        for index, item in enumerate(raw_patterns)
    )


def run_triple_extraction_eval(
    records_path: str | Path,
    *,
    extractor_name: str,
    patterns_path: str | Path | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Run a triple extraction evaluation and return a JSON-ready payload."""
    records = load_triple_extraction_records(records_path)
    patterns = load_regex_patterns(patterns_path)
    extractor = build_triple_extractor(extractor_name, patterns=patterns)
    report = evaluate_triple_extractor(records, extractor, max_examples=max_examples)
    return {
        "workflow": "triple_extraction_eval",
        "records_path": str(records_path),
        "patterns_path": None if patterns_path is None else str(patterns_path),
        "extractor": extractor_name,
        "pattern_count": len(patterns),
        "report": report,
    }


def _record_claim(record: Mapping[str, Any], *, index: int) -> Claim:
    text = record.get("text", record.get("claim", record.get("statement")))
    if text is None or not str(text).strip():
        raise ValueError(f"record {index} must contain text, claim, or statement.")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"record {index} metadata must be a mapping when provided.")
    claim_id = record.get("claim_id", record.get("id", f"r{index}"))
    return Claim(str(text), claim_id=str(claim_id), metadata=dict(metadata))


def _record_expected_triples(record: Mapping[str, Any], claim: Claim) -> tuple[ClaimTriple, ...]:
    raw = record.get("expected_triples", record.get("triples", record.get("expected")))
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        items = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = tuple(raw)
    else:
        raise ValueError("expected triples must be a mapping or sequence of mappings.")
    triples = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("expected triples must contain mappings.")
        payload = dict(item)
        payload.setdefault("claim_id", claim.claim_id)
        payload.setdefault("source_text", claim.text)
        triples.append(ClaimTriple.from_dict(payload))
    return tuple(triples)


def _triple_key(triple: ClaimTriple) -> tuple[str, str, str]:
    return (_normalize_slot(triple.subject), _normalize_predicate(triple.predicate), _normalize_slot(triple.object))


def _normalize_slot(value: Any) -> str:
    return " ".join(str(value).casefold().strip().split())


def _normalize_predicate(value: Any) -> str:
    return "_".join(str(value).casefold().strip().replace("-", "_").split())


def _key_dict(key: tuple[str, str, str]) -> dict[str, str]:
    return {"subject": key[0], "predicate": key[1], "object": key[2]}


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _as_mapping(value: Any, *, index: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"pattern {index} must be a mapping.")
    return value


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate dependency-free claim triple extractors")
    parser.add_argument("--records", required=True, help="JSON/JSONL records with text and expected_triples")
    parser.add_argument(
        "--extractor",
        default="rule_based",
        help="rule_based, regex, regex_rule_based, or composite",
    )
    parser.add_argument("--patterns", default=None, help="optional JSON regex pattern list")
    parser.add_argument("--json", required=True, help="output JSON report path")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)

    payload = run_triple_extraction_eval(
        args.records,
        extractor_name=args.extractor,
        patterns_path=args.patterns,
        max_examples=args.max_examples,
    )
    _write_json(Path(args.json), payload, compact=bool(args.compact_json))
    report = payload["report"]
    print(
        "triple_extraction_eval_ok "
        f"extractor={payload['extractor']} "
        f"records={report['record_count']} "
        f"precision={report['precision']:.3f} "
        f"recall={report['recall']:.3f} "
        f"f1={report['f1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
