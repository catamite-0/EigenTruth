"""Evaluate verifier robustness on counterfactual claim probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.adapters import QuestionAnswerVerifier, StructuredFactVerifier  # noqa: E402
from eigentruth.json_utils import to_jsonable  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify import (  # noqa: E402
    Claim,
    CounterfactualProbe,
    InMemoryVerifier,
    VerificationStatus,
    Verifier,
    audit_counterfactual_verification,
    generate_counterfactual_probes,
    normalize_claim_text,
)


def load_counterfactual_probes(path: str | Path) -> tuple[CounterfactualProbe, ...]:
    """Load counterfactual probes from JSON or JSONL."""
    path = Path(path)
    records = _load_json_records(path)
    return tuple(CounterfactualProbe.from_dict(record) for record in records)


def load_claims_for_counterfactual_generation(path: str | Path) -> tuple[Claim, ...]:
    """Load claims from JSON or JSONL for counterfactual probe generation."""
    path = Path(path)
    records = _load_json_records(path)
    return tuple(_claim_from_record(record, index=index) for index, record in enumerate(records))


def load_counterfactual_probes_from_verified_records(
    path: str | Path,
    *,
    max_pairs: int | None = None,
) -> tuple[CounterfactualProbe, ...]:
    """Build QA counterfactual probes from supported/refuted verified records."""
    if max_pairs is not None and int(max_pairs) < 1:
        raise ValueError("max_pairs must be positive when set.")
    records = _load_json_records(Path(path))
    by_question: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for index, row in enumerate(records):
        item = _probe_claim_from_verified_record(row, index=index)
        if item is None:
            continue
        bucket = by_question.setdefault(item["question_key"], {"supported": [], "refuted": []})
        bucket[item["status"]].append(item)

    probes: list[CounterfactualProbe] = []
    for question_key, bucket in by_question.items():
        if not bucket["supported"] or not bucket["refuted"]:
            continue
        original = bucket["supported"][0]
        for counterfactual in bucket["refuted"]:
            probe_index = len(probes)
            probes.append(CounterfactualProbe(
                original=original["claim"],
                counterfactual=counterfactual["claim"],
                probe_id=f"verified_records:{probe_index}",
                probe_type="structured_qa_answer_mismatch",
                expected_original_status=VerificationStatus.SUPPORTED,
                expected_counterfactual_status=VerificationStatus.REFUTED,
                expected_flip=True,
                metadata={
                    "source": "verified_records",
                    "source_verified_records_path": str(path),
                    "question_key": question_key,
                    "original_record_index": original["record_index"],
                    "counterfactual_record_index": counterfactual["record_index"],
                },
            ))
            if max_pairs is not None and len(probes) >= int(max_pairs):
                return tuple(probes)
    return tuple(probes)


def build_counterfactual_verifier(
    verifier_name: str,
    probes: Sequence[CounterfactualProbe],
    *,
    fact_corpus_path: str | Path | None = None,
    in_memory_facts_path: str | Path | None = None,
    default_status: VerificationStatus | str = VerificationStatus.INSUFFICIENT_EVIDENCE,
) -> Verifier:
    """Build a local verifier for counterfactual audit fixtures."""
    name = verifier_name.strip().casefold().replace("-", "_")
    if name in {"in_memory", "exact_match"}:
        facts = _load_in_memory_facts(in_memory_facts_path) if in_memory_facts_path else _facts_from_expected(probes)
        return InMemoryVerifier(
            facts=facts,
            default_status=_coerce_status(default_status, field_name="default_status"),
        )
    if name in {"structured_fact", "structured_facts"}:
        if fact_corpus_path is None:
            raise ValueError("structured_fact verifier requires --fact-corpus.")
        payload = json.loads(Path(fact_corpus_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("fact corpus must contain a JSON object.")
        return StructuredFactVerifier.from_corpus(payload)
    if name in {"structured_qa", "qa", "question_answer"}:
        if fact_corpus_path is None:
            raise ValueError("structured_qa verifier requires --fact-corpus.")
        payload = json.loads(Path(fact_corpus_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("QA corpus must contain a JSON object.")
        return QuestionAnswerVerifier.from_corpus(payload)
    raise ValueError("verifier must be one of: in_memory, structured_fact, structured_qa.")


def run_counterfactual_verification_eval(
    records_path: str | Path | None = None,
    *,
    claims_path: str | Path | None = None,
    verified_records_path: str | Path | None = None,
    max_verified_record_pairs: int | None = None,
    max_generated_probes_per_claim: int = 3,
    generated_probe_types: Sequence[str] = ("metadata", "entity_swap", "quantity", "year", "negation"),
    verifier_name: str = "in_memory",
    fact_corpus_path: str | Path | None = None,
    in_memory_facts_path: str | Path | None = None,
    default_status: VerificationStatus | str = VerificationStatus.INSUFFICIENT_EVIDENCE,
    max_examples: int = 20,
    output_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    register_name: str | None = None,
    register_version: str = "0.1",
    compact_json: bool = False,
) -> dict[str, Any]:
    """Run counterfactual verifier audit and return a JSON-ready payload."""
    probes = _load_or_generate_probes(
        records_path=records_path,
        claims_path=claims_path,
        verified_records_path=verified_records_path,
        max_verified_record_pairs=max_verified_record_pairs,
        max_generated_probes_per_claim=max_generated_probes_per_claim,
        generated_probe_types=generated_probe_types,
    )
    verifier = build_counterfactual_verifier(
        verifier_name,
        probes,
        fact_corpus_path=fact_corpus_path,
        in_memory_facts_path=in_memory_facts_path,
        default_status=default_status,
    )
    report = audit_counterfactual_verification(
        verifier,
        probes,
        max_examples=max_examples,
        metadata={
            "records_path": None if records_path is None else str(records_path),
            "claims_path": None if claims_path is None else str(claims_path),
            "verified_records_path": None if verified_records_path is None else str(verified_records_path),
            "generated_probe_count": sum(1 for probe in probes if probe.metadata.get("source") == "generated"),
            "verified_record_probe_count": sum(
                1 for probe in probes if probe.metadata.get("source") == "verified_records"
            ),
            "verifier": verifier_name,
            "fact_corpus_path": None if fact_corpus_path is None else str(fact_corpus_path),
            "in_memory_facts_path": None if in_memory_facts_path is None else str(in_memory_facts_path),
        },
    ).to_dict()
    payload: dict[str, Any] = {
        "workflow": "counterfactual_verification_eval",
        "records_path": None if records_path is None else str(records_path),
        "claims_path": None if claims_path is None else str(claims_path),
        "verified_records_path": None if verified_records_path is None else str(verified_records_path),
        "generated_probe_count": sum(1 for probe in probes if probe.metadata.get("source") == "generated"),
        "verified_record_probe_count": sum(
            1 for probe in probes if probe.metadata.get("source") == "verified_records"
        ),
        "verifier": verifier_name,
        "fact_corpus_path": None if fact_corpus_path is None else str(fact_corpus_path),
        "in_memory_facts_path": None if in_memory_facts_path is None else str(in_memory_facts_path),
        "report": report,
        "paths": {},
    }
    if output_path is not None:
        payload["paths"]["report"] = str(output_path)
    if registry_path is not None:
        if output_path is None:
            raise ValueError("registry_path requires output_path.")
        name = (register_name or Path(output_path).stem).strip()
        if not name:
            raise ValueError("register_name must be non-empty when provided.")
        payload["registry_record"] = f"report:{name}:{register_version}"
        payload["paths"]["registry"] = str(registry_path)
    if artifact_manifest_path is not None:
        if output_path is None:
            raise ValueError("artifact_manifest_path requires output_path.")
        manifest_path = Path(artifact_manifest_path)
        payload["paths"]["artifact_manifest"] = str(manifest_path)
        _write_json(Path(output_path), payload, compact=compact_json)
        artifacts = {
            "counterfactual_verification_report": output_path,
        }
        if records_path is not None:
            artifacts["records"] = Path(records_path)
        if claims_path is not None:
            artifacts["claims"] = Path(claims_path)
        if verified_records_path is not None:
            artifacts["verified_records"] = Path(verified_records_path)
        if fact_corpus_path is not None:
            artifacts["fact_corpus"] = Path(fact_corpus_path)
        if in_memory_facts_path is not None:
            artifacts["in_memory_facts"] = Path(in_memory_facts_path)
        manifest_metadata = {
            "workflow": "counterfactual_verification_eval",
            "verifier": verifier_name,
            "record_count": report["summary"]["record_count"],
            "pass_rate": report["summary"]["pass_rate"],
            "false_invariance_rate": report["summary"]["false_invariance_rate"],
        }
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata=manifest_metadata,
        )
        payload["artifact_manifest_summary"] = manifest["summary"]
        _write_json(Path(output_path), payload, compact=compact_json)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata=manifest_metadata,
        )
        payload["artifact_manifest_summary"] = manifest["summary"]
        _write_json(manifest_path, manifest, compact=False)
        _write_json(Path(output_path), payload, compact=compact_json)
    elif output_path is not None:
        _write_json(Path(output_path), payload, compact=compact_json)
    if registry_path is not None:
        registry = ArtifactRegistry.load_json(registry_path)
        registry.record_report(
            name=name,
            version=register_version,
            path=output_path,
            metadata={
                "workflow": "counterfactual_verification_eval",
                "verifier": verifier_name,
                "record_count": report["summary"]["record_count"],
                "pass_rate": report["summary"]["pass_rate"],
                "false_invariance_rate": report["summary"]["false_invariance_rate"],
                "artifact_manifest": payload["paths"].get("artifact_manifest"),
            },
        ).save_json()
    return payload


def _load_json_records(path: Path) -> tuple[Mapping[str, Any], ...]:
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
        raw_records = payload.get("records", payload.get("probes", payload.get("examples")))
    else:
        raw_records = payload
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes, bytearray)):
        raise ValueError("counterfactual records must be a list or contain records/probes/examples.")
    records = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"record {index} must be a mapping.")
        records.append(item)
    return tuple(records)


def _load_or_generate_probes(
    *,
    records_path: str | Path | None,
    claims_path: str | Path | None,
    verified_records_path: str | Path | None,
    max_verified_record_pairs: int | None,
    max_generated_probes_per_claim: int,
    generated_probe_types: Sequence[str],
) -> tuple[CounterfactualProbe, ...]:
    probes: list[CounterfactualProbe] = []
    if records_path is not None:
        probes.extend(load_counterfactual_probes(records_path))
    if verified_records_path is not None:
        probes.extend(load_counterfactual_probes_from_verified_records(
            verified_records_path,
            max_pairs=max_verified_record_pairs,
        ))
    if claims_path is not None:
        claims = load_claims_for_counterfactual_generation(claims_path)
        probes.extend(generate_counterfactual_probes(
            claims,
            max_probes_per_claim=max_generated_probes_per_claim,
            probe_types=generated_probe_types,
        ))
    if not probes:
        raise ValueError("counterfactual eval requires --records, --verified-records, --claims, or a combination.")
    return tuple(probes)


def _probe_claim_from_verified_record(
    row: Mapping[str, Any],
    *,
    index: int,
) -> dict[str, Any] | None:
    record = _mapping(row.get("record"))
    final = _mapping(record.get("final", record.get("qa")))
    status = str(final.get("status", "")).strip().casefold().replace("-", "_")
    if status not in {"supported", "refuted"}:
        return None
    statement = _mapping(_nested(record, "metadata", "statement"))
    claim = _mapping(record.get("claim"))
    claim_metadata = dict(_mapping(claim.get("metadata")))
    question = _optional_text(
        statement.get("question")
        or claim_metadata.get("question")
        or _nested(claim_metadata, "statement", "question")
    )
    answer = _optional_text(
        statement.get("answer")
        or claim_metadata.get("answer")
        or _nested(claim_metadata, "statement", "answer")
    )
    if question is None or answer is None:
        return None
    text = _optional_text(statement.get("text") or claim.get("text")) or f"{question} {answer}"
    row_index = row.get("record_index", index)
    question_key = normalize_claim_text(question)
    return {
        "status": status,
        "question_key": question_key,
        "record_index": row_index,
        "claim": Claim(
            text=text,
            claim_id=str(claim.get("claim_id") or claim.get("id") or f"verified_record_{row_index}"),
            metadata={
                **claim_metadata,
                "question": question,
                "answer": answer,
                "statement": {
                    **statement,
                    "question": question,
                    "answer": answer,
                    "text": text,
                },
                "verified_record_index": row_index,
                "verified_record_status": status,
            },
        ),
    }


def _claim_from_record(record: Mapping[str, Any], *, index: int) -> Claim:
    text = record.get("text", record.get("claim", record.get("statement")))
    if text is None or not str(text).strip():
        raise ValueError(f"claim record {index} must contain text, claim, or statement.")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"claim record {index} metadata must be a mapping.")
    span = record.get("span")
    if span is not None:
        if not isinstance(span, Sequence) or isinstance(span, (str, bytes, bytearray)) or len(span) != 2:
            raise ValueError(f"claim record {index} span must be a two-item sequence.")
        span = (int(span[0]), int(span[1]))
    claim_id = record.get("claim_id", record.get("id", f"c{index + 1}"))
    return Claim(
        text=str(text),
        claim_id=None if claim_id is None else str(claim_id),
        span=span,
        metadata=dict(metadata),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_in_memory_facts(path: str | Path | None) -> dict[str, VerificationStatus]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    facts: dict[str, VerificationStatus] = {}
    if isinstance(payload, Mapping) and isinstance(payload.get("facts"), Sequence):
        for index, item in enumerate(payload["facts"]):
            if not isinstance(item, Mapping):
                raise ValueError(f"in-memory fact {index} must be a mapping.")
            text = item.get("text", item.get("claim", item.get("statement")))
            if text is None or not str(text).strip():
                raise ValueError(f"in-memory fact {index} must contain text, claim, or statement.")
            facts[normalize_claim_text(str(text))] = _coerce_status(
                item.get("status", VerificationStatus.SUPPORTED),
                field_name=f"in-memory fact {index} status",
            )
        return facts
    if isinstance(payload, Mapping):
        for text, status in payload.items():
            facts[normalize_claim_text(str(text))] = _coerce_status(status, field_name=f"status for {text!r}")
        return facts
    raise ValueError("in-memory facts must be a mapping or contain a facts list.")


def _facts_from_expected(probes: Sequence[CounterfactualProbe]) -> dict[str, VerificationStatus]:
    facts: dict[str, VerificationStatus] = {}
    for probe in probes:
        if probe.expected_original_status is not None:
            facts[normalize_claim_text(probe.original.text)] = probe.expected_original_status
        if probe.expected_counterfactual_status is not None:
            facts[normalize_claim_text(probe.counterfactual.text)] = probe.expected_counterfactual_status
    return facts


def _coerce_status(value: VerificationStatus | str, *, field_name: str) -> VerificationStatus:
    if isinstance(value, VerificationStatus):
        return value
    text = str(value).strip().casefold().replace("-", "_")
    try:
        return VerificationStatus(text)
    except ValueError as exc:
        choices = ", ".join(status.value for status in VerificationStatus)
        raise ValueError(f"{field_name} must be one of: {choices}") from exc


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    path.write_text(
        json.dumps(to_jsonable(dict(payload)), indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate verifier robustness on counterfactual probes")
    parser.add_argument("--records", default=None, help="JSON/JSONL counterfactual probe records")
    parser.add_argument("--claims", default=None, help="JSON/JSONL claims to generate counterfactual probes from")
    parser.add_argument(
        "--verified-records",
        default=None,
        help="JSONL verifier records; supported/refuted rows with the same question become probes",
    )
    parser.add_argument("--max-verified-record-pairs", type=int, default=None)
    parser.add_argument("--generate-probes", action="store_true",
                        help="generate counterfactual probes from --claims; retained for explicit CLI readability")
    parser.add_argument("--max-generated-probes-per-claim", type=int, default=3)
    parser.add_argument("--generated-probe-types", default="metadata,entity_swap,quantity,year,negation",
                        help="comma-separated generated probe types")
    parser.add_argument("--verifier", default="in_memory", help="in_memory, structured_fact, or structured_qa")
    parser.add_argument("--fact-corpus", default=None, help="structured fact corpus JSON path")
    parser.add_argument("--in-memory-facts", default=None, help="optional exact-match facts JSON path")
    parser.add_argument("--default-status", default=VerificationStatus.INSUFFICIENT_EVIDENCE.value)
    parser.add_argument("--json", required=True, help="output JSON report path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest path")
    parser.add_argument("--registry", default=None, help="optional ArtifactRegistry JSON path")
    parser.add_argument("--register-name", default=None)
    parser.add_argument("--register-version", default="0.1")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)

    payload = run_counterfactual_verification_eval(
        args.records,
        claims_path=args.claims,
        verified_records_path=args.verified_records,
        max_verified_record_pairs=args.max_verified_record_pairs,
        max_generated_probes_per_claim=args.max_generated_probes_per_claim,
        generated_probe_types=tuple(
            item.strip()
            for item in str(args.generated_probe_types).split(",")
            if item.strip()
        ),
        verifier_name=args.verifier,
        fact_corpus_path=args.fact_corpus,
        in_memory_facts_path=args.in_memory_facts,
        default_status=args.default_status,
        max_examples=args.max_examples,
        output_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        register_name=args.register_name,
        register_version=args.register_version,
        compact_json=bool(args.compact_json),
    )
    summary = payload["report"]["summary"]
    print(
        "counterfactual_verification_eval_ok "
        f"verifier={payload['verifier']} "
        f"records={summary['record_count']} "
        f"pass_rate={summary['pass_rate']:.3f} "
        f"false_invariance_rate={summary['false_invariance_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
