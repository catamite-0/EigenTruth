"""Run a covered-facts structured QA route benchmark from a Wikidata QA corpus.

This workflow turns externally sourced Wikidata QA facts into a balanced
true/false score dump, then verifies those rows through the existing
``QuestionAnswerVerifier`` route. It is intentionally a covered-facts benchmark:
negative rows are generated as answers that mismatch the known source corpus for
the same question, not as broad open-domain TruthfulQA labels.
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

from benchmarks.eval_verifier_ensemble import build_verifier_ensemble_report  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402
from eigentruth.verify import normalize_claim_text  # noqa: E402

DEFAULT_SIGNAL = "truth_proj"
DEFAULT_ALPHA = 0.10
STATEMENT_PROVENANCE_METADATA_KEYS = (
    "alignment_candidate_id",
    "alignment_source_document_id",
    "review_id",
    "review_status",
    "reviewed_at",
    "reviewer",
    "source_family",
    "structured_evidence_slots",
)


def build_wikidata_covered_fact_score_dump(
    qa_corpus: Mapping[str, Any],
    *,
    limit: int | None = None,
    signal: str = DEFAULT_SIGNAL,
    statement_style: str = "qa",
) -> dict[str, Any]:
    """Return a balanced true/false score dump from a Wikidata QA corpus."""
    statement_style = _normalize_statement_style(statement_style)
    documents = _qa_documents(qa_corpus)
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be positive when set.")
        documents = documents[:int(limit)]
    if len(documents) < 2:
        raise ValueError("at least two QA documents are required to generate false answers.")

    known_answers = _known_answers_by_question(documents)
    known_answer_values = _known_answer_values_by_question(documents)
    statements: list[dict[str, Any]] = []
    labels: list[int] = []
    scores: list[float] = []
    skipped_false_answer = 0
    for idx, document in enumerate(documents):
        question = str(document["question"])
        answer = str(document["answer"])
        metadata = dict(document.get("metadata", {}))
        source = document.get("source")
        true_statements = _statements(
            question=question,
            answer=answer,
            source=None if source is None else str(source),
            metadata=metadata,
            label_generation="wikidata_known_answer",
            statement_style=statement_style,
            known_answers=known_answer_values.get(normalize_claim_text(question), ()),
        )
        statements.extend(true_statements)
        labels.extend(0 for _ in true_statements)
        scores.extend(0.0 for _ in true_statements)

        false_answer = _false_answer_for(
            document,
            documents=documents,
            known_answers=known_answers,
        )
        if false_answer is None:
            skipped_false_answer += 1
            continue
        false_statements = _statements(
            question=question,
            answer=false_answer["answer"],
            source=None if false_answer.get("source") is None else str(false_answer["source"]),
            metadata={
                **metadata,
                "false_answer_source": false_answer.get("source"),
                "false_answer_statement_property": false_answer.get("statement_property"),
            },
            label_generation="wikidata_known_answer_mismatch",
            statement_style=statement_style,
            known_answers=known_answer_values.get(normalize_claim_text(question), ()),
        )
        statements.extend(false_statements)
        labels.extend(1 for _ in false_statements)
        scores.extend(0.0 for _ in false_statements)

    if not any(label == 1 for label in labels):
        raise ValueError("no false-answer rows could be generated.")

    property_summary = _score_dump_property_summary(
        documents=documents,
        statements=statements,
        labels=labels,
    )
    return {
        "config": {
            "model": "wikidata-covered-facts",
            "layer": -1,
            "workflow": "wikidata_structured_qa_route_workflow",
            "signal": signal,
            "statement_style": statement_style,
            "label_semantics": {
                "0": "known Wikidata QA answer",
                "1": "answer mismatches known Wikidata QA answer(s) for the same question",
            },
            "source": {
                "corpus_type": qa_corpus.get("corpus_type"),
                "provider": _mapping(qa_corpus.get("source")).get("provider"),
                "templates": _mapping(qa_corpus.get("source")).get("templates"),
            },
        },
        "labels": labels,
        "scores": {signal: scores},
        "statements": statements,
        "summary": {
            "n_source_documents": len(documents),
            "n_records": len(labels),
            "n_true": sum(1 for label in labels if label == 0),
            "n_false": sum(1 for label in labels if label == 1),
            "n_skipped_false_answer": skipped_false_answer,
            "property_count": len(property_summary),
            "by_property": property_summary,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    qa_corpus_path = Path(args.qa_corpus)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signal = str(args.signal)
    alpha = float(args.alpha)
    route = _normalize_route(args.route)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    qa_corpus = json.loads(qa_corpus_path.read_text(encoding="utf-8"))
    if not isinstance(qa_corpus, Mapping):
        raise ValueError("qa_corpus must be a JSON object.")
    score_dump = build_wikidata_covered_fact_score_dump(
        qa_corpus,
        limit=args.limit,
        signal=signal,
        statement_style=_statement_style_for_route(route, getattr(args, "fact_claim_style", "canonical")),
    )
    route_stem = route.replace("_", "-")
    score_dump_path = Path(args.score_dump_json or output_dir / "covered-facts-scores.json")
    verifier_report_path = Path(args.verifier_report_json or output_dir / f"{route_stem}-verifier-report.json")
    verified_records_path = Path(args.verified_records_jsonl or output_dir / "verified-records.jsonl")
    summary_path = Path(args.json or output_dir / f"{route_stem}-route-summary.json")
    manifest_path = Path(args.artifact_manifest or output_dir / "artifact-manifest.json")

    _write_json(score_dump_path, score_dump, compact=bool(args.compact_json))
    verifier_report = build_verifier_ensemble_report(
        ((str(args.score_name), score_dump_path),),
        signal=signal,
        qa_corpus_path=qa_corpus_path if route == "structured_qa" else None,
        fact_corpus_path=qa_corpus_path if route == "structured_fact" else None,
        alphas=(alpha,),
        repeats=1,
        seed=int(args.seed),
        verified_records_path=verified_records_path,
    )
    _write_json(verifier_report_path, verifier_report, compact=bool(args.compact_json))
    summary = _summary_payload(
        qa_corpus_path=qa_corpus_path,
        score_dump_path=score_dump_path,
        verifier_report_path=verifier_report_path,
        verified_records_path=verified_records_path,
        score_dump=score_dump,
        verifier_report=verifier_report,
        signal=signal,
        alpha=alpha,
        route=route,
    )
    _write_json(summary_path, summary, compact=bool(args.compact_json))
    manifest = build_artifact_manifest(
        {
            "qa_corpus": qa_corpus_path,
            "covered_fact_score_dump": score_dump_path,
            "verifier_report": verifier_report_path,
            "verified_records_jsonl": verified_records_path,
            "route_summary": summary_path,
        },
        root=manifest_path.parent,
        metadata={
            "workflow": "wikidata_structured_qa_route_workflow",
            "route": route,
            "status": summary["status"],
            "score_name": str(args.score_name),
            "signal": signal,
            "alpha": alpha,
            "n_records": score_dump["summary"]["n_records"],
            "n_true": score_dump["summary"]["n_true"],
            "n_false": score_dump["summary"]["n_false"],
            "property_count": summary["property_count"],
            f"{route}_property_count": summary["property_count"],
            f"{route}_decision_accuracy": summary[f"{route}_metrics"].get("decision_accuracy"),
            f"{route}_false_supported_rate": summary[f"{route}_metrics"].get("false_supported_rate"),
            f"{route}_false_refuted_rate": summary[f"{route}_metrics"].get("false_refuted_rate"),
            "promotes_covered_facts_route": summary["status"] == "promote",
            **_covered_fact_property_manifest_metadata(summary),
        },
    )
    _write_json(manifest_path, manifest, compact=False)
    print(
        "wikidata_structured_qa_route_workflow_ok "
        f"route={route} status={summary['status']} records={score_dump['summary']['n_records']} output={summary_path}"
    )
    return summary


def _summary_payload(
    *,
    qa_corpus_path: Path,
    score_dump_path: Path,
    verifier_report_path: Path,
    verified_records_path: Path,
    score_dump: Mapping[str, Any],
    verifier_report: Mapping[str, Any],
    signal: str,
    alpha: float,
    route: str,
) -> dict[str, Any]:
    runs = verifier_report.get("runs", ())
    if not runs:
        raise ValueError("verifier report did not contain any runs.")
    run = _mapping(runs[0])
    route_quality = _mapping(run.get("route_quality"))
    route_metrics = _mapping(route_quality.get(route))
    selected_counts = _mapping(_mapping(run.get("route_summary")).get("selected_counts"))
    status = (
        "promote"
        if (
            route_metrics.get("decision_accuracy") == 1.0
            and route_metrics.get("true_supported_rate") == 1.0
            and route_metrics.get("false_refuted_rate") == 1.0
            and route_metrics.get("false_supported_rate") == 0.0
        )
        else "blocked"
    )
    property_metrics = _property_metrics_from_verified_records(
        statements=score_dump["statements"],
        verified_records_path=verified_records_path,
        score_property_summary=_mapping(score_dump["summary"].get("by_property")),
    )
    return {
        "schema_version": 1,
        "workflow": "wikidata_structured_qa_route_workflow",
        "status": status,
        "route": route,
        "scope": "covered Wikidata QA facts, not open-domain TruthfulQA route coverage",
        "signal": signal,
        "alpha": alpha,
        "qa_corpus_path": str(qa_corpus_path),
        "covered_fact_score_dump_path": str(score_dump_path),
        "verifier_report_path": str(verifier_report_path),
        "verified_records_jsonl_path": str(verified_records_path),
        "score_dump_summary": dict(score_dump["summary"]),
        "selected_route_counts": dict(selected_counts),
        "route_metrics": dict(route_metrics),
        f"{route}_metrics": dict(route_metrics),
        "property_count": len(property_metrics),
        "property_metrics": property_metrics,
        "verification_status_counts": dict(_mapping(run.get("verification_status_counts"))),
        "qa_verifier": _mapping(run.get("qa")),
        "fact_verifier": _mapping(run.get("fact")),
        "next_step": (
            "Use this covered-facts route as the property-level correction path; "
            "keep lexical retrieval gated separately for broad open-domain coverage."
        ),
    }


def _covered_fact_property_manifest_metadata(summary: Mapping[str, Any]) -> dict[str, Any]:
    property_metrics = _mapping(summary.get("property_metrics"))
    if not property_metrics:
        return {}

    metadata: dict[str, Any] = {
        "covered_fact_property_count": len(property_metrics),
        "covered_fact_property_ids": sorted(str(key) for key in property_metrics),
    }
    numeric_rollups: dict[str, list[float]] = {
        "n_records": [],
        "n_source_documents": [],
        "decision_accuracy": [],
        "false_supported_rate": [],
        "false_refuted_rate": [],
    }
    metric_fields = (
        "statement_property_label",
        "n_source_documents",
        "n_records",
        "n_true",
        "n_false",
        "decision_accuracy",
        "false_supported_rate",
        "false_refuted_rate",
        "true_supported_rate",
        "true_refuted_rate",
        "insufficient_evidence_rate",
        "decision_error_rate",
    )
    for property_id in sorted(str(key) for key in property_metrics):
        metrics = _mapping(property_metrics.get(property_id))
        prefix = f"covered_fact_property_{_metadata_key_component(property_id)}"
        for field in metric_fields:
            metadata[f"{prefix}_{field}"] = metrics.get(field)
        for field in numeric_rollups:
            value = metrics.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_rollups[field].append(float(value))

    metadata.update({
        "covered_fact_property_min_records": _min_or_none(numeric_rollups["n_records"]),
        "covered_fact_property_min_source_documents": _min_or_none(
            numeric_rollups["n_source_documents"]
        ),
        "covered_fact_property_min_decision_accuracy": _min_or_none(
            numeric_rollups["decision_accuracy"]
        ),
        "covered_fact_property_max_false_supported_rate": _max_or_none(
            numeric_rollups["false_supported_rate"]
        ),
        "covered_fact_property_min_false_refuted_rate": _min_or_none(
            numeric_rollups["false_refuted_rate"]
        ),
    })
    return metadata


def _score_dump_property_summary(
    *,
    documents: Sequence[Mapping[str, Any]],
    statements: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
) -> dict[str, dict[str, Any]]:
    by_property: dict[str, dict[str, Any]] = {}
    for document in documents:
        metadata = _mapping(document.get("metadata"))
        payload = by_property.setdefault(
            _statement_property_id(metadata),
            _empty_property_summary(metadata),
        )
        payload["n_source_documents"] += 1
    for statement, label in zip(statements, labels):
        metadata = _mapping(_mapping(statement.get("metadata")))
        payload = by_property.setdefault(
            _statement_property_id(metadata),
            _empty_property_summary(metadata),
        )
        payload["n_records"] += 1
        if int(label) == 1:
            payload["n_false"] += 1
        else:
            payload["n_true"] += 1
    return {
        key: by_property[key]
        for key in sorted(by_property)
    }


def _property_metrics_from_verified_records(
    *,
    statements: Sequence[Mapping[str, Any]],
    verified_records_path: Path,
    score_property_summary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    verified_records = _load_verified_records_jsonl(verified_records_path)
    if len(verified_records) != len(statements):
        raise ValueError(
            "verified records and score dump statements must have the same length "
            f"({len(verified_records)} != {len(statements)})."
        )
    by_property: dict[str, dict[str, Any]] = {}
    for statement, verified in zip(statements, verified_records):
        metadata = _mapping(_mapping(statement.get("metadata")))
        property_id = _statement_property_id(metadata)
        payload = by_property.setdefault(
            property_id,
            _empty_property_metrics(
                metadata,
                score_summary=_mapping(score_property_summary.get(property_id)),
            ),
        )
        label = int(verified.get("label", 0))
        record = _mapping(verified.get("record"))
        final = _mapping(record.get("final"))
        route = _mapping(record.get("route"))
        status = str(final.get("status", "unknown"))
        selected_route = str(route.get("selected_route", "unknown"))
        payload["n_records"] += 1
        payload["status_counts"][status] = payload["status_counts"].get(status, 0) + 1
        payload["selected_route_counts"][selected_route] = (
            payload["selected_route_counts"].get(selected_route, 0) + 1
        )
        label_key = "false" if label == 1 else "true"
        payload["label_status_matrix"][label_key][status] = (
            payload["label_status_matrix"][label_key].get(status, 0) + 1
        )
        if label == 1:
            payload["n_false"] += 1
        else:
            payload["n_true"] += 1
    for payload in by_property.values():
        _finalize_property_metrics(payload)
    return {
        key: by_property[key]
        for key in sorted(by_property)
    }


def _load_verified_records_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            records.append(dict(payload))
    return records


def _empty_property_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "statement_property": _statement_property_id(metadata),
        "statement_property_label": metadata.get("statement_property_label"),
        "n_source_documents": 0,
        "n_records": 0,
        "n_true": 0,
        "n_false": 0,
    }


def _empty_property_metrics(
    metadata: Mapping[str, Any],
    *,
    score_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "statement_property": _statement_property_id(metadata),
        "statement_property_label": metadata.get("statement_property_label")
        or score_summary.get("statement_property_label"),
        "n_source_documents": score_summary.get("n_source_documents", 0),
        "n_records": 0,
        "n_true": 0,
        "n_false": 0,
        "status_counts": {},
        "selected_route_counts": {},
        "label_status_matrix": {
            "true": {},
            "false": {},
        },
    }


def _finalize_property_metrics(payload: dict[str, Any]) -> None:
    matrix = payload["label_status_matrix"]
    true_total = int(payload["n_true"])
    false_total = int(payload["n_false"])
    true_supported = int(matrix["true"].get("supported", 0))
    true_refuted = int(matrix["true"].get("refuted", 0))
    false_supported = int(matrix["false"].get("supported", 0))
    false_refuted = int(matrix["false"].get("refuted", 0))
    insufficient = (
        int(matrix["true"].get("insufficient_evidence", 0))
        + int(matrix["false"].get("insufficient_evidence", 0))
    )
    decided = true_supported + true_refuted + false_supported + false_refuted
    correct = true_supported + false_refuted
    wrong = true_refuted + false_supported
    payload.update({
        "true_supported_rate": _safe_div(true_supported, true_total),
        "true_refuted_rate": _safe_div(true_refuted, true_total),
        "false_refuted_rate": _safe_div(false_refuted, false_total),
        "false_supported_rate": _safe_div(false_supported, false_total),
        "insufficient_evidence_rate": _safe_div(insufficient, int(payload["n_records"])),
        "decision_accuracy": _safe_div(correct, decided),
        "decision_error_rate": _safe_div(wrong, decided),
        "n_decided_supported_or_refuted": decided,
    })


def _statement_property_id(metadata: Mapping[str, Any]) -> str:
    return _clean_text(metadata.get("statement_property")) or "unknown"


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _qa_documents(qa_corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_documents = qa_corpus.get("documents", qa_corpus.get("records", ()))
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes, bytearray)):
        raise ValueError("qa_corpus must contain a documents or records list.")
    documents = []
    seen: set[tuple[str, str]] = set()
    for idx, item in enumerate(raw_documents, start=1):
        if not isinstance(item, Mapping):
            continue
        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("answer"))
        if question is None or answer is None:
            continue
        key = (normalize_claim_text(question), normalize_claim_text(answer))
        if key in seen:
            continue
        seen.add(key)
        documents.append({
            "question": question,
            "answer": answer,
            "source": item.get("source"),
            "metadata": dict(_mapping(item.get("metadata"))),
        })
    if not documents:
        raise ValueError("qa_corpus did not contain any usable question/answer documents.")
    return documents


def _known_answers_by_question(documents: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for document in documents:
        known.setdefault(
            normalize_claim_text(str(document["question"])),
            set(),
        ).add(normalize_claim_text(str(document["answer"])))
    return known


def _known_answer_values_by_question(documents: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    known: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for document in documents:
        question_key = normalize_claim_text(str(document["question"]))
        answer = str(document["answer"])
        answer_key = normalize_claim_text(answer)
        seen.setdefault(question_key, set())
        if answer_key in seen[question_key]:
            continue
        known.setdefault(question_key, []).append(answer)
        seen[question_key].add(answer_key)
    return {key: tuple(values) for key, values in known.items()}


def _false_answer_for(
    document: Mapping[str, Any],
    *,
    documents: Sequence[Mapping[str, Any]],
    known_answers: Mapping[str, set[str]],
) -> dict[str, Any] | None:
    question_key = normalize_claim_text(str(document["question"]))
    current_answer = normalize_claim_text(str(document["answer"]))
    target_property = _mapping(document.get("metadata")).get("statement_property")
    candidates = list(documents)
    if target_property is not None:
        same_property = [
            item for item in candidates
            if _mapping(item.get("metadata")).get("statement_property") == target_property
        ]
        if same_property:
            candidates = same_property
    for candidate in candidates:
        candidate_question_key = normalize_claim_text(str(candidate["question"]))
        candidate_answer_key = normalize_claim_text(str(candidate["answer"]))
        if candidate_question_key == question_key:
            continue
        if candidate_answer_key == current_answer:
            continue
        if candidate_answer_key in known_answers.get(question_key, set()):
            continue
        return {
            "answer": str(candidate["answer"]),
            "source": candidate.get("source"),
            "statement_property": _mapping(candidate.get("metadata")).get("statement_property"),
        }
    for candidate in documents:
        candidate_question_key = normalize_claim_text(str(candidate["question"]))
        candidate_answer_key = normalize_claim_text(str(candidate["answer"]))
        if candidate_question_key == question_key:
            continue
        if candidate_answer_key in known_answers.get(question_key, set()):
            continue
        return {
            "answer": str(candidate["answer"]),
            "source": candidate.get("source"),
            "statement_property": _mapping(candidate.get("metadata")).get("statement_property"),
        }
    return None


def _statements(
    *,
    question: str,
    answer: str,
    source: str | None,
    metadata: Mapping[str, Any],
    label_generation: str,
    statement_style: str,
    known_answers: Sequence[str] = (),
) -> list[dict[str, Any]]:
    if statement_style == "qa":
        templates = (("qa", f"{question} {answer}"),)
    elif statement_style == "natural_fact":
        templates = (("canonical", _natural_fact_statement(answer=answer, metadata=metadata)),)
    elif statement_style == "natural_fact_paraphrase":
        templates = _natural_fact_paraphrases(
            answer=answer,
            metadata=metadata,
            known_answers=known_answers,
        )
    else:
        raise ValueError(f"unsupported statement_style: {statement_style!r}")
    return [
        _statement_payload(
            question=question,
            answer=answer,
            text=text,
            source=source,
            metadata=metadata,
            label_generation=label_generation,
            statement_style=statement_style,
            claim_template_id=claim_template_id,
        )
        for claim_template_id, text in templates
    ]


def _statement(
    *,
    question: str,
    answer: str,
    source: str | None,
    metadata: Mapping[str, Any],
    label_generation: str,
    statement_style: str,
) -> dict[str, Any]:
    return _statements(
        question=question,
        answer=answer,
        source=source,
        metadata=metadata,
        label_generation=label_generation,
        statement_style=statement_style,
    )[0]


def _statement_payload(
    *,
    question: str,
    answer: str,
    text: str,
    source: str | None,
    metadata: Mapping[str, Any],
    label_generation: str,
    statement_style: str,
    claim_template_id: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "text": text,
        "metadata": {
            "provider": "wikidata",
            "source": source,
            "label_generation": label_generation,
            "statement_style": statement_style,
            "claim_template_id": claim_template_id,
            "statement_property": metadata.get("statement_property"),
            "statement_property_label": metadata.get("statement_property_label"),
            "question_template": metadata.get("question_template"),
            "answer_field": metadata.get("answer_field"),
            "country": metadata.get("country"),
            "country_qid": metadata.get("country_qid"),
            "value_qid": metadata.get("value_qid"),
            **{
                key: metadata[key]
                for key in STATEMENT_PROVENANCE_METADATA_KEYS
                if key in metadata and metadata[key] is not None
            },
            **{
                key: value for key, value in metadata.items()
                if key.startswith("false_answer_")
            },
        },
    }


def _natural_fact_paraphrases(
    *,
    answer: str,
    metadata: Mapping[str, Any],
    known_answers: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    country = _required_country(metadata)
    property_key = _property_key(metadata)
    if property_key == "p36" or property_key == "capital":
        return (
            ("canonical", f"{answer} is the capital of {country}."),
            ("subject_first", f"The capital of {country} is {answer}."),
            ("possessive", f"{country}'s capital is {answer}."),
        )
    if property_key == "p37" or property_key == "official language":
        templates = [
            ("canonical", f"{answer} is an official language of {country}."),
            ("subject_first", f"The official languages of {country} include {answer}."),
            ("possessive", f"{country}'s official languages include {answer}."),
        ]
        list_answer = _list_answer_for(answer, known_answers=known_answers)
        if list_answer is not None:
            templates.append(("object_list", f"The official languages of {country} include {list_answer}."))
        return tuple(templates)
    if property_key == "p38" or property_key == "currency":
        templates = [
            ("canonical", f"{answer} is a currency of {country}."),
            ("subject_first", f"The currency of {country} is {answer}."),
            ("possessive", f"{country}'s currency is {answer}."),
            ("uses_currency", f"{country} uses {answer} as its currency."),
        ]
        list_answer = _list_answer_for(answer, known_answers=known_answers)
        if list_answer is not None:
            templates.append(("object_list", f"The currencies of {country} include {list_answer}."))
        return tuple(templates)
    raise ValueError(f"unsupported natural_fact Wikidata property: {_property_label(metadata)!r}")


def _natural_fact_statement(*, answer: str, metadata: Mapping[str, Any]) -> str:
    country = _required_country(metadata)
    property_key = _property_key(metadata)
    if property_key == "p36" or property_key == "capital":
        return f"{answer} is the capital of {country}."
    if property_key == "p37" or property_key == "official language":
        return f"{answer} is an official language of {country}."
    if property_key == "p38" or property_key == "currency":
        return f"{answer} is a currency of {country}."
    raise ValueError(f"unsupported natural_fact Wikidata property: {_property_label(metadata)!r}")


def _statement_style_for_route(route: str, fact_claim_style: Any) -> str:
    if route != "structured_fact":
        return "qa"
    style = str(fact_claim_style).strip()
    if style == "canonical":
        return "natural_fact"
    if style == "paraphrase_robustness":
        return "natural_fact_paraphrase"
    raise ValueError("fact_claim_style must be canonical or paraphrase_robustness.")


def _normalize_route(value: Any) -> str:
    route = str(value).strip()
    if route not in {"structured_qa", "structured_fact"}:
        raise ValueError("route must be structured_qa or structured_fact.")
    return route


def _normalize_statement_style(value: Any) -> str:
    style = str(value).strip()
    if style not in {"qa", "natural_fact", "natural_fact_paraphrase"}:
        raise ValueError("statement_style must be qa, natural_fact, or natural_fact_paraphrase.")
    return style


def _required_country(metadata: Mapping[str, Any]) -> str:
    country = _clean_text(metadata.get("country"))
    if country is None:
        raise ValueError("natural_fact statements require metadata.country.")
    return country


def _property_label(metadata: Mapping[str, Any]) -> str:
    property_id = _clean_text(metadata.get("statement_property"))
    property_label = _clean_text(metadata.get("statement_property_label"))
    return property_id or property_label or ""


def _property_key(metadata: Mapping[str, Any]) -> str:
    return normalize_claim_text(_property_label(metadata))


def _list_answer_for(answer: str, *, known_answers: Sequence[str]) -> str | None:
    values: list[str] = []
    seen = set()
    answer_key = normalize_claim_text(answer)
    for value in known_answers:
        value_key = normalize_claim_text(str(value))
        if not value_key or value_key in seen:
            continue
        values.append(str(value))
        seen.add(value_key)
    if answer_key not in seen:
        if values:
            values = [values[0], answer]
        else:
            return None
    if len(values) == 1:
        return values[0]
    return _join_answer_list(values)


def _join_answer_list(values: Sequence[str]) -> str:
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _metadata_key_component(value: Any) -> str:
    text = str(value).strip().lower()
    key = "".join(char if char.isalnum() else "_" for char in text).strip("_")
    return key or "unknown"


def _min_or_none(values: Sequence[float]) -> float | None:
    return min(values) if values else None


def _max_or_none(values: Sequence[float]) -> float | None:
    return max(values) if values else None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if compact:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        else:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wikidata covered-facts structured QA route benchmark")
    parser.add_argument("--qa-corpus", required=True, help="Wikidata structured QA corpus JSON")
    parser.add_argument("--output-dir", required=True, help="directory for generated workflow artifacts")
    parser.add_argument("--score-name", default="wikidata-covered-facts")
    parser.add_argument(
        "--route",
        choices=("structured_qa", "structured_fact"),
        default="structured_qa",
        help="route to benchmark; structured_fact emits natural-language fact claims",
    )
    parser.add_argument(
        "--fact-claim-style",
        choices=("canonical", "paraphrase_robustness"),
        default="canonical",
        help="structured_fact claim generation style; paraphrase_robustness emits multiple natural-language variants",
    )
    parser.add_argument("--signal", default=DEFAULT_SIGNAL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="optional number of QA facts to consume")
    parser.add_argument("--score-dump-json", default=None)
    parser.add_argument("--verifier-report-json", default=None)
    parser.add_argument("--verified-records-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--json", default=None, help="optional route summary output path")
    parser.add_argument("--compact-json", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
