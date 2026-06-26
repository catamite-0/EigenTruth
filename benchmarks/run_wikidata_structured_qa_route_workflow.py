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
    statements: list[dict[str, Any]] = []
    labels: list[int] = []
    scores: list[float] = []
    skipped_false_answer = 0
    for idx, document in enumerate(documents):
        question = str(document["question"])
        answer = str(document["answer"])
        metadata = dict(document.get("metadata", {}))
        source = document.get("source")
        true_statement = _statement(
            question=question,
            answer=answer,
            source=None if source is None else str(source),
            metadata=metadata,
            label_generation="wikidata_known_answer",
            statement_style=statement_style,
        )
        statements.append(true_statement)
        labels.append(0)
        scores.append(0.0)

        false_answer = _false_answer_for(
            document,
            documents=documents,
            known_answers=known_answers,
        )
        if false_answer is None:
            skipped_false_answer += 1
            continue
        false_statement = _statement(
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
        )
        statements.append(false_statement)
        labels.append(1)
        scores.append(0.0)

    if not any(label == 1 for label in labels):
        raise ValueError("no false-answer rows could be generated.")

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
        statement_style="natural_fact" if route == "structured_fact" else "qa",
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
            f"{route}_decision_accuracy": summary[f"{route}_metrics"].get("decision_accuracy"),
            f"{route}_false_supported_rate": summary[f"{route}_metrics"].get("false_supported_rate"),
            f"{route}_false_refuted_rate": summary[f"{route}_metrics"].get("false_refuted_rate"),
            "promotes_covered_facts_route": summary["status"] == "promote",
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
        "verification_status_counts": dict(_mapping(run.get("verification_status_counts"))),
        "qa_verifier": _mapping(run.get("qa")),
        "fact_verifier": _mapping(run.get("fact")),
        "next_step": (
            "Use this covered-facts route as the property-level correction path; "
            "keep lexical retrieval gated separately for broad open-domain coverage."
        ),
    }


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


def _statement(
    *,
    question: str,
    answer: str,
    source: str | None,
    metadata: Mapping[str, Any],
    label_generation: str,
    statement_style: str,
) -> dict[str, Any]:
    text = (
        _natural_fact_statement(answer=answer, metadata=metadata)
        if statement_style == "natural_fact"
        else f"{question} {answer}"
    )
    return {
        "question": question,
        "answer": answer,
        "text": text,
        "metadata": {
            "provider": "wikidata",
            "source": source,
            "label_generation": label_generation,
            "statement_style": statement_style,
            "statement_property": metadata.get("statement_property"),
            "statement_property_label": metadata.get("statement_property_label"),
            "question_template": metadata.get("question_template"),
            "answer_field": metadata.get("answer_field"),
            "country": metadata.get("country"),
            "country_qid": metadata.get("country_qid"),
            "value_qid": metadata.get("value_qid"),
            **{
                key: value for key, value in metadata.items()
                if key.startswith("false_answer_")
            },
        },
    }


def _natural_fact_statement(*, answer: str, metadata: Mapping[str, Any]) -> str:
    country = _clean_text(metadata.get("country"))
    if country is None:
        raise ValueError("natural_fact statements require metadata.country.")
    property_id = _clean_text(metadata.get("statement_property"))
    property_label = _clean_text(metadata.get("statement_property_label"))
    property_key = normalize_claim_text(property_id or property_label or "")
    if property_key == "p36" or property_key == "capital":
        return f"{answer} is the capital of {country}."
    if property_key == "p37" or property_key == "official language":
        return f"{answer} is an official language of {country}."
    if property_key == "p38" or property_key == "currency":
        return f"{answer} is a currency of {country}."
    raise ValueError(f"unsupported natural_fact Wikidata property: {property_id or property_label!r}")


def _normalize_route(value: Any) -> str:
    route = str(value).strip()
    if route not in {"structured_qa", "structured_fact"}:
        raise ValueError("route must be structured_qa or structured_fact.")
    return route


def _normalize_statement_style(value: Any) -> str:
    style = str(value).strip()
    if style not in {"qa", "natural_fact"}:
        raise ValueError("statement_style must be qa or natural_fact.")
    return style


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
