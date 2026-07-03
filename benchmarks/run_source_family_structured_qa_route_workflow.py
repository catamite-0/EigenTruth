"""Run a covered-facts structured QA route audit for source-family corpora.

The workflow consumes a label-free structured QA corpus, such as the output of
``build_source_family_qa_corpus.py``, generates balanced known-answer and
known-mismatch rows, and verifies them through the existing
``QuestionAnswerVerifier`` route. It is deliberately scoped to covered facts:
generated false rows are mismatched answers for questions present in the corpus,
not broad open-domain TruthfulQA labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.eval_verifier_ensemble import build_verifier_ensemble_report  # noqa: E402
from eigentruth.json_utils import strict_json_dumps, to_jsonable  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402
from eigentruth.verify import normalize_claim_text  # noqa: E402

WORKFLOW = "source_family_structured_qa_route_workflow"
DEFAULT_SIGNAL = "truth_proj"
DEFAULT_ALPHA = 0.10
ROUTE = "structured_qa"


def build_source_family_covered_fact_score_dump(
    qa_corpus: Mapping[str, Any],
    *,
    limit: int | None = None,
    signal: str = DEFAULT_SIGNAL,
) -> dict[str, Any]:
    """Return a balanced true/false score dump from a structured QA corpus."""
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

    for document in documents:
        question = str(document["question"])
        answer = str(document["answer"])
        metadata = dict(_mapping(document.get("metadata")))
        source = None if document.get("source") is None else str(document["source"])
        statements.append(
            _statement_payload(
                question=question,
                answer=answer,
                text=f"{question} {answer}",
                source=source,
                metadata=metadata,
                label_generation="source_family_known_answer",
                known_answers=known_answer_values.get(normalize_claim_text(question), ()),
            )
        )
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
        false_metadata = {
            **metadata,
            "false_answer_source": false_answer.get("source"),
            "false_answer_provider": false_answer.get("provider"),
            "false_answer_source_family": false_answer.get("source_family"),
            "false_answer_fact_group": false_answer.get("fact_group"),
        }
        statements.append(
            _statement_payload(
                question=question,
                answer=str(false_answer["answer"]),
                text=f"{question} {false_answer['answer']}",
                source=None if false_answer.get("source") is None else str(false_answer["source"]),
                metadata=false_metadata,
                label_generation="source_family_known_answer_mismatch",
                known_answers=known_answer_values.get(normalize_claim_text(question), ()),
            )
        )
        labels.append(1)
        scores.append(0.0)

    if not any(label == 1 for label in labels):
        raise ValueError("no false-answer rows could be generated.")

    group_summary = _score_dump_group_summary(documents=documents, statements=statements, labels=labels)
    provider_counts = Counter(_provider_from_metadata(_mapping(doc.get("metadata"))) for doc in documents)
    family_counts = Counter(_source_family_from_metadata(_mapping(doc.get("metadata"))) for doc in documents)
    return {
        "config": {
            "model": "source-family-covered-facts",
            "layer": -1,
            "workflow": WORKFLOW,
            "signal": signal,
            "statement_style": "qa",
            "label_semantics": {
                "0": "known source-family structured QA answer",
                "1": "answer mismatches known structured QA answer(s) for the same question",
            },
            "source": {
                "corpus_type": qa_corpus.get("corpus_type"),
                "builder": _mapping(qa_corpus.get("source")).get("builder"),
                "accepted_providers": _mapping(qa_corpus.get("source")).get("accepted_providers"),
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
            "provider_count": len(provider_counts),
            "source_family_count": len(family_counts),
            "fact_group_count": len(group_summary),
            "by_provider": dict(sorted(provider_counts.items())),
            "by_source_family": dict(sorted(family_counts.items())),
            "by_fact_group": group_summary,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    qa_corpus_path = Path(args.qa_corpus)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(args.alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if args.registry is not None and (not args.name or not args.version):
        raise ValueError("--registry requires --name and --version.")
    metadata = _parse_metadata(getattr(args, "metadata", ()) or ())

    qa_corpus = json.loads(qa_corpus_path.read_text(encoding="utf-8"))
    if not isinstance(qa_corpus, Mapping):
        raise ValueError("qa_corpus must be a JSON object.")

    signal = str(args.signal)
    score_dump = build_source_family_covered_fact_score_dump(
        qa_corpus,
        limit=args.limit,
        signal=signal,
    )
    score_dump_path = Path(args.score_dump_json or output_dir / "covered-facts-scores.json")
    verifier_report_path = Path(args.verifier_report_json or output_dir / "structured-qa-verifier-report.json")
    verified_records_path = Path(args.verified_records_jsonl or output_dir / "verified-records.jsonl")
    summary_path = Path(args.json or output_dir / "structured-qa-route-summary.json")
    manifest_path = Path(args.artifact_manifest or output_dir / "artifact-manifest.json")

    _write_json(score_dump_path, score_dump, compact=bool(args.compact_json))
    verifier_report = build_verifier_ensemble_report(
        ((str(args.score_name), score_dump_path),),
        signal=signal,
        qa_corpus_path=qa_corpus_path,
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
        qa_corpus=qa_corpus,
        score_dump=score_dump,
        verifier_report=verifier_report,
        score_name=str(args.score_name),
        signal=signal,
        alpha=alpha,
        metadata=metadata,
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
            "workflow": WORKFLOW,
            "route": ROUTE,
            "status": summary["status"],
            "score_name": str(args.score_name),
            "signal": signal,
            "alpha": alpha,
            "n_records": score_dump["summary"]["n_records"],
            "n_true": score_dump["summary"]["n_true"],
            "n_false": score_dump["summary"]["n_false"],
            "provider_count": summary["provider_count"],
            "source_family_count": summary["source_family_count"],
            "fact_group_count": summary["fact_group_count"],
            "promotes_covered_facts_route": summary["status"] == "promote",
            "structured_qa_decision_accuracy": summary["structured_qa_metrics"].get("decision_accuracy"),
            "structured_qa_false_supported_rate": summary["structured_qa_metrics"].get(
                "false_supported_rate"
            ),
            "structured_qa_false_refuted_rate": summary["structured_qa_metrics"].get("false_refuted_rate"),
            **metadata,
        },
    )
    _write_json(manifest_path, manifest, compact=False)
    if args.registry is not None:
        assert args.name is not None and args.version is not None
        ArtifactRegistry.load_json(args.registry).record_report(
            name=str(args.name),
            version=str(args.version),
            path=summary_path,
            metadata={
                "workflow": WORKFLOW,
                "status": summary["status"],
                "route": ROUTE,
                "artifact_manifest": str(manifest_path),
                "n_records": score_dump["summary"]["n_records"],
                "provider_count": summary["provider_count"],
                "source_family_count": summary["source_family_count"],
                "fact_group_count": summary["fact_group_count"],
                **metadata,
            },
        ).save_json()
    print(
        "source_family_structured_qa_route_workflow_ok "
        f"route={ROUTE} status={summary['status']} "
        f"records={score_dump['summary']['n_records']} output={summary_path}"
    )
    return summary


def _summary_payload(
    *,
    qa_corpus_path: Path,
    score_dump_path: Path,
    verifier_report_path: Path,
    verified_records_path: Path,
    qa_corpus: Mapping[str, Any],
    score_dump: Mapping[str, Any],
    verifier_report: Mapping[str, Any],
    score_name: str,
    signal: str,
    alpha: float,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    runs = verifier_report.get("runs", ())
    if not runs:
        raise ValueError("verifier report did not contain any runs.")
    run = _mapping(runs[0])
    route_quality = _mapping(run.get("route_quality"))
    route_metrics = _mapping(route_quality.get(ROUTE))
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
    grouped_metrics = _group_metrics_from_verified_records(
        statements=score_dump["statements"],
        verified_records_path=verified_records_path,
    )
    provider_metrics = _rollup_group_metrics(grouped_metrics, key_field="provider")
    source_family_metrics = _rollup_group_metrics(grouped_metrics, key_field="source_family")
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "route": ROUTE,
        "scope": "covered source-family structured QA facts, not open-domain TruthfulQA route coverage",
        "score_name": score_name,
        "signal": signal,
        "alpha": alpha,
        "qa_corpus_path": str(qa_corpus_path),
        "covered_fact_score_dump_path": str(score_dump_path),
        "verifier_report_path": str(verifier_report_path),
        "verified_records_jsonl_path": str(verified_records_path),
        "qa_corpus_summary": dict(_mapping(qa_corpus.get("summary"))),
        "score_dump_summary": dict(score_dump["summary"]),
        "metadata": dict(metadata),
        "selected_route_counts": dict(selected_counts),
        "route_metrics": dict(route_metrics),
        "structured_qa_metrics": dict(route_metrics),
        "provider_count": len(provider_metrics),
        "source_family_count": len(source_family_metrics),
        "fact_group_count": len(grouped_metrics),
        "provider_metrics": provider_metrics,
        "source_family_metrics": source_family_metrics,
        "fact_group_metrics": grouped_metrics,
        "verification_status_counts": dict(_mapping(run.get("verification_status_counts"))),
        "qa_verifier": _mapping(run.get("qa")),
        "next_step": (
            "Use this only as covered-fact route-quality evidence. Map product claims "
            "or blind spots into these exact questions before creating a correction handoff."
        ),
    }


def _group_metrics_from_verified_records(
    *,
    statements: Sequence[Mapping[str, Any]],
    verified_records_path: Path,
) -> dict[str, dict[str, Any]]:
    verified_records = _load_verified_records_jsonl(verified_records_path)
    if len(verified_records) != len(statements):
        raise ValueError(
            "verified records and score dump statements must have the same length "
            f"({len(verified_records)} != {len(statements)})."
        )
    by_group: dict[str, dict[str, Any]] = {}
    for statement, verified in zip(statements, verified_records):
        metadata = _mapping(_mapping(statement.get("metadata")))
        group_id = _fact_group(metadata)
        payload = by_group.setdefault(group_id, _empty_group_metrics(metadata))
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
    for payload in by_group.values():
        _finalize_metrics(payload)
    return {key: by_group[key] for key in sorted(by_group)}


def _rollup_group_metrics(grouped_metrics: Mapping[str, Mapping[str, Any]], *, key_field: str) -> dict[str, Any]:
    rolled: dict[str, dict[str, Any]] = {}
    for metrics in grouped_metrics.values():
        key = str(metrics.get(key_field, "unknown"))
        payload = rolled.setdefault(
            key,
            {
                key_field: key,
                "n_records": 0,
                "n_true": 0,
                "n_false": 0,
                "status_counts": {},
                "selected_route_counts": {},
                "label_status_matrix": {"true": {}, "false": {}},
            },
        )
        payload["n_records"] += int(metrics.get("n_records", 0))
        payload["n_true"] += int(metrics.get("n_true", 0))
        payload["n_false"] += int(metrics.get("n_false", 0))
        _merge_counts(payload["status_counts"], _mapping(metrics.get("status_counts")))
        _merge_counts(payload["selected_route_counts"], _mapping(metrics.get("selected_route_counts")))
        _merge_counts(
            payload["label_status_matrix"]["true"],
            _mapping(_mapping(metrics.get("label_status_matrix")).get("true")),
        )
        _merge_counts(
            payload["label_status_matrix"]["false"],
            _mapping(_mapping(metrics.get("label_status_matrix")).get("false")),
        )
    for payload in rolled.values():
        _finalize_metrics(payload)
    return {key: rolled[key] for key in sorted(rolled)}


def _score_dump_group_summary(
    *,
    documents: Sequence[Mapping[str, Any]],
    statements: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
) -> dict[str, dict[str, Any]]:
    by_group: dict[str, dict[str, Any]] = {}
    for document in documents:
        metadata = _mapping(document.get("metadata"))
        payload = by_group.setdefault(_fact_group(metadata), _empty_group_summary(metadata))
        payload["n_source_documents"] += 1
    for statement, label in zip(statements, labels):
        metadata = _mapping(_mapping(statement.get("metadata")))
        payload = by_group.setdefault(_fact_group(metadata), _empty_group_summary(metadata))
        payload["n_records"] += 1
        if int(label) == 1:
            payload["n_false"] += 1
        else:
            payload["n_true"] += 1
    return {key: by_group[key] for key in sorted(by_group)}


def _empty_group_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fact_group": _fact_group(metadata),
        "provider": _provider_from_metadata(metadata),
        "source_family": _source_family_from_metadata(metadata),
        "fact_type": _fact_type(metadata),
        "n_source_documents": 0,
        "n_records": 0,
        "n_true": 0,
        "n_false": 0,
    }


def _empty_group_metrics(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = _empty_group_summary(metadata)
    payload.update({
        "status_counts": {},
        "selected_route_counts": {},
        "label_status_matrix": {"true": {}, "false": {}},
    })
    return payload


def _finalize_metrics(payload: dict[str, Any]) -> None:
    matrix = _mapping(payload.get("label_status_matrix"))
    true_matrix = _mapping(matrix.get("true"))
    false_matrix = _mapping(matrix.get("false"))
    true_total = int(payload["n_true"])
    false_total = int(payload["n_false"])
    true_supported = int(true_matrix.get("supported", 0))
    true_refuted = int(true_matrix.get("refuted", 0))
    false_supported = int(false_matrix.get("supported", 0))
    false_refuted = int(false_matrix.get("refuted", 0))
    insufficient = (
        int(true_matrix.get("insufficient_evidence", 0))
        + int(false_matrix.get("insufficient_evidence", 0))
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


def _qa_documents(qa_corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_documents = qa_corpus.get("documents", qa_corpus.get("records", ()))
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes, bytearray)):
        raise ValueError("qa_corpus must contain a documents or records list.")
    documents = []
    seen: set[tuple[str, str]] = set()
    for item in raw_documents:
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
        metadata = dict(_mapping(item.get("metadata")))
        metadata.setdefault("source", item.get("source"))
        documents.append({
            "question": question,
            "answer": answer,
            "source": item.get("source"),
            "metadata": metadata,
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
    metadata = _mapping(document.get("metadata"))
    provider = _provider_from_metadata(metadata)
    tiers = (
        [item for item in documents if _fact_group(_mapping(item.get("metadata"))) == _fact_group(metadata)],
        [
            item for item in documents
            if _provider_from_metadata(_mapping(item.get("metadata"))) == provider
        ],
        list(documents),
    )
    for candidates in tiers:
        for candidate in candidates:
            candidate_question_key = normalize_claim_text(str(candidate["question"]))
            candidate_answer_key = normalize_claim_text(str(candidate["answer"]))
            if candidate_question_key == question_key:
                continue
            if candidate_answer_key == current_answer:
                continue
            if candidate_answer_key in known_answers.get(question_key, set()):
                continue
            candidate_metadata = _mapping(candidate.get("metadata"))
            return {
                "answer": str(candidate["answer"]),
                "source": candidate.get("source"),
                "provider": _provider_from_metadata(candidate_metadata),
                "source_family": _source_family_from_metadata(candidate_metadata),
                "fact_group": _fact_group(candidate_metadata),
            }
    return None


def _statement_payload(
    *,
    question: str,
    answer: str,
    text: str,
    source: str | None,
    metadata: Mapping[str, Any],
    label_generation: str,
    known_answers: Sequence[str],
) -> dict[str, Any]:
    safe_metadata = _jsonable_mapping(metadata)
    safe_metadata.update({
        "source": source,
        "label_generation": label_generation,
        "statement_style": "qa",
        "claim_template_id": "qa",
        "fact_group": _fact_group(metadata),
        "fact_type": _fact_type(metadata),
        "provider": _provider_from_metadata(metadata),
        "source_family": _source_family_from_metadata(metadata),
        "known_answers": tuple(str(value) for value in known_answers),
    })
    return {
        "question": question,
        "answer": answer,
        "text": text,
        "metadata": safe_metadata,
    }


def _fact_group(metadata: Mapping[str, Any]) -> str:
    return ":".join((
        _metadata_key_component(_provider_from_metadata(metadata)),
        _metadata_key_component(_source_family_from_metadata(metadata)),
        _metadata_key_component(_fact_type(metadata)),
    ))


def _fact_type(metadata: Mapping[str, Any]) -> str:
    return (
        _clean_text(metadata.get("statement_property"))
        or _clean_text(metadata.get("indicator"))
        or _clean_text(metadata.get("extraction_rule"))
        or "unknown_fact"
    )


def _provider_from_metadata(metadata: Mapping[str, Any]) -> str:
    return _clean_text(metadata.get("provider")) or "unknown_provider"


def _source_family_from_metadata(metadata: Mapping[str, Any]) -> str:
    return _clean_text(metadata.get("source_family")) or "unknown_family"


def _jsonable_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): to_jsonable(value)
        for key, value in metadata.items()
        if value is not None
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


def _merge_counts(target: dict[str, int], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _metadata_key_component(value: Any) -> str:
    text = str(value).strip().lower()
    key = "".join(char if char.isalnum() else "_" for char in text).strip("_")
    return key or "unknown"


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-corpus", required=True, help="source-family structured QA corpus JSON")
    parser.add_argument("--output-dir", required=True, help="directory for generated workflow artifacts")
    parser.add_argument("--score-name", default="source-family-covered-facts")
    parser.add_argument("--signal", default=DEFAULT_SIGNAL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="optional number of QA facts to consume")
    parser.add_argument("--score-dump-json", default=None)
    parser.add_argument("--verifier-report-json", default=None)
    parser.add_argument("--verified-records-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--json", default=None, help="optional route summary output path")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
