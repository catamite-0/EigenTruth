"""Audit retrieval corpus provenance before treating evidence as grounding.

This is a dependency-free gate for local retrieval experiments. It checks
whether a supplied corpus looks like external/domain evidence, a controlled
dataset-derived baseline, or an answer-echo/oracle-risk stress corpus. The audit
does not retrieve or verify claims; it only inspects source score-dump
statements, corpus metadata, document text, and artifact fingerprints.
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

from eigentruth.eval.score_dump import ScoreDump, load_score_dump, score_dump_file_metadata  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402

AUDIT_ROLES = ("grounding", "controlled_baseline", "stress_control")
ANSWER_ECHO_CORPUS_TYPE = "retrieval_stress_answer_echo"
CONTROLLED_DATASET_CORPUS_TYPES = {"truthfulqa_correct_answer_evidence"}
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_TEXT_FIELDS = ("text", "content", "document", "answer", "question_answer")
_LABEL_METADATA_KEYS = ("score_label", "label", "labels")
_ROW_LINK_KEYS = ("row_index", "source_index", "record_index")


def audit_retrieval_corpus_provenance(
    score_dump: ScoreDump,
    *,
    scores_path: str | Path,
    corpus_paths: Sequence[str | Path],
    audit_role: str = "grounding",
    max_exact_answer_copy_rate: float = 0.80,
    max_claim_id_link_rate: float = 0.0,
    max_label_metadata_rate: float = 0.0,
) -> dict[str, Any]:
    """Return a structured provenance audit for one or more retrieval corpora."""
    if audit_role not in AUDIT_ROLES:
        raise ValueError(f"audit_role must be one of: {', '.join(AUDIT_ROLES)}.")
    _validate_rate(max_exact_answer_copy_rate, name="max_exact_answer_copy_rate")
    _validate_rate(max_claim_id_link_rate, name="max_claim_id_link_rate")
    _validate_rate(max_label_metadata_rate, name="max_label_metadata_rate")
    if not corpus_paths:
        raise ValueError("corpus_paths must contain at least one path.")
    if not score_dump.statements:
        raise ValueError("retrieval corpus provenance audit requires statement-bearing score dumps.")

    sources = _source_records(score_dump)
    corpora = [_load_corpus(path) for path in corpus_paths]
    documents = [document for corpus in corpora for document in corpus["documents"]]
    if not documents:
        raise ValueError("retrieval corpus provenance audit requires at least one corpus document.")

    document_reports = [_document_report(document, sources) for document in documents]
    summary = _summary(corpora, document_reports)
    thresholds = {
        "max_exact_answer_copy_rate": float(max_exact_answer_copy_rate),
        "max_claim_id_link_rate": float(max_claim_id_link_rate),
        "max_label_metadata_rate": float(max_label_metadata_rate),
    }
    evidence_class = _evidence_class(summary)
    gate = _gate(
        audit_role=audit_role,
        evidence_class=evidence_class,
        summary=summary,
        thresholds=thresholds,
    )
    return {
        "schema_version": 1,
        "workflow": "retrieval_corpus_provenance_audit",
        "audit_role": audit_role,
        "evidence_class": evidence_class,
        "status": gate["status"],
        "passed": gate["passed"],
        "gate": gate,
        "thresholds": thresholds,
        "summary": summary,
        "input_provenance": {
            "score_dump": score_dump_file_metadata(scores_path, score_dump),
            "corpora": [
                {
                    "path": str(corpus["path"]),
                    "corpus_type": corpus["corpus_type"],
                    "label_usage": corpus["label_usage"],
                    "n_documents": len(corpus["documents"]),
                }
                for corpus in corpora
            ],
        },
        "document_samples": document_reports[:10],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    scores_path = Path(args.scores)
    score_dump = load_score_dump(
        scores_path,
        allow_missing_scores=True,
        require_statements=True,
    )
    report = audit_retrieval_corpus_provenance(
        score_dump,
        scores_path=scores_path,
        corpus_paths=tuple(Path(path) for path in args.corpus),
        audit_role=args.audit_role,
        max_exact_answer_copy_rate=args.max_exact_answer_copy_rate,
        max_claim_id_link_rate=args.max_claim_id_link_rate,
        max_label_metadata_rate=args.max_label_metadata_rate,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.artifact_manifest:
        manifest_path = Path(args.artifact_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "scores": scores_path,
            "provenance_report": output_path,
        }
        for idx, path in enumerate(tuple(Path(path) for path in args.corpus), start=1):
            artifacts[f"corpus.{idx}.{path.stem}"] = path
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": "retrieval_corpus_provenance_audit",
                "audit_role": args.audit_role,
                "evidence_class": report["evidence_class"],
                "status": report["status"],
                "passed": report["passed"],
                "summary": report["summary"],
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "retrieval_corpus_provenance_audit_ok "
        f"status={report['status']} class={report['evidence_class']} output={output_path}"
    )
    return report


def _load_corpus(path: str | Path) -> dict[str, Any]:
    corpus_path = Path(path)
    if corpus_path.suffix.lower() == ".jsonl":
        documents = []
        with corpus_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = {"text": line, "source": f"{corpus_path}:{line_no}"}
                documents.append(_coerce_document(item, source_default=f"{corpus_path}:{line_no}"))
        payload: Mapping[str, Any] = {}
    elif corpus_path.suffix.lower() == ".json":
        with corpus_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, Mapping):
            payload = loaded
            raw_documents = loaded.get("documents", loaded.get("records", ()))
        elif isinstance(loaded, Sequence) and not isinstance(loaded, (str, bytes, bytearray)):
            payload = {}
            raw_documents = loaded
        else:
            raise ValueError(f"corpus {corpus_path} must be a JSON object or list.")
        documents = [_coerce_document(item, source_default=str(corpus_path)) for item in raw_documents]
    else:
        payload = {}
        documents = []
        with corpus_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if text:
                    documents.append({
                        "text": text,
                        "source": f"{corpus_path}:{line_no}",
                        "metadata": {},
                    })
    return {
        "path": corpus_path,
        "corpus_type": None if payload.get("corpus_type") is None else str(payload.get("corpus_type")),
        "label_usage": _mapping(payload.get("label_usage")),
        "documents": documents,
    }


def _coerce_document(value: Any, *, source_default: str) -> dict[str, Any]:
    if isinstance(value, str):
        return {"text": value, "source": source_default, "metadata": {}}
    if not isinstance(value, Mapping):
        raise ValueError("corpus document must be a string or JSON object.")
    text = ""
    for key in _TEXT_FIELDS:
        if value.get(key):
            text = str(value[key]).strip()
            break
    if not text:
        raise ValueError("corpus document is missing text/content/document/answer.")
    return {
        "text": text,
        "source": str(value.get("source", source_default)),
        "metadata": _mapping(value.get("metadata")),
    }


def _source_records(score_dump: ScoreDump) -> tuple[dict[str, Any], ...]:
    records = []
    for idx, statement in enumerate(score_dump.statements):
        claim_id = str(statement.get("claim_id") or f"c{idx + 1}")
        answer = str(statement.get("answer") or statement.get("text") or statement.get("claim") or "")
        text = str(statement.get("claim") or statement.get("text") or statement.get("answer") or "")
        question = str(statement.get("question", ""))
        records.append({
            "index": idx,
            "claim_id": claim_id,
            "answer": answer,
            "text": text,
            "question": question,
            "answer_tokens": set(_tokens(answer)),
            "text_tokens": set(_tokens(text)),
            "answer_norm": _norm(answer),
            "text_norm": _norm(text),
        })
    return tuple(records)


def _document_report(document: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text = str(document.get("text", ""))
    metadata = _mapping(document.get("metadata"))
    text_norm = _norm(text)
    text_tokens = set(_tokens(text))
    exact_answer_matches = []
    answer_overlaps = []
    source_text_matches = []
    for source in sources:
        answer_norm = str(source.get("answer_norm", ""))
        source_text_norm = str(source.get("text_norm", ""))
        if answer_norm and len(answer_norm) >= 8 and (answer_norm in text_norm or text_norm in answer_norm):
            exact_answer_matches.append(source["claim_id"])
        if source_text_norm and len(source_text_norm) >= 12 and (
            source_text_norm in text_norm or text_norm in source_text_norm
        ):
            source_text_matches.append(source["claim_id"])
        answer_tokens = set(source.get("answer_tokens", set()))
        if answer_tokens:
            answer_overlaps.append(len(text_tokens & answer_tokens) / len(answer_tokens))
    claim_id = metadata.get("claim_id")
    has_claim_id_link = claim_id is not None and str(claim_id) in {str(source["claim_id"]) for source in sources}
    has_row_link = any(key in metadata for key in _ROW_LINK_KEYS)
    has_label_metadata = any(key in metadata for key in _LABEL_METADATA_KEYS)
    stress_control = metadata.get("stress_control")
    return {
        "source": str(document.get("source", "")),
        "metadata_keys": sorted(str(key) for key in metadata),
        "has_claim_id_link": bool(has_claim_id_link),
        "has_row_link": bool(has_row_link),
        "has_label_metadata": bool(has_label_metadata),
        "stress_control": None if stress_control is None else str(stress_control),
        "exact_answer_match_count": len(exact_answer_matches),
        "source_text_match_count": len(source_text_matches),
        "max_answer_token_overlap": max(answer_overlaps) if answer_overlaps else 0.0,
        "matched_claim_ids": exact_answer_matches[:5],
    }


def _summary(corpora: Sequence[Mapping[str, Any]], document_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_documents = len(document_reports)
    corpus_types = sorted(
        str(corpus.get("corpus_type"))
        for corpus in corpora
        if corpus.get("corpus_type") is not None
    )
    top_label_usage = tuple(_mapping(corpus.get("label_usage")) for corpus in corpora)
    labels_used_for_documents = any(item.get("labels_used_for_documents") is True for item in top_label_usage)
    labels_copied_to_document_metadata = any(
        item.get("labels_copied_to_document_metadata") is True
        for item in top_label_usage
    )
    stress_control_documents = sum(
        1 for report in document_reports
        if report.get("stress_control") is not None
    )
    exact_answer_copy_documents = sum(
        1 for report in document_reports
        if int(report.get("exact_answer_match_count", 0)) > 0
    )
    source_text_copy_documents = sum(
        1 for report in document_reports
        if int(report.get("source_text_match_count", 0)) > 0
    )
    claim_id_link_documents = sum(1 for report in document_reports if report.get("has_claim_id_link"))
    row_link_documents = sum(1 for report in document_reports if report.get("has_row_link"))
    label_metadata_documents = sum(1 for report in document_reports if report.get("has_label_metadata"))
    return {
        "n_corpora": len(corpora),
        "n_documents": n_documents,
        "corpus_types": corpus_types,
        "labels_used_for_documents": labels_used_for_documents,
        "labels_copied_to_document_metadata": labels_copied_to_document_metadata,
        "stress_control_documents": stress_control_documents,
        "stress_control_rate": _rate(stress_control_documents, n_documents),
        "exact_answer_copy_documents": exact_answer_copy_documents,
        "exact_answer_copy_rate": _rate(exact_answer_copy_documents, n_documents),
        "source_text_copy_documents": source_text_copy_documents,
        "source_text_copy_rate": _rate(source_text_copy_documents, n_documents),
        "claim_id_link_documents": claim_id_link_documents,
        "claim_id_link_rate": _rate(claim_id_link_documents, n_documents),
        "row_link_documents": row_link_documents,
        "row_link_rate": _rate(row_link_documents, n_documents),
        "label_metadata_documents": label_metadata_documents,
        "label_metadata_rate": _rate(label_metadata_documents, n_documents),
        "mean_max_answer_token_overlap": (
            sum(float(report.get("max_answer_token_overlap", 0.0)) for report in document_reports) / n_documents
            if n_documents else 0.0
        ),
    }


def _evidence_class(summary: Mapping[str, Any]) -> str:
    corpus_types = set(str(item) for item in summary.get("corpus_types", ()))
    if ANSWER_ECHO_CORPUS_TYPE in corpus_types or int(summary.get("stress_control_documents", 0)) > 0:
        return "answer_echo_stress_control"
    if corpus_types & CONTROLLED_DATASET_CORPUS_TYPES:
        return "controlled_dataset_baseline"
    return "external_candidate"


def _gate(
    *,
    audit_role: str,
    evidence_class: str,
    summary: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    blocking = []
    warnings = []
    if summary.get("labels_used_for_documents") is True:
        blocking.append("labels_used_for_documents must be false")
    if summary.get("labels_copied_to_document_metadata") is True:
        blocking.append("labels_copied_to_document_metadata must be false")
    _check_rate(
        blocking,
        "label_metadata_rate",
        float(summary.get("label_metadata_rate", 0.0)),
        float(thresholds["max_label_metadata_rate"]),
    )
    _check_rate(
        blocking,
        "claim_id_link_rate",
        float(summary.get("claim_id_link_rate", 0.0)),
        float(thresholds["max_claim_id_link_rate"]),
    )
    if evidence_class == "answer_echo_stress_control":
        warnings.append("corpus is an answer-echo stress control, not grounding evidence")
    if evidence_class == "controlled_dataset_baseline":
        warnings.append("corpus is a controlled dataset-derived baseline, not external/domain-shifted evidence")
    exact_answer_copy_rate = float(summary.get("exact_answer_copy_rate", 0.0))
    if audit_role == "grounding" and exact_answer_copy_rate > float(thresholds["max_exact_answer_copy_rate"]):
        blocking.append(f"exact_answer_copy_rate above {thresholds['max_exact_answer_copy_rate']}")
    elif exact_answer_copy_rate > 0.0:
        warnings.append("some corpus documents copy source answer text")

    if audit_role == "grounding":
        if evidence_class != "external_candidate":
            blocking.append(f"evidence_class is {evidence_class!r}, expected 'external_candidate'")
    elif audit_role == "controlled_baseline":
        if evidence_class == "answer_echo_stress_control":
            blocking.append("answer-echo stress controls are not controlled grounding baselines")
    elif audit_role == "stress_control":
        if evidence_class != "answer_echo_stress_control":
            blocking.append(f"evidence_class is {evidence_class!r}, expected 'answer_echo_stress_control'")
    passed = not blocking
    return {
        "role": audit_role,
        "passed": passed,
        "status": "pass" if passed else "fail",
        "blocking_reasons": blocking,
        "warnings": warnings,
        "external_domain_shift_ready": passed and evidence_class == "external_candidate",
    }


def _check_rate(blocking: list[str], metric: str, value: float, limit: float) -> None:
    if value > limit:
        blocking.append(f"{metric} above {limit}")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(str(text)))


def _norm(text: str) -> str:
    return " ".join(_tokens(text))


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _validate_rate(value: float, *, name: str) -> None:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit retrieval corpus provenance for grounding claims")
    parser.add_argument("--scores", required=True, help="statement-bearing score dump")
    parser.add_argument("--corpus", action="append", required=True, help="corpus JSON/JSONL/text path; repeatable")
    parser.add_argument("--output", required=True, help="audit report JSON path")
    parser.add_argument("--audit-role", choices=AUDIT_ROLES, default="grounding")
    parser.add_argument("--max-exact-answer-copy-rate", type=float, default=0.80)
    parser.add_argument("--max-claim-id-link-rate", type=float, default=0.0)
    parser.add_argument("--max-label-metadata-rate", type=float, default=0.0)
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest for scores/corpus/report")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
