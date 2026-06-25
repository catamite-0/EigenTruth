"""Build retrieval stress corpora from statement-bearing score dumps.

The default corpus is an answer-echo corpus: every scored statement contributes
its own answer text as a retrievable document. This is intentionally a negative
control for local retrieval experiments. If a verifier looks strong only when it
retrieves from the same model answers it is auditing, the run is measuring
self-support rather than grounded factuality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.eval.score_dump import ScoreDump, load_score_dump, score_dump_file_metadata  # noqa: E402

DOCUMENT_FIELDS = ("answer", "question_answer", "text")


def build_retrieval_stress_corpus(
    score_dump: ScoreDump,
    *,
    source_scores_path: str | Path,
    document_field: str = "answer",
    include_label_metadata: bool = False,
    corpus_name: str = "answer_echo",
) -> dict[str, Any]:
    """Return a local retrieval stress corpus built from score-dump statements."""
    if document_field not in DOCUMENT_FIELDS:
        raise ValueError(f"document_field must be one of: {', '.join(DOCUMENT_FIELDS)}.")
    if not score_dump.statements:
        raise ValueError("retrieval stress corpora require statement-bearing score dumps.")
    documents = []
    skipped_empty = 0
    for idx, statement in enumerate(score_dump.statements):
        text = _document_text(statement, document_field=document_field)
        if not text:
            skipped_empty += 1
            continue
        metadata: dict[str, Any] = {
            "corpus_name": corpus_name,
            "document_field": document_field,
            "row_index": idx,
            "claim_id": str(statement.get("claim_id") or f"c{idx + 1}"),
            "question": str(statement.get("question", "")),
            "answer": str(statement.get("answer", "")),
            "stress_control": "score_dump_answer_echo",
        }
        if include_label_metadata:
            metadata["score_label"] = score_dump.labels[idx]
        documents.append({
            "text": text,
            "source": f"{corpus_name}:{document_field}:{idx}",
            "metadata": metadata,
        })
    if not documents:
        raise ValueError("stress corpus would be empty; selected document field has no text.")
    return {
        "schema_version": 1,
        "corpus_type": "retrieval_stress_answer_echo",
        "description": (
            "Retrieval stress corpus built from the same statement answers being audited. "
            "Use this as a negative control: strong support here can indicate self-support, "
            "not external grounding."
        ),
        "label_usage": {
            "labels_used_for_documents": False,
            "labels_copied_to_document_metadata": bool(include_label_metadata),
        },
        "config": {
            "builder": "build_retrieval_stress_corpus",
            "source_scores_path": str(source_scores_path),
            "document_field": document_field,
            "corpus_name": corpus_name,
            "include_label_metadata": bool(include_label_metadata),
        },
        "input_provenance": {
            "score_dump": score_dump_file_metadata(source_scores_path, score_dump),
        },
        "summary": {
            "n_documents": len(documents),
            "n_source_records": score_dump.n_total,
            "n_skipped_empty": skipped_empty,
            "document_field": document_field,
        },
        "documents": documents,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    scores_path = Path(args.scores)
    score_dump = load_score_dump(
        scores_path,
        allow_missing_scores=True,
        require_statements=True,
    )
    payload = build_retrieval_stress_corpus(
        score_dump,
        source_scores_path=scores_path,
        document_field=args.document_field,
        include_label_metadata=bool(args.include_label_metadata),
        corpus_name=args.corpus_name,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = payload["summary"]
    print(f"Wrote retrieval stress corpus to {output_path} ({summary['n_documents']} documents)")
    return payload


def _document_text(statement: Mapping[str, Any], *, document_field: str) -> str:
    if document_field == "answer":
        return str(statement.get("answer", "")).strip()
    if document_field == "question_answer":
        return f"{statement.get('question', '')} {statement.get('answer', '')}".strip()
    if document_field == "text":
        return str(statement.get("claim") or statement.get("text") or statement.get("answer") or "").strip()
    raise ValueError(f"document_field must be one of: {', '.join(DOCUMENT_FIELDS)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build score-dump retrieval stress corpus")
    parser.add_argument("--scores", required=True, help="statement-bearing score dump")
    parser.add_argument("--output", required=True, help="path to write corpus JSON")
    parser.add_argument("--document-field", choices=DOCUMENT_FIELDS, default="answer")
    parser.add_argument("--corpus-name", default="answer_echo")
    parser.add_argument(
        "--include-label-metadata",
        action="store_true",
        help="copy labels into document metadata for audit-only debugging",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
