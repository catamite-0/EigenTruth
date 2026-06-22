"""Build a local TruthfulQA correct-answer evidence corpus.

This is a no-model utility for creating a reproducible local fact corpus before
plugging in real retrieval systems. It writes correct-answer statements from the
same deterministic TruthfulQA split used by ``eval_truthfulqa.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.eval_truthfulqa import Statement, load_truthfulqa


def build_truthfulqa_corpus(
    *,
    manifold_questions: int,
    limit: int,
    include_manifold: bool = True,
    include_eval: bool = True,
) -> dict[str, Any]:
    """Return a JSON corpus of TruthfulQA correct-answer statements."""
    manifold_true, _, eval_statements = load_truthfulqa(manifold_questions, limit)
    documents = []
    if include_manifold:
        for idx, text in enumerate(manifold_true, start=1):
            documents.append({
                "text": text,
                "source": f"truthfulqa:manifold:{idx}",
                "metadata": {
                    "dataset": "TruthfulQA",
                    "split": "manifold",
                    "is_false": 0,
                    "row_index": idx - 1,
                },
            })
    if include_eval:
        for idx, statement in enumerate(eval_statements, start=1):
            if int(statement.is_false) != 0:
                continue
            documents.append({
                "text": _statement_text(statement),
                "source": f"truthfulqa:eval:{idx}",
                "metadata": {
                    "dataset": "TruthfulQA",
                    "split": "eval",
                    "is_false": 0,
                    "row_index": idx - 1,
                    "question": statement.question,
                    "answer": statement.answer,
                },
            })
    if not documents:
        raise ValueError("corpus would be empty; enable manifold and/or eval correct-answer documents.")
    return {
        "schema_version": 1,
        "corpus_type": "truthfulqa_correct_answer_evidence",
        "description": (
            "TruthfulQA correct-answer statements for local retrieval baselines. "
            "This is dataset-derived evidence, not label-derived per-claim oracle refutation."
        ),
        "config": {
            "manifold_questions": int(manifold_questions),
            "limit": int(limit),
            "include_manifold": bool(include_manifold),
            "include_eval": bool(include_eval),
        },
        "summary": {
            "n_documents": len(documents),
            "n_manifold_documents": sum(1 for document in documents if document["metadata"]["split"] == "manifold"),
            "n_eval_documents": sum(1 for document in documents if document["metadata"]["split"] == "eval"),
        },
        "documents": documents,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    payload = build_truthfulqa_corpus(
        manifold_questions=args.manifold_questions,
        limit=args.limit,
        include_manifold=args.include_manifold,
        include_eval=args.include_eval,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    summary = payload["summary"]
    print(f"Wrote TruthfulQA corpus to {output_path} ({summary['n_documents']} documents)")
    return payload


def _statement_text(statement: Statement) -> str:
    return f"{statement.question} {statement.answer}".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local TruthfulQA correct-answer corpus")
    parser.add_argument("--manifold-questions", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-manifold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-eval", action=argparse.BooleanOptionalAction, default=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
