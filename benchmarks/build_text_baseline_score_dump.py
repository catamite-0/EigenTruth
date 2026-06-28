"""Build score-dump text baselines from statement metadata.

The baselines here are deliberately simple. They exist as red-team controls for
hallucination-detection experiments: new geometry, verifier, or self-check
signals should be compared against cheap text-length and lexical-overlap
features before being treated as meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.eval.score_dump import ScoreDump, load_score_dump, write_score_dump_jsonl

DEFAULT_TEXT_BASELINE_SIGNALS = (
    "answer_char_length",
    "answer_token_count",
    "claim_char_length",
    "claim_token_count",
    "question_answer_token_overlap",
    "answer_negation_flag",
    "answer_number_count",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "false",
    "incorrect",
    "wrong",
    "cannot",
    "can't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "不是",
    "没有",
    "并非",
    "错误",
    "不正确",
}


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"{name} must contain unique values.")
    return parts


def build_text_baseline_score_dump(
    score_dump: ScoreDump,
    *,
    source_scores_path: str | Path,
    keep_signals: Sequence[str] | None = None,
    baseline_signals: Sequence[str] = DEFAULT_TEXT_BASELINE_SIGNALS,
) -> ScoreDump:
    """Return a score dump with simple text-baseline score columns appended."""
    if not score_dump.statements:
        raise ValueError("text baselines require statement-bearing score dumps.")
    selected_keep_signals = tuple(score_dump.scores) if keep_signals is None else tuple(keep_signals)
    missing = [signal for signal in selected_keep_signals if signal not in score_dump.scores]
    if missing:
        raise ValueError(f"score dump is missing requested signal(s): {missing}.")
    selected_baselines = tuple(baseline_signals)
    if len(set(selected_baselines)) != len(selected_baselines):
        raise ValueError("baseline_signals must contain unique values.")
    overlap = set(selected_keep_signals) & set(selected_baselines)
    if overlap:
        raise ValueError(f"text baseline signal(s) overlap existing score signals: {sorted(overlap)}.")

    baseline_columns = {signal: [] for signal in selected_baselines}
    for statement in score_dump.statements:
        features = text_baseline_features(statement)
        for signal in selected_baselines:
            if signal not in features:
                raise ValueError(f"unknown text baseline signal {signal!r}.")
            baseline_columns[signal].append(_finite_float(features[signal], name=signal))

    config = dict(score_dump.config)
    config["text_baseline_score_dump"] = {
        "builder": "build_text_baseline_score_dump",
        "source_scores_path": str(source_scores_path),
        "signals": list(selected_baselines),
    }
    scores = {
        **{signal: tuple(float(value) for value in score_dump.scores[signal]) for signal in selected_keep_signals},
        **{signal: tuple(values) for signal, values in baseline_columns.items()},
    }
    extras = dict(score_dump.extras)
    extras["text_baseline_metadata"] = {
        "source_scores_path": str(source_scores_path),
        "signals": list(selected_baselines),
        "signal_definitions": text_baseline_signal_definitions(),
    }
    return ScoreDump(
        labels=score_dump.labels,
        scores=scores,
        config=config,
        sweep_scores=score_dump.sweep_scores,
        statements=score_dump.statements,
        extras=extras,
    )


def text_baseline_features(statement: Mapping[str, Any]) -> dict[str, float]:
    """Return dependency-free text baseline features for one statement."""
    question = _text_field(statement, "question")
    answer = _answer_text(statement)
    claim = _claim_text(statement, answer=answer, question=question)
    answer_tokens = _tokens(answer)
    question_tokens = _tokens(question)
    claim_tokens = _tokens(claim)
    overlap = _answer_overlap(answer_tokens, question_tokens)
    return {
        "answer_char_length": float(len(answer)),
        "answer_token_count": float(len(answer_tokens)),
        "claim_char_length": float(len(claim)),
        "claim_token_count": float(len(claim_tokens)),
        "question_answer_token_overlap": overlap,
        "answer_negation_flag": 1.0 if any(token in _NEGATION_TOKENS for token in answer_tokens) else 0.0,
        "answer_number_count": float(len(_NUMBER_RE.findall(answer))),
    }


def text_baseline_signal_definitions() -> dict[str, str]:
    """Return human-readable definitions for each text baseline."""
    return {
        "answer_char_length": "Character count of the answer field; higher is a longer-answer baseline.",
        "answer_token_count": "Regex token count of the answer field; higher is a longer-answer baseline.",
        "claim_char_length": "Character count of the claim/text fallback used for verification.",
        "claim_token_count": "Regex token count of the claim/text fallback used for verification.",
        "question_answer_token_overlap": (
            "Fraction of answer tokens that also appear in the question; lower can indicate lexical drift."
        ),
        "answer_negation_flag": "1 when the answer contains a simple negation token, else 0.",
        "answer_number_count": "Count of numeric substrings in the answer field.",
    }


def write_score_dump(dump: ScoreDump, output_path: str | Path, *, output_format: str) -> None:
    """Write output score dump as JSON or JSONL manifest."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        output.write_text(json.dumps(dump.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if output_format == "jsonl":
        write_score_dump_jsonl(dump, output)
        return
    raise ValueError("output_format must be 'json' or 'jsonl'.")


def build_report(
    *,
    input_scores: str | Path,
    output: str | Path,
    output_format: str,
    keep_signals: Sequence[str] | None,
    baseline_signals: Sequence[str],
) -> dict[str, Any]:
    """Build text baseline score dump and return a compact report."""
    score_dump = load_score_dump(
        input_scores,
        allow_missing_scores=False,
        require_statements=True,
    )
    enhanced = build_text_baseline_score_dump(
        score_dump,
        source_scores_path=input_scores,
        keep_signals=keep_signals,
        baseline_signals=baseline_signals,
    )
    write_score_dump(enhanced, output, output_format=output_format)
    return {
        "schema_version": 1,
        "input_scores": str(input_scores),
        "output": str(output),
        "output_format": output_format,
        "n_total": enhanced.n_total,
        "signals": list(enhanced.scores),
        "text_baseline_signals": list(baseline_signals),
        "summary": {
            signal: {
                "min": min(enhanced.scores[signal]),
                "max": max(enhanced.scores[signal]),
                "mean": sum(enhanced.scores[signal]) / len(enhanced.scores[signal]),
            }
            for signal in baseline_signals
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from CLI-style arguments."""
    keep_signals = _parse_csv(args.keep_signals, name="keep_signals")
    baseline_signals = _parse_csv(args.baseline_signals, name="baseline_signals") or DEFAULT_TEXT_BASELINE_SIGNALS
    report = build_report(
        input_scores=args.scores,
        output=args.output,
        output_format=args.output_format,
        keep_signals=keep_signals,
        baseline_signals=baseline_signals,
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote text-baseline score dump to {args.output}")
    return report


def _answer_text(statement: Mapping[str, Any]) -> str:
    answer = _text_field(statement, "answer")
    if answer:
        return answer
    text = _text_field(statement, "claim") or _text_field(statement, "text")
    if text:
        return text
    raise ValueError("statement record is missing answer/claim/text.")


def _claim_text(statement: Mapping[str, Any], *, answer: str, question: str) -> str:
    claim = _text_field(statement, "claim") or _text_field(statement, "text")
    if claim:
        return claim
    return f"{question} {answer}".strip() if question else answer


def _text_field(statement: Mapping[str, Any], field: str) -> str:
    value = statement.get(field)
    return "" if value is None else str(value).strip()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(text))


def _answer_overlap(answer_tokens: Sequence[str], question_tokens: Sequence[str]) -> float:
    if not answer_tokens:
        return 0.0
    question_set = set(question_tokens)
    if not question_set:
        return 0.0
    return sum(1 for token in answer_tokens if token in question_set) / len(answer_tokens)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric and must not be bool.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def main() -> None:
    parser = argparse.ArgumentParser(description="Build text-baseline score-dump columns")
    parser.add_argument("--scores", required=True, help="statement-bearing score dump")
    parser.add_argument("--output", required=True, help="output score dump path")
    parser.add_argument("--output-format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument("--keep-signals", default=None)
    parser.add_argument("--baseline-signals", default=",".join(DEFAULT_TEXT_BASELINE_SIGNALS))
    parser.add_argument("--json", default=None, help="optional compact report path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
