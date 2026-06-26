"""TruthfulQA-style forced-answer trajectory diagnostics.

This benchmark reuses statement-bearing score dumps and measures whether the
hidden-state trajectory across answer-token prediction positions separates true
and false candidate answers. It is a real-data follow-up to
``trajectory_convergence_sanity.py`` while keeping optional model dependencies
outside the core package.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.core import trajectory_convergence_metrics  # noqa: E402
from eigentruth.eval import roc_auc, spearman_correlation  # noqa: E402
from eigentruth.eval.score_dump import ScoreDump, load_score_dump, score_dump_file_metadata  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class TrajectoryStatement:
    """One labeled statement for forced-answer trajectory scoring."""

    question: str
    answer: str
    label: int
    index: int
    metadata: Mapping[str, Any]


def load_trajectory_statements(path: str | Path, *, limit: int | None = None) -> tuple[TrajectoryStatement, ...]:
    """Load question/answer statements from a score dump."""
    dump = load_score_dump(path, require_statements=True)
    return trajectory_statements_from_score_dump(dump, limit=limit)


def trajectory_statements_from_score_dump(
    dump: ScoreDump,
    *,
    limit: int | None = None,
) -> tuple[TrajectoryStatement, ...]:
    """Convert a statement-bearing score dump into trajectory records."""
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive when provided.")
    if not dump.statements:
        raise ValueError("score dump must contain statements.")
    records: list[TrajectoryStatement] = []
    for index, (label, statement) in enumerate(zip(dump.labels, dump.statements, strict=True)):
        if limit is not None and len(records) >= int(limit):
            break
        question = str(statement.get("question") or "").strip()
        answer = str(statement.get("answer") or "").strip()
        if not question or not answer:
            continue
        records.append(TrajectoryStatement(
            question=question,
            answer=answer,
            label=int(label),
            index=index,
            metadata={"statement": dict(statement)},
        ))
    if not records:
        raise ValueError("score dump did not contain any usable question/answer statements.")
    return tuple(records)


def trajectory_truthfulqa_report(
    records: Sequence[TrajectoryStatement],
    *,
    model: Any,
    tokenizer: Any,
    layer: int = -1,
    min_answer_tokens: int = 3,
    max_answer_tokens: int | None = None,
    device: str | torch.device = "cpu",
    min_abs_spearman: float = 0.3,
    min_auroc: float = 0.55,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate forced-answer hidden-state trajectories over labeled records."""
    if int(min_answer_tokens) < 3:
        raise ValueError("min_answer_tokens must be >= 3.")
    if max_answer_tokens is not None and int(max_answer_tokens) < int(min_answer_tokens):
        raise ValueError("max_answer_tokens must be >= min_answer_tokens when provided.")
    if not records:
        raise ValueError("at least one trajectory record is required.")
    target_device = torch.device(device)
    if hasattr(model, "to"):
        model = model.to(target_device)
    if hasattr(model, "eval"):
        model.eval()

    evaluated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            try:
                evaluation = _evaluate_record_trajectory(
                    record,
                    model=model,
                    tokenizer=tokenizer,
                    layer=int(layer),
                    min_answer_tokens=int(min_answer_tokens),
                    max_answer_tokens=max_answer_tokens,
                    device=target_device,
                )
            except ValueError as exc:
                skipped.append({
                    "index": int(record.index),
                    "label": int(record.label),
                    "reason": str(exc),
                })
                continue
            evaluated.append(evaluation)

    summary = _trajectory_summary(
        evaluated,
        skipped,
        min_abs_spearman=float(min_abs_spearman),
        min_auroc=float(min_auroc),
    )
    return {
        "workflow": "truthfulqa_forced_answer_trajectory",
        "config": {
            "layer": int(layer),
            "min_answer_tokens": int(min_answer_tokens),
            "max_answer_tokens": None if max_answer_tokens is None else int(max_answer_tokens),
            "min_abs_spearman": float(min_abs_spearman),
            "min_auroc": float(min_auroc),
        },
        "summary": summary,
        "records": evaluated,
        "skipped_records": skipped,
        "metadata": dict(metadata or {}),
    }


def _evaluate_record_trajectory(
    record: TrajectoryStatement,
    *,
    model: Any,
    tokenizer: Any,
    layer: int,
    min_answer_tokens: int,
    max_answer_tokens: int | None,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = _tokenize(tokenizer, record.question, add_special_tokens=True)
    answer_text = record.answer if record.answer.startswith((" ", "\n", "\t")) else f" {record.answer}"
    answer_ids = _tokenize(tokenizer, answer_text, add_special_tokens=False)
    if max_answer_tokens is not None:
        answer_ids = answer_ids[:int(max_answer_tokens)]
    if len(prompt_ids) < 1:
        raise ValueError("prompt tokenization produced no tokens")
    if len(answer_ids) < int(min_answer_tokens):
        raise ValueError("answer token count below min_answer_tokens")
    input_ids = prompt_ids + answer_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    output = model(
        input_ids=input_tensor,
        attention_mask=torch.ones_like(input_tensor),
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = _hidden_states_from_output(output)
    logits = _logits_from_output(output)
    selected_layer = _resolve_layer(layer, len(hidden_states))
    hidden = torch.as_tensor(hidden_states[selected_layer], dtype=torch.float32, device=device)
    if hidden.ndim != 3 or int(hidden.shape[0]) != 1:
        raise ValueError("selected hidden state must be shaped [1, sequence, hidden_dim]")
    predictor_positions = torch.arange(
        len(prompt_ids) - 1,
        len(prompt_ids) + len(answer_ids) - 1,
        dtype=torch.long,
        device=device,
    )
    if int(predictor_positions[0].item()) < 0 or int(predictor_positions[-1].item()) >= int(hidden.shape[1]):
        raise ValueError("answer predictor positions are outside the hidden-state sequence")
    states = hidden[0, predictor_positions, :].detach().cpu()
    metrics = trajectory_convergence_metrics(
        states,
        metadata={
            "record_index": int(record.index),
            "label": int(record.label),
            "layer": int(layer),
            "resolved_layer": int(selected_layer),
        },
    )
    answer_targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    selected_logits = torch.as_tensor(logits, dtype=torch.float32, device=device)[0, predictor_positions, :]
    log_probs = torch.log_softmax(selected_logits, dim=-1)
    token_nll = -log_probs.gather(dim=-1, index=answer_targets[:, None]).squeeze(-1)
    mean_nll = float(token_nll.mean().item())
    return {
        "index": int(record.index),
        "label": int(record.label),
        "question": record.question,
        "answer": record.answer,
        "n_prompt_tokens": len(prompt_ids),
        "n_answer_tokens": len(answer_ids),
        "nll_answer": mean_nll,
        "trajectory": metrics.to_dict(),
    }


def _trajectory_summary(
    evaluated: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    *,
    min_abs_spearman: float,
    min_auroc: float,
) -> dict[str, Any]:
    labels = [int(record["label"]) for record in evaluated]
    convergence_scores = [float(record["trajectory"]["convergence_score"]) for record in evaluated]
    nll_scores = [float(record["nll_answer"]) for record in evaluated]
    higher_auroc = _safe_roc_auc(convergence_scores, labels)
    lower_auroc = _safe_roc_auc([-score for score in convergence_scores], labels)
    nll_auroc = _safe_roc_auc(nll_scores, labels)
    best_direction = "higher" if higher_auroc >= lower_auroc else "lower"
    best_auroc = max(higher_auroc, lower_auroc)
    spearman_false = _safe_spearman(convergence_scores, labels)
    status = "pass" if (
        abs(float(spearman_false)) >= float(min_abs_spearman)
        or float(best_auroc) >= float(min_auroc)
    ) else "fail"
    return {
        "status": status,
        "n_total": len(evaluated) + len(skipped),
        "n_evaluated": len(evaluated),
        "n_skipped": len(skipped),
        "skip_reasons": dict(Counter(str(item["reason"]) for item in skipped)),
        "n_true": sum(1 for label in labels if label == 0),
        "n_false": sum(1 for label in labels if label == 1),
        "trajectory_score_direction_for_false": best_direction,
        "trajectory_score_best_auroc": float(best_auroc),
        "trajectory_score_higher_is_false_auroc": float(higher_auroc),
        "trajectory_score_lower_is_false_auroc": float(lower_auroc),
        "spearman_convergence_false_label": float(spearman_false),
        "nll_answer_higher_is_false_auroc": float(nll_auroc),
        "mean_false_convergence_score": _mean(
            float(record["trajectory"]["convergence_score"]) for record in evaluated if int(record["label"]) == 1
        ),
        "mean_true_convergence_score": _mean(
            float(record["trajectory"]["convergence_score"]) for record in evaluated if int(record["label"]) == 0
        ),
        "mean_false_nll_answer": _mean(
            float(record["nll_answer"]) for record in evaluated if int(record["label"]) == 1
        ),
        "mean_true_nll_answer": _mean(
            float(record["nll_answer"]) for record in evaluated if int(record["label"]) == 0
        ),
    }


def _safe_roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    if len(set(int(label) for label in labels)) < 2:
        return 0.5
    return float(roc_auc(scores, labels))


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(set(float(value) for value in left)) < 2 or len(set(float(value) for value in right)) < 2:
        return 0.0
    value = float(spearman_correlation(left, right))
    if not math.isfinite(value):
        return 0.0
    return value


def _mean(values: Sequence[float]) -> float | None:
    materialized = [float(value) for value in values]
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _tokenize(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return [int(token) for token in tokenizer.encode(text, add_special_tokens=add_special_tokens)]
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else getattr(encoded, "input_ids", None)
    if input_ids is None:
        raise ValueError("tokenizer must expose encode() or return input_ids.")
    if isinstance(input_ids, torch.Tensor):
        values = input_ids.flatten().tolist()
    else:
        values = input_ids[0] if input_ids and isinstance(input_ids[0], Sequence) else input_ids
    return [int(token) for token in values]


def _hidden_states_from_output(output: Any) -> Sequence[Any]:
    hidden_states = (
        output.get("hidden_states")
        if isinstance(output, Mapping)
        else getattr(output, "hidden_states", None)
    )
    if not isinstance(hidden_states, Sequence) or not hidden_states:
        raise ValueError("model output must expose a non-empty hidden_states sequence.")
    return hidden_states


def _logits_from_output(output: Any) -> Any:
    logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
    if logits is None:
        raise ValueError("model output must expose logits.")
    return logits


def _resolve_layer(layer: int, n_layers: int) -> int:
    normalized = int(layer)
    if normalized < 0:
        normalized += int(n_layers)
    if normalized < 0 or normalized >= int(n_layers):
        raise ValueError(f"layer {layer} is outside hidden_states range with {n_layers} layers.")
    return normalized


class _OfflineTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {
            "<pad>": 0,
            "<bos>": 1,
            "false": 2,
            "wrong": 3,
            "unstable": 4,
            "true": 5,
            "stable": 6,
            "correct": 7,
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        pieces = str(text).strip().lower().replace(".", " .").replace("?", " ?").split()
        ids = [self._token_id(piece) for piece in pieces]
        return ([self._token_id("<bos>")] + ids) if add_special_tokens else ids

    def _token_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]


class _OfflineTrajectoryModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 256, hidden_dim: int = 6) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Any:
        del attention_mask, output_hidden_states, return_dict
        batch, seq_len = input_ids.shape
        hidden = torch.zeros(batch, seq_len, self.hidden_dim, dtype=torch.float32, device=input_ids.device)
        logits = torch.zeros(batch, seq_len, self.vocab_size, dtype=torch.float32, device=input_ids.device)
        for batch_idx in range(batch):
            unstable_seen = False
            for pos in range(seq_len):
                token_id = int(input_ids[batch_idx, pos].item())
                unstable_seen = unstable_seen or token_id in {self._false_id, self._wrong_id, self._unstable_id}
                if unstable_seen:
                    scale = 1.0 + (0.12 * float(pos))
                    hidden[batch_idx, pos] = torch.tensor(
                        [
                            ((-1.0) ** pos) * scale,
                            math.sin(float(pos) * 1.7) * scale,
                            math.cos(float(pos) * 1.3) * scale,
                            float(pos % 3) * 0.5,
                            -scale,
                            0.25 * float(pos),
                        ],
                        device=input_ids.device,
                    )
                else:
                    progress = 1.0 - (0.45 ** max(pos, 1))
                    hidden[batch_idx, pos] = torch.tensor(
                        [progress, 0.5 * progress, 0.25 * progress, 0.0, 0.0, 0.0],
                        device=input_ids.device,
                    )
                logits[batch_idx, pos, token_id % self.vocab_size] = 4.0
        return type("OfflineOutput", (), {"logits": logits, "hidden_states": (hidden * 0.5, hidden)})()

    @property
    def _false_id(self) -> int:
        return 2

    @property
    def _wrong_id(self) -> int:
        return 3

    @property
    def _unstable_id(self) -> int:
        return 4


def _offline_records() -> tuple[TrajectoryStatement, ...]:
    rows = [
        ("What is the stable fact?", "true stable correct answer", 0),
        ("Which answer is grounded?", "true stable correct value", 0),
        ("What fact is supported?", "true stable clean record", 0),
        ("Which claim is reliable?", "true stable correct route", 0),
        ("What is the bad claim?", "false unstable wrong answer", 1),
        ("Which answer is unsupported?", "false unstable wrong value", 1),
        ("What fact is fabricated?", "false unstable wrong record", 1),
        ("Which claim is unreliable?", "false unstable wrong route", 1),
    ]
    return tuple(
        TrajectoryStatement(question=question, answer=answer, label=label, index=index, metadata={})
        for index, (question, answer, label) in enumerate(rows)
    )


def _load_hf_model_and_tokenizer(model_id: str, *, device: str | torch.device) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without optional extra.
        raise RuntimeError("transformers is required for non-offline trajectory evaluation.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.to(device), tokenizer


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.offline:
        records = _offline_records()
        model: Any = _OfflineTrajectoryModel()
        tokenizer: Any = _OfflineTokenizer()
        source_metadata: dict[str, Any] = {"source": "offline synthetic trajectory fixture"}
    else:
        if not args.scores:
            raise ValueError("--scores is required unless --offline is set.")
        if not args.model:
            raise ValueError("--model is required unless --offline is set.")
        records = load_trajectory_statements(args.scores, limit=args.limit)
        model, tokenizer = _load_hf_model_and_tokenizer(args.model, device=args.device)
        source_metadata = {
            "source_scores": score_dump_file_metadata(Path(args.scores)),
            "model": args.model,
        }
    if args.limit is not None and args.offline:
        records = records[:int(args.limit)]
    report = trajectory_truthfulqa_report(
        records,
        model=model,
        tokenizer=tokenizer,
        layer=args.layer,
        min_answer_tokens=args.min_answer_tokens,
        max_answer_tokens=args.max_answer_tokens,
        device=args.device,
        min_abs_spearman=args.min_abs_spearman,
        min_auroc=args.min_auroc,
        metadata={
            **source_metadata,
            "research_note": (
                "Forced-answer hidden-state trajectory over answer-token prediction positions; "
                "labels come from the source statement benchmark."
            ),
        },
    )
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.artifact_manifest:
        if not args.json:
            raise ValueError("--artifact-manifest requires --json.")
        manifest_path = Path(args.artifact_manifest)
        artifacts = {"trajectory_truthfulqa_report": Path(args.json)}
        if args.scores:
            artifacts["source_scores"] = Path(args.scores)
        manifest = build_artifact_manifest(
            artifacts,
            root=manifest_path.parent,
            metadata={
                "workflow": "truthfulqa_forced_answer_trajectory",
                "model": args.model if not args.offline else "offline",
            },
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not getattr(args, "quiet", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate forced-answer hidden-state trajectory diagnostics")
    parser.add_argument("--scores", default=None, help="statement-bearing score dump JSON or JSONL manifest")
    parser.add_argument("--model", default=None, help="Hugging Face causal LM id/path")
    parser.add_argument("--offline", action="store_true", help="run deterministic toy fixture without downloads")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-answer-tokens", type=int, default=3)
    parser.add_argument("--max-answer-tokens", type=int, default=None)
    parser.add_argument("--min-abs-spearman", type=float, default=0.3)
    parser.add_argument("--min-auroc", type=float, default=0.55)
    parser.add_argument("--json", default=None, help="optional path to write JSON report")
    parser.add_argument("--artifact-manifest", default=None, help="optional manifest path for --json output")
    parser.add_argument("--quiet", action="store_true", help="write requested artifacts without printing full JSON")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
