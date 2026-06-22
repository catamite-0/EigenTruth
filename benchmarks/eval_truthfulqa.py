"""EigenTruth 基准评测 — 隐状态几何信号能否分离真/假陈述 (AUROC)。
EigenTruth benchmark — can hidden-state geometry separate true vs. false statements (AUROC)?

研究问题 / Research questions:
    1. 流形马氏距离 (maha_last) 能否把"假陈述"排在"真陈述"之上？ 它打得过困惑度基线 (nll) 吗？
       Does manifold Mahalanobis distance rank false statements above true ones, and beat a
       perplexity baseline?
    2. 双曲离散度 (disp_hse) 是否优于欧氏离散度 (disp_euclid)？ —— 即"双曲几何有没有用"的消融。
       Does hyperbolic dispersion (disp_hse) beat Euclidean dispersion (disp_euclid)? — the
       "does the hyperbolic projection earn its keep?" ablation.
    3. 对比方向投影 (truth_proj，即工具自带的 contrastive_direction 用作 mass-mean 探针，
       参见 Marks & Tegmark) 是否是更强的检测器？最佳目标层在哪 (--sweep)？
       Is the contrastive-direction projection (the tool's own steering direction used as a
       mass-mean probe, cf. Marks & Tegmark) an even stronger detector, and which layer is
       best (--sweep)?

方法 / Method (SAPLMA 式、确定性、无需 LLM 裁判 / SAPLMA-style, deterministic, judge-free):
    - 从**留出**题目的*正确*答案构建真值流形（无标签泄漏）。
      Build the truth manifold from the *correct* answers of **held-out** questions (no leakage).
    - 对其余题目的每条候选答案（正确=负类, 错误=正类/幻觉）做单次前向，提取目标层隐状态。
      For each candidate answer of the remaining questions (correct=negative, incorrect=positive),
      run a single forward pass and read the target-layer hidden states.
    - 每条陈述计算多种内部诊断分数，分别报告 AUROC。
      Score each statement with multiple internal diagnostics and report AUROC per signal.

局限 / Limitations:
    - 强制候选答案的"陈述级"打分是开放生成幻觉的*代理*，干净地检验表征假设但不等同于在线检测。
      Forced-answer statement scoring is a *proxy* for open-generation hallucination; it cleanly
      tests the representation hypothesis but is not the same as online detection.
    - 句内 token 离散度和 eigenscore 是基于多次采样语义不确定性 / INSIDE EigenScore 的廉价代理。
      Within-statement token dispersion and eigenscore are cheap proxies for sample-based
      semantic uncertainty / INSIDE EigenScore.
    - 小模型 + 几百条样本 → AUROC 置信区间较宽，结果仅供参考，非定论。
      Small model + a few hundred items → wide AUROC CIs; results are indicative, not conclusive.

用法 / Usage:
    # 真实基准 / real benchmark (downloads model + TruthfulQA):
    python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct --layer -8 --limit 200
    # 快速管线自检 / fast pipeline smoke check (tiny model, bundled statements, no dataset):
    python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, Sequence

import torch

from eigentruth.calibration import DEFAULT_SCORE_DIRECTIONS
from eigentruth.core import TruthSubspace, internal_eigenscore
from eigentruth.core.math_engine import (
    TruthManifold,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,
    poincare_map,
)
from eigentruth.eval.conformal import directional_conformal_threshold
from eigentruth.eval.metrics import euclidean_dispersion, roc_auc, selective_classification_report
from eigentruth.intervention.hooks import TruthProbe

SIGNALS = [
    "maha_last",
    "truth_proj",
    "subspace_resid",
    "disp_euclid",
    "disp_hse",
    "eigenscore",
    "nll_answer",
]
INSIDE_SIGNAL = "inside_eigenscore"
REPORT_ALPHA = 0.10
HIDDEN_STATE_CAPTURE_METHODS = ("outputs", "hooks")
PROFILE_GROUPS = {
    "startup": ("load_data", "load_model"),
    "tokenization": (
        "load_statement_encoding_cache",
        "tokenize_statements",
        "save_statement_encoding_cache",
    ),
    "model_forward": ("build_layer_stats", "forced_answer_forward", "inside_generation"),
    "cache_io": (
        "read_cache_metadata",
        "load_layer_stats_cache",
        "save_layer_stats_cache",
        "load_eval_reps_cache",
        "init_eval_reps_cache_writer",
        "read_eval_reps_cache_batch",
        "write_eval_reps_cache_batch",
        "save_eval_reps_cache",
    ),
    "postprocess": ("score_postprocess", "reporting", "sweep_reporting"),
}


@dataclass
class Statement:
    question: str
    answer: str
    is_false: int  # 1 = 错误答案(正类/幻觉) / incorrect (positive), 0 = 正确答案(负类)


@dataclass(frozen=True)
class StatementEncoding:
    input_ids: tuple[int, ...]
    n_answer_tokens: int

    def __post_init__(self) -> None:
        input_ids = tuple(int(token_id) for token_id in self.input_ids)
        n_answer_tokens = int(self.n_answer_tokens)
        if not input_ids:
            raise ValueError("statement encoding input_ids must not be empty.")
        if n_answer_tokens <= 0:
            raise ValueError("statement encoding n_answer_tokens must be positive.")
        if n_answer_tokens > len(input_ids):
            raise ValueError("statement encoding answer span exceeds input length.")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "n_answer_tokens", n_answer_tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_ids": list(self.input_ids),
            "n_answer_tokens": int(self.n_answer_tokens),
        }


def _selective_reports(
    scores: Mapping[str, Sequence[float]],
    labels: Sequence[int],
    *,
    alpha: float = REPORT_ALPHA,
    directions: Mapping[str, str] | None = None,
) -> dict[str, dict]:
    labels_t = torch.tensor(labels)
    reports = {}
    for signal, values in scores.items():
        direction = (directions or DEFAULT_SCORE_DIRECTIONS).get(signal, "higher")
        signal_scores = torch.tensor(values, dtype=torch.float64)
        true_scores = signal_scores[labels_t == 0]
        threshold = directional_conformal_threshold(true_scores, alpha, direction)
        reports[signal] = {
            "alpha": alpha,
            **selective_classification_report(signal_scores, labels_t, threshold, direction=direction),
        }
    return reports


def _inside_enabled(args) -> bool:
    return int(getattr(args, "inside_samples", 0)) >= 2


def _inside_trigger_enabled(args) -> bool:
    return bool(getattr(args, "inside_trigger_signal", None))


def _enabled_signals(args) -> list[str]:
    signals = list(SIGNALS)
    if _inside_enabled(args):
        signals.append(INSIDE_SIGNAL)
    return signals


def _sweep_signal_names(args) -> list[str]:
    signals = ["maha_last", "truth_proj", "subspace_resid", "eigenscore"]
    if _inside_enabled(args):
        signals.append(INSIDE_SIGNAL)
    return signals


def _sweep_output_enabled(args) -> bool:
    return bool(getattr(args, "sweep", False) or getattr(args, "sweep_layers", None))


def _parse_sweep_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    layers = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            layers.append(int(text))
        except ValueError as exc:
            raise ValueError("--sweep-layers must be a comma-separated list of integer layer indexes.") from exc
    if not layers:
        raise ValueError("--sweep-layers must contain at least one layer index.")
    return layers


def _resolve_sweep_layers(target_layer: int, n_layers: int, *, sweep: bool, sweep_layers: str | None) -> list[int]:
    requested = _parse_sweep_layers(sweep_layers)
    if requested is not None:
        layers = [target_layer, *requested]
    elif sweep:
        layers = [target_layer, *[-(i + 1) for i in range(n_layers)]]
    else:
        layers = [target_layer]

    resolved = []
    seen = set()
    for layer in layers:
        if not (-(n_layers + 1) <= layer <= n_layers):
            raise ValueError(
                f"sweep layer {layer} is out of range for a model with {n_layers} transformer layers "
                f"(valid hidden-state indexes: {-n_layers - 1}..{n_layers})."
            )
        if layer not in seen:
            seen.add(layer)
            resolved.append(layer)
    return resolved


def _normalize_hidden_state_index(layer: int, n_layers: int) -> int:
    """Map a hidden-state tuple index to its non-negative position."""
    n_hidden_states = int(n_layers) + 1
    normalized = int(layer)
    if normalized < 0:
        normalized = n_hidden_states + normalized
    if not (0 <= normalized < n_hidden_states):
        raise IndexError(
            f"hidden-state index {layer} is out of range for {n_layers} transformer layers."
        )
    return normalized


def _hook_capture_layer_map(model, layers: Sequence[int]) -> dict[int, int]:
    """Return requested hidden-state indexes mapped to transformer block indexes.

    HF decoder hidden-state tuples contain the embedding state at index 0 and,
    for common decoder-only models, the final hidden state after final norm at
    index ``num_hidden_layers``. A decoder block hook can exactly capture only
    intermediate post-block states, i.e. hidden-state indexes 1..N-1.
    """
    transformer_layers = TruthProbe._find_layers(model)
    n_layers = len(transformer_layers)
    mapped: dict[int, int] = {}
    unsupported = []
    for layer in layers:
        normalized = _normalize_hidden_state_index(int(layer), n_layers)
        if normalized == 0:
            unsupported.append((int(layer), "embedding hidden state"))
            continue
        if normalized == n_layers:
            unsupported.append((int(layer), "final post-norm hidden state"))
            continue
        mapped[int(layer)] = normalized - 1
    if unsupported:
        details = ", ".join(f"{layer} ({reason})" for layer, reason in unsupported)
        raise ValueError(
            "--hidden-state-capture=hooks cannot capture hidden-state index(es): "
            f"{details}. Use --hidden-state-capture=outputs for exact HF hidden_states semantics."
        )
    return mapped


def _extract_layer_hidden(output):
    hidden, _ = TruthProbe._unpack_output(output)
    return hidden


def _forward_with_selected_hidden_states(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: Sequence[int],
    hidden_state_capture: str,
):
    if hidden_state_capture not in HIDDEN_STATE_CAPTURE_METHODS:
        raise ValueError(f"hidden_state_capture must be one of {HIDDEN_STATE_CAPTURE_METHODS}.")

    if hidden_state_capture == "outputs":
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return out, {int(layer): out.hidden_states[int(layer)] for layer in layers}

    layer_to_module = _hook_capture_layer_map(model, layers)
    transformer_layers = TruthProbe._find_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for requested_layer, module_idx in layer_to_module.items():
        def _capture(_module, _input, output, *, layer=requested_layer):
            captured[layer] = _extract_layer_hidden(output)

        handles.append(transformer_layers[module_idx].register_forward_hook(_capture))

    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer for layer in layers if int(layer) not in captured]
    if missing:
        raise ValueError(f"hook hidden-state capture missed layer(s): {missing}.")
    return out, captured


def _candidate_verification_prompt(stmt: Statement) -> str:
    if stmt.question:
        return (
            f"Question: {stmt.question.strip()}\n"
            f"Candidate answer: {stmt.answer.strip()}\n"
            "Is the candidate answer factually correct? Answer yes or no and give a brief reason."
        )
    return (
        f"Statement: {stmt.answer.strip()}\n"
        "Is this statement factually correct? Answer yes or no and give a brief reason."
    )


def _chunked(items: Sequence, size: int):
    if size < 1:
        raise ValueError("batch size must be >= 1.")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _statement_length(stmt: Statement) -> int:
    return len(stmt.question) + len(stmt.answer)


def _statement_to_dump(stmt: Statement) -> dict[str, object]:
    text = stmt.answer.strip()
    if stmt.question.strip():
        text = f"{stmt.question.strip()} {text}".strip()
    return {
        "question": stmt.question,
        "answer": stmt.answer,
        "text": text,
        "is_false": int(stmt.is_false),
    }


def _batched_statements(statements: Sequence[Statement], size: int, *, length_bucketed: bool):
    if not length_bucketed:
        yield from _chunked(statements, size)
        return
    ordered = sorted(statements, key=_statement_length)
    yield from _chunked(ordered, size)


def _batched_statements_after_offset(
    statements: Sequence[Statement],
    size: int,
    *,
    length_bucketed: bool,
    offset: int,
):
    """Yield batches beginning after a processed statement offset."""
    cursor = 0
    offset = max(0, int(offset))
    for batch in _batched_statements(statements, size, length_bucketed=length_bucketed):
        end = cursor + len(batch)
        if end <= offset:
            cursor = end
            continue
        start = max(0, offset - cursor)
        remaining = batch[start:]
        if remaining:
            yield remaining
        cursor = end


def _inside_seed(base_seed: int, eval_batch_idx: int, inside_batch_idx: int) -> int:
    return int(base_seed) + int(eval_batch_idx) * 1_000_003 + int(inside_batch_idx)


def _profile_requested(args) -> bool:
    return bool(getattr(args, "profile", False) or getattr(args, "profile_json", None))


def _progress_report_due(completed: int, total: int, every: int, last_reported: int) -> bool:
    """Return whether a progress line should be printed for this counter state."""
    if every <= 0 or total <= 0:
        return False
    completed = min(int(completed), int(total))
    last_reported = min(int(last_reported), int(total))
    if completed <= last_reported:
        return False
    if completed >= total:
        return True
    next_mark = ((last_reported // every) + 1) * every
    return completed >= next_mark


def _format_progress(label: str, completed: int, total: int, elapsed_seconds: float) -> str:
    """Format a compact progress line for long benchmark phases."""
    completed = int(completed)
    total = int(total)
    elapsed_seconds = max(float(elapsed_seconds), 0.0)
    pct = (completed / total * 100.0) if total else 100.0
    rate = (completed / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    return f"   {label}: {completed}/{total} ({pct:.1f}%) elapsed={elapsed_seconds:.1f}s rate={rate:.2f}/s"


def _inside_trigger_indexes(records: Sequence[Mapping], args) -> set[int]:
    """Return batch-local record indexes selected for sampled INSIDE scoring."""
    signal = getattr(args, "inside_trigger_signal", None)
    if not signal:
        return set(range(len(records)))
    if not records:
        return set()

    values = [float(record["primary_scores"][signal]) for record in records]
    direction = DEFAULT_SCORE_DIRECTIONS.get(signal, "higher")
    threshold = getattr(args, "inside_trigger_threshold", None)
    if threshold is not None:
        threshold = float(threshold)
        if direction == "higher":
            return {idx for idx, value in enumerate(values) if value > threshold}
        return {idx for idx, value in enumerate(values) if value < threshold}

    fraction = getattr(args, "inside_trigger_top_fraction", None)
    if fraction is None:
        return set(range(len(records)))
    k = max(1, math.ceil(len(records) * float(fraction)))
    reverse = direction == "higher"
    ranked = sorted(range(len(records)), key=lambda idx: values[idx], reverse=reverse)
    return set(ranked[:k])


def _empty_inside_scores(layers: Sequence[int]) -> dict[int, float]:
    return {int(layer): 0.0 for layer in layers}


def _score_reps_batch(
    statements: Sequence[Statement],
    reps_batch: Sequence[Optional[Mapping]],
    *,
    layers: Sequence[int],
    target_layer: int,
    manifolds: Mapping[int, TruthManifold],
    subspaces: Mapping[int, TruthSubspace],
) -> list[dict]:
    valid = [(stmt, reps) for stmt, reps in zip(statements, reps_batch) if reps is not None]
    records = [
        {
            "stmt": stmt,
            "layer_scores": {},
            "primary_scores": {},
            "inside_scores": None,
            "inside_sampled": False,
        }
        for stmt, _ in valid
    ]
    if not records:
        return records

    for layer in layers:
        manifold = manifolds[layer]
        states = torch.stack([
            reps["last"][layer].to(manifold.mean.device) for _, reps in valid
        ])
        maha_values = mahalanobis_distance(states, manifold.mean, manifold.cov_inv).detach().cpu().tolist()
        if manifold.contrastive_direction is not None:
            proj_values = (-(states @ manifold.contrastive_direction.to(states.device))).detach().cpu().tolist()
        else:
            proj_values = [0.0] * len(records)

        subspace = subspaces.get(layer)
        if subspace is not None and subspace.is_ready():
            resid_values = subspace.residual_distance(states).detach().cpu().tolist()
        else:
            resid_values = [0.0] * len(records)

        eigenscore_values = [float(reps["eigenscore_by_layer"][layer]) for _, reps in valid]
        for record, maha, proj, resid, eigenscore in zip(
            records, maha_values, proj_values, resid_values, eigenscore_values
        ):
            record["layer_scores"][layer] = {
                "maha_last": float(maha),
                "truth_proj": float(proj),
                "subspace_resid": float(resid),
                "eigenscore": float(eigenscore),
            }

    for record, (_, reps) in zip(records, valid):
        ans = reps["ans_hs"]
        record["primary_scores"] = {
            **record["layer_scores"][target_layer],
            "disp_euclid": float(euclidean_dispersion(ans).item()),
            "disp_hse": float(hyperbolic_semantic_entropy(poincare_map(ans)).item()),
            "nll_answer": float(reps["nll"]),
        }

    return records


@contextmanager
def _profile_phase(profile: dict[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - started)


def _profile_payload(
    profile: Mapping[str, float],
    total_seconds: float,
    *,
    n_eval_records: int | None = None,
    n_warmup_true: int | None = None,
    n_warmup_false: int | None = None,
) -> dict:
    phases = {name: round(float(seconds), 6) for name, seconds in profile.items()}
    return {
        "total_seconds": round(float(total_seconds), 6),
        "phases": phases,
        "summary": _profile_summary(
            phases,
            total_seconds,
            n_eval_records=n_eval_records,
            n_warmup_true=n_warmup_true,
            n_warmup_false=n_warmup_false,
        ),
    }


def _profile_summary(
    profile: Mapping[str, float],
    total_seconds: float,
    *,
    n_eval_records: int | None = None,
    n_warmup_true: int | None = None,
    n_warmup_false: int | None = None,
) -> dict:
    total = max(float(total_seconds), 0.0)
    phases = {name: max(float(seconds), 0.0) for name, seconds in profile.items()}
    phase_total = sum(phases.values())
    top_phases = [
        {
            "name": name,
            "seconds": round(seconds, 6),
            "share": _profile_share(seconds, total),
        }
        for name, seconds in sorted(phases.items(), key=lambda item: item[1], reverse=True)[:5]
    ]
    groups = {}
    for group, names in PROFILE_GROUPS.items():
        seconds = sum(phases.get(name, 0.0) for name in names)
        groups[group] = {
            "seconds": round(seconds, 6),
            "share": _profile_share(seconds, total),
        }

    throughput = {}
    eval_records = max(int(n_eval_records or 0), 0)
    warmup_records = max(int(n_warmup_true or 0), 0) + max(int(n_warmup_false or 0), 0)
    if eval_records and phases.get("forced_answer_forward", 0.0) > 0:
        throughput["forced_answer_records_per_second"] = round(
            eval_records / phases["forced_answer_forward"],
            6,
        )
    if warmup_records and phases.get("build_layer_stats", 0.0) > 0:
        throughput["warmup_records_per_second"] = round(
            warmup_records / phases["build_layer_stats"],
            6,
        )
    if eval_records and total > 0:
        throughput["end_to_end_eval_records_per_second"] = round(eval_records / total, 6)

    return {
        "bottleneck": top_phases[0]["name"] if top_phases else None,
        "top_phases": top_phases,
        "groups": groups,
        "throughput": throughput,
        "accounted_seconds": round(phase_total, 6),
        "accounted_share": _profile_share(phase_total, total),
        "unaccounted_seconds": round(max(total - phase_total, 0.0), 6),
    }


def _profile_share(seconds: float, total_seconds: float) -> float:
    if total_seconds <= 0:
        return 0.0
    return round(float(seconds) / float(total_seconds), 6)


def _read_cache_metadata(path: str | Path) -> dict:
    cache_path = Path(path)
    if _is_sharded_eval_reps_cache(cache_path):
        manifest = _load_eval_reps_manifest(cache_path)
        return dict(manifest.get("metadata", {}))
    if cache_path.is_dir():
        raise ValueError(f"cache directory is missing manifest.json: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    return dict(cache.get("metadata", {}))


def _warmup_text_fingerprint(true_texts: Sequence[str], false_texts: Sequence[str]) -> str:
    payload = json.dumps(
        {"true": list(true_texts), "false": list(false_texts)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _statement_fingerprint(statements: Sequence[Statement]) -> str:
    payload = json.dumps(
        [
            {"question": stmt.question, "answer": stmt.answer, "is_false": int(stmt.is_false)}
            for stmt in statements
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _layer_stats_cache_metadata(
    args,
    *,
    layers: Sequence[int],
    n_layers: int,
    true_texts: Sequence[str],
    false_texts: Sequence[str],
) -> dict:
    return {
        "format": 1,
        "model": args.model,
        "dtype": args.dtype,
        "offline": bool(args.offline),
        "n_layers": int(n_layers),
        "layers": [int(layer) for layer in layers],
        "max_length": int(args.max_length),
        "subspace_rank": int(args.subspace_rank),
        "length_bucketed_batches": bool(args.length_bucketed_batches),
        "n_true": len(true_texts),
        "n_false": len(false_texts),
        "warmup_fingerprint": _warmup_text_fingerprint(true_texts, false_texts),
    }


def _statement_encoding_cache_metadata(
    args,
    *,
    true_texts: Sequence[str],
    false_texts: Sequence[str],
    eval_statements: Sequence[Statement],
) -> dict:
    return {
        "format": 1,
        "model": args.model,
        "offline": bool(args.offline),
        "max_length": int(args.max_length),
        "n_true": len(true_texts),
        "n_false": len(false_texts),
        "n_eval": len(eval_statements),
        "warmup_fingerprint": _warmup_text_fingerprint(true_texts, false_texts),
        "eval_fingerprint": _statement_fingerprint(eval_statements),
    }


def _encoding_list_to_json(
    encodings: Sequence[Optional[StatementEncoding]],
) -> list[dict[str, object] | None]:
    return [None if encoding is None else encoding.to_dict() for encoding in encodings]


def _encoding_list_from_json(values: Sequence[object]) -> list[Optional[StatementEncoding]]:
    return [_coerce_statement_encoding(value) for value in values]


def save_statement_encoding_cache(
    path: str | Path,
    *,
    metadata: Mapping,
    true_encodings: Sequence[Optional[StatementEncoding]],
    false_encodings: Sequence[Optional[StatementEncoding]],
    eval_encodings: Sequence[Optional[StatementEncoding]],
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata),
        "true": _encoding_list_to_json(true_encodings),
        "false": _encoding_list_to_json(false_encodings),
        "eval": _encoding_list_to_json(eval_encodings),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def load_statement_encoding_cache(
    path: str | Path,
    *,
    expected_metadata: Mapping,
) -> tuple[
    list[Optional[StatementEncoding]],
    list[Optional[StatementEncoding]],
    list[Optional[StatementEncoding]],
    dict,
]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    metadata = dict(payload.get("metadata", {}))
    _validate_cache_metadata(metadata, expected_metadata, cache_name="statement encoding cache")
    true_encodings = _encoding_list_from_json(payload.get("true", []))
    false_encodings = _encoding_list_from_json(payload.get("false", []))
    eval_encodings = _encoding_list_from_json(payload.get("eval", []))
    if len(true_encodings) != int(expected_metadata["n_true"]):
        raise ValueError("statement encoding cache true record count does not match this run.")
    if len(false_encodings) != int(expected_metadata["n_false"]):
        raise ValueError("statement encoding cache false record count does not match this run.")
    if len(eval_encodings) != int(expected_metadata["n_eval"]):
        raise ValueError("statement encoding cache eval record count does not match this run.")
    return true_encodings, false_encodings, eval_encodings, metadata


def _tensor_to_cpu(value):
    return value.detach().cpu() if value is not None else None


def _manifold_state(manifold: TruthManifold) -> dict:
    return {
        "mean": _tensor_to_cpu(manifold.mean),
        "_M2": _tensor_to_cpu(manifold._M2),  # noqa: SLF001 - benchmark cache mirrors core serialization.
        "n": int(manifold.n),
        "hidden_dim": int(manifold.hidden_dim),
        "ridge_lambda": float(manifold.ridge_lambda),
        "false_mean": _tensor_to_cpu(manifold.false_mean),
        "contrastive_direction": _tensor_to_cpu(manifold.contrastive_direction),
    }


def _manifold_from_state(state: Mapping, device: torch.device) -> TruthManifold:
    manifold = TruthManifold()
    manifold.mean = state["mean"]
    manifold._M2 = state.get("_M2")  # noqa: SLF001 - benchmark cache mirrors core serialization.
    manifold.n = int(state["n"])
    manifold.hidden_dim = int(state["hidden_dim"])
    manifold.ridge_lambda = float(state.get("ridge_lambda", 0.1))
    manifold.false_mean = state.get("false_mean")
    manifold.contrastive_direction = state.get("contrastive_direction")
    manifold._dirty = True  # noqa: SLF001
    return manifold.to(device)


def _subspace_state(subspace: TruthSubspace) -> dict:
    return {
        "mean": _tensor_to_cpu(subspace.mean),
        "basis": _tensor_to_cpu(subspace.basis),
        "rank": int(subspace.rank),
        "false_mean": _tensor_to_cpu(subspace.false_mean),
        "contrastive_direction": _tensor_to_cpu(subspace.contrastive_direction),
    }


def _subspace_from_state(state: Mapping, device: torch.device) -> TruthSubspace:
    return TruthSubspace(
        mean=state.get("mean"),
        basis=state.get("basis"),
        rank=int(state.get("rank", 0)),
        false_mean=state.get("false_mean"),
        contrastive_direction=state.get("contrastive_direction"),
    ).to(device)


def _validate_cache_metadata(actual: Mapping, expected: Mapping, *, cache_name: str) -> None:
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in expected
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected {values['expected']!r}, got {values['actual']!r}"
            for key, values in mismatches.items()
        )
        raise ValueError(f"{cache_name} metadata does not match this run ({details}).")


def _tensor_list_state(values: Mapping[int, Sequence[torch.Tensor]]) -> dict[int, list[torch.Tensor]]:
    return {
        int(layer): [_tensor_to_cpu(value) for value in layer_values]
        for layer, layer_values in values.items()
    }


def _tensor_list_from_state(state: Mapping, device: torch.device) -> dict[int, list[torch.Tensor]]:
    return {
        int(layer): [value.to(device) for value in layer_values]
        for layer, layer_values in state.items()
    }


def _tensor_mapping_state(values: Mapping[int, torch.Tensor | None]) -> dict[int, torch.Tensor | None]:
    return {int(layer): _tensor_to_cpu(value) for layer, value in values.items()}


def _tensor_mapping_from_state(state: Mapping, device: torch.device) -> dict[int, torch.Tensor | None]:
    return {
        int(layer): None if value is None else value.to(device)
        for layer, value in state.items()
    }


def save_layer_stats_cache(
    path: str | Path,
    manifolds: Mapping[int, TruthManifold],
    subspaces: Mapping[int, TruthSubspace],
    *,
    metadata: Mapping,
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": dict(metadata),
            "manifolds": {int(layer): _manifold_state(manifold) for layer, manifold in manifolds.items()},
            "subspaces": {int(layer): _subspace_state(subspace) for layer, subspace in subspaces.items()},
        },
        cache_path,
    )


def load_layer_stats_cache(
    path: str | Path,
    *,
    expected_metadata: Mapping,
    device: torch.device,
) -> tuple[dict[int, TruthManifold], dict[int, TruthSubspace], dict]:
    cache = torch.load(Path(path), map_location=device, weights_only=True)
    metadata = dict(cache.get("metadata", {}))
    _validate_cache_metadata(metadata, expected_metadata, cache_name="layer stats cache")
    manifolds = {
        int(layer): _manifold_from_state(state, device)
        for layer, state in cache.get("manifolds", {}).items()
    }
    subspaces = {
        int(layer): _subspace_from_state(state, device)
        for layer, state in cache.get("subspaces", {}).items()
    }
    missing = set(expected_metadata["layers"]) - set(manifolds)
    if missing:
        raise ValueError(f"layer stats cache is missing manifold(s) for layer(s): {sorted(missing)}.")
    return manifolds, subspaces, metadata


def save_warmup_checkpoint(
    path: str | Path,
    *,
    metadata: Mapping,
    manifolds: Mapping[int, TruthManifold],
    true_state_lists: Mapping[int, Sequence[torch.Tensor]],
    false_state_lists: Mapping[int, Sequence[torch.Tensor]],
    false_sums: Mapping[int, torch.Tensor | None],
    n_false: int,
    true_done: int,
    false_done: int,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": 1,
            "metadata": dict(metadata),
            "progress": {"true_done": int(true_done), "false_done": int(false_done)},
            "manifolds": {int(layer): _manifold_state(manifold) for layer, manifold in manifolds.items()},
            "true_state_lists": _tensor_list_state(true_state_lists),
            "false_state_lists": _tensor_list_state(false_state_lists),
            "false_sums": _tensor_mapping_state(false_sums),
            "n_false": int(n_false),
        },
        checkpoint_path,
    )


def load_warmup_checkpoint(
    path: str | Path,
    *,
    expected_metadata: Mapping,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if int(checkpoint.get("format", 0)) != 1:
        raise ValueError("warmup checkpoint has an unsupported format.")
    metadata = dict(checkpoint.get("metadata", {}))
    _validate_cache_metadata(metadata, expected_metadata, cache_name="warmup checkpoint")

    expected_layers = {int(layer) for layer in expected_metadata["layers"]}
    manifolds = {
        int(layer): _manifold_from_state(state, device)
        for layer, state in checkpoint.get("manifolds", {}).items()
    }
    true_state_lists = _tensor_list_from_state(checkpoint.get("true_state_lists", {}), device)
    false_state_lists = _tensor_list_from_state(checkpoint.get("false_state_lists", {}), device)
    false_sums = _tensor_mapping_from_state(checkpoint.get("false_sums", {}), device)
    missing = expected_layers - set(manifolds)
    if missing:
        raise ValueError(f"warmup checkpoint is missing manifold(s) for layer(s): {sorted(missing)}.")
    for layer in expected_layers:
        true_state_lists.setdefault(layer, [])
        false_state_lists.setdefault(layer, [])
        false_sums.setdefault(layer, None)

    progress = dict(checkpoint.get("progress", {}))
    true_done = int(progress.get("true_done", 0))
    false_done = int(progress.get("false_done", 0))
    n_true = int(expected_metadata.get("n_true", 0))
    n_false_expected = int(expected_metadata.get("n_false", 0))
    if not (0 <= true_done <= n_true and 0 <= false_done <= n_false_expected):
        raise ValueError("warmup checkpoint progress is out of bounds for this run.")
    if false_done > 0 and true_done < n_true:
        raise ValueError("warmup checkpoint cannot resume false statements before true warmup is complete.")

    return {
        "metadata": metadata,
        "manifolds": manifolds,
        "true_state_lists": true_state_lists,
        "false_state_lists": false_state_lists,
        "false_sums": false_sums,
        "n_false": int(checkpoint.get("n_false", 0)),
        "true_done": true_done,
        "false_done": false_done,
    }


def _eval_reps_cache_metadata(
    args,
    *,
    layers: Sequence[int],
    n_layers: int,
    eval_statements: Sequence[Statement],
) -> dict:
    return {
        "format": 1,
        "model": args.model,
        "dtype": args.dtype,
        "offline": bool(args.offline),
        "n_layers": int(n_layers),
        "layers": [int(layer) for layer in layers],
        "max_length": int(args.max_length),
        "eigenscore_alpha": float(args.eigenscore_alpha),
        "length_bucketed_batches": bool(args.length_bucketed_batches),
        "n_eval": len(eval_statements),
        "eval_fingerprint": _statement_fingerprint(eval_statements),
    }


def _reps_to_cache_state(reps: Optional[Mapping]) -> Optional[dict]:
    if reps is None:
        return None
    return {
        "last": {int(layer): _tensor_to_cpu(value) for layer, value in reps["last"].items()},
        "ans_hs": _tensor_to_cpu(reps["ans_hs"]),
        "eigenscore_by_layer": {
            int(layer): float(value) for layer, value in reps["eigenscore_by_layer"].items()
        },
        "nll": float(reps["nll"]),
    }


def _reps_from_cache_state(state: Optional[Mapping]) -> Optional[dict]:
    if state is None:
        return None
    return {
        "last": {int(layer): value for layer, value in state["last"].items()},
        "ans_hs": state["ans_hs"],
        "eigenscore_by_layer": {
            int(layer): float(value) for layer, value in state["eigenscore_by_layer"].items()
        },
        "nll": float(state["nll"]),
    }


def _eval_reps_manifest_path(path: str | Path) -> Path:
    return Path(path) / "manifest.json"


def _is_sharded_eval_reps_cache(path: str | Path) -> bool:
    return Path(path).is_dir() and _eval_reps_manifest_path(path).exists()


def _load_eval_reps_manifest(path: str | Path) -> dict:
    with open(_eval_reps_manifest_path(path), encoding="utf-8") as f:
        return json.load(f)


def _existing_eval_reps_shard_size(path: str | Path) -> int:
    if not _is_sharded_eval_reps_cache(path):
        return 0
    return int(_load_eval_reps_manifest(path).get("shard_size", 0))


def _validate_eval_reps_manifest(manifest: Mapping, expected_records: int) -> None:
    if int(manifest.get("format", 0)) != 2:
        raise ValueError("sharded eval reps cache manifest has an unsupported format.")
    if int(manifest.get("record_count", -1)) != int(expected_records):
        raise ValueError(
            f"eval reps cache record count does not match this run "
            f"(expected {expected_records}, got {manifest.get('record_count')})."
        )
    offset = 0
    for shard in manifest.get("shards", []):
        start = int(shard.get("start", -1))
        count = int(shard.get("count", -1))
        if start != offset or count < 0:
            raise ValueError("sharded eval reps cache manifest has non-contiguous shard ranges.")
        offset += count
    if offset != int(expected_records):
        raise ValueError("sharded eval reps cache manifest shard counts do not match record_count.")


class EvalRepsCacheReader:
    def __init__(
        self,
        path: str | Path,
        *,
        expected_metadata: Mapping,
        expected_records: int,
    ) -> None:
        self.path = Path(path)
        self.metadata: dict
        self.record_count = int(expected_records)
        self._records: list[Optional[dict]] | None = None
        self._shards: list[dict] = []
        if _is_sharded_eval_reps_cache(self.path):
            manifest = _load_eval_reps_manifest(self.path)
            self.metadata = dict(manifest.get("metadata", {}))
            _validate_cache_metadata(self.metadata, expected_metadata, cache_name="eval reps cache")
            _validate_eval_reps_manifest(manifest, expected_records)
            self.record_count = int(manifest["record_count"])
            self._shards = [dict(shard) for shard in manifest.get("shards", [])]
        else:
            if self.path.is_dir():
                raise ValueError(f"eval reps cache directory is missing manifest.json: {self.path}")
            cache = torch.load(self.path, map_location="cpu", weights_only=True)
            self.metadata = dict(cache.get("metadata", {}))
            _validate_cache_metadata(self.metadata, expected_metadata, cache_name="eval reps cache")
            self._records = [_reps_from_cache_state(record) for record in cache.get("records", [])]
            if len(self._records) != int(expected_records):
                raise ValueError(
                    f"eval reps cache record count does not match this run "
                    f"(expected {expected_records}, got {len(self._records)})."
                )
            self.record_count = len(self._records)

    def read_range(self, start: int, count: int) -> list[Optional[dict]]:
        start = int(start)
        count = int(count)
        if start < 0 or count < 0 or start + count > self.record_count:
            raise ValueError("eval reps cache read range is out of bounds.")
        if count == 0:
            return []
        if self._records is not None:
            return self._records[start:start + count]

        end = start + count
        records: list[Optional[dict]] = []
        for shard in self._shards:
            shard_start = int(shard["start"])
            shard_count = int(shard["count"])
            shard_end = shard_start + shard_count
            if shard_end <= start:
                continue
            if shard_start >= end:
                break
            shard_payload = torch.load(self.path / shard["path"], map_location="cpu", weights_only=True)
            raw_records = shard_payload.get("records", [])
            if int(shard_payload.get("start", -1)) != shard_start or len(raw_records) != shard_count:
                raise ValueError("sharded eval reps cache shard payload does not match manifest.")
            local_start = max(start, shard_start) - shard_start
            local_end = min(end, shard_end) - shard_start
            records.extend(_reps_from_cache_state(record) for record in raw_records[local_start:local_end])
        if len(records) != count:
            raise ValueError(f"eval reps cache returned {len(records)} records for a {count}-record range.")
        return records

    def read_all(self) -> list[Optional[dict]]:
        return self.read_range(0, self.record_count)


class EvalRepsCacheWriter:
    def __init__(self, path: str | Path, *, metadata: Mapping, shard_size: int) -> None:
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.shard_size = int(shard_size)
        if self.shard_size < 1:
            raise ValueError("eval reps cache shard size must be >=1.")
        if self.path.exists() and not self.path.is_dir():
            raise ValueError("sharded eval reps cache path must be a directory path.")
        self.path.mkdir(parents=True, exist_ok=True)
        manifest_path = _eval_reps_manifest_path(self.path)
        if manifest_path.exists():
            manifest_path.unlink()
        for stale in self.path.glob("records-*.pt"):
            stale.unlink()
        self._buffer: list[Optional[dict]] = []
        self._record_count = 0
        self._shards: list[dict] = []

    def extend(self, reps_records: Sequence[Optional[Mapping]]) -> None:
        self._buffer.extend(_reps_to_cache_state(reps) for reps in reps_records)
        while len(self._buffer) >= self.shard_size:
            self._flush(self.shard_size)

    def close(self) -> None:
        self._flush(len(self._buffer))
        manifest = {
            "format": 2,
            "metadata": self.metadata,
            "record_count": self._record_count,
            "shard_size": self.shard_size,
            "shards": self._shards,
        }
        with open(_eval_reps_manifest_path(self.path), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    def _flush(self, count: int) -> None:
        if count <= 0:
            return
        records = self._buffer[:count]
        del self._buffer[:count]
        shard_name = f"records-{len(self._shards):06d}.pt"
        shard_payload = {
            "start": self._record_count,
            "records": records,
        }
        torch.save(shard_payload, self.path / shard_name)
        self._shards.append({
            "path": shard_name,
            "start": self._record_count,
            "count": len(records),
        })
        self._record_count += len(records)


def save_eval_reps_cache(
    path: str | Path,
    reps_records: Sequence[Optional[Mapping]],
    *,
    metadata: Mapping,
    shard_size: int = 0,
) -> None:
    if int(shard_size) > 0:
        writer = EvalRepsCacheWriter(path, metadata=metadata, shard_size=int(shard_size))
        writer.extend(reps_records)
        writer.close()
        return
    cache_path = Path(path)
    if cache_path.exists() and cache_path.is_dir():
        raise ValueError("single-file eval reps cache path must not be an existing directory.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": dict(metadata),
            "records": [_reps_to_cache_state(reps) for reps in reps_records],
        },
        cache_path,
    )


def load_eval_reps_cache(
    path: str | Path,
    *,
    expected_metadata: Mapping,
    expected_records: int,
) -> tuple[list[Optional[dict]], dict]:
    reader = EvalRepsCacheReader(path, expected_metadata=expected_metadata, expected_records=expected_records)
    return reader.read_all(), reader.metadata


# ---------------------------------------------------------------------------
# 离线烟雾集（仅用于验证管线，不构成基准）/ Offline smoke set (pipeline check only)
# ---------------------------------------------------------------------------

_TRUE_SMOKE = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth revolves around the Sun.",
    "Humans have two lungs.",
    "The Pacific is the largest ocean on Earth.",
    "Ice is frozen water.",
    "A triangle has three sides.",
    "The Sun rises in the east.",
    "Honey is made by bees.",
    "Mount Everest is the tallest mountain above sea level.",
]
_FALSE_SMOKE = [
    "The capital of France is Berlin.",
    "Water boils at 30 degrees Celsius at sea level.",
    "The Sun revolves around the Earth.",
    "Humans have five lungs.",
    "The Pacific is the smallest ocean on Earth.",
    "Ice is boiling water.",
    "A triangle has seven sides.",
    "The Sun rises in the west.",
    "Honey is made by spiders.",
    "Mount Everest is the shortest mountain on Earth.",
]


def load_offline() -> tuple[List[str], List[str], List[Statement]]:
    """返回 (流形构建用真陈述, 对比方向用假陈述, 评测陈述)。"""
    manifold_true = _TRUE_SMOKE[:6]
    manifold_false = _FALSE_SMOKE[:6]
    eval_stmts: List[Statement] = []
    for t in _TRUE_SMOKE[6:]:
        eval_stmts.append(Statement("", t, 0))
    for f in _FALSE_SMOKE[6:]:
        eval_stmts.append(Statement("", f, 1))
    return manifold_true, manifold_false, eval_stmts


def load_truthfulqa(
    manifold_questions: int, limit: int
) -> tuple[List[str], List[str], List[Statement]]:
    """加载 TruthfulQA multiple_choice，切分为流形集 / 评测集（题目层面不重叠）。

    流形集题目的正确答案用于构建真值流形，错误答案仅用于 mass-mean 对比方向。
    Correct answers of manifold-split questions build the truth manifold; their
    incorrect answers are used only for the mass-mean contrastive direction.
    """
    from datasets import load_dataset  # lazy

    # 新旧版 datasets 的数据集 id 不同，依次尝试 / dataset id differs across versions
    last_err: Optional[Exception] = None
    ds = None
    for dataset_id in ("truthfulqa/truthful_qa", "truthful_qa"):
        try:
            ds = load_dataset(dataset_id, "multiple_choice")["validation"]
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if ds is None:
        raise RuntimeError(
            f"Could not load TruthfulQA. Last error: {last_err}. "
            f"Try `pip install -U datasets` or run with --offline for a pipeline check."
        )

    manifold_true: List[str] = []
    manifold_false: List[str] = []
    eval_stmts: List[Statement] = []
    for i, row in enumerate(ds):
        q = row["question"]
        targets = row["mc2_targets"]
        choices, labels = targets["choices"], targets["labels"]
        if i < manifold_questions:
            for c, lab in zip(choices, labels):
                if lab == 1:
                    manifold_true.append(f"{q} {c}")
                else:
                    manifold_false.append(f"{q} {c}")
        else:
            for c, lab in zip(choices, labels):
                eval_stmts.append(Statement(q, c, is_false=int(lab == 0)))
        if limit and (i - manifold_questions + 1) >= limit and i >= manifold_questions:
            break
    return manifold_true, manifold_false, eval_stmts


# ---------------------------------------------------------------------------
# 模型与表征提取 / Model and representation extraction
# ---------------------------------------------------------------------------

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_model(model_name: str, device: torch.device, dtype: str = "float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dtype= 替代已弃用的 torch_dtype=；bfloat16 减半权重内存；
    # low_cpu_mem_usage 避免加载时 2× 峰值内存（低内存机器关键）
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=_DTYPES[dtype], low_cpu_mem_usage=True
    )
    model.to(device).eval()
    return model, tokenizer


def resolve_target_layer(layer: int, n_layers: int, *, offline: bool) -> int:
    """Return a hidden-state layer index valid for the loaded model."""
    if -(n_layers + 1) <= layer <= n_layers:
        return layer
    if offline and layer == -8:
        print(f"[!] Requested layer {layer} is unavailable for this tiny smoke model; using -1.\n")
        return -1
    raise ValueError(
        f"layer {layer} is out of range for a model with {n_layers} transformer layers "
        f"(valid hidden-state indexes: {-n_layers - 1}..{n_layers})."
    )


def _statement_token_ids(tokenizer, stmt: Statement, max_length: int) -> Optional[tuple[list[int], int]]:
    encoding = _statement_encoding(tokenizer, stmt, max_length)
    if encoding is None:
        return None
    return list(encoding.input_ids), int(encoding.n_answer_tokens)


def _statement_encoding(tokenizer, stmt: Statement, max_length: int) -> Optional[StatementEncoding]:
    q_ids = tokenizer(stmt.question, add_special_tokens=True).input_ids if stmt.question \
        else tokenizer(tokenizer.bos_token or tokenizer.eos_token or " ").input_ids
    a_ids = tokenizer(" " + stmt.answer.strip(), add_special_tokens=False).input_ids
    if len(a_ids) == 0:
        return None
    ids = (q_ids + a_ids)[:max_length]
    n_ans = len(ids) - len(q_ids)
    if n_ans <= 0:
        return None
    return StatementEncoding(tuple(ids), n_ans)


def _input_ids_from_tokenizer_output(output) -> list[list[int]]:
    input_ids = output["input_ids"] if isinstance(output, Mapping) else output.input_ids
    if isinstance(input_ids, torch.Tensor):
        return input_ids.detach().cpu().tolist()
    return [list(ids) for ids in input_ids]


def _batch_statement_token_ids(
    tokenizer,
    statements: Sequence[Statement],
    max_length: int,
) -> list[Optional[tuple[list[int], int]]]:
    return [
        None if encoding is None else (list(encoding.input_ids), int(encoding.n_answer_tokens))
        for encoding in _batch_statement_encodings(tokenizer, statements, max_length)
    ]


def _batch_statement_encodings(
    tokenizer,
    statements: Sequence[Statement],
    max_length: int,
) -> list[Optional[StatementEncoding]]:
    if not statements:
        return []

    encoded: list[Optional[StatementEncoding]] = [None] * len(statements)
    fallback_prompt = tokenizer.bos_token or tokenizer.eos_token or " "

    question_positions = [idx for idx, stmt in enumerate(statements) if stmt.question]
    if question_positions:
        question_output = tokenizer(
            [statements[idx].question for idx in question_positions],
            add_special_tokens=True,
        )
        question_ids_by_position = dict(zip(
            question_positions,
            _input_ids_from_tokenizer_output(question_output),
        ))
    else:
        question_ids_by_position = {}

    fallback_q_ids = tokenizer(fallback_prompt).input_ids if len(question_positions) < len(statements) else []

    answer_output = tokenizer(
        [" " + stmt.answer.strip() for stmt in statements],
        add_special_tokens=False,
    )
    answer_ids_batch = _input_ids_from_tokenizer_output(answer_output)

    for idx, (stmt, answer_ids) in enumerate(zip(statements, answer_ids_batch)):
        if len(answer_ids) == 0:
            continue
        question_ids = question_ids_by_position.get(idx, fallback_q_ids)
        ids = (list(question_ids) + list(answer_ids))[:max_length]
        n_ans = len(ids) - len(question_ids)
        if n_ans <= 0:
            continue
        encoded[idx] = StatementEncoding(tuple(ids), n_ans)
    return encoded


def _coerce_statement_encoding(value) -> Optional[StatementEncoding]:
    if value is None:
        return None
    if isinstance(value, StatementEncoding):
        return value
    if isinstance(value, Mapping):
        return StatementEncoding(value["input_ids"], int(value["n_answer_tokens"]))
    ids, n_ans = value
    return StatementEncoding(ids, int(n_ans))


def _encoding_to_token_ids(value) -> Optional[tuple[list[int], int]]:
    encoding = _coerce_statement_encoding(value)
    if encoding is None:
        return None
    return list(encoding.input_ids), int(encoding.n_answer_tokens)


def _batched_statement_pairs(
    statements: Sequence[Statement],
    encodings: Sequence[Optional[StatementEncoding]] | None,
    size: int,
    *,
    length_bucketed: bool,
):
    if size < 1:
        raise ValueError("batch size must be >= 1.")
    if encodings is not None and len(encodings) != len(statements):
        raise ValueError("statement encodings must have the same length as statements.")
    if encodings is None:
        pairs = [(stmt, None) for stmt in statements]
    else:
        pairs = list(zip(statements, encodings))
    if length_bucketed:
        pairs = sorted(pairs, key=lambda pair: _statement_length(pair[0]))
    yield from _chunked(pairs, size)


def _batched_statement_pairs_after_offset(
    statements: Sequence[Statement],
    encodings: Sequence[Optional[StatementEncoding]] | None,
    size: int,
    *,
    length_bucketed: bool,
    offset: int,
):
    cursor = 0
    offset = max(0, int(offset))
    for batch in _batched_statement_pairs(statements, encodings, size, length_bucketed=length_bucketed):
        end = cursor + len(batch)
        if end <= offset:
            cursor = end
            continue
        start = max(0, offset - cursor)
        remaining = batch[start:]
        if remaining:
            yield remaining
        cursor = end


@torch.no_grad()
def batched_statement_reps(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    compute_answer_metrics: bool = True,
    eigenscore_alpha: float = 1e-3,
    hidden_state_capture: str = "outputs",
    encoded_statements: Sequence[Optional[StatementEncoding]] | None = None,
) -> list[Optional[dict]]:
    """Batch forced-answer forwards while preserving per-statement result shape."""
    encoded: list[tuple[int, list[int], int]] = []
    results: list[Optional[dict]] = [None] * len(statements)
    if encoded_statements is None:
        tokenized_statements = _batch_statement_token_ids(tokenizer, statements, max_length)
    else:
        if len(encoded_statements) != len(statements):
            raise ValueError("encoded_statements must have the same length as statements.")
        tokenized_statements = [_encoding_to_token_ids(encoding) for encoding in encoded_statements]
    for idx, tokenized in enumerate(tokenized_statements):
        if tokenized is None:
            continue
        ids, n_ans = tokenized
        encoded.append((idx, ids, n_ans))
    if not encoded:
        return results

    pad_token_id = _pad_token_id(tokenizer)
    batch_len = max(len(ids) for _, ids, _ in encoded)
    input_ids = torch.full((len(encoded), batch_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, (_, ids, _) in enumerate(encoded):
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[row, :len(ids)] = ids_t
        attention_mask[row, :len(ids)] = 1

    out, hidden_by_layer = _forward_with_selected_hidden_states(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        layers=layers,
        hidden_state_capture=hidden_state_capture,
    )
    for row, (original_idx, ids, n_ans) in enumerate(encoded):
        seq_len = len(ids)
        last_by_layer = {
            layer: hidden_by_layer[layer][row, seq_len - 1, :].float().cpu()
            for layer in layers
        }
        if not compute_answer_metrics:
            results[original_idx] = {"last": last_by_layer}
            continue

        ans_start = seq_len - n_ans
        ans_hs = hidden_by_layer[layers[0]][row, ans_start:seq_len, :].float().cpu()
        eigenscore_by_layer = {
            layer: float(internal_eigenscore(
                hidden_by_layer[layer][row, ans_start:seq_len, :].float(),
                alpha=eigenscore_alpha,
            ).item())
            for layer in layers
        }

        logits = out.logits[row, :seq_len, :].float()
        logp = torch.log_softmax(logits[:-1], dim=-1)
        targets = input_ids[row, 1:seq_len]
        tok_logp = logp[torch.arange(logp.shape[0], device=device), targets]
        ans_logp = tok_logp[-n_ans:] if n_ans <= tok_logp.shape[0] else tok_logp
        nll = float((-ans_logp.mean()).item())
        results[original_idx] = {
            "last": last_by_layer,
            "ans_hs": ans_hs,
            "eigenscore_by_layer": eigenscore_by_layer,
            "nll": nll,
        }
    return results


def statement_reps(model, tokenizer, stmt: Statement, layers: List[int],
                   device: torch.device, max_length: int, *,
                   compute_answer_metrics: bool = True,
                   eigenscore_alpha: float = 1e-3,
                   hidden_state_capture: str = "outputs") -> Optional[dict]:
    """Single-statement compatibility wrapper around batched_statement_reps."""
    return batched_statement_reps(
        model,
        tokenizer,
        [stmt],
        layers,
        device,
        max_length,
        compute_answer_metrics=compute_answer_metrics,
        eigenscore_alpha=eigenscore_alpha,
        hidden_state_capture=hidden_state_capture,
        encoded_statements=None,
    )[0]


def _pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return int(tokenizer.eos_token_id)
    raise ValueError("tokenizer must define either pad_token_id or eos_token_id for sampling.")


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index or torch.cuda.current_device()]


@torch.no_grad()
def sampled_response_embeddings_batch(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    hidden_state_capture: str = "outputs",
) -> list[Optional[dict[int, torch.Tensor]]]:
    """Generate multiple continuations per statement and pool response embeddings."""
    if n_samples < 2:
        return [None] * len(statements)
    if max_new_tokens < 1:
        raise ValueError("inside max_new_tokens must be >= 1.")
    if temperature <= 0.0:
        raise ValueError("inside temperature must be > 0.")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("inside top_p must be in (0, 1].")
    if pooling not in {"last", "mean"}:
        raise ValueError("inside pooling must be 'last' or 'mean'.")
    if not statements:
        return []

    prompts = [_candidate_verification_prompt(stmt) for stmt in statements]
    pad_token_id = _pad_token_id(tokenizer)
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    with torch.random.fork_rng(devices=_fork_rng_devices(device)):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            num_return_sequences=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_width = input_ids.shape[1]
    new_width = max(generated.shape[1] - prompt_width, 1)
    generated_attention = torch.cat([
        attention_mask.repeat_interleave(n_samples, dim=0),
        torch.ones((len(statements) * n_samples, new_width), dtype=attention_mask.dtype, device=device),
    ], dim=1)
    if generated_attention.shape[1] != generated.shape[1]:
        generated_attention = torch.ones_like(generated, dtype=attention_mask.dtype, device=device)

    _out, hidden_by_layer = _forward_with_selected_hidden_states(
        model,
        input_ids=generated,
        attention_mask=generated_attention,
        layers=layers,
        hidden_state_capture=hidden_state_capture,
    )
    results: list[Optional[dict[int, torch.Tensor]]] = [{layer: torch.empty(0) for layer in layers}
                                                        for _ in statements]
    response_start = min(prompt_width, generated.shape[1] - 1)
    response_end = generated.shape[1]
    for layer in layers:
        states = hidden_by_layer[layer].float()
        for stmt_idx in range(len(statements)):
            pooled = []
            for sample_idx in range(n_samples):
                row = stmt_idx * n_samples + sample_idx
                if pooling == "mean" and response_end > response_start:
                    pooled.append(states[row, response_start:response_end, :].mean(dim=0))
                else:
                    pooled.append(states[row, response_end - 1, :])
            results[stmt_idx][layer] = torch.stack(pooled).cpu()
    return results


@torch.no_grad()
def sampled_response_embeddings(
    model,
    tokenizer,
    stmt: Statement,
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    hidden_state_capture: str = "outputs",
) -> Optional[dict[int, torch.Tensor]]:
    return sampled_response_embeddings_batch(
        model,
        tokenizer,
        [stmt],
        layers,
        device,
        max_length,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pooling=pooling,
        seed=seed,
        hidden_state_capture=hidden_state_capture,
    )[0]


def sampled_inside_scores_batch(
    model,
    tokenizer,
    statements: Sequence[Statement],
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    eigenscore_alpha: float,
    hidden_state_capture: str = "outputs",
) -> list[Optional[dict[int, float]]]:
    embeddings_batch = sampled_response_embeddings_batch(
        model,
        tokenizer,
        statements,
        layers,
        device,
        max_length,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pooling=pooling,
        seed=seed,
        hidden_state_capture=hidden_state_capture,
    )
    scores_batch: list[Optional[dict[int, float]]] = []
    for embeddings in embeddings_batch:
        if embeddings is None:
            scores_batch.append(None)
            continue
        scores_batch.append({
            layer: float(internal_eigenscore(values, alpha=eigenscore_alpha).item())
            for layer, values in embeddings.items()
        })
    return scores_batch


def sampled_inside_scores(
    model,
    tokenizer,
    stmt: Statement,
    layers: List[int],
    device: torch.device,
    max_length: int,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pooling: str,
    seed: int,
    eigenscore_alpha: float,
    hidden_state_capture: str = "outputs",
) -> Optional[dict[int, float]]:
    return sampled_inside_scores_batch(
        model,
        tokenizer,
        [stmt],
        layers,
        device,
        max_length,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pooling=pooling,
        seed=seed,
        eigenscore_alpha=eigenscore_alpha,
        hidden_state_capture=hidden_state_capture,
    )[0]


def build_layer_stats(model, tokenizer, true_texts: List[str], false_texts: List[str],
                      layers: List[int], device: torch.device, max_length: int,
                      subspace_rank: int, batch_size: int, length_bucketed: bool = False,
                      progress_every: int = 50, checkpoint_path: str | Path | None = None,
                      checkpoint_metadata: Mapping | None = None, resume_checkpoint: bool = True,
                      checkpoint_every: int = 50,
                      hidden_state_capture: str = "outputs",
                      true_encodings: Sequence[Optional[StatementEncoding]] | None = None,
                      false_encodings: Sequence[Optional[StatementEncoding]] | None = None) -> tuple[dict, dict]:
    """逐层构建真值流形与 mass-mean 对比方向（与 EigenTruthWrapper.warmup 同构）。
    Per-layer truth manifolds plus the mass-mean contrastive direction
    (mirrors EigenTruthWrapper.warmup; cf. Marks & Tegmark mass-mean probing).
    """
    checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path else None
    checkpoint_metadata = dict(checkpoint_metadata or {})
    if checkpoint_path_obj and resume_checkpoint and checkpoint_path_obj.exists():
        loaded_checkpoint = load_warmup_checkpoint(
            checkpoint_path_obj,
            expected_metadata=checkpoint_metadata,
            device=device,
        )
        manifolds = loaded_checkpoint["manifolds"]
        true_state_lists = loaded_checkpoint["true_state_lists"]
        false_state_lists = loaded_checkpoint["false_state_lists"]
        false_sums = loaded_checkpoint["false_sums"]
        n_false = int(loaded_checkpoint["n_false"])
        true_done = int(loaded_checkpoint["true_done"])
        false_done = int(loaded_checkpoint["false_done"])
        print(f"   loaded warmup checkpoint: {checkpoint_path_obj} "
              f"(true={true_done}/{len(true_texts)}, false={false_done}/{len(false_texts)})")
    else:
        manifolds = {layer: TruthManifold() for layer in layers}
        true_state_lists = {layer: [] for layer in layers}
        false_state_lists = {layer: [] for layer in layers}
        false_sums: dict = {layer: None for layer in layers}
        n_false = 0
        true_done = 0
        false_done = 0

    true_statements = [Statement("", text, 0) for text in true_texts]
    true_last_reported = true_done
    true_last_checkpoint = true_done
    true_started = time.perf_counter()
    for batch_pairs in _batched_statement_pairs_after_offset(
        true_statements,
        true_encodings,
        batch_size,
        length_bucketed=length_bucketed,
        offset=true_done,
    ):
        batch = [stmt for stmt, _encoding in batch_pairs]
        batch_encodings = None if true_encodings is None else [encoding for _stmt, encoding in batch_pairs]
        reps_batch = batched_statement_reps(
            model,
            tokenizer,
            batch,
            layers,
            device,
            max_length,
            compute_answer_metrics=False,
            hidden_state_capture=hidden_state_capture,
            encoded_statements=batch_encodings,
        )
        for reps in reps_batch:
            if reps is None:
                continue
            for layer in layers:
                h = reps["last"][layer]
                manifolds[layer].update(h)
                true_state_lists[layer].append(h)
        true_done += len(batch)
        if _progress_report_due(true_done, len(true_statements), progress_every, true_last_reported):
            true_last_reported = min(true_done, len(true_statements))
            print(_format_progress("warmup true", true_last_reported, len(true_statements),
                                   time.perf_counter() - true_started))
        if checkpoint_path_obj and _progress_report_due(
            true_done, len(true_statements), checkpoint_every, true_last_checkpoint
        ):
            true_last_checkpoint = min(true_done, len(true_statements))
            save_warmup_checkpoint(
                checkpoint_path_obj,
                metadata=checkpoint_metadata,
                manifolds=manifolds,
                true_state_lists=true_state_lists,
                false_state_lists=false_state_lists,
                false_sums=false_sums,
                n_false=n_false,
                true_done=true_last_checkpoint,
                false_done=false_done,
            )
            print(f"   saved warmup checkpoint: {checkpoint_path_obj} "
                  f"(true={true_last_checkpoint}/{len(true_statements)}, false={false_done}/{len(false_texts)})")

    false_statements = [Statement("", text, 1) for text in false_texts]
    false_last_reported = false_done
    false_last_checkpoint = false_done
    false_started = time.perf_counter()
    for batch_pairs in _batched_statement_pairs_after_offset(
        false_statements,
        false_encodings,
        batch_size,
        length_bucketed=length_bucketed,
        offset=false_done,
    ):
        batch = [stmt for stmt, _encoding in batch_pairs]
        batch_encodings = None if false_encodings is None else [encoding for _stmt, encoding in batch_pairs]
        reps_batch = batched_statement_reps(
            model,
            tokenizer,
            batch,
            layers,
            device,
            max_length,
            compute_answer_metrics=False,
            hidden_state_capture=hidden_state_capture,
            encoded_statements=batch_encodings,
        )
        for reps in reps_batch:
            if reps is None:
                continue
            n_false += 1
            for layer in layers:
                h = reps["last"][layer]
                false_state_lists[layer].append(h)
                false_sums[layer] = h if false_sums[layer] is None else false_sums[layer] + h
        false_done += len(batch)
        if _progress_report_due(false_done, len(false_statements), progress_every, false_last_reported):
            false_last_reported = min(false_done, len(false_statements))
            print(_format_progress("warmup false", false_last_reported, len(false_statements),
                                   time.perf_counter() - false_started))
        if checkpoint_path_obj and _progress_report_due(
            false_done, len(false_statements), checkpoint_every, false_last_checkpoint
        ):
            false_last_checkpoint = min(false_done, len(false_statements))
            save_warmup_checkpoint(
                checkpoint_path_obj,
                metadata=checkpoint_metadata,
                manifolds=manifolds,
                true_state_lists=true_state_lists,
                false_state_lists=false_state_lists,
                false_sums=false_sums,
                n_false=n_false,
                true_done=true_done,
                false_done=false_last_checkpoint,
            )
            print(f"   saved warmup checkpoint: {checkpoint_path_obj} "
                  f"(true={true_done}/{len(true_texts)}, false={false_last_checkpoint}/{len(false_statements)})")

    subspaces = {}
    for layer in layers:
        m = manifolds[layer]
        if n_false > 0 and m.mean is not None:
            m.false_mean = (false_sums[layer] / n_false).to(torch.float32)
            raw = m.mean - m.false_mean
            m.contrastive_direction = raw / torch.norm(raw).clamp(min=1e-8)

        true_states = true_state_lists[layer]
        false_states = false_state_lists[layer]
        if len(true_states) >= 2 and false_states:
            subspaces[layer] = TruthSubspace.fit_contrastive(
                torch.stack(true_states), torch.stack(false_states), rank=subspace_rank
            )
        elif len(true_states) >= 2:
            subspaces[layer] = TruthSubspace.fit(torch.stack(true_states), rank=subspace_rank)
    return manifolds, subspaces


# ---------------------------------------------------------------------------
# 主流程 / Main
# ---------------------------------------------------------------------------

def run(args) -> dict:
    # Windows 控制台默认 cp1252，确保非 ASCII 不崩溃；行缓冲让进度实时可见（被杀也不丢日志）
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    profile: dict[str, float] = {}
    total_started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    with _profile_phase(profile, "load_data"):
        if args.offline:
            manifold_true, manifold_false, eval_stmts = load_offline()
            print("[!] OFFLINE SMOKE MODE - pipeline check only, NOT a benchmark.\n")
        else:
            manifold_true, manifold_false, eval_stmts = load_truthfulqa(
                args.manifold_questions, args.limit
            )

    stats_cache_path = Path(args.layer_stats_cache) if args.layer_stats_cache else None
    eval_reps_cache_path = Path(args.eval_reps_cache) if args.eval_reps_cache else None
    statement_encoding_cache_path = Path(args.statement_encoding_cache) if args.statement_encoding_cache else None
    warmup_checkpoint_path = Path(args.warmup_checkpoint) if args.warmup_checkpoint else None
    true_encodings: list[Optional[StatementEncoding]] | None = None
    false_encodings: list[Optional[StatementEncoding]] | None = None
    eval_encodings: list[Optional[StatementEncoding]] | None = None
    model = None
    tokenizer = None
    if args.cache_only:
        if stats_cache_path is None or eval_reps_cache_path is None:
            raise ValueError("cache-only mode requires both layer-stats and eval-reps caches.")
        device = torch.device("cpu")
        with _profile_phase(profile, "read_cache_metadata"):
            stats_meta = _read_cache_metadata(stats_cache_path)
            eval_meta = _read_cache_metadata(eval_reps_cache_path)
        if stats_meta.get("n_layers") != eval_meta.get("n_layers"):
            raise ValueError("cache-only mode requires layer-stats and eval-reps caches with matching n_layers.")
        n_layers = int(stats_meta["n_layers"])
        print("Cache-only scoring: skipping model load and forced-answer forward.")
    else:
        print(f"Loading {args.model} on {device} (dtype={args.dtype}) ...")
        with _profile_phase(profile, "load_model"):
            model, tokenizer = load_model(args.model, device, args.dtype)
        n_layers = int(model.config.num_hidden_layers)

    args.layer = resolve_target_layer(args.layer, n_layers, offline=args.offline)
    # --sweep: 主层在前；--sweep-layers 可限制候选层带，避免大模型全层后处理成本。
    layers = _resolve_sweep_layers(
        args.layer,
        n_layers,
        sweep=args.sweep,
        sweep_layers=args.sweep_layers,
    )
    if model is not None and args.hidden_state_capture == "hooks":
        _hook_capture_layer_map(model, layers)
        print("Using hook-based hidden-state capture for selected non-final layers.")

    if statement_encoding_cache_path is not None and not args.cache_only:
        statement_encoding_cache_metadata = _statement_encoding_cache_metadata(
            args,
            true_texts=manifold_true,
            false_texts=manifold_false,
            eval_statements=eval_stmts,
        )
        if statement_encoding_cache_path.exists() and not args.refresh_statement_encoding_cache:
            with _profile_phase(profile, "load_statement_encoding_cache"):
                true_encodings, false_encodings, eval_encodings, _ = load_statement_encoding_cache(
                    statement_encoding_cache_path,
                    expected_metadata=statement_encoding_cache_metadata,
                )
            print(f"   loaded statement encoding cache: {statement_encoding_cache_path}")
        else:
            with _profile_phase(profile, "tokenize_statements"):
                true_encodings = _batch_statement_encodings(
                    tokenizer,
                    [Statement("", text, 0) for text in manifold_true],
                    args.max_length,
                )
                false_encodings = _batch_statement_encodings(
                    tokenizer,
                    [Statement("", text, 1) for text in manifold_false],
                    args.max_length,
                )
                eval_encodings = _batch_statement_encodings(tokenizer, eval_stmts, args.max_length)
            with _profile_phase(profile, "save_statement_encoding_cache"):
                save_statement_encoding_cache(
                    statement_encoding_cache_path,
                    metadata=statement_encoding_cache_metadata,
                    true_encodings=true_encodings,
                    false_encodings=false_encodings,
                    eval_encodings=eval_encodings,
                )
            print(f"   saved statement encoding cache: {statement_encoding_cache_path}")

    print(f"Building per-layer truth stats from {len(manifold_true)} true / "
          f"{len(manifold_false)} false statements ({len(layers)} layer(s)) ...")
    stats_cache_metadata = _layer_stats_cache_metadata(
        args,
        layers=layers,
        n_layers=n_layers,
        true_texts=manifold_true,
        false_texts=manifold_false,
    )
    if stats_cache_path and stats_cache_path.exists() and not args.refresh_layer_stats_cache:
        with _profile_phase(profile, "load_layer_stats_cache"):
            manifolds, subspaces, _ = load_layer_stats_cache(
                stats_cache_path,
                expected_metadata=stats_cache_metadata,
                device=device,
            )
        print(f"   loaded layer stats cache: {stats_cache_path}")
    else:
        with _profile_phase(profile, "build_layer_stats"):
            manifolds, subspaces = build_layer_stats(
                model, tokenizer, manifold_true, manifold_false, layers, device, args.max_length,
                args.subspace_rank, args.batch_size, args.length_bucketed_batches,
                progress_every=args.progress_every,
                checkpoint_path=warmup_checkpoint_path,
                checkpoint_metadata=stats_cache_metadata,
                resume_checkpoint=not args.refresh_layer_stats_cache,
                checkpoint_every=args.warmup_checkpoint_every,
                hidden_state_capture=args.hidden_state_capture,
                true_encodings=true_encodings,
                false_encodings=false_encodings,
            )
        if stats_cache_path:
            with _profile_phase(profile, "save_layer_stats_cache"):
                save_layer_stats_cache(
                    stats_cache_path,
                    manifolds,
                    subspaces,
                    metadata=stats_cache_metadata,
                )
            print(f"   saved layer stats cache: {stats_cache_path}")
    primary = manifolds[args.layer]
    if not primary.is_ready():
        print("[X] Manifold not ready (need >=2 statements). Aborting.")
        sys.exit(1)
    print(f"   manifold: n={primary.n}, hidden_dim={primary.hidden_dim}, "
          f"contrastive_direction={'yes' if primary.contrastive_direction is not None else 'no'}  "
          f"subspace={'yes' if args.layer in subspaces else 'no'}\n")

    signals = _enabled_signals(args)
    sweep_signal_names = _sweep_signal_names(args)
    scores: dict[str, List[float]] = {s: [] for s in signals}
    sweep_scores: dict = {
        layer: {signal: [] for signal in sweep_signal_names} for layer in layers
    }
    labels: List[int] = []
    scored_statements: list[dict[str, object]] = []
    inside_sampled: List[bool] = []
    inside_triggered_total = 0
    inside_skipped_total = 0

    print(f"Scoring {len(eval_stmts)} eval statements ...")
    scored = 0
    eval_batch_pairs = list(_batched_statement_pairs(
        eval_stmts,
        eval_encodings,
        args.batch_size,
        length_bucketed=args.length_bucketed_batches,
    ))
    eval_batches = [[stmt for stmt, _encoding in batch_pairs] for batch_pairs in eval_batch_pairs]
    expected_eval_records = sum(len(batch) for batch in eval_batches)
    eval_reps_cache_metadata = _eval_reps_cache_metadata(
        args,
        layers=layers,
        n_layers=n_layers,
        eval_statements=eval_stmts,
    )
    eval_reps_reader: EvalRepsCacheReader | None = None
    eval_reps_writer: EvalRepsCacheWriter | None = None
    new_eval_reps: list[Optional[Mapping]] = []
    if eval_reps_cache_path and eval_reps_cache_path.exists() and not args.refresh_eval_reps_cache:
        with _profile_phase(profile, "load_eval_reps_cache"):
            eval_reps_reader = EvalRepsCacheReader(
                eval_reps_cache_path,
                expected_metadata=eval_reps_cache_metadata,
                expected_records=expected_eval_records,
            )
        print(f"   loaded eval reps cache: {eval_reps_cache_path}")
    elif eval_reps_cache_path:
        write_shard_size = int(args.eval_reps_cache_shard_size)
        if write_shard_size <= 0 and args.refresh_eval_reps_cache:
            write_shard_size = _existing_eval_reps_shard_size(eval_reps_cache_path)
        if write_shard_size > 0:
            with _profile_phase(profile, "init_eval_reps_cache_writer"):
                eval_reps_writer = EvalRepsCacheWriter(
                    eval_reps_cache_path,
                    metadata=eval_reps_cache_metadata,
                    shard_size=write_shard_size,
                )

    eval_reps_offset = 0
    eval_last_reported = 0
    eval_started = time.perf_counter()
    for batch_idx, (batch, batch_pairs) in enumerate(zip(eval_batches, eval_batch_pairs)):
        if eval_reps_reader is not None:
            with _profile_phase(profile, "read_eval_reps_cache_batch"):
                reps_batch = eval_reps_reader.read_range(eval_reps_offset, len(batch))
        else:
            batch_encodings = None if eval_encodings is None else [encoding for _stmt, encoding in batch_pairs]
            with _profile_phase(profile, "forced_answer_forward"):
                reps_batch = batched_statement_reps(
                    model,
                    tokenizer,
                    batch,
                    layers,
                    device,
                    args.max_length,
                    eigenscore_alpha=args.eigenscore_alpha,
                    hidden_state_capture=args.hidden_state_capture,
                    encoded_statements=batch_encodings,
                )
            if eval_reps_writer is not None:
                with _profile_phase(profile, "write_eval_reps_cache_batch"):
                    eval_reps_writer.extend(reps_batch)
            elif eval_reps_cache_path:
                new_eval_reps.extend(reps_batch)
        eval_reps_offset += len(batch)

        with _profile_phase(profile, "score_postprocess"):
            batch_records = _score_reps_batch(
                batch,
                reps_batch,
                layers=layers,
                target_layer=args.layer,
                manifolds=manifolds,
                subspaces=subspaces,
            )

        if _inside_enabled(args):
            if _inside_trigger_enabled(args):
                triggered = _inside_trigger_indexes(batch_records, args)
            else:
                triggered = set(range(len(batch_records)))
            inside_triggered_total += len(triggered)
            inside_skipped_total += len(batch_records) - len(triggered)

            triggered_positions = [idx for idx in range(len(batch_records)) if idx in triggered]
            for inside_batch_idx, position_batch in enumerate(_chunked(triggered_positions, args.inside_batch_size)):
                inside_batch = [batch_records[position]["stmt"] for position in position_batch]
                with _profile_phase(profile, "inside_generation"):
                    sampled_batch = sampled_inside_scores_batch(
                        model,
                        tokenizer,
                        inside_batch,
                        layers,
                        device,
                        args.max_length,
                        n_samples=args.inside_samples,
                        max_new_tokens=args.inside_max_new_tokens,
                        temperature=args.inside_temperature,
                        top_p=args.inside_top_p,
                        pooling=args.inside_pooling,
                        seed=_inside_seed(args.seed, batch_idx, inside_batch_idx),
                        eigenscore_alpha=args.eigenscore_alpha,
                        hidden_state_capture=args.hidden_state_capture,
                    )
                for position, sampled in zip(position_batch, sampled_batch):
                    batch_records[position]["inside_scores"] = sampled
                    batch_records[position]["inside_sampled"] = sampled is not None

            for position, record in enumerate(batch_records):
                if position not in triggered:
                    record["inside_scores"] = _empty_inside_scores(layers)

        with _profile_phase(profile, "score_postprocess"):
            for record in batch_records:
                inside_scores = record["inside_scores"]
                if _inside_enabled(args) and inside_scores is None:
                    continue

                for layer in layers:
                    layer_scores = record["layer_scores"][layer]
                    sweep_scores[layer]["maha_last"].append(layer_scores["maha_last"])
                    sweep_scores[layer]["truth_proj"].append(layer_scores["truth_proj"])
                    sweep_scores[layer]["subspace_resid"].append(layer_scores["subspace_resid"])
                    sweep_scores[layer]["eigenscore"].append(layer_scores["eigenscore"])
                    if inside_scores is not None:
                        sweep_scores[layer][INSIDE_SIGNAL].append(float(inside_scores[layer]))

                primary_scores = record["primary_scores"]
                scores["maha_last"].append(primary_scores["maha_last"])
                scores["truth_proj"].append(primary_scores["truth_proj"])
                scores["subspace_resid"].append(primary_scores["subspace_resid"])
                scores["disp_euclid"].append(primary_scores["disp_euclid"])
                scores["disp_hse"].append(primary_scores["disp_hse"])
                scores["eigenscore"].append(primary_scores["eigenscore"])
                if inside_scores is not None:
                    scores[INSIDE_SIGNAL].append(sweep_scores[args.layer][INSIDE_SIGNAL][-1])
                scores["nll_answer"].append(primary_scores["nll_answer"])
                labels.append(record["stmt"].is_false)
                scored_statements.append(_statement_to_dump(record["stmt"]))
                if _inside_enabled(args):
                    inside_sampled.append(bool(record["inside_sampled"]))
                scored += 1

                if _progress_report_due(scored, len(eval_stmts), args.progress_every, eval_last_reported):
                    eval_last_reported = min(scored, len(eval_stmts))
                    print(_format_progress("eval", eval_last_reported, len(eval_stmts),
                                           time.perf_counter() - eval_started))

    if eval_reps_writer is not None:
        with _profile_phase(profile, "save_eval_reps_cache"):
            eval_reps_writer.close()
        print(f"   saved sharded eval reps cache: {eval_reps_cache_path}")
    elif eval_reps_cache_path and eval_reps_reader is None:
        with _profile_phase(profile, "save_eval_reps_cache"):
            save_eval_reps_cache(
                eval_reps_cache_path,
                new_eval_reps,
                metadata=eval_reps_cache_metadata,
            )
        print(f"   saved eval reps cache: {eval_reps_cache_path}")

    with _profile_phase(profile, "reporting"):
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        results = {s: roc_auc(scores[s], labels) for s in signals}
        selective = _selective_reports(scores, labels, alpha=REPORT_ALPHA)

    # ---- 输出 ----
    print("\n" + "=" * 56)
    print("  AUROC  (positive = false/hallucinated statement)")
    print(f"  model={args.model}  layer={args.layer}  n_pos={n_pos}  n_neg={n_neg}")
    print("=" * 56)
    print(f"  {'signal':<18}{'AUROC':>10}   interpretation")
    print("  " + "-" * 52)
    for s in signals:
        print(f"  {s:<18}{results[s]:>10.3f}")
    print("  " + "-" * 52)
    # 关键对比 / key comparisons
    if not (results["disp_hse"] != results["disp_hse"]):  # not NaN
        delta = results["disp_hse"] - results["disp_euclid"]
        verdict = "hyperbolic HELPS" if delta > 0.01 else (
            "hyperbolic HURTS" if delta < -0.01 else "no meaningful difference")
        print(f"  disp_hse - disp_euclid = {delta:+.3f}  ->  {verdict}")
    geo = max(results[s] for s in signals if s != "nll_answer")
    if not (results["nll_answer"] != results["nll_answer"]):
        print(f"  best geometry ({geo:.3f}) vs nll baseline ({results['nll_answer']:.3f})  ->  "
              f"{'geometry wins' if geo > results['nll_answer'] + 0.01 else 'baseline competitive'}")
    print("=" * 56)

    sweep_payload = None
    with _profile_phase(profile, "sweep_reporting"):
        if _sweep_output_enabled(args):
            sweep_payload = {}
            print("\n  Layer sweep (AUROC):")
            header = f"  {'layer':>6}" + "".join(f" {name:>17}" for name in sweep_signal_names)
            print(header)
            for layer in sorted(layers):
                layer_payload = {
                    signal: roc_auc(sweep_scores[layer][signal], labels)
                    for signal in sweep_signal_names
                }
                sweep_payload[str(layer)] = layer_payload
                values = "".join(f" {layer_payload[signal]:>17.3f}" for signal in sweep_signal_names)
                print(f"  {layer:>6}{values}")

    payload = {
        "config": {"model": args.model, "layer": args.layer, "offline": args.offline,
                   "manifold_n": primary.n, "n_manifold_false": len(manifold_false),
                   "hidden_dim": primary.hidden_dim, "subspace_rank": args.subspace_rank,
                   "n_pos": n_pos, "n_neg": n_neg, "seed": args.seed,
                   "eigenscore_alpha": args.eigenscore_alpha,
                   "batch_size": args.batch_size,
                   "inside_samples": args.inside_samples,
                   "inside_batch_size": args.inside_batch_size,
                   "inside_max_new_tokens": args.inside_max_new_tokens,
                   "inside_temperature": args.inside_temperature,
                   "inside_top_p": args.inside_top_p,
                   "inside_pooling": args.inside_pooling,
                   "inside_trigger_signal": args.inside_trigger_signal,
                   "inside_trigger_threshold": args.inside_trigger_threshold,
                   "inside_trigger_top_fraction": args.inside_trigger_top_fraction,
                   "length_bucketed_batches": args.length_bucketed_batches,
                   "hidden_state_capture": args.hidden_state_capture,
                   "cache_only": args.cache_only,
                   "statement_encoding_cache": args.statement_encoding_cache,
                   "refresh_statement_encoding_cache": args.refresh_statement_encoding_cache,
                   "layer_stats_cache": args.layer_stats_cache,
                   "refresh_layer_stats_cache": args.refresh_layer_stats_cache,
                   "eval_reps_cache": args.eval_reps_cache,
                   "eval_reps_cache_shard_size": args.eval_reps_cache_shard_size,
                   "refresh_eval_reps_cache": args.refresh_eval_reps_cache,
                   "progress_every": args.progress_every,
                   "warmup_checkpoint": args.warmup_checkpoint,
                   "warmup_checkpoint_every": args.warmup_checkpoint_every,
                   "sweep_layers": layers if (args.sweep or args.sweep_layers) else None},
        "auroc": results,
        "selective": selective,
        "sweep": sweep_payload,
    }
    if _inside_enabled(args):
        payload["inside_sampling"] = {
            "mode": "triggered" if _inside_trigger_enabled(args) else "all",
            "signal": args.inside_trigger_signal,
            "threshold": args.inside_trigger_threshold,
            "top_fraction": args.inside_trigger_top_fraction,
            "sampled": int(sum(inside_sampled)),
            "not_sampled": int(len(inside_sampled) - sum(inside_sampled)),
            "triggered": int(inside_triggered_total),
            "skipped_by_trigger": int(inside_skipped_total),
            "fill_value_for_untriggered": 0.0 if _inside_trigger_enabled(args) else None,
        }
    if _profile_requested(args):
        payload["profile"] = _profile_payload(
            profile,
            time.perf_counter() - total_started,
            n_eval_records=len(labels),
            n_warmup_true=len(manifold_true),
            n_warmup_false=len(manifold_false),
        )
        print("\n  Profile timings (seconds):")
        for name, seconds in payload["profile"]["phases"].items():
            print(f"  {name:<24}{seconds:>10.3f}")
        print(f"  {'total':<24}{payload['profile']['total_seconds']:>10.3f}")
        summary = payload["profile"]["summary"]
        bottleneck = summary.get("bottleneck")
        if bottleneck:
            print(f"  {'bottleneck':<24}{bottleneck:>10}")
        model_forward = summary["groups"]["model_forward"]
        if model_forward["seconds"] > 0:
            print(f"  {'model_forward_share':<24}{model_forward['share']:>10.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote structured results to {args.json}")
    if args.profile_json:
        profile_data = payload.get(
            "profile",
            _profile_payload(
                profile,
                time.perf_counter() - total_started,
                n_eval_records=len(labels),
                n_warmup_true=len(manifold_true),
                n_warmup_false=len(manifold_false),
            ),
        )
        with open(args.profile_json, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)
        print(f"\nWrote profile timings to {args.profile_json}")
    if args.dump_scores:
        # 逐陈述原始分数：供共形校准等后处理复用，无需再跑模型
        # Raw per-statement scores: enables post-hoc analyses (e.g. conformal
        # calibration) without re-running the model
        dump = {"config": payload["config"], "labels": labels, "scores": scores, "statements": scored_statements}
        if _inside_enabled(args):
            dump["inside_sampled"] = inside_sampled
            dump["inside_sampling"] = payload["inside_sampling"]
        if _sweep_output_enabled(args):
            dump["sweep_scores"] = {str(layer): sweep_scores[layer] for layer in layers}
        with open(args.dump_scores, "w", encoding="utf-8") as f:
            json.dump(dump, f)
        print(f"Dumped raw per-statement scores to {args.dump_scores}")
    print("\nJSON:", json.dumps(payload["auroc"]))
    return payload


def main():
    p = argparse.ArgumentParser(description="EigenTruth TruthfulQA AUROC benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="float32", choices=list(_DTYPES),
                   help="model weight dtype; bfloat16 halves memory on low-RAM machines")
    p.add_argument("--layer", type=int, default=-8, help="target layer index (negative ok)")
    p.add_argument("--sweep", action="store_true",
                   help="score geometry signals at every layer; with default hidden-state capture, "
                        "one forward pass returns all hidden states")
    p.add_argument("--sweep-layers", default=None,
                   help="comma-list of layer indexes to score in addition to --layer; "
                        "limits sweep cost and implies a sweep payload")
    p.add_argument("--limit", type=int, default=200, help="max eval questions (0 = all)")
    p.add_argument("--manifold-questions", type=int, default=80,
                   help="held-out questions whose correct answers build the manifold")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1,
                   help="forced-answer forward batch size; increase for faster benchmarks if memory allows")
    p.add_argument("--length-bucketed-batches", action="store_true",
                   help="sort statements by approximate text length before batching to reduce padding")
    p.add_argument("--hidden-state-capture", default="outputs", choices=HIDDEN_STATE_CAPTURE_METHODS,
                   help="how forced-answer forwards collect hidden states: 'outputs' preserves exact "
                        "HF output_hidden_states semantics; 'hooks' stores only selected non-final "
                        "decoder-layer states and can reduce memory pressure")
    p.add_argument("--subspace-rank", type=int, default=2,
                   help="rank for TruthSubspace residual scoring")
    p.add_argument("--eigenscore-alpha", type=float, default=1e-3,
                   help="regularization alpha for EigenScore-style log-det scores")
    p.add_argument("--inside-samples", type=int, default=0,
                   help="enable multi-sample INSIDE proxy with this many sampled continuations; "
                        "0 disables it, values >=2 enable inside_eigenscore")
    p.add_argument("--inside-batch-size", type=int, default=1,
                   help="number of prompts to sample in one generate() call for --inside-samples")
    p.add_argument("--inside-max-new-tokens", type=int, default=12,
                   help="max sampled continuation length for --inside-samples")
    p.add_argument("--inside-temperature", type=float, default=0.7,
                   help="sampling temperature for --inside-samples")
    p.add_argument("--inside-top-p", type=float, default=0.9,
                   help="top-p sampling cutoff for --inside-samples")
    p.add_argument("--inside-pooling", default="last", choices=("last", "mean"),
                   help="sentence embedding pooling for sampled continuations")
    p.add_argument("--inside-trigger-signal", default=None, choices=SIGNALS,
                   help="optional cheap primary-layer signal used to decide which statements receive "
                        "sampled INSIDE scoring")
    p.add_argument("--inside-trigger-threshold", type=float, default=None,
                   help="sample INSIDE only when --inside-trigger-signal crosses this anomaly threshold")
    p.add_argument("--inside-trigger-top-fraction", type=float, default=None,
                   help="sample INSIDE for the most anomalous fraction of each eval batch according to "
                        "--inside-trigger-signal")
    p.add_argument("--offline", action="store_true",
                   help="use bundled smoke statements (pipeline check, not a benchmark)")
    p.add_argument("--json", default=None, help="optional path to write structured results")
    p.add_argument("--profile", action="store_true",
                   help="include phase timing diagnostics in stdout and --json output")
    p.add_argument("--profile-json", default=None,
                   help="optional path to write only phase timing diagnostics")
    p.add_argument("--progress-every", type=int, default=50,
                   help="print warmup/eval progress every N statements; use 0 to disable")
    p.add_argument("--cache-only", action="store_true",
                   help="score only from --layer-stats-cache and --eval-reps-cache; skip model loading")
    p.add_argument("--statement-encoding-cache", default=None,
                   help="optional JSON path to load or create cached statement token ids and answer spans")
    p.add_argument("--refresh-statement-encoding-cache", action="store_true",
                   help="rebuild and overwrite --statement-encoding-cache instead of loading it")
    p.add_argument("--layer-stats-cache", default=None,
                   help="optional .pt path to load or create cached warmup manifolds/subspaces")
    p.add_argument("--refresh-layer-stats-cache", action="store_true",
                   help="rebuild and overwrite --layer-stats-cache instead of loading it; "
                        "also ignores any existing --warmup-checkpoint")
    p.add_argument("--warmup-checkpoint", default=None,
                   help="optional .pt path for resumable warmup checkpoint state")
    p.add_argument("--warmup-checkpoint-every", type=int, default=50,
                   help="save --warmup-checkpoint every N processed warmup statements; use 0 to disable writes")
    p.add_argument("--eval-reps-cache", default=None,
                   help="optional .pt path to load or create cached forced-answer hidden states/metrics")
    p.add_argument("--eval-reps-cache-shard-size", type=int, default=0,
                   help="write --eval-reps-cache as a sharded directory with this many records per shard")
    p.add_argument("--refresh-eval-reps-cache", action="store_true",
                   help="rebuild and overwrite --eval-reps-cache instead of loading it")
    p.add_argument("--dump-scores", default=None,
                   help="optional path to dump raw per-statement scores+labels "
                        "(enables post-hoc analyses, e.g. conformal calibration)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be >=1")
    if args.inside_batch_size < 1:
        p.error("--inside-batch-size must be >=1")
    if args.inside_samples == 1:
        p.error("--inside-samples must be 0 or >=2")
    if args.inside_trigger_signal and args.inside_samples < 2:
        p.error("--inside-trigger-signal requires --inside-samples >=2")
    if (args.inside_trigger_threshold is not None or args.inside_trigger_top_fraction is not None) \
            and not args.inside_trigger_signal:
        p.error("--inside-trigger-threshold/--inside-trigger-top-fraction require --inside-trigger-signal")
    if args.inside_trigger_threshold is not None and args.inside_trigger_top_fraction is not None:
        p.error("choose only one of --inside-trigger-threshold or --inside-trigger-top-fraction")
    if args.inside_trigger_signal and args.inside_trigger_threshold is None \
            and args.inside_trigger_top_fraction is None:
        p.error("--inside-trigger-signal requires a threshold or top fraction")
    if args.inside_trigger_top_fraction is not None and not (0.0 < args.inside_trigger_top_fraction <= 1.0):
        p.error("--inside-trigger-top-fraction must be in (0, 1]")
    if args.refresh_statement_encoding_cache and not args.statement_encoding_cache:
        p.error("--refresh-statement-encoding-cache requires --statement-encoding-cache")
    if args.refresh_layer_stats_cache and not args.layer_stats_cache:
        p.error("--refresh-layer-stats-cache requires --layer-stats-cache")
    if args.refresh_eval_reps_cache and not args.eval_reps_cache:
        p.error("--refresh-eval-reps-cache requires --eval-reps-cache")
    if args.eval_reps_cache_shard_size < 0:
        p.error("--eval-reps-cache-shard-size must be >=0")
    if args.eval_reps_cache_shard_size > 0 and not args.eval_reps_cache:
        p.error("--eval-reps-cache-shard-size requires --eval-reps-cache")
    if args.progress_every < 0:
        p.error("--progress-every must be >=0")
    if args.warmup_checkpoint_every < 0:
        p.error("--warmup-checkpoint-every must be >=0")
    if args.cache_only:
        if not args.layer_stats_cache or not args.eval_reps_cache:
            p.error("--cache-only requires --layer-stats-cache and --eval-reps-cache")
        if args.refresh_layer_stats_cache or args.refresh_eval_reps_cache:
            p.error("--cache-only cannot be combined with refresh cache flags")
        if args.refresh_statement_encoding_cache:
            p.error("--cache-only cannot refresh statement encodings because it skips tokenizer loading")
        if args.inside_samples:
            p.error("--cache-only cannot run sampled INSIDE; omit --inside-samples")
        if not Path(args.layer_stats_cache).exists():
            p.error("--cache-only layer stats cache does not exist")
        if not Path(args.eval_reps_cache).exists():
            p.error("--cache-only eval reps cache does not exist")
    run(args)


if __name__ == "__main__":
    main()
